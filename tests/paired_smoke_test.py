"""
Smoke test for the CROSS-AD paired pipeline. Outputs are SMOKE-TEST ONLY (synthetic
features) and go to paths.smoke_results; they are not results.

Checks
  1. pairing: paired_cohort.csv is one-to-one (MRI_ID, EEG_ID unique), labels are task
     classes, the T1 and EEG files exist, included rows match the manifest
  2. tensor shapes: every CrossAD mode returns (n, n_classes); gate weights sum to 1
  3. participant separation: synthetic data where some participants have 2-3 visits;
     every outer and inner split keeps all visits of a participant on one side
  4. one training/evaluation pass: 1 repeat x outer folds x all models, few epochs,
     metrics, confusion matrices, bootstrap and output files written
  5. if real feature tables exist: they merge with the cohort and have the expected shapes

  python -m tests.paired_smoke_test --config configs/paired_config.yaml
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml

from crossad.paired import paired_train as pt


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  PASS  {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paired_config.yaml")
    args = ap.parse_args()
    t0 = time.time()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    work, res = cfg["paths"]["work"], cfg["paths"]["results"]
    out = cfg["paths"]["smoke_results"]
    with open(os.path.join(res, "audit_summary.json")) as f:
        classes = json.load(f)["task_classes"]
    n_cls = len(classes)

    print("1. pairing")
    coh = pd.read_csv(os.path.join(work, "paired_cohort.csv"))
    man = pd.read_csv(os.path.join(work, "pairing_manifest.csv"))
    check(len(coh) > 0, f"paired cohort has {len(coh)} participants {coh['dx'].value_counts().to_dict()}")
    check(coh["mri_id"].is_unique and coh["eeg_id"].is_unique, "MRI_ID and EEG_ID one-to-one")
    check(coh["dx"].isin(classes).all(), f"all labels in task classes {classes}")
    check(coh["t1_path"].map(os.path.exists).all(), "every T1 file exists")
    check(coh["eeg_path"].map(os.path.exists).all(), "every EEG file exists")
    check(set(coh["mri_id"]) == set(man.loc[man["included"], "mri_id"]),
          "cohort equals manifest rows marked included")
    check((coh["eeg_records_dx"].fillna(coh["dx"]) == coh["dx"]).all(),
          "MRI-table and EEG-Records labels agree")

    print("2. tensor shapes")
    d_e, d_m = 26, 16
    if os.path.exists(os.path.join(work, "eeg_features.csv")) and \
       os.path.exists(os.path.join(work, "mri_features.csv")):
        df_real, ecols, mcols, dropped = pt.load_paired(work, classes)
        d_e, d_m = len(ecols), len(mcols)
        check(len(df_real) == len(coh) - len(dropped),
              f"real features merge: {len(df_real)} rows, EEG {d_e}, MRI {d_m}, dropped {len(dropped)}")
        check(not df_real[ecols + mcols].isna().all(axis=0).any(), "no all-missing feature column")
    hp = cfg["model"]
    for mode in ("eeg", "mri", "concat", "gate"):
        net = pt.CrossAD(mode, d_e, d_m, n_cls, hp).eval()
        o = net(torch.randn(7, d_e), torch.randn(7, d_m))
        check(tuple(o.shape) == (7, n_cls), f"CrossAD[{mode}] output {tuple(o.shape)}")
        if mode == "gate":
            check(torch.allclose(net.last_gate.sum(1), torch.ones(7)), "gate weights sum to 1")

    print("3. participant separation (synthetic multi-visit data)")
    rng = np.random.default_rng(0)
    rows = []
    for r in coh.itertuples():
        for v in range(1 + rng.integers(0, 3)):          # 1-3 synthetic visits per participant
            rows.append({"mri_id": f"{r.mri_id}_v{v}", "participant": r.mri_id, "dx": r.dx,
                         "age": r.age, "sex": r.sex})
    df = pd.DataFrame(rows)
    yy = df["dx"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    for j in range(d_e):
        df[f"e{j}"] = rng.normal(size=len(df)) + 0.8 * yy * (j < 3)
    for j in range(d_m):
        df[f"vol_m{j}"] = rng.normal(size=len(df)) + 0.8 * yy * (j < 3)
    df.loc[df.sample(frac=0.05, random_state=0).index, "e0"] = np.nan   # imputation path
    ecols = [f"e{j}" for j in range(d_e)]
    mcols = [f"vol_m{j}" for j in range(d_m)]
    X, y, groups = pt.arrays(df, ecols, mcols, classes)
    check(X["eeg"].shape == (len(df), d_e) and X["mri"].shape == (len(df), d_m),
          f"arrays: {len(df)} visits / {len(set(groups))} participants, EEG {X['eeg'].shape}, MRI {X['mri'].shape}")
    C = cfg["cv"]
    splits = pt.make_splits(y, groups, C["outer_folds"], 2, C["seed"])
    for rep, k, tr, te in splits:
        assert not set(groups[tr]) & set(groups[te])
        for itr, ite in pt.inner_splits(y[tr], groups[tr], C["inner_folds"], 0):
            assert not set(groups[tr][itr]) & set(groups[tr][ite])
    check(True, f"{len(splits)} outer splits (+ inner splits): no participant on both sides")
    n_miss = sum(len(np.unique(y[te])) < n_cls for _, _, _, te in splits)
    print(f"  info  synthetic multi-visit folds missing a class: {n_miss}/{len(splits)}")
    y_real = coh["dx"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    real_splits = pt.make_splits(y_real, coh["mri_id"].to_numpy(), C["outer_folds"],
                                 C["repeats"], C["seed"])
    check(all(len(np.unique(y_real[te])) == n_cls for _, _, _, te in real_splits),
          f"real cohort ({len(real_splits)} outer splits): every test fold has every class")

    print("4. one training/evaluation pass")
    scfg = copy.deepcopy(cfg)
    scfg["model"].update(epochs=5, weight_decay=[0.01, 0.1], logreg_C=[0.1, 1.0])
    scfg["cv"].update(repeats=1, bootstrap=50)
    scfg["_n_eeg"], scfg["_n_mri"] = d_e, d_m
    splits1 = [s for s in splits if s[0] == 0]
    preds, gates, chosen = pt.run_cv(X, y, groups, df["mri_id"].to_numpy(), splits1, scfg,
                                     n_cls, list(pt.LOGREG) + list(pt.NN), log=lambda *_: None)
    check(len(preds) == len(df) * (len(pt.LOGREG) + len(pt.NN)),
          f"one out-of-fold prediction per visit per model ({len(preds)})")
    pc = [f"p{c}" for c in range(n_cls)]
    check(np.allclose(preds[pc].sum(1), 1), "probabilities sum to 1")
    summ, boot, info = pt.evaluate_and_save(preds, gates, chosen,
                                            pt.splits_frame(splits1, df, y, classes),
                                            df, scfg, classes, out, work, "smoke_", t0)
    info.update(tag="SMOKE TEST - synthetic features, not a result")
    with open(os.path.join(out, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(out, "README_SMOKE.txt"), "w") as f:
        f.write("SMOKE TEST OUTPUT. Features are synthetic (real participant IDs/labels, "
                "1-3 fake visits each, signal planted in 3 columns). Not a result.\n")
    for fn in ("metrics_summary.csv", "confusion_matrices.csv", "fusion_vs_single.csv",
               "gate_weights_summary.csv", "split_summary.csv", "config_used.yaml"):
        check(os.path.exists(os.path.join(out, fn)), f"wrote {out}/{fn}")
    print("\n" + summ[["macro_f1_mean", "balanced_accuracy_mean"]].round(3).to_string())
    print(f"\nSMOKE TEST PASSED in {time.time() - t0:.0f}s (synthetic features; not a result)")


if __name__ == "__main__":
    main()
