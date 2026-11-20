"""
CROSS-AD pretrain-then-fuse ("model 1").

Stage 1  pretrain one encoder per modality on its unpaired BrainLat pool
         MRI pool: CN/AD participants with a T1 (16 SynthSeg AD-signature volumes / ICV)
         EEG pool: CN/AD resting-state recordings (26 region-level spectral features)
         Encoder = paired_train.Encoder (Linear-ReLU-Dropout-Linear-ReLU) + linear head,
         balanced cross-entropy; weight decay chosen on a stratified validation split of
         the pool, then refitted on the whole pool.
Stage 2  freeze both encoders, embed the 25 paired participants, fit a logistic-regression
         head on the paired training fold only, score the paired test fold.

Leakage control: the outer splits are exactly those of paired_train.py (checked). For each
outer split the test participants are removed from BOTH pools before pretraining, so no test
participant's MRI or EEG ever reaches an encoder, scaler or head. Paired training
participants stay in the pools.

Models (all scored on the same 10 x 5-fold outer splits as the paired-only run)
  pt_mri_head        pretrained MRI encoder + its own head, no paired fitting
  pt_eeg_head        pretrained EEG encoder + its own head, no paired fitting
  pt_late_avg        mean of the two pretrained heads' probabilities, no paired fitting
  pt_mri_lr          frozen MRI embedding  -> logistic regression on paired train
  pt_eeg_lr          frozen EEG embedding  -> logistic regression on paired train
  pt_fusion_concat   frozen [EEG | MRI] embeddings -> logistic regression   (main model)
  pt_fusion_stack    [EEG head logit, MRI head logit] -> logistic regression
  pt_fusion_concat@<pool>, pt_mri_head@<pool>   MRI pool variants (site confounding check)
Compared with the paired-only models (paired_train.py oof_predictions.csv) by the same
participant-level bootstrap.

  python -m crossad.pretrain.pretrain_train --config configs/pretrain_config.yaml
"""
import argparse
import json
import os
import platform
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from crossad.paired.paired_train import (CrossAD, bootstrap_diff, confusion, inner_splits, load_paired,
                          macro_f1, make_splits, per_repeat)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

PAIRED_ONLY = ["logreg_eeg", "logreg_mri", "logreg_concat", "mlp_eeg", "mlp_mri",
               "fusion_concat", "fusion_gate"]
COMPARISONS = [
    ("pt_fusion_concat", "fusion_concat"),     # does pretraining help fusion?
    ("pt_fusion_concat", "logreg_concat"),
    ("pt_fusion_concat", "pt_mri_lr"),         # does fusion help, given pretraining?
    ("pt_fusion_concat", "pt_eeg_lr"),
    ("pt_fusion_concat", "mlp_mri"),           # best paired-only single modality
    ("pt_mri_lr", "mlp_mri"),                  # does pretraining help each modality?
    ("pt_eeg_lr", "mlp_eeg"),
    ("pt_late_avg", "fusion_concat"),
    ("pt_fusion_stack", "pt_fusion_concat"),
]


# ------------------------------------------------------------------ data
def load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    with open(cfg["paired_config"]) as f:
        cfg["paired"] = yaml.safe_load(f)
    return cfg


def load_pools(work, ecols, mcols):
    mp = pd.read_csv(os.path.join(work, "mri_pool.csv"))
    mf = pd.read_csv(os.path.join(work, "mri_pool_features.csv"))
    mri = mp.merge(mf[["mri_id"] + mcols], on="mri_id", how="inner")
    ep = pd.read_csv(os.path.join(work, "eeg_pool.csv"))
    ef = pd.read_csv(os.path.join(work, "eeg_pool_features.csv")) \
           .rename(columns={"mri_id": "participant"})
    eeg = ep.merge(ef[["participant"] + ecols], on="participant", how="inner")
    lost = {"mri": len(mp) - len(mri), "eeg": len(ep) - len(eeg)}
    return mri, eeg, lost


def check_paired_features(df, mri, eeg, ecols, mcols):
    """Pool features of the paired participants must equal the paired run's features."""
    out = {}
    for name, pool, cols, key in (("mri", mri, mcols, "mri_id"), ("eeg", eeg, ecols, "participant")):
        a = df.set_index("mri_id")[cols]
        b = pool.set_index(key).loc[a.index, cols]
        out[name] = float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy())))
    return out


def near_duplicates(df, mri, mcols, r_thr=0.999):
    """Unpaired MRI pool scans whose regional profile is ~identical to a paired scan."""
    pool = mri[~mri["paired"]]
    P = StandardScaler().fit(mri[mcols]).transform
    A, B = P(df[mcols]), P(pool[mcols])
    A = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
    B = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
    R = A @ B.T / A.shape[1]
    hits = [{"paired": df["mri_id"].iat[i], "pool": pool["mri_id"].iat[j], "r": R[i, j]}
            for i, j in zip(*np.where(R > r_thr))]
    return hits, float(R.max())


# ------------------------------------------------------------------ stage 1: pretraining
def train_encoder(X, y, mode, hp, wd, seed):
    torch.manual_seed(seed)
    d = X.shape[1]
    model = CrossAD(mode, d, d, 2, hp)
    cnt = np.bincount(y, minlength=2).astype(float)
    w = torch.tensor(len(y) / (2 * np.maximum(cnt, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=wd)
    xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
    xe, xm = (xt, None) if mode == "eeg" else (None, xt)
    model.train()
    for _ in range(hp["epochs"]):
        opt.zero_grad()
        loss_fn(model(xe, xm), yt).backward()
        opt.step()
    return model.eval()


class Frozen:
    """Preprocessing + pretrained encoder + head, fitted on one pool, never updated again."""

    def __init__(self, X, y, mode, hp, seed):
        self.mode = mode
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X)
        Z = self.pre.transform(X)
        sss = StratifiedShuffleSplit(1, test_size=hp["val_fraction"], random_state=seed)
        tr, va = next(sss.split(Z, y))
        self.val = {}
        for wd in hp["weight_decay"]:
            m = train_encoder(Z[tr], y[tr], mode, hp, wd, seed)
            self.val[wd] = macro_f1(y[va], self._prob(m, Z[va]))
        # ties go to the larger weight decay (stronger regularization)
        self.wd = max(sorted(self.val, reverse=True), key=lambda k: self.val[k])
        self.model = train_encoder(Z, y, mode, hp, self.wd, seed)
        self.n = len(y)

    def _enc(self, m):
        return m.enc_eeg if self.mode == "eeg" else m.enc_mri

    def _prob(self, m, Z):
        with torch.no_grad():
            return torch.softmax(m.head(self._enc(m)(torch.tensor(Z, dtype=torch.float32))),
                                 1).numpy()

    def embed(self, X):
        Z = torch.tensor(self.pre.transform(X), dtype=torch.float32)
        with torch.no_grad():
            e = self._enc(self.model)(Z)
            return e.numpy(), self.model.head(e).numpy()          # embedding, logits

    def prob(self, X):
        return self._prob(self.model, self.pre.transform(X))


# ------------------------------------------------------------------ stage 2: paired head
def fit_head(F, y, tr, te, groups, Cs, inner_folds, seed):
    """Scaler + logistic regression on frozen features; C tuned on inner folds of tr."""
    def fit(a, b, C):
        p = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, class_weight="balanced", max_iter=5000, random_state=seed))
        return p.fit(F[a], y[a]).predict_proba(F[b])
    if len(Cs) > 1:
        inner = inner_splits(y[tr], groups[tr], inner_folds, seed)
        sc = [np.mean([macro_f1(y[tr][ib], fit(tr[ia], tr[ib], C)) for ia, ib in inner])
              for C in Cs]
        C = Cs[max(range(len(Cs)), key=lambda i: sc[i])]          # ties -> smallest C
    else:
        C = Cs[0]
    return fit(tr, te, C), C


# ------------------------------------------------------------------ main loop
def one_split(split, cfg, df, ecols, mcols, mri, eeg, y, groups):
    rep, k, tr, te = split
    H, classes = cfg["pretrain"], cfg["classes"]
    hp = {k_: H[k_] for k_ in ("emb_dim", "hidden", "dropout", "lr", "epochs", "weight_decay",
                               "val_fraction")}
    Cs = cfg["fusion"]["logreg_C"]
    inner_k = cfg["paired"]["cv"]["inner_folds"]
    cmap = {c: i for i, c in enumerate(classes)}
    Xe, Xm = df[ecols].to_numpy(float), df[mcols].to_numpy(float)
    preds, enc_log, chosen = [], [], []
    seed = cfg["paired"]["cv"]["seed"] + 100 * rep + k
    test_ids = set(df["mri_id"].iloc[te])
    # ---- stage 1: pretrain on the pools minus this split's test participants
    e_pool = eeg[~eeg["participant"].isin(test_ids)]
    enc = {"eeg": Frozen(e_pool[ecols].to_numpy(float), e_pool["dx"].map(cmap).to_numpy(),
                         "eeg", hp, seed)}
    for name, sites in H["mri_pools"].items():
        m_pool = mri[~mri["mri_id"].isin(test_ids)]
        if sites:
            m_pool = m_pool[m_pool["site"].isin(sites)]
        assert not test_ids & set(m_pool["mri_id"]), "test participant in MRI pool"
        enc[f"mri@{name}"] = Frozen(m_pool[mcols].to_numpy(float),
                                    m_pool["dx"].map(cmap).to_numpy(), "mri", hp, seed)
    assert not test_ids & set(e_pool["participant"]), "test participant in EEG pool"
    for name, f in enc.items():
        enc_log.append({"repeat": rep, "fold": k, "encoder": name, "n_pool": f.n,
                        "weight_decay": f.wd,
                        **{f"val_f1_wd{wd}": v for wd, v in f.val.items()},
                        "val_f1_chosen": f.val[f.wd]})

    # ---- stage 2
    main = f"mri@{H['main_mri_pool']}"
    ze, le = enc["eeg"].embed(Xe)
    out = {"pt_eeg_head": enc["eeg"].prob(Xe[te])}
    feats = {"pt_eeg_lr": ze}
    for name in H["mri_pools"]:
        zm, lm = enc[f"mri@{name}"].embed(Xm)
        sfx = "" if f"mri@{name}" == main else f"@{name}"
        out[f"pt_mri_head{sfx}"] = enc[f"mri@{name}"].prob(Xm[te])
        feats[f"pt_fusion_concat{sfx}"] = np.hstack([ze, zm])
        if not sfx:
            feats["pt_mri_lr"] = zm
            feats["pt_fusion_stack"] = np.column_stack([le[:, 1] - le[:, 0],
                                                        lm[:, 1] - lm[:, 0]])
    out["pt_late_avg"] = (out["pt_eeg_head"] + out["pt_mri_head"]) / 2
    for m, F in feats.items():
        out[m], C = fit_head(F, y, tr, te, groups, Cs, inner_k, seed)
        chosen.append({"model": m, "repeat": rep, "fold": k, "hyperparameter": C})

    for m, prob in out.items():
        for j, i in enumerate(te):
            preds.append({"model": m, "repeat": rep, "fold": k, "row": i,
                          "id": df["mri_id"].iat[i], "group": groups[i], "y": y[i],
                          "p0": prob[j, 0], "p1": prob[j, 1]})
    print(f"  repeat {rep} fold {k} done (pool n: eeg {enc['eeg'].n}, mri {enc[main].n})",
          flush=True)
    return preds, enc_log, chosen


def run(cfg, df, ecols, mcols, mri, eeg, y, groups, splits, jobs=1):
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    f = partial(one_split, cfg=cfg, df=df, ecols=ecols, mcols=mcols, mri=mri, eeg=eeg,
                y=y, groups=groups)
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            res = list(ex.map(f, splits))
    else:
        res = [f(s) for s in splits]
    return tuple(pd.DataFrame([r for part in res for r in part[i]]) for i in range(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_config.yaml")
    ap.add_argument("--repeats", type=int, default=0, help="override (quick check)")
    ap.add_argument("--jobs", type=int, default=1, help="outer splits run in parallel")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_cfg(args.config)
    P = cfg["paired"]
    pwork, work, res = P["paths"]["work"], cfg["paths"]["work"], cfg["paths"]["results"]
    os.makedirs(res, exist_ok=True)
    classes = cfg["classes"]

    df, ecols, mcols, dropped = load_paired(pwork, classes)
    assert not dropped
    mri, eeg, lost = load_pools(work, ecols, mcols)
    feat_diff = check_paired_features(df, mri, eeg, ecols, mcols)
    print(f"pool rows without features: {lost}; paired feature max |diff|: {feat_diff}")
    assert max(feat_diff.values()) < 1e-6, "pool features differ from the paired run"
    dups, rmax = near_duplicates(df, mri, mcols)
    print(f"MRI near-duplicates of paired scans (r>0.999): {dups}  max r {rmax:.4f}")
    mri = mri[~mri["mri_id"].isin({d["pool"] for d in dups})]

    y = df["dx"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    groups = df["mri_id"].to_numpy()
    C = P["cv"]
    reps = args.repeats or C["repeats"]
    splits = make_splits(y, groups, C["outer_folds"], C["repeats"], C["seed"])
    # identical folds to the paired-only run
    old = pd.read_csv(os.path.join(pwork, "splits.csv"))
    for rep, k, tr, te in splits:
        o = old[(old["repeat"] == rep) & (old["fold"] == k) & (old["split"] == "test")]
        assert sorted(o["row"]) == sorted(te), f"split mismatch repeat {rep} fold {k}"
    splits = [s for s in splits if s[0] < reps]

    preds, enc_log, chosen = run(cfg, df, ecols, mcols, mri, eeg, y, groups, splits,
                                   args.jobs)
    os.makedirs(work, exist_ok=True)
    preds.to_csv(os.path.join(work, "oof_predictions.csv"), index=False)

    # paired-only predictions on the same splits, for comparison
    po = pd.read_csv(os.path.join(pwork, "oof_predictions.csv"))
    po = po[po["model"].isin(PAIRED_ONLY) & (po["repeat"] < reps)]
    allp = pd.concat([preds, po], ignore_index=True)
    rep = per_repeat(allp, 2, classes)
    rep.to_csv(os.path.join(res, "metrics_by_repeat.csv"), index=False)
    summ = rep.drop(columns="repeat").groupby("model").agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    order = list(dict.fromkeys(preds["model"])) + PAIRED_ONLY
    summ = summ.reindex([m for m in order if m in summ.index])
    summ.to_csv(os.path.join(res, "metrics_summary.csv"))
    confusion(allp, 2, classes).to_csv(os.path.join(res, "confusion_matrices.csv"), index=False)
    boot = []
    for a, b in COMPARISONS:
        boot += bootstrap_diff(allp, a, b, 2, classes, P["cv"]["bootstrap"], P["cv"]["seed"])
    boot = pd.DataFrame(boot)
    boot.to_csv(os.path.join(res, "comparisons.csv"), index=False)
    enc_sum = enc_log.groupby("encoder").agg(
        n_pool_median=("n_pool", "median"), val_f1_mean=("val_f1_chosen", "mean"),
        val_f1_std=("val_f1_chosen", "std"))
    enc_sum = enc_sum.join(enc_log.groupby(["encoder", "weight_decay"]).size()
                           .unstack(fill_value=0).add_prefix("n_folds_wd_"))
    enc_sum.to_csv(os.path.join(res, "pretrain_encoders.csv"))
    chosen.groupby(["model", "hyperparameter"]).size().rename("n_folds") \
          .to_csv(os.path.join(res, "selected_hyperparameters.csv"))
    pool_dx = lambda d: d["dx"].value_counts().to_dict()
    info = {"classes": classes, "n_paired": len(df), "repeats": reps,
            "outer_folds": C["outer_folds"], "splits_identical_to_paired_run": True,
            "mri_pool": {"n": len(mri), "by_dx": pool_dx(mri),
                         "by_site": mri["site"].value_counts().to_dict()},
            "eeg_pool": {"n": len(eeg), "by_dx": pool_dx(eeg),
                         "by_site": eeg["site"].value_counts().to_dict()},
            "pool_rows_without_features": lost,
            "paired_feature_max_abs_diff": feat_diff,
            "mri_near_duplicates_removed": len(dups),  # count only: IDs stay in the log, not in results/ "mri_max_profile_r_paired_vs_pool": rmax,
            "n_eeg_features": len(ecols), "n_mri_features": len(mcols),
            "test_participants_in_pretraining": 0,
            "runtime_s": round(time.time() - t0, 1), "python": platform.python_version(),
            "torch": torch.__version__}
    with open(os.path.join(res, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    with open(os.path.join(res, "config_used.yaml"), "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if k != "paired"}, f, sort_keys=False)
    pd.set_option("display.width", 220)
    print("\n" + summ.round(3).to_string())
    print("\n" + boot.round(3).to_string())
    print("\n" + enc_sum.round(3).to_string())


if __name__ == "__main__":
    main()
