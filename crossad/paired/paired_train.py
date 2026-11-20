"""
CROSS-AD paired EEG + MRI: baselines, single-modality models and fusion on one cohort.

Every model sees the same participants and the same outer splits:
  repeated stratified group k-fold (groups = participant; every visit and every EEG epoch
  of a participant lives in one split -- epochs are aggregated per visit before training).

Models
  logreg_age_sex   reference: age + sex only (AD and CN differ in age in BrainLat)
  logreg_eeg       regularized feature baseline, EEG features      (L2, C tuned)
  logreg_mri       regularized feature baseline, MRI features
  logreg_concat    regularized feature baseline, EEG + MRI features
  mlp_eeg          EEG encoder -> embedding -> linear head
  mlp_mri          MRI encoder -> embedding -> linear head
  fusion_concat    both encoders, concatenated embeddings -> linear head
  fusion_gate      both encoders, per-visit softmax gate over the two embeddings -> head
                   (gate weights are logged; they are a model component, not importance)

Inside each outer training set only: median imputation, z-scoring, optional SelectKBest,
balanced class weights, and hyperparameter choice by inner stratified group k-fold
(macro-F1). Metrics per repeat from out-of-fold predictions: macro-F1, balanced accuracy,
per-class sensitivity, ROC AUC (binary), confusion matrices. Fusion vs single modality:
participant-level stratified bootstrap of the repeat-averaged difference.

  python -m crossad.paired.paired_train --config configs/paired_config.yaml                 # real data
  python -m tests.paired_smoke_test --config configs/paired_config.yaml            # smoke test first
"""
import argparse
import hashlib
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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

LOGREG = {"logreg_age_sex": ("demo",), "logreg_eeg": ("eeg",), "logreg_mri": ("mri",),
          "logreg_concat": ("eeg", "mri")}
NN = {"mlp_eeg": "eeg", "mlp_mri": "mri", "fusion_concat": "concat", "fusion_gate": "gate"}
COMPARISONS = [("fusion_concat", "mlp_mri"), ("fusion_concat", "mlp_eeg"),
               ("fusion_gate", "mlp_mri"), ("fusion_gate", "mlp_eeg"),
               ("fusion_gate", "fusion_concat"), ("logreg_concat", "logreg_mri"),
               ("logreg_concat", "logreg_eeg")]


# ------------------------------------------------------------------ data
def load_paired(work, classes):
    coh = pd.read_csv(os.path.join(work, "paired_cohort.csv"))
    eeg = pd.read_csv(os.path.join(work, "eeg_features.csv"))
    mri = pd.read_csv(os.path.join(work, "mri_features.csv"))
    df = coh[["mri_id", "dx", "age", "sex"]].merge(eeg, on=["mri_id", "dx"]) \
                                           .merge(mri, on=["mri_id", "dx"])
    dropped = sorted(set(coh["mri_id"]) - set(df["mri_id"]))
    df = df[df["dx"].isin(classes)].reset_index(drop=True)
    ecols = [c for c in eeg.columns if c not in ("mri_id", "dx")]
    mcols = [c for c in mri.columns if c.startswith("vol_")]
    return df, ecols, mcols, dropped


def arrays(df, ecols, mcols, classes):
    X = {"eeg": df[ecols].to_numpy(float), "mri": df[mcols].to_numpy(float),
         "demo": df[["age", "sex"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)}
    y = df["dx"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    groups = df["participant"].to_numpy() if "participant" in df else df["mri_id"].to_numpy()
    return X, y, groups


# ------------------------------------------------------------------ splits
def make_splits(y, groups, folds, repeats, seed):
    """List of (repeat, fold, train_idx, test_idx); participants never cross splits."""
    out = []
    for rep in range(repeats):
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + rep)
        for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            assert not set(groups[tr]) & set(groups[te]), "participant in train and test"
            out.append((rep, k, tr, te))
        assert sorted(np.concatenate([s[3] for s in out if s[0] == rep])) == list(range(len(y)))
    return out


def inner_splits(y, groups, folds, seed):
    n_min = min(np.bincount(y)[np.bincount(y) > 0])
    folds = max(2, min(folds, n_min))
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


# ------------------------------------------------------------------ preprocessing
def prep(Xtr, Xte, ytr, k=None):
    steps = [SimpleImputer(strategy="median"), StandardScaler()]
    if k and k < Xtr.shape[1]:
        steps.append(SelectKBest(f_classif, k=k))
    p = make_pipeline(*steps).fit(Xtr, ytr)
    return p.transform(Xtr), p.transform(Xte)


def prep_blocks(X, tr, te, y, blocks, k):
    return {b: prep(X[b][tr], X[b][te], y[tr], None if b == "demo" else k) for b in blocks}


# ------------------------------------------------------------------ models
class Encoder(nn.Module):
    def __init__(self, d, hidden, emb, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(hidden, emb), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class CrossAD(nn.Module):
    """mode: eeg | mri | concat | gate."""

    def __init__(self, mode, d_eeg, d_mri, n_cls, hp):
        super().__init__()
        self.mode = mode
        h, e, p = hp["hidden"], hp["emb_dim"], hp["dropout"]
        self.enc_eeg = Encoder(d_eeg, h, e, p) if mode != "mri" else None
        self.enc_mri = Encoder(d_mri, h, e, p) if mode != "eeg" else None
        self.gate = nn.Linear(2 * e, 2) if mode == "gate" else None
        self.head = nn.Linear(2 * e if mode == "concat" else e, n_cls)
        self.last_gate = None

    def forward(self, xe, xm):
        if self.mode == "eeg":
            return self.head(self.enc_eeg(xe))
        if self.mode == "mri":
            return self.head(self.enc_mri(xm))
        ze, zm = self.enc_eeg(xe), self.enc_mri(xm)
        if self.mode == "concat":
            return self.head(torch.cat([ze, zm], 1))
        w = torch.softmax(self.gate(torch.cat([ze, zm], 1)), 1)      # (n, 2): [eeg, mri]
        self.last_gate = w.detach()
        return self.head(w[:, :1] * ze + w[:, 1:] * zm)


def fit_nn(mode, Xe, Xm, y, Xe_te, Xm_te, n_cls, hp, wd, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    model = CrossAD(mode, Xe.shape[1], Xm.shape[1], n_cls, hp)
    cnt = np.bincount(y, minlength=n_cls).astype(float)
    w = torch.tensor(len(y) / (n_cls * np.maximum(cnt, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=wd)
    xe, xm, yt = t(Xe), t(Xm), torch.tensor(y)
    bs = hp.get("batch_size") or len(y)
    model.train()
    for _ in range(hp["epochs"]):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(model(xe[b], xm[b]), yt[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(t(Xe_te), t(Xm_te)), 1).numpy()
    gate = model.last_gate.numpy() if mode == "gate" else None
    return prob, gate


def fit_logreg(Xtr, ytr, Xte, C, seed):
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
    return m.fit(Xtr, ytr).predict_proba(Xte)


def macro_f1(y, prob):
    return f1_score(y, prob.argmax(1), average="macro", zero_division=0)


def tune(model, X, y, groups, tr, cfg, n_cls, seed):
    """Choose a hyperparameter on inner folds of the outer training set (macro-F1)."""
    H, C = cfg["model"], cfg["cv"]
    grid = H["logreg_C"] if model in LOGREG else H["weight_decay"]
    if len(grid) == 1:
        return grid[0]
    inner = inner_splits(y[tr], groups[tr], C["inner_folds"], seed)
    scores = []
    for g in grid:
        s = []
        for itr, ite in inner:
            a, b = tr[itr], tr[ite]
            if model in LOGREG:
                P = prep_blocks(X, a, b, y, LOGREG[model], H["select_k"])
                Xa = np.hstack([P[k][0] for k in LOGREG[model]])
                Xb = np.hstack([P[k][1] for k in LOGREG[model]])
                prob = fit_logreg(Xa, y[a], Xb, g, seed)
            else:
                P = prep_blocks(X, a, b, y, ("eeg", "mri"), H["select_k"])
                prob, _ = fit_nn(NN[model], P["eeg"][0], P["mri"][0], y[a],
                                 P["eeg"][1], P["mri"][1], n_cls, H, g, seed)
            s.append(macro_f1(y[b], prob))
        scores.append(np.mean(s))
    # max() keeps the first best in iteration order, so ties go to stronger
    # regularization: grids are ascending, smallest C first / largest weight decay first
    order = range(len(grid)) if model in LOGREG else reversed(range(len(grid)))
    return grid[max(order, key=lambda i: scores[i])]


def run_cv(X, y, groups, ids, splits, cfg, n_cls, models, log=print):
    H = cfg["model"]
    preds, gates, chosen = [], [], []
    for rep, k, tr, te in splits:
        seed = cfg["cv"]["seed"] + 100 * rep + k
        for m in models:
            hp = tune(m, X, y, groups, tr, cfg, n_cls, seed)
            chosen.append({"model": m, "repeat": rep, "fold": k, "hyperparameter": hp})
            if m in LOGREG:
                P = prep_blocks(X, tr, te, y, LOGREG[m], H["select_k"])
                prob = fit_logreg(np.hstack([P[b][0] for b in LOGREG[m]]), y[tr],
                                  np.hstack([P[b][1] for b in LOGREG[m]]), hp, seed)
                gate = None
            else:
                P = prep_blocks(X, tr, te, y, ("eeg", "mri"), H["select_k"])
                prob, gate = fit_nn(NN[m], P["eeg"][0], P["mri"][0], y[tr],
                                    P["eeg"][1], P["mri"][1], n_cls, H, hp, seed)
            assert prob.shape == (len(te), n_cls), (m, prob.shape)
            for j, i in enumerate(te):
                preds.append({"model": m, "repeat": rep, "fold": k, "row": i, "id": ids[i],
                              "group": groups[i], "y": y[i],
                              **{f"p{c}": prob[j, c] for c in range(n_cls)}})
                if gate is not None:
                    gates.append({"repeat": rep, "row": i, "y": y[i], "w_eeg": gate[j, 0],
                                  "w_mri": gate[j, 1]})
        log(f"  repeat {rep} fold {k} done")
    return pd.DataFrame(preds), pd.DataFrame(gates), pd.DataFrame(chosen)


# ------------------------------------------------------------------ evaluation
def metrics(y, prob, n_cls, classes):
    pred = prob.argmax(1)
    m = {"macro_f1": f1_score(y, pred, average="macro", zero_division=0),
         "balanced_accuracy": balanced_accuracy_score(y, pred)}
    rec = recall_score(y, pred, labels=range(n_cls), average=None, zero_division=0)
    m.update({f"sensitivity_{c}": r for c, r in zip(classes, rec)})
    if n_cls == 2 and len(np.unique(y)) == 2:
        m["roc_auc"] = roc_auc_score(y, prob[:, 1])
    return m


def per_repeat(preds, n_cls, classes):
    rows = []
    pc = [f"p{c}" for c in range(n_cls)]
    for (m, rep), g in preds.groupby(["model", "repeat"]):
        rows.append({"model": m, "repeat": rep,
                     **metrics(g["y"].to_numpy(), g[pc].to_numpy(), n_cls, classes)})
    return pd.DataFrame(rows)


def confusion(preds, n_cls, classes):
    rows = []
    pc = [f"p{c}" for c in range(n_cls)]
    n_rep = preds["repeat"].nunique()
    for m, g in preds.groupby("model"):
        cm = confusion_matrix(g["y"], g[pc].to_numpy().argmax(1), labels=range(n_cls))
        for i in range(n_cls):
            for j in range(n_cls):
                rows.append({"model": m, "true": classes[i], "pred": classes[j],
                             "count_all_repeats": int(cm[i, j]),
                             "mean_per_repeat": cm[i, j] / n_rep})
    return pd.DataFrame(rows)


def bootstrap_diff(preds, a, b, n_cls, classes, n_boot, seed):
    """Participant-level stratified bootstrap of mean-over-repeats metric(a) - metric(b)."""
    pc = [f"p{c}" for c in range(n_cls)]
    piv = {}
    for m in (a, b):
        g = preds[preds["model"] == m].sort_values(["repeat", "row"])
        piv[m] = g
    rows_y = piv[a].drop_duplicates("row").set_index("row")["y"]
    rows = rows_y.index.to_numpy()
    reps = sorted(piv[a]["repeat"].unique())
    P = {m: np.stack([piv[m][piv[m]["repeat"] == r].set_index("row").loc[rows, pc].to_numpy()
                      for r in reps]) for m in (a, b)}                # (n_rep, n, n_cls)
    yv = rows_y.to_numpy()

    keys = ("macro_f1", "balanced_accuracy")

    def stat(idx):
        d = {k: [] for k in keys}
        for r in range(len(reps)):
            ma = metrics(yv[idx], P[a][r][idx], n_cls, classes)
            mb = metrics(yv[idx], P[b][r][idx], n_cls, classes)
            for k in keys:
                d[k].append(ma[k] - mb[k])
        return {k: float(np.mean(v)) for k, v in d.items()}

    obs = stat(np.arange(len(yv)))
    rng = np.random.default_rng(seed)
    strata = [np.where(yv == c)[0] for c in range(n_cls)]
    boot = {k: [] for k in obs}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata if len(s)])
        for k, v in stat(idx).items():
            boot[k].append(v)
    res = []
    for k in obs:
        bs = np.array(boot[k])
        res.append({"model_a": a, "model_b": b, "metric": k, "diff_a_minus_b": obs[k],
                    "ci95_low": np.percentile(bs, 2.5), "ci95_high": np.percentile(bs, 97.5),
                    "boot_frac_gt0": float((bs > 0).mean()), "n_participants": len(yv)})
    return res


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def evaluate_and_save(preds, gates, chosen, splits_df, df, cfg, classes, out, work, tag, t0):
    n_cls = len(classes)
    os.makedirs(out, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    preds.to_csv(os.path.join(work, f"{tag}oof_predictions.csv"), index=False)
    split_path = os.path.join(work, f"{tag}splits.csv")
    splits_df.to_csv(split_path, index=False)

    rep = per_repeat(preds, n_cls, classes)
    rep.to_csv(os.path.join(out, "metrics_by_repeat.csv"), index=False)
    summ = rep.drop(columns="repeat").groupby("model").agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    order = list(LOGREG) + list(NN)
    summ = summ.reindex([m for m in order if m in summ.index])
    summ.to_csv(os.path.join(out, "metrics_summary.csv"))
    confusion(preds, n_cls, classes).to_csv(os.path.join(out, "confusion_matrices.csv"),
                                            index=False)

    # best single-modality model (by mean macro-F1) vs each fusion model
    single = [m for m in ("logreg_eeg", "logreg_mri", "mlp_eeg", "mlp_mri") if m in summ.index]
    best = summ.loc[single, "macro_f1_mean"].idxmax() if single else None
    comps = [c for c in COMPARISONS if c[0] in summ.index and c[1] in summ.index]
    if best:
        comps += [(f, best) for f in ("fusion_concat", "fusion_gate", "logreg_concat")
                  if f in summ.index and (f, best) not in comps]
    boot = []
    for a, b in comps:
        boot += bootstrap_diff(preds, a, b, n_cls, classes, cfg["cv"]["bootstrap"],
                               cfg["cv"]["seed"])
    boot = pd.DataFrame(boot)
    if best is not None and len(boot):
        boot["b_is_best_single"] = boot["model_b"] == best
    boot.to_csv(os.path.join(out, "fusion_vs_single.csv"), index=False)

    if len(gates):
        g = gates.groupby("y")[["w_eeg", "w_mri"]].agg(["mean", "std"])
        g.index = [classes[i] for i in g.index]
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        g.to_csv(os.path.join(out, "gate_weights_summary.csv"))
    chosen.groupby(["model", "hyperparameter"]).size().rename("n_folds") \
          .to_csv(os.path.join(out, "selected_hyperparameters.csv"))
    ssum = splits_df.groupby(["repeat", "fold", "split", "dx"]).size().unstack(fill_value=0)
    ssum.to_csv(os.path.join(out, "split_summary.csv"))
    info = {"tag": tag or "real", "classes": classes,
            "n_participants": int(df["mri_id"].nunique()),
            "n_rows": len(df), "by_dx": df["dx"].value_counts().to_dict(),
            "n_eeg_features": int(cfg["_n_eeg"]), "n_mri_features": int(cfg["_n_mri"]),
            "best_single_modality_model": best, "splits_sha256": sha(split_path),
            "participant_overlap_between_train_and_test": 0,
            "runtime_s": round(time.time() - t0, 1), "python": platform.python_version(),
            "torch": torch.__version__}
    with open(os.path.join(out, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(out, "config_used.yaml"), "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f,
                       sort_keys=False)
    return summ, boot, info


def splits_frame(splits, df, y, classes):
    rows = []
    for rep, k, tr, te in splits:
        for split, idx in (("train", tr), ("test", te)):
            for i in idx:
                rows.append({"repeat": rep, "fold": k, "split": split,
                             "participant": df["participant"].iat[i] if "participant" in df
                             else df["mri_id"].iat[i], "row": int(i), "dx": classes[y[i]]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paired_config.yaml")
    args = ap.parse_args()
    t0 = time.time()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    work, out = cfg["paths"]["work"], cfg["paths"]["results"]
    with open(os.path.join(out, "audit_summary.json")) as f:
        classes = json.load(f)["task_classes"]
    df, ecols, mcols, dropped = load_paired(work, classes)
    if dropped:
        print(f"paired participants without both feature sets (dropped): {len(dropped)}")
    cfg["_n_eeg"], cfg["_n_mri"] = len(ecols), len(mcols)
    X, y, groups = arrays(df, ecols, mcols, classes)
    print(f"cohort: {df['dx'].value_counts().to_dict()}  EEG {X['eeg'].shape}  MRI {X['mri'].shape}")
    C = cfg["cv"]
    splits = make_splits(y, groups, C["outer_folds"], C["repeats"], C["seed"])
    preds, gates, chosen = run_cv(X, y, groups, df["mri_id"].to_numpy(), splits, cfg,
                                  len(classes), list(LOGREG) + list(NN))
    summ, boot, info = evaluate_and_save(preds, gates, chosen, splits_frame(splits, df, y, classes),
                                         df, cfg, classes, out, work, "", t0)
    pd.set_option("display.width", 200)
    print("\n" + summ.round(3).to_string())
    print("\n" + boot.round(3).to_string())
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
