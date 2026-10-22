"""
Option 1 prototype: unpaired EEG and MRI cohorts trained into one shared embedding.

Each modality has its own encoder (MLP) into a shared 16-d space. One AD/CN classifier
head is shared by both modalities, and a cross-modal supervised contrastive loss pulls
AD-EEG embeddings towards AD-MRI embeddings (and CN towards CN). No subject is paired:
at test time every subject is predicted from its own modality only. The question is
whether training alongside the other modality improves unimodal prediction.

Tracks (same splits for every model, so differences are paired)
  eeg_cv     repeated stratified k-fold over all EEG subjects (BrainLat + ds004504,
             stratified by dataset x dx); all MRI subjects are auxiliary training data
  eeg_xfer   train on one EEG dataset (+ MRI), test on the other EEG dataset
  mri_cv     repeated stratified k-fold over ADNI; all EEG subjects are auxiliary

Models
  logreg               L2 logistic regression on the target modality
  mlp                  target-modality encoder + head, no other modality (the fair reference)
  joint                both encoders, shared head, + cross-modal contrastive loss
  joint_nocon          ablation: shared head only (no contrastive loss)
  joint_shuffled       ablation: auxiliary-modality labels permuted -> any gain should vanish
  joint_demo           ablation: auxiliary nodes carry only age + sex -> gain from demographics?

Preprocessing is fitted on training rows only: median imputation and z-scoring (EEG per
dataset; for eeg_xfer the unseen dataset is z-scored with its own feature statistics,
which uses no labels). MRI volumes (_CV, _SV) are divided by intracranial volume first.
Age, sex and education are not inputs (they are confounded with dx in BrainLat).

Run on the SCC:
  module load python3/3.10.12 pytorch/1.13.1
  python train_option1.py --out results/option1            # full run (~10-20 min on 4 cores)
  python train_option1.py --out results/option1_quick --quick
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

EEG_FILES = {"BrainLat": "processed/eeg_full/eeg_features.csv",
             "ds004504": "processed/ds004504/eeg_features.csv"}
MRI_FILE = "processed/mri/adni_mri_cohort.csv"
MRI_DROP = ["FifthVentricle_SV", "NonWMHypoIntensities_SV"]      # >90% missing
NN_MODELS = ["mlp", "joint", "joint_nocon", "joint_shuffled", "joint_demo"]

# fixed a priori, not tuned on test folds
HP = dict(emb=16, hidden_eeg=32, hidden_mri=64, dropout=0.3, lr=1e-3, wd=1e-3,
          epochs=300, lam=0.5, tau=0.1, w_aux=1.0)


# ------------------------------------------------------------------ data
def load_data(base):
    eeg = []
    for name, path in EEG_FILES.items():
        d = pd.read_csv(os.path.join(base, path))
        d["dataset"] = name
        eeg.append(d)
    eeg = pd.concat(eeg, ignore_index=True)
    eeg_cols = [c for c in eeg.columns if c.startswith(("relpow_", "theta_alpha_ratio_"))]
    eeg_cols.append("peak_alpha_freq")
    for c in eeg_cols:
        if c.startswith("theta_alpha_ratio_"):
            eeg[c] = np.log(eeg[c])                                # strongly right-skewed

    mri = pd.read_csv(os.path.join(base, MRI_FILE), low_memory=False)
    mri_cols = [c for c in mri.columns if re.search(r"_(CV|SA|TA|TS|SV)$", c)
                and c not in MRI_DROP and c != "Icv_CV"]
    for c in mri_cols:
        if c.endswith(("_CV", "_SV")):
            mri[c] = mri[c] / mri["Icv_CV"] * 1e3
    mri = mri.copy()                                               # de-fragment
    mri["dataset"] = "ADNI"
    for d in (eeg, mri):
        d["y"] = (d["dx"] == "AD").astype(int)
        d["sex_m"] = (d["sex"] == "M").astype(float)
    return eeg, eeg_cols, mri, mri_cols


def fit_transform(train, test, cols, by=None, test_own_stats=False):
    """Median-impute and z-score with training statistics (per group in `by`).
    test_own_stats: z-score test groups unseen in training with their own (label-free) stats."""
    def stats(df):
        med = df[cols].median()
        x = df[cols].fillna(med)
        return med, x.mean(), x.std().replace(0, 1)

    def apply(df, s):
        med, mu, sd = s
        return ((df[cols].fillna(med) - mu) / sd).fillna(0).to_numpy(np.float32)

    groups = [None] if by is None else sorted(set(train[by]) | set(test[by]))
    Xtr = np.zeros((len(train), len(cols)), np.float32)
    Xte = np.zeros((len(test), len(cols)), np.float32)
    for g in groups:
        mtr = np.ones(len(train), bool) if g is None else (train[by] == g).to_numpy()
        mte = np.ones(len(test), bool) if g is None else (test[by] == g).to_numpy()
        if mtr.any():
            s = stats(train[mtr])
        elif test_own_stats and mte.any():
            s = stats(test[mte])
        else:
            continue
        if mtr.any():
            Xtr[mtr] = apply(train[mtr], s)
        if mte.any():
            Xte[mte] = apply(test[mte], s)
    return Xtr, Xte


# ------------------------------------------------------------------ model
class Encoder(nn.Module):
    def __init__(self, d_in, hidden, emb, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, emb))

    def forward(self, x):
        return self.net(x)


def cross_supcon(za, ya, zb, yb, tau):
    """Supervised contrastive loss between modalities only: anchors in A, candidates in B
    (positives = same label), and the symmetric B->A term."""
    za, zb = F.normalize(za, dim=1), F.normalize(zb, dim=1)
    sim = za @ zb.T / tau
    pos = (ya[:, None] == yb[None, :]).float()

    def one_way(s, p):
        logp = s - torch.logsumexp(s, dim=1, keepdim=True)
        return (-(p * logp).sum(1) / p.sum(1).clamp(min=1)).mean()

    return 0.5 * (one_way(sim, pos) + one_way(sim.T, pos.T))


def weighted_bce(logit, y):
    pw = (y == 0).sum() / (y == 1).sum().clamp(min=1)
    return F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)


def train_nn(Xt, yt, Xa=None, ya=None, lam=0.0, seed=0, hp=HP):
    """Target modality (Xt, yt) and optional auxiliary modality (Xa, ya). Full-batch Adam.
    Returns a function mapping target-modality features to P(AD)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    ht = hp["hidden_eeg"] if Xt.shape[1] < 100 else hp["hidden_mri"]
    enc_t = Encoder(Xt.shape[1], ht, hp["emb"], hp["dropout"])
    head = nn.Linear(hp["emb"], 1)
    params = list(enc_t.parameters()) + list(head.parameters())
    if Xa is not None:
        ha = hp["hidden_eeg"] if Xa.shape[1] < 100 else hp["hidden_mri"]
        enc_a = Encoder(Xa.shape[1], ha, hp["emb"], hp["dropout"])
        params += list(enc_a.parameters())
        Xa_t, ya_t = torch.from_numpy(Xa), torch.from_numpy(ya.astype(np.float32))
    opt = torch.optim.Adam(params, lr=hp["lr"], weight_decay=hp["wd"])
    Xt_t, yt_t = torch.from_numpy(Xt), torch.from_numpy(yt.astype(np.float32))

    for _ in range(hp["epochs"]):
        enc_t.train()
        zt = enc_t(Xt_t)
        loss = weighted_bce(head(zt).squeeze(1), yt_t)
        if Xa is not None:
            enc_a.train()
            za = enc_a(Xa_t)
            loss = loss + hp["w_aux"] * weighted_bce(head(za).squeeze(1), ya_t)
            if lam > 0:
                loss = loss + lam * cross_supcon(zt, yt_t, za, ya_t, hp["tau"])
        opt.zero_grad()
        loss.backward()
        opt.step()

    enc_t.eval()

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(head(enc_t(torch.from_numpy(X))).squeeze(1)).numpy()
    return predict


def run_models(models, Xt_tr, yt_tr, Xt_te, Xa, ya, Xdemo, seed):
    """Fit every requested model on one split; return {model: P(AD) on the test rows}."""
    out = {}
    rng = np.random.default_rng(seed)
    for m in models:
        if m == "logreg":
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
            out[m] = clf.fit(Xt_tr, yt_tr).predict_proba(Xt_te)[:, 1]
            continue
        if m == "mlp":
            f = train_nn(Xt_tr, yt_tr, seed=seed)
        elif m == "joint":
            f = train_nn(Xt_tr, yt_tr, Xa, ya, lam=HP["lam"], seed=seed)
        elif m == "joint_nocon":
            f = train_nn(Xt_tr, yt_tr, Xa, ya, lam=0.0, seed=seed)
        elif m == "joint_shuffled":
            f = train_nn(Xt_tr, yt_tr, Xa, rng.permutation(ya), lam=HP["lam"], seed=seed)
        elif m == "joint_demo":
            f = train_nn(Xt_tr, yt_tr, Xdemo, ya, lam=HP["lam"], seed=seed)
        out[m] = f(Xt_te)
    return out


def demo_features(df):
    x = df[["age", "sex_m"]].astype(float)
    x = x.fillna(x.median())
    return ((x - x.mean()) / x.std()).to_numpy(np.float32)


def aux_features(df, cols, by=None):
    X, _ = fit_transform(df, df.iloc[:0], cols, by=by)
    return X


# ------------------------------------------------------------------ tracks
def cv_track(name, target, tcols, aux, acols, models, repeats, folds, by, strat):
    """Out-of-fold predictions for every target subject, per repeat."""
    Xa, ya = aux_features(aux, acols, by="dataset"), aux["y"].to_numpy()
    Xdemo = demo_features(aux)
    rows = []
    for r in range(repeats):
        skf = StratifiedKFold(folds, shuffle=True, random_state=r)
        oof = {m: np.full(len(target), np.nan) for m in models}
        for k, (tr, te) in enumerate(skf.split(target, strat)):
            Xtr, Xte = fit_transform(target.iloc[tr], target.iloc[te], tcols, by=by)
            preds = run_models(models, Xtr, target["y"].to_numpy()[tr], Xte, Xa, ya, Xdemo,
                               seed=1000 * r + k)
            for m, p in preds.items():
                oof[m][te] = p
        for m in models:
            rows.append(pd.DataFrame({"track": name, "repeat": r, "model": m,
                                      "subject": target["id_eeg"] if "id_eeg" in target
                                      else target["ptid"], "dataset": target["dataset"],
                                      "y": target["y"], "p_ad": oof[m]}))
        print(f"  [{name}] repeat {r + 1}/{repeats} done", flush=True)
    return pd.concat(rows, ignore_index=True)


def xfer_track(eeg, ecols, mri, mcols, models, seeds):
    Xa, ya = aux_features(mri, mcols), mri["y"].to_numpy()
    Xdemo = demo_features(mri)
    rows = []
    for src, dst in (("ds004504", "BrainLat"), ("BrainLat", "ds004504")):
        tr, te = eeg[eeg.dataset == src], eeg[eeg.dataset == dst]
        Xtr, Xte = fit_transform(tr, te, ecols, by="dataset", test_own_stats=True)
        for s in range(seeds):
            preds = run_models(models, Xtr, tr["y"].to_numpy(), Xte, Xa, ya, Xdemo, seed=s)
            for m, p in preds.items():
                rows.append(pd.DataFrame({"track": f"eeg_xfer:{src}->{dst}", "repeat": s,
                                          "model": m, "subject": te["id_eeg"],
                                          "dataset": dst, "y": te["y"], "p_ad": p}))
        print(f"  [eeg_xfer] {src} -> {dst} done", flush=True)
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ summary
def score(df):
    y, p = df["y"].to_numpy(), df["p_ad"].to_numpy()
    if len(np.unique(y)) < 2:
        return pd.Series({"auc": np.nan, "bal_acc": np.nan, "n": len(df)})
    return pd.Series({"auc": roc_auc_score(y, p), "bal_acc": balanced_accuracy_score(y, p >= 0.5),
                      "n": len(df)})


def summarize(pred, reference="mlp"):
    """Per track x evaluation subset x model: mean/sd over repeats, and the paired AUC
    difference vs the reference model (same repeat, same subjects)."""
    pred = pred.copy()
    subsets = [pred.assign(subset="all")]
    multi = pred.groupby("track")["dataset"].transform("nunique") > 1
    subsets.append(pred[multi].assign(subset=pred.loc[multi, "dataset"]))
    per = (pd.concat(subsets).groupby(["track", "subset", "model", "repeat"])
           .apply(score).reset_index())
    ref = per[per.model == reference][["track", "subset", "repeat", "auc"]]
    per = per.merge(ref.rename(columns={"auc": "auc_ref"}), on=["track", "subset", "repeat"],
                    how="left")
    per["d_auc"] = per["auc"] - per["auc_ref"]
    summ = per.groupby(["track", "subset", "model"]).agg(
        n=("n", "first"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
        bal_acc_mean=("bal_acc", "mean"), d_auc_vs_mlp=("d_auc", "mean"),
        d_auc_sd=("d_auc", "std"), frac_better=("d_auc", lambda d: (d > 0).mean()),
        repeats=("repeat", "nunique")).reset_index()
    order = {m: i for i, m in enumerate(["logreg"] + NN_MODELS)}
    return summ.sort_values(["track", "subset", "model"], key=lambda s: s.map(order)
                            if s.name == "model" else s), per


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/project/mlresearch/Runkun")
    ap.add_argument("--out", default="results/option1")
    ap.add_argument("--quick", action="store_true", help="2 repeats, few epochs: smoke test")
    ap.add_argument("--tracks", default="eeg_cv,eeg_xfer,mri_cv")
    args = ap.parse_args()
    out = os.path.join(args.base, args.out)
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("NSLOTS", 1)))

    reps = dict(eeg=10, mri=3, xfer=10)
    if args.quick:
        HP["epochs"] = 30
        reps = dict(eeg=2, mri=1, xfer=2)

    eeg, ecols, mri, mcols = load_data(args.base)
    print(f"EEG: {len(eeg)} subjects {eeg.groupby(['dataset', 'dx']).size().to_dict()}, "
          f"{len(ecols)} features")
    print(f"MRI: {len(mri)} subjects {mri['dx'].value_counts().to_dict()}, {len(mcols)} features")
    print(f"Hyperparameters (fixed): {HP}")

    models = ["logreg"] + NN_MODELS
    tracks = args.tracks.split(",")
    t0, preds = time.time(), []
    if "eeg_cv" in tracks:
        strat = eeg["dataset"] + "_" + eeg["dx"]
        preds.append(cv_track("eeg_cv", eeg, ecols, mri, mcols, models, reps["eeg"], 5,
                              by="dataset", strat=strat))
    if "eeg_xfer" in tracks:
        preds.append(xfer_track(eeg, ecols, mri, mcols, models, reps["xfer"]))
    if "mri_cv" in tracks:
        preds.append(cv_track("mri_cv", mri, mcols, eeg, ecols,
                              ["logreg", "mlp", "joint", "joint_nocon", "joint_shuffled"],
                              reps["mri"], 5, by=None, strat=mri["y"]))
    pred = pd.concat(preds, ignore_index=True)
    pred.to_csv(os.path.join(out, "oof_predictions.csv"), index=False)

    summ, per = summarize(pred)
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    per.to_csv(os.path.join(out, "per_repeat_scores.csv"), index=False)
    pd.Series(HP).to_csv(os.path.join(out, "hyperparameters.csv"), header=["value"])
    pd.set_option("display.width", 200)
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min. AUC over repeats; d_auc = paired "
          f"difference vs the target-only MLP (same splits); frac_better = share of repeats "
          f"where the model beats it.\n")
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
