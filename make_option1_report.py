"""
Brief methods + results report for the option 1 prototype (train_option1.py).

Reads results/option1/{summary,per_repeat_scores,hyperparameters}.csv and writes
  reports/option1_results.html and reports/option1_results.pdf (needs `module load wkhtmltopdf`)
  reports/figures/option1_*.png

Usage:  python make_option1_report.py [--base /project/mlresearch/Runkun]
"""
import argparse
import os
import shutil
import subprocess
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# same figure style, palette and helpers as the data report
from make_data_report import ACKNOWLEDGMENTS_HTML, COL, GRID, INK, INK2, SURFACE, save, table

TRACKS = [("eeg_cv", "all", "EEG cross-validation (132 subjects)"),
          ("eeg_xfer:BrainLat->ds004504", "all", "EEG transfer: BrainLat → ds004504"),
          ("eeg_xfer:ds004504->BrainLat", "all", "EEG transfer: ds004504 → BrainLat"),
          ("mri_cv", "all", "MRI cross-validation (870 subjects)")]
LABEL = {"logreg": "Logistic regression", "mlp": "Network, target modality only",
         "joint": "Joint EEG + MRI", "joint_nocon": "Joint, no contrastive loss",
         "joint_shuffled": "Joint, auxiliary labels shuffled",
         "joint_demo": "Joint, auxiliary = age/sex only"}
ORDER = list(LABEL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/project/mlresearch/Runkun")
    args = ap.parse_args()
    res = os.path.join(args.base, "results/option1")
    out = os.path.join(args.base, "reports")
    figdir = os.path.join(out, "figures")
    os.makedirs(figdir, exist_ok=True)
    summ = pd.read_csv(os.path.join(res, "summary.csv"))
    per = pd.read_csv(os.path.join(res, "per_repeat_scores.csv"))
    hp = pd.read_csv(os.path.join(res, "hyperparameters.csv"), index_col=0)["value"]

    # ---------- figure 1: AUC per model and track
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), sharey=True)
    for ax, (tr, sub, title) in zip(axs.flat, TRACKS):
        s = summ[(summ.track == tr) & (summ.subset == sub)].set_index("model")
        models = [m for m in ORDER if m in s.index]
        y = np.arange(len(models))[::-1]
        for yy, m in zip(y, models):
            c = COL["AD"] if m == "joint" else COL["CN"]
            ax.errorbar(s.loc[m, "auc_mean"], yy, xerr=s.loc[m, "auc_sd"], fmt="o", ms=7,
                        color=c, ecolor=c, elinewidth=2, capsize=0, zorder=3)
            sd = 0 if np.isnan(s.loc[m, "auc_sd"]) else s.loc[m, "auc_sd"]
            ax.text(s.loc[m, "auc_mean"] + sd + 0.004, yy, f"{s.loc[m, 'auc_mean']:.3f}",
                    ha="left", va="center", fontsize=7.5, color=INK2)
        ref = s.loc["logreg", "auc_mean"]
        ax.axvline(ref, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.set_yticks(np.arange(len(ORDER))[::-1])
        ax.set_yticklabels([LABEL[m] for m in ORDER])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        lo = s.auc_mean.min() - 0.06
        ax.set_xlim(lo, s.auc_mean.max() + s.auc_sd.max() + 0.03)
    for ax in axs[1]:
        ax.set_xlabel("AUC (mean ± SD over repeats)")
    fig.tight_layout(h_pad=2)
    f1 = save(fig, "option1_auc.png", figdir)

    # ---------- figure 2: paired AUC differences of the joint model
    w = (per[per.subset == "all"].pivot_table(index=["track", "repeat"], columns="model",
                                              values="auc"))
    comps = [("logreg", "vs logistic regression"), ("mlp", "vs network, target only"),
             ("joint_shuffled", "vs shuffled-label ablation"),
             ("joint_demo", "vs age/sex-only ablation")]
    rows = []
    for tr, _, title in TRACKS:
        for ref, lab in comps:
            if ref not in w.columns:
                continue
            d = (w.loc[tr, "joint"] - w.loc[tr, ref]).dropna()
            if len(d):
                rows.append({"track": title, "comparison": lab, "mean": d.mean(),
                             "sd": d.std(), "frac": (d > 0).mean(), "n": len(d)})
    diffs = pd.DataFrame(rows)
    fig, axs = plt.subplots(2, 2, figsize=(10, 4.6), sharey=True)
    for ax, (_, _, title) in zip(axs.flat, TRACKS):
        d = diffs[diffs.track == title].reset_index(drop=True)
        y = np.arange(len(comps))[::-1]
        for _, r in d.iterrows():
            yy = y[[c[1] for c in comps].index(r.comparison)]
            ax.errorbar(r["mean"], yy, xerr=r["sd"], fmt="o", ms=7, color=COL["AD"],
                        ecolor=COL["AD"], elinewidth=2, zorder=3)
        ax.axvline(0, color=INK, linewidth=1, zorder=2)
        ax.set_yticks(y); ax.set_yticklabels([c[1] for c in comps])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        m = max(0.06, (d["mean"].abs() + d["sd"].fillna(0)).max() * 1.15)
        ax.set_xlim(-m, m)
    for ax in axs[1]:
        ax.set_xlabel("ΔAUC, joint − comparison")
    fig.tight_layout(h_pad=2)
    f2 = save(fig, "option1_paired_diff.png", figdir)

    # ---------- tables
    t = summ[summ.subset.eq("all")].copy()
    t["Track"] = t.track.map({k: v for k, _, v in TRACKS})
    t["Model"] = t.model.map(LABEL)
    t["AUC"] = t.auc_mean.map("{:.3f}".format) + " ± " + t.auc_sd.fillna(0).map("{:.3f}".format)
    t["Balanced acc."] = t.bal_acc_mean.map("{:.3f}".format)
    t["Repeats"] = t.repeats
    t["_o"] = t.model.map(ORDER.index)
    t["_t"] = t.track.map([k for k, _, _ in TRACKS].index)
    t = t.sort_values(["_t", "_o"])[["Track", "Model", "AUC", "Balanced acc.", "Repeats"]]

    eeg_ds = summ[(summ.track == "eeg_cv") & (summ.subset != "all")]
    within = eeg_ds.pivot(index="model", columns="subset", values="auc_mean").reindex(ORDER)
    within.index = within.index.map(LABEL)
    within = within.map("{:.3f}".format)
    within.index.name = "Model"

    d = diffs.copy()
    d["ΔAUC (mean ± SD)"] = d["mean"].map("{:+.3f}".format) + " ± " + d.sd.map("{:.3f}".format)
    d["Joint better in"] = (d.frac * d.n).round().astype(int).astype(str) + " / " + d.n.astype(str)
    d = d[["track", "comparison", "ΔAUC (mean ± SD)", "Joint better in"]].rename(
        columns={"track": "Track", "comparison": "Joint model"})

    def g(tr, m, col="auc_mean"):
        return summ[(summ.track == tr) & (summ.subset == "all") & (summ.model == m)][col].iloc[0]

    def dd(tr, ref):
        return diffs[(diffs.track == dict((k, v) for k, _, v in TRACKS)[tr])
                     & (diffs.comparison.str.contains(ref))]["mean"].iloc[0]

    body = f"""
<header>
  <p class="eyebrow">CROSS-AD · Option 1 prototype</p>
  <h1>Shared EEG–MRI embedding from unpaired cohorts: results</h1>
  <p class="meta">{date.today():%d %B %Y} · code <code>train_option1.py</code> (job script
  <code>train_option1.qsub</code>) · outputs <code>results/option1/</code></p>
</header>

<section class="callout">
<b>Bottom line.</b> Training EEG together with unpaired ADNI MRI raised EEG AUC over an EEG-only network
({g('eeg_cv','mlp'):.3f} → {g('eeg_cv','joint'):.3f}), but the ablations show the gain is not MRI information:
shuffling the MRI labels keeps most of it, and replacing MRI with age/sex alone does slightly better than
MRI ({dd('eeg_cv','age'):+.3f}). The joint model only matches plain EEG logistic regression
({dd('eeg_cv','logistic'):+.3f}). The contrastive loss acts as a regularizer for a small network;
we find no evidence that unpaired MRI improves EEG prediction (or vice versa).
</section>

<h2>1. Data</h2>
<table class="tbl">
<thead><tr><th>Modality</th><th>Cohort</th><th>Subjects (AD / CN)</th><th>Input features</th></tr></thead>
<tbody>
<tr><td>EEG</td><td>BrainLat (2 sites) + OpenNeuro ds004504 (1 site), resting state</td><td>71 / 61<br>(35/32 + 36/29)</td>
<td>26 spectral features: relative δ, θ, α, β power (global + 4 scalp regions), log θ/α ratio (global + regions), peak alpha frequency</td></tr>
<tr><td>MRI</td><td>ADNI, baseline 3T, CN age/sex-matched 2:1</td><td>294 / 576</td>
<td>321 UCSF FreeSurfer 7 regional measures (volume, area, thickness); volumes ÷ intracranial volume; 2 near-empty columns dropped</td></tr>
</tbody></table>
<p class="note">No subject has both modalities. EEG was preprocessed identically for both datasets (average reference,
1–30 Hz, 256 Hz, ≤300 s, Welch PSD). Age, sex and education are <i>not</i> model inputs because they are confounded
with diagnosis in BrainLat. Details: <code>reports/data_introduction.pdf</code>.</p>

<h2>2. Method</h2>
<div class="two">
<div>
<h3>Model</h3>
<ul>
<li><b>Encoders</b>: one MLP per modality (EEG 26→32→16; MRI 321→64→16; ReLU, dropout {hp['dropout']}) into a shared 16-d embedding.</li>
<li><b>Shared head</b>: one linear AD/CN classifier used by both modalities, so “AD” means the same direction in the embedding.</li>
<li><b>Cross-modal supervised contrastive loss</b> (τ = {hp['tau']}): each EEG embedding is pulled towards MRI embeddings with the same label and away from the other label, and vice versa. No subject pairing is used — only labels.</li>
<li><b>Loss</b> = BCE<sub>target</sub> + {hp['w_aux']}·BCE<sub>auxiliary</sub> + {hp['lam']}·contrastive; BCE class-weighted for imbalance.</li>
<li>Full-batch Adam, lr {hp['lr']}, weight decay {hp['wd']}, {int(float(hp['epochs']))} epochs. All hyperparameters fixed before running; nothing was tuned on test folds.</li>
<li>At test time a subject is predicted <b>from its own modality only</b>.</li>
</ul>
</div>
<div>
<h3>Evaluation</h3>
<ul>
<li><b>EEG CV</b>: 5-fold × 10 repeats over all EEG subjects, stratified by dataset × diagnosis; all MRI subjects are auxiliary training data.</li>
<li><b>EEG transfer</b>: train on one EEG dataset (+ MRI), test on the other; 10 seeds.</li>
<li><b>MRI CV</b>: 5-fold × 3 repeats over ADNI; all EEG subjects auxiliary.</li>
<li>Imputation and z-scoring fitted on training rows only (EEG per dataset; the unseen transfer dataset is z-scored with its own, label-free statistics).</li>
<li>Metric: AUC of out-of-fold predictions per repeat; every model uses the same splits, so differences are paired.</li>
<li><b>Comparisons</b>: logistic regression and the same network without the other modality; ablations without the contrastive loss, with auxiliary labels shuffled, and with auxiliary nodes that carry only age + sex.</li>
</ul>
</div>
</div>

<h2 class="newpage">3. Results</h2>
<figure><img src="data:image/png;base64,{f1}" alt="AUC per model for each evaluation track">
<figcaption>Figure 1. AUC per model (mean ± SD over repeats). Orange = joint EEG + MRI model; dashed line = logistic regression.</figcaption></figure>
<figure><img src="data:image/png;base64,{f2}" alt="Paired AUC differences of the joint model">
<figcaption>Figure 2. Paired AUC difference of the joint model against each comparison (same splits). Right of 0 = joint better. If MRI carried useful information, the joint model should beat the shuffled and age/sex-only ablations clearly; it does not.</figcaption></figure>
<div class="keep"><h3>Joint model vs comparisons (paired)</h3>{table(d)}</div>
<p class="note">Per-model AUC for every track is in Figure 1 and <code>results/option1/summary.csv</code>.</p>

<div class="keep"><h3>EEG CV AUC within each EEG dataset</h3>{table(within, index=True)}</div>

<h2>4. Interpretation</h2>
<ul>
<li><b>EEG</b> separates AD from CN with AUC ≈ {g('eeg_cv','logreg'):.2f} overall (≈0.80 within ds004504, ≈0.65 within BrainLat) and transfers across datasets at {g('eeg_xfer:ds004504->BrainLat','logreg'):.2f}–{g('eeg_xfer:BrainLat->ds004504','logreg'):.2f}.</li>
<li><b>MRI</b> regional features reach AUC ≈ {g('mri_cv','mlp'):.2f}; adding EEG as auxiliary data changes nothing ({dd('mri_cv','network'):+.3f}).</li>
<li>The joint model's gain over the EEG-only network (+{dd('eeg_cv','network'):.3f}) disappears against the ablations: the shared head alone adds 0.000, the shuffled-label ablation keeps most of the gain, and age/sex-only auxiliary nodes do better than MRI. The contrastive term regularizes the small EEG network, which otherwise overfits; logistic regression gets the same benefit for free.</li>
<li>This is the controlled negative result anticipated in the proposal: with unpaired cohorts, the label-aligned shared embedding does not transfer disease information between modalities beyond regularization.</li>
</ul>

<h2>5. Limitations and next steps</h2>
<ul>
<li>Hyperparameters were fixed, not tuned (nested CV would tune them without test leakage); the EEG-only network had no early stopping, so logistic regression is the fairer baseline.</li>
<li>132 EEG subjects: AUC differences of ≈0.01 are within noise (SD across repeats 0.01–0.03).</li>
<li>BrainLat confounds (recording type matches diagnosis, AD older and less educated) limit what the BrainLat numbers mean.</li>
<li>Next: nested-CV tuning, EEG connectivity features, global-only EEG features for cross-dataset robustness, feature importance; and paired EEG–MRI fusion once BrainLat MRI download access is granted.</li>
</ul>
{ACKNOWLEDGMENTS_HTML}
"""

    css = f"""
:root {{ --bg:#f6f5f2; --card:{SURFACE}; --ink:{INK}; --ink2:{INK2}; --line:{GRID}; color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
       font: 15px/1.5 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
main {{ max-width: 1000px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 26px; margin: 4px 0 6px; }}
h2 {{ font-size: 19px; margin: 28px 0 8px; padding-top: 10px; border-top: 1px solid var(--line); }}
h3 {{ font-size: 15px; margin: 16px 0 6px; }}
.eyebrow {{ text-transform: uppercase; letter-spacing: .08em; font-size: 12px; color: var(--ink2); margin: 0; }}
.meta, .note {{ color: var(--ink2); font-size: 13px; }}
code {{ font: 12px ui-monospace, Menlo, monospace; background: #ecebe6; padding: 1px 4px; border-radius: 4px; }}
.callout {{ background: var(--card); border: 1px solid var(--line); border-left: 4px solid {COL['AD']};
           border-radius: 8px; padding: 12px 16px; margin: 16px 0; }}
.two {{ display: flex; gap: 24px; }} .two > div {{ flex: 1; }}
ul {{ padding-left: 20px; margin: 6px 0; }} li {{ margin: 3px 0; }}
.tbl {{ width:100%; border-collapse: collapse; background: var(--card); font-size: 12.5px;
        border: 1px solid var(--line); margin: 8px 0; }}
.tbl th, .tbl td {{ text-align:left; padding: 6px 9px; border-bottom: 1px solid var(--line); vertical-align: top; }}
.tbl th {{ background: #efeee9; font-weight: 600; }}
figure {{ margin: 14px 0; background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
figure img {{ width: 100%; display: block; }}
figcaption {{ color: var(--ink2); font-size: 12.5px; margin-top: 6px; }}
@media (max-width: 700px) {{ .two {{ display: block; }} }}
@media print {{
  body {{ background: #fff; font-size: 12px; }} main {{ max-width: none; padding: 0; }}
  .two {{ display: -webkit-box; }} .two > div {{ -webkit-box-flex: 1; width: 50%; padding-right: 12px; }}
  .tbl {{ font-size: 10.5px; }}
  h2, h3 {{ page-break-after: avoid; }}
  h2.newpage {{ page-break-before: always; border-top: 0; margin-top: 0; }}
  figure, tr, .keep, .callout {{ page-break-inside: avoid; }}
}}
"""
    body = body.replace(" <b>", "&nbsp;<b>").replace(" <i>", "&nbsp;<i>")
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Option 1 Results</title><style>{css}</style></head>"
            f"<body><main>{body}</main></body></html>")
    path = os.path.join(out, "option1_results.html")
    with open(path, "w") as f:
        f.write(page)
    print(f"wrote {path}")
    if shutil.which("wkhtmltopdf"):
        pdf = path[:-5] + ".pdf"
        subprocess.run(["wkhtmltopdf", "--quiet", "--print-media-type", "-s", "Letter",
                        "-T", "14mm", "-B", "14mm", "-L", "12mm", "-R", "12mm",
                        "--footer-center", "[page] / [topage]", "--footer-font-size", "8",
                        path, pdf], check=True)
        print(f"wrote {pdf}")
    else:
        print("wkhtmltopdf not found: PDF skipped (module load wkhtmltopdf)")


if __name__ == "__main__":
    main()
