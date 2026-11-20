"""
PDF progress report: two-channel (EEG + MRI) pretrain-then-fuse model vs the single-channel
and paired-only models, with methods, testing protocol, results and interpretation.

Reads the out-of-fold predictions of the paired run and the pretrain run (participant-level,
in paths.work of each config, outside the repo) and the aggregate tables in results/.
The PDF itself contains aggregate numbers only (no participant IDs).

  python -m crossad.reports.make_two_channel_report \
      [--paired-config configs/paired_config.yaml] [--pretrain-config configs/pretrain_config.yaml] \
      [--out reports/two_channel_report.pdf]
"""
import argparse
import os
import tempfile
from datetime import date

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, recall_score
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)

ap = argparse.ArgumentParser()
ap.add_argument("--paired-config", default="configs/paired_config.yaml")
ap.add_argument("--pretrain-config", default="configs/pretrain_config.yaml")
ap.add_argument("--out", default="reports/two_channel_report.pdf")
args = ap.parse_args()
with open(args.paired_config) as f:
    PCFG = yaml.safe_load(f)
with open(args.pretrain_config) as f:
    TCFG = yaml.safe_load(f)
OUT = args.out
TMP = tempfile.mkdtemp(prefix="two_channel_figs_")
RES = TCFG["paths"]["results"]

P = pd.read_csv(os.path.join(PCFG["paths"]["work"], "oof_predictions.csv"))
T = pd.read_csv(os.path.join(TCFG["paths"]["work"], "oof_predictions.csv"))
CMP = pd.read_csv(os.path.join(RES, "comparisons.csv"))
CMP = CMP[CMP.metric == "macro_f1"].set_index(["model_a", "model_b"])
ENC = pd.read_csv(os.path.join(RES, "pretrain_encoders.csv")).set_index("encoder")


def cmp(a, b):
    r = CMP.loc[(a, b)]
    return r.diff_a_minus_b, r.ci95_low, r.ci95_high, r.boot_frac_gt0


def ci(a, b):
    d, lo, hi, _ = cmp(a, b)
    return f"{d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})"



def metrics(d):
    rows = []
    for _, g in d.groupby("repeat"):
        yh = (g.p1 >= 0.5).astype(int)
        rows.append(dict(f1=f1_score(g.y, yh, average="macro"),
                         ba=balanced_accuracy_score(g.y, yh),
                         auc=roc_auc_score(g.y, g.p1),
                         sCN=recall_score(g.y, yh, pos_label=0),
                         sAD=recall_score(g.y, yh, pos_label=1)))
    o = pd.DataFrame(rows)
    return dict(f1=o.f1.mean(), sd=o.f1.std(ddof=1), ba=o.ba.mean(), auc=o.auc.mean(),
                sCN=o.sCN.mean(), sAD=o.sAD.mean())


M = {}
for D in (P, T):
    for m, g in D.groupby("model"):
        M[m] = metrics(g)

# ---------- palette ----------
GREY = "#9a9a94"
LIGHT = "#cfcfc9"
BLUE = "#2a6fdb"
INK = "#1d1d1b"
SUB = "#5c5c58"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": LIGHT,
                     "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB})

# ---------- Figure 1: single vs two channel, previous vs now ----------
groups = [("MRI only (1 channel)", "mlp_mri", "pt_mri_lr"),
          ("EEG only (1 channel)", "mlp_eeg", "pt_eeg_lr"),
          ("MRI + EEG (2 channels)", "fusion_concat", "pt_fusion_concat")]
fig, ax = plt.subplots(figsize=(7.0, 2.9))
for i, (lab, prev, now) in enumerate(groups):
    y = 2 - i
    two = i == 2
    for m, dy, col, tag in [(prev, 0.17, LIGHT if not two else "#a9c4ee", "previous (paired-only)"),
                            (now, -0.17, GREY if not two else BLUE, "now (pretrained encoders)")]:
        v = M[m]
        ax.plot([v["f1"] - v["sd"], v["f1"] + v["sd"]], [y + dy] * 2, color=col, lw=2.2,
                solid_capstyle="round")
        ax.plot(v["f1"], y + dy, "o", color=col, ms=8)
        ax.text(v["f1"] + v["sd"] + 0.008, y + dy, f'{v["f1"]:.3f}  {tag}', va="center",
                fontsize=8, color=INK if (two and m == now) else SUB,
                fontweight="bold" if (two and m == now) else "normal")
ax.axvline(0.5, color=LIGHT, ls="--", lw=1)
ax.text(0.5, 2.62, "chance", ha="center", fontsize=8, color=SUB)
ax.set_yticks([2, 1, 0])
ax.set_yticklabels([g[0] for g in groups], fontsize=9, color=INK)
ax.set_xlim(0.4, 1.2)
ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_ylim(-0.5, 2.8)
ax.set_xlabel("macro-F1 (mean ± SD over 10 repeats of 5-fold CV, n = 25)")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("Two channels reach 0.840 macro-F1; the best single channel 0.784",
             loc="left", fontsize=10.5, color=INK, fontweight="bold")
fig.tight_layout()
fig.savefig(TMP + "/fig1.png", dpi=220, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ---------- Figure 2: per-participant correctness ----------
def correct(D, m):
    g = D[D.model == m].groupby("id")
    acc = g.apply(lambda x: ((x.p1 >= 0.5).astype(int) == x.y).mean())
    return acc, g.y.first()

aM, y = correct(T, "pt_mri_lr")
aE, _ = correct(T, "pt_eeg_lr")
aF, _ = correct(T, "pt_fusion_concat")
df = pd.DataFrame({"MRI only": aM, "EEG only": aE, "MRI + EEG": aF, "y": y})
df["key"] = (df["MRI only"] >= 0.5).astype(int) * 2 + (df["EEG only"] >= 0.5)
df = df.sort_values(["key", "MRI only", "EEG only"])
wrongM = df["MRI only"] < 0.5
wrongE = df["EEG only"] < 0.5
wrongF = df["MRI + EEG"] < 0.5
stats = dict(nM=int(wrongM.sum()), nE=int(wrongE.sum()), nBoth=int((wrongM & wrongE).sum()),
             nF=int(wrongF.sum()), fixed=int((wrongM & ~wrongF).sum()),
             broke=int((~wrongM & wrongF).sum()))

fig, ax = plt.subplots(figsize=(7.0, 2.15))
cols = ["MRI only", "EEG only", "MRI + EEG"]
for r, c in enumerate(cols):
    for j, (pid, row) in enumerate(df.iterrows()):
        v = row[c]
        ok = v >= 0.5
        face = (BLUE if c == "MRI + EEG" else GREY) if ok else "#ffffff"
        edge = (BLUE if c == "MRI + EEG" else GREY)
        ax.add_patch(plt.Rectangle((j + 0.08, 2 - r + 0.12), 0.84, 0.76, facecolor=face,
                                   edgecolor=edge, lw=1.0, alpha=0.35 + 0.65 * v if ok else 1))
        if not ok:
            ax.text(j + 0.5, 2 - r + 0.5, "×", ha="center", va="center", color="#c0392b",
                    fontsize=9, fontweight="bold")
ax.set_xlim(0, len(df))
ax.set_ylim(-0.1, 3.05)
ax.set_yticks([2.5, 1.5, 0.5])
ax.set_yticklabels(cols, color=INK)
ax.set_xticks([j + 0.5 for j in range(len(df))])
ax.set_xticklabels([("AD" if v else "CN") for v in df.y], fontsize=6.5)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(f'MRI and EEG fail on different people: {stats["nM"]} MRI errors, '
             f'{stats["nE"]} EEG errors, {stats["nBoth"]} shared',
             loc="left", fontsize=10.5, color=INK, fontweight="bold")
ax.set_xlabel("25 paired participants (label = true diagnosis). Filled = correct in ≥ 50% of 10 repeats; × = wrong", fontsize=7.5)
fig.tight_layout()
fig.savefig(TMP + "/fig2.png", dpi=220, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ---------- Figure 3: bootstrap differences ----------
comp = [(lab,) + cmp(a, b)[:3] for lab, a, b in [
    ("2-ch (pretrained) vs 1-ch EEG (pretrained)", "pt_fusion_concat", "pt_eeg_lr"),
    ("2-ch (pretrained) vs 1-ch MRI MLP (previous)", "pt_fusion_concat", "mlp_mri"),
    ("2-ch (pretrained) vs 2-ch MLP fusion (previous)", "pt_fusion_concat", "fusion_concat"),
    ("2-ch (pretrained) vs 1-ch MRI (pretrained)", "pt_fusion_concat", "pt_mri_lr"),
    ("1-ch MRI: pretrained vs previous MLP", "pt_mri_lr", "mlp_mri"),
    ("1-ch EEG: pretrained vs previous MLP", "pt_eeg_lr", "mlp_eeg")]]
fig, ax = plt.subplots(figsize=(7.0, 2.4))
for i, (lab, d, lo, hi) in enumerate(comp):
    yy = len(comp) - 1 - i
    sig = lo > 0 or hi < 0
    c = BLUE if sig else GREY
    ax.plot([lo, hi], [yy, yy], color=c, lw=2.2, solid_capstyle="round")
    ax.plot(d, yy, "o", color=c, ms=7)
    ax.text(0.40, yy, f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=8,
            color=INK if sig else SUB, fontweight="bold" if sig else "normal")
ax.axvline(0, color=SUB, lw=0.8)
ax.set_yticks(range(len(comp)))
ax.set_yticklabels([c[0] for c in comp][::-1], color=INK, fontsize=8)
ax.set_xlim(-0.25, 0.68)
ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3])
ax.set_xlabel("Δ macro-F1 with participant-level bootstrap 95% CI (2,000 resamples)")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("Only the gap over EEG alone is clear with n = 25",
             loc="left", fontsize=10.5, color=INK, fontweight="bold")
fig.tight_layout()
fig.savefig(TMP + "/fig3.png", dpi=220, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)


# ---------- Figure 4: ROC curves (performance visualisation) ----------
from sklearn.metrics import roc_curve, auc as _auc

fig, ax = plt.subplots(figsize=(3.6, 3.4))
curves = [("pt_eeg_lr", T, "EEG only", LIGHT, 1.6),
          ("pt_mri_lr", T, "MRI only", GREY, 1.6),
          ("pt_fusion_concat", T, "MRI + EEG", BLUE, 2.4)]
for m, D, lab, col, lw in curves:
    g = D[D.model == m]
    fpr, tpr, _ = roc_curve(g.y, g.p1)
    ax.plot(fpr, tpr, color=col, lw=lw, label=f"{lab}  (pooled AUC {_auc(fpr, tpr):.2f})")
ax.plot([0, 1], [0, 1], color=LIGHT, ls="--", lw=1)
ax.set_xlabel("False-positive rate (CN called AD)")
ax.set_ylabel("True-positive rate (AD detected)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.01)
ax.legend(loc="lower right", fontsize=7.5, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("ROC, pretrained models", loc="left", fontsize=10, color=INK, fontweight="bold")
fig.tight_layout()
fig.savefig(TMP + "/fig4.png", dpi=220, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# confusion matrix of the main model, summed over the 10 repeats
g = T[T.model == "pt_fusion_concat"]
yh = (g.p1 >= 0.5).astype(int)
cm = pd.crosstab(g.y, yh).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values
fig, ax = plt.subplots(figsize=(3.0, 3.4))
rown = cm / cm.sum(axis=1, keepdims=True)
for i in range(2):
    for j in range(2):
        good = i == j
        ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=BLUE if good else "#c0392b",
                                   alpha=0.15 + 0.6 * rown[i, j], edgecolor="white", lw=2))
        ax.text(j + 0.5, 1 - i + 0.58, f"{rown[i, j]:.0%}", ha="center", va="center",
                fontsize=13, color=INK, fontweight="bold")
        ax.text(j + 0.5, 1 - i + 0.32, f"{cm[i, j]} of {cm[i].sum()}", ha="center", va="center",
                fontsize=7.5, color=SUB)
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5])
ax.set_xticklabels(["predicted CN", "predicted AD"])
ax.set_yticks([1.5, 0.5])
ax.set_yticklabels(["true CN", "true AD"])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Main model, 10 repeats", loc="left", fontsize=10, color=INK, fontweight="bold")
fig.tight_layout()
fig.savefig(TMP + "/fig5.png", dpi=220, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ---------- PDF ----------
ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=17, leading=21, alignment=0,
                    textColor=colors.HexColor(INK), spaceAfter=4)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, leading=16,
                    textColor=colors.HexColor(INK), spaceBefore=10, spaceAfter=4)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=10.5, leading=13,
                    textColor=colors.HexColor(INK), spaceBefore=6, spaceAfter=2)
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.6, leading=13.2,
                      textColor=colors.HexColor(INK), spaceAfter=5)
META = ParagraphStyle("m", parent=BODY, textColor=colors.HexColor(SUB), fontSize=8.8)
BUL = ParagraphStyle("bl", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=3)
CELL = ParagraphStyle("c", parent=BODY, fontSize=8.3, leading=10.4, spaceAfter=0)
CELLB = ParagraphStyle("cb", parent=CELL, fontName="Helvetica-Bold")
CAP = ParagraphStyle("cap", parent=META, fontSize=7.8, leading=10)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(data, widths, bold_rows=()):
    rows = [[Paragraph(str(c), CELLB if (r == 0 or r in bold_rows) else CELL) for c in row]
            for r, row in enumerate(data)]
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(SUB)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor(LIGHT)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        *[("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eaf1fc")) for r in bold_rows],
    ]))
    return t


def img(path, w):
    iw, ih = ImageReader(path).getSize()
    return Image(path, width=w, height=w * ih / iw)


def fmt(m):
    v = M[m]
    return [f'{v["f1"]:.3f} ± {v["sd"]:.3f}', f'{v["ba"]:.3f}', f'{v["auc"]:.3f}',
            f'{v["sCN"]:.2f}', f'{v["sAD"]:.2f}']


mf, mm, me = M["pt_fusion_concat"], M["pt_mri_lr"], M["pt_eeg_lr"]
W = 7.0 * inch
story = []
story.append(Paragraph("CROSS-AD progress report: two-channel EEG + MRI model", H1))
story.append(Paragraph(
    "CAS CS 506 · AD vs cognitively normal (CN) on BrainLat · covers the work since the "
    "Option 1 report (22 Oct 2026): the paired-cohort run (13 Nov, "
    "<font face='Courier'>derivatives/paired</font>) and the pretrain-then-fuse run (18 Nov, "
    f"<font face='Courier'>derivatives/pretrain</font>) · generated {date.today():%d %b %Y}", META))
story.append(Spacer(1, 6))

# ---------------- Summary ----------------
story.append(Paragraph("Summary", H2))
story.append(Paragraph(
    f'We now have a genuinely paired EEG + MRI model. On the 25 BrainLat participants who have both '
    f'modalities, the two-channel model reaches <b>macro-F1 {mf["f1"]:.3f}</b> (ROC AUC '
    f'{mf["auc"]:.3f}), against {mm["f1"]:.3f} for the best MRI-only model and '
    f'{M["mlp_eeg"]["f1"]:.3f} for the best EEG-only model, all scored on identical test folds.', BODY))
story += bullets([
    '<b>What changed since Option 1.</b> Option 1 could only align <i>unpaired</i> cohorts (ADNI MRI, '
    'BrainLat/ds004504 EEG) and found no cross-modal benefit. We have since built a paired cohort '
    'from BrainLat, segmented its raw T1 scans ourselves, and pretrained each channel on all '
    'unpaired BrainLat data before fusing on the paired people.',
    f'<b>Why two channels help.</b> MRI and EEG fail on different people: {stats["nM"]} MRI errors, '
    f'{stats["nE"]} EEG errors, only {stats["nBoth"]} shared. Fusion fixes {stats["fixed"]} of the '
    f'{stats["nM"]} MRI errors.',
    '<b>Methods tried:</b> eight paired-only models (logistic regression, MLPs, two fusion MLPs, an '
    'age + sex baseline) and four pretrain-then-fuse variants, plus site-restricted re-runs.',
    f'<b>How sure we are.</b> The gain over EEG alone is statistically clear: '
    f'{ci("pt_fusion_concat", "pt_eeg_lr")}. The gain over MRI alone, '
    f'{ci("pt_fusion_concat", "pt_mri_lr")}, is likely but its CI includes zero: 25 test people '
    'are too few to prove it.',
])

# ---------------- 1. Progress ----------------
story.append(Paragraph("1. Progress since the Option 1 report", H2))
story.append(table([
    ["Date", "Step", "Outcome"],
    ["22 Oct", "Option 1: shared EEG–MRI embedding from unpaired cohorts (ADNI MRI, 870; "
     "BrainLat + ds004504 EEG, 132)",
     "MRI AUC 0.91, EEG AUC 0.73. Ablations showed no information transfer between modalities: "
     "without subject pairing, fusion cannot learn complementarity."],
    ["Oct–Nov", "Built a paired cohort: matched BrainLat MRI and EEG records for 761 candidates",
     "25 AD/CN participants (15 AD, 10 CN, all site CLB) with both a usable EEG and a T1. Most "
     "exclusions: no documented EEG ID (441) or non-AD/CN diagnosis (246)."],
    ["13 Nov", "Paired-only run: SynthSeg MRI features + EEG features for the 25; 8 models",
     "Best single channel MRI MLP 0.748; best fusion MLP concat 0.777. Fusion gain small: models "
     "trained on ~20 people per fold."],
    ["18 Nov", "Pretrain-then-fuse: SynthSeg on 380 more T1s; EEG for 72 recordings; encoders "
     "pretrained on all unpaired BrainLat data, then fused on the paired cohort",
     f"Two-channel model {mf['f1']:.3f} macro-F1. Found and removed a duplicate scan "
     "(an unpaired scan identical to a paired participant's T1). Site-restricted pools confirm the result."],
], [0.65 * inch, 2.9 * inch, 3.45 * inch]))

# ---------------- 2. Features ----------------
story.append(Paragraph("2. Data and features", H2))
story.append(Paragraph(
    "Both channels use a small set of hand-crafted, clinically motivated features rather than raw "
    "signals or images. With at most a few hundred people per modality, and 25 paired, an end-to-end "
    "network would overfit. Interpretable features also let us check that the model uses known "
    "disease markers.", BODY))
story.append(table([
    ["Channel", "Features", "Why these"],
    ["MRI (16)", "FreeSurfer 8.2 SynthSeg (<font face='Courier'>--parc --robust</font>) on each T1; "
     "hippocampus, amygdala, inferior-lateral and lateral ventricles, entorhinal, parahippocampal, "
     "fusiform, inferior/middle temporal, temporal pole, precuneus, isthmus and posterior cingulate, "
     "inferior parietal, total cortex, white matter. Left + right summed, divided by intracranial "
     "volume (ICV).",
     "These are the regions where AD atrophy appears first (medial temporal lobe) and spreads "
     "(temporal–parietal cortex, ventricles enlarge). Dividing by ICV removes head size. BrainLat "
     "ships raw T1s with no FreeSurfer tables, so we segmented them; SynthSeg is robust to "
     "different scanners and resolutions. Option 1's 321 ADNI features would be far too many "
     "for 25 people."],
    ["EEG (26)", "Resting state, 1–30 Hz, 256 Hz, average reference, ≤300 s, 2 s epochs with bad-"
     "channel and bad-epoch rejection. Relative δ, θ, α, β power and θ/α ratio in global, frontal, "
     "central, temporal and posterior regions; peak alpha frequency. Median over epochs.",
     "AD slows the EEG: θ power rises, α power falls, and the alpha peak moves lower. Relative "
     "power and regional averages are robust to amplitude differences between sites and "
     "electrode caps. This is the same feature set as Option 1, so the results are comparable."],
], [0.7 * inch, 3.05 * inch, 3.25 * inch]))
story.append(Paragraph(
    "QC: all 72 EEG recordings passed (median 150 clean epochs, 2 bad channels); 7 of 405 MRI "
    "segmentations were flagged by SynthSeg QC (5 AD, 2 CN). Features for the 25 paired people are "
    "identical between the two runs (max |difference| 0 for MRI, 8.9e-16 for EEG).", CAP))

# ---------------- 3. Modeling methods ----------------
story.append(Paragraph("3. Modeling methods", H2))
story.append(Paragraph(
    "We tried two families of models. Each method is listed with the reason it was chosen.", BODY))
story.append(Paragraph("3.1 Paired-only models (13 Nov run)", H3))
story.append(table([
    ["Model", "Input", "Why"],
    ["Logistic regression (L2, class-balanced)", "MRI, EEG, or both concatenated",
     "Few parameters, so the strongest baseline for ~20 training people; coefficients are "
     "interpretable."],
    ["MLP", "MRI or EEG", "Tests whether nonlinear interactions between regions or bands help."],
    ["MLP fusion, concat", "MRI + EEG", "Simplest early fusion: one network sees both channels."],
    ["MLP fusion, gated", "MRI + EEG", "Learns per-person weights for each channel, in case one "
     "channel is unreliable for some people."],
    ["Logistic regression on age + sex", "demographics", "Sanity check: shows how much signal "
     "comes from demographics alone."],
], [1.9 * inch, 1.5 * inch, 3.6 * inch]))
story.append(Paragraph("3.2 Pretrain-then-fuse (18 Nov run, main model)", H3))
story.append(Paragraph(
    "The paired-only models learn everything from about 20 people per fold. That is the bottleneck, "
    "but only the <i>fusion</i> step needs paired data; each channel's encoder does not. So we "
    "split the model into two stages:", BODY))
story += bullets([
    "<b>Stage 1, pretrain each encoder on its own pool.</b> MRI: 404 BrainLat CN/AD scans from 8 "
    "sites. EEG: 72 recordings from 2 sites. Each encoder is a small MLP (MRI 16→16→8, EEG 26→16→8; "
    "ReLU, dropout 0.3; 408 and 568 parameters) with a temporary 8→2 head, trained with "
    "class-balanced cross-entropy (full-batch AdamW, lr 0.003, 300 epochs). Weight decay "
    "{0.001, 0.01, 0.1} is picked on a 20% validation split, then the encoder is refitted on the "
    "whole pool.",
    "<b>Stage 2, fuse on the paired cohort.</b> Encoders are frozen. Each person's two 8-dim "
    "embeddings are concatenated, z-scored and fed to an L2 logistic regression (17 parameters, "
    "class-balanced). C ∈ {0.01, 0.1, 1} is tuned by inner CV on the training fold.",
    "<b>Why this design:</b> the encoders learn from 16× (MRI) and 3× (EEG) more people. The part "
    "fitted on paired data is small enough not to overfit 20 people, and the 8-dim bottleneck "
    "keeps the encoders from memorising their pools.",
    "<b>Variants</b> to check the design: logit stacking (fuse the two pretrained heads' outputs "
    "instead of embeddings), late averaging (no paired fitting at all), single-channel versions "
    "(one encoder → LR), and MRI pools restricted to sites with both diagnoses (CLB+COB+PE) or to "
    "CLB only, because several sites contribute only AD or only CN.",
])

# ---------------- 4. Testing ----------------
story.append(Paragraph("4. How performance was tested", H2))
story += bullets([
    "<b>Cross-validation:</b> 10 repeats × 5-fold stratified CV on the 25 paired people. All 19 "
    "models use exactly the same splits (verified fold by fold), so their differences are paired.",
    "<b>No leakage:</b> in every split, the test people are removed from both pretraining pools "
    "before z-scoring, encoder training and head fitting. A near-duplicate scan screen (r > 0.999) "
    "caught one unpaired scan identical to a paired participant's; it was removed.",
    "<b>Metrics:</b> macro-F1 is the primary metric because classes are imbalanced (15 AD, 10 CN) "
    "and both errors matter. We also report balanced accuracy, ROC AUC and per-class sensitivity, "
    "as mean ± SD over the 10 repeats.",
    "<b>Model comparisons:</b> participant-level stratified bootstrap (2,000 resamples) of the "
    "repeat-averaged difference, giving a 95% CI and P(Δ > 0).",
    "<b>Controls:</b> age + sex baseline, site-restricted pools, and the paired-only models as "
    "the no-pretraining reference.",
])

# ---------------- 5. Results ----------------
story.append(Paragraph("5. Results", H2))
story.append(img(TMP + "/fig1.png", 6.6 * inch))
story.append(Paragraph("Dots: mean macro-F1 over 10 repeats; whiskers ± 1 SD. Same 25 people and "
                       "folds in both runs.", CAP))
story.append(Spacer(1, 4))
hdr = ["Model", "Channels", "Run", "macro-F1 ± SD", "Bal. acc", "ROC AUC", "Sens. CN", "Sens. AD"]
rows = [hdr]
spec = [("Pretrained encoders → concat → LR (main)", "MRI + EEG", "18 Nov", "pt_fusion_concat"),
        ("  same, MRI pool CLB only", "MRI + EEG", "18 Nov", "pt_fusion_concat@clb_only"),
        ("  same, MRI pool CLB+COB+PE", "MRI + EEG", "18 Nov", "pt_fusion_concat@mixed_sites"),
        ("Pretrained logits → LR (stacking)", "MRI + EEG", "18 Nov", "pt_fusion_stack"),
        ("Average of pretrained heads", "MRI + EEG", "18 Nov", "pt_late_avg"),
        ("Pretrained MRI encoder → LR", "MRI", "18 Nov", "pt_mri_lr"),
        ("MLP fusion (concat)", "MRI + EEG", "13 Nov", "fusion_concat"),
        ("LR on concatenated features", "MRI + EEG", "13 Nov", "logreg_concat"),
        ("MRI MLP", "MRI", "13 Nov", "mlp_mri"),
        ("LR on MRI features", "MRI", "13 Nov", "logreg_mri"),
        ("MLP fusion (gate)", "MRI + EEG", "13 Nov", "fusion_gate"),
        ("EEG MLP", "EEG", "13 Nov", "mlp_eeg"),
        ("Pretrained EEG encoder → LR", "EEG", "18 Nov", "pt_eeg_lr"),
        ("LR on EEG features", "EEG", "13 Nov", "logreg_eeg"),
        ("LR on age + sex (baseline)", "none", "13 Nov", "logreg_age_sex")]
for lab, ch, run, m in spec:
    rows.append([lab, ch, run] + fmt(m))
story.append(table(rows, [2.25 * inch, 0.78 * inch, 0.58 * inch, 0.95 * inch, 0.55 * inch,
                          0.6 * inch, 0.55 * inch, 0.55 * inch], bold_rows=(1,)))
story.append(Paragraph("Main model and its variants first, then by macro-F1; threshold 0.5. Pretrained-head-only "
                       "single-channel variants (no paired fitting) are in train.log.", CAP))

perf = Table([[img(TMP + "/fig4.png", 3.45 * inch), img(TMP + "/fig5.png", 2.9 * inch)]],
             colWidths=[3.6 * inch, 3.4 * inch])
perf.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM")]))
story.append(KeepTogether([perf, Paragraph(
    "Left: ROC curves from the out-of-fold predictions of all 10 repeats pooled (pooled AUC can "
    "differ slightly from the per-repeat mean in the table). Right: confusion matrix of the main "
    "model, summed over repeats; rows sum to 100%.", CAP)]))
story.append(KeepTogether([img(TMP + "/fig3.png", 6.4 * inch),
                           Paragraph("Blue = 95% CI excludes zero. From derivatives/pretrain/train.log. "
                                     "Also significant: concat beats stacking by "
                                     f"{-cmp('pt_fusion_stack', 'pt_fusion_concat')[0]:.3f} macro-F1 "
                                     f"(95% CI {-cmp('pt_fusion_stack', 'pt_fusion_concat')[2]:.3f} to "
                                     f"{-cmp('pt_fusion_stack', 'pt_fusion_concat')[1]:.3f}).", CAP)]))

# ---------------- 6. Interpretation ----------------
story.append(Paragraph("6. Interpretation", H2))
story.append(Paragraph("6.1 Why two channels beat one", H3))
story.append(Paragraph(
    "MRI measures structural atrophy; EEG measures functional slowing. They capture different "
    "aspects of the disease, so they fail on different people.", BODY))
story.append(img(TMP + "/fig2.png", W))
story += bullets([
    f'Only {stats["nBoth"]} of 25 people is misclassified by both channels. When MRI is wrong, EEG '
    f'is usually right, and the fused model uses that: {stats["fixed"]} of {stats["nM"]} MRI errors '
    f'are corrected at the cost of {stats["broke"]} new error ({stats["nF"]} left).',
    'The fused model is also more balanced: sensitivity 0.87 for CN and 0.83 for AD, against 0.82 / '
    '0.77 for MRI alone.',
    'In the paired-only run the fusion MLP had to learn this from ~20 people and corrected only 2 of '
    '5 MRI errors. With pretrained encoders, a 17-parameter head is enough.',
    'Concatenating embeddings beats stacking the two heads\' outputs (significant). The 8-dim '
    'embeddings keep information that the single AD/CN logit throws away.',
])
story.append(Paragraph("6.2 Why the results under- or outperform other approaches", H3))
story += bullets([
    '<b>MRI here (AUC 0.85) vs our Option 1 MRI (AUC 0.91 on 870 ADNI people).</b> The features '
    'are not the problem: same disease regions, fewer of them. The difference is data: 25 test '
    'people from one site, and 16 summary volumes instead of 321 FreeSurfer measures. A model '
    'trained on hundreds of ADNI subjects is the standard setting for MRI AD classification; '
    'ours is a much harder, smaller-sample setting.',
    '<b>EEG underperforms (macro-F1 ≈ 0.64–0.66, AUC ≈ 0.69–0.75).</b> This matches Option 1, '
    'where BrainLat EEG alone reached AUC 0.65 (vs 0.80 on ds004504). BrainLat recordings come '
    'from 2 sites with different protocols, and only 72 exist, so EEG pretraining did not help '
    f'(pool validation macro-F1 {ENC.loc["eeg", "val_f1_mean"]:.3f}, near chance).',
    '<b>Two-channel vs Option 1 joint model.</b> Option 1\'s shared embedding gained nothing from the '
    'other modality because no person had both. Here every test person has both scans, so the model '
    'can combine evidence within a person. That is the reason fusion now works.',
    f'<b>Pretrain-then-fuse vs paired-only fusion.</b> {cmp("pt_fusion_concat", "fusion_concat")[0]:+.3f} '
    'macro-F1: the encoders learn from 16× '
    'more MRI data. The pretrained MRI encoder alone, with no paired fitting, already matches the '
    'paired-only MRI MLP.',
    '<b>Not driven by site or demographics.</b> Age + sex alone score 0.475 (chance). Restricting '
    f'the MRI pool to mixed-diagnosis sites ({M["pt_fusion_concat@mixed_sites"]["f1"]:.3f}) or to CLB '
    f'only ({M["pt_fusion_concat@clb_only"]["f1"]:.3f}) changes little.',
])

# ---------------- 7. Limitations ----------------
story.append(Paragraph("7. Limitations and next steps", H2))
story += bullets([
    "All final numbers rest on 25 people from one site; CIs on model differences are about ±0.1 wide.",
    "EEG is the weak channel. Growing the EEG pool (e.g. adding ds004504 for pretraining) is the most "
    "likely way to make fusion clearly beat MRI alone.",
    "The fusion head's C is tuned on inner folds whose people were in the pretraining pool. This can "
    "only bias the choice of C, never the test fold.",
    "Next: feature importance for the fused model (which regions and bands drive each correction), "
    "more paired participants, and an external check on another paired dataset if one becomes available.",
])

story.append(Paragraph("Data sources and acknowledgments", H2))
story.append(Paragraph(
    "BrainLat: Prado P. et al., <i>The BrainLat project, a multimodal neuroimaging dataset of "
    "neurodegeneration from underrepresented backgrounds</i>, Scientific Data 10, 889 (2023); "
    "Synapse syn51549340. Option 1 results used ADNI (adni.loni.usc.edu; NIH U01 AG024904, DOD "
    "W81XWH-12-2-0012) and OpenNeuro ds004504 (Miltiadous A. et al., Data 8(6), 95, 2023). ADNI "
    "investigators contributed to the design and implementation of ADNI and/or provided data but did "
    "not participate in analysis or writing of this report.", CAP))


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor(SUB))
    c.drawString(0.75 * inch, 0.5 * inch, "CROSS-AD progress report · two-channel EEG + MRI")
    c.drawRightString(7.75 * inch, 0.5 * inch, f"Page {d.page}")
    c.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.7 * inch, bottomMargin=0.75 * inch,
                        title="CROSS-AD progress report: two-channel EEG + MRI model")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT, stats, cm.tolist())
