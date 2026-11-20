"""
HTML report for the pretrain-then-fuse experiment (model 1): cohorts, model structure,
training protocol, results and comparison with the paired-only models.
Reads only aggregate tables (results/pretrain, results/paired); writes one self-contained
HTML file to paths.report_dir.

  python -m crossad.reports.make_pretrain_report --config configs/pretrain_config.yaml
"""
import argparse
import html
import json
import os

import pandas as pd
import yaml

FAMILY = {  # model -> (family, label)
    "pt_fusion_concat": ("pt", "Pretrained encoders → concat → LR  (main)"),
    "pt_fusion_stack": ("pt", "Pretrained heads' logits → LR (stacking)"),
    "pt_mri_lr": ("pt", "Pretrained MRI encoder → LR"),
    "pt_eeg_lr": ("pt", "Pretrained EEG encoder → LR"),
    "pt_fusion_concat@mixed_sites": ("pt", "Concat fusion, MRI pool = CLB+COB+PE"),
    "pt_fusion_concat@clb_only": ("pt", "Concat fusion, MRI pool = CLB only"),
    "pt_late_avg": ("zs", "Average of pretrained heads (no paired fitting)"),
    "pt_mri_head": ("zs", "Pretrained MRI head (no paired fitting)"),
    "pt_eeg_head": ("zs", "Pretrained EEG head (no paired fitting)"),
    "pt_mri_head@mixed_sites": ("zs", "MRI head, pool = CLB+COB+PE"),
    "pt_mri_head@clb_only": ("zs", "MRI head, pool = CLB only"),
    "fusion_concat": ("po", "Paired-only MLP fusion (concat)"),
    "fusion_gate": ("po", "Paired-only MLP fusion (gate)"),
    "logreg_concat": ("po", "Paired-only LR, EEG+MRI features"),
    "mlp_mri": ("po", "Paired-only MRI MLP"),
    "mlp_eeg": ("po", "Paired-only EEG MLP"),
    "logreg_mri": ("po", "Paired-only LR, MRI features"),
    "logreg_eeg": ("po", "Paired-only LR, EEG features"),
}
FAM_NAME = {"pt": "Pretrained, head fitted on paired train",
            "zs": "Pretrained, no paired fitting",
            "po": "Paired-only (previous run)"}
FAM_VAR = {"pt": "--series-1", "zs": "--series-3", "po": "--series-2"}

CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--panel:#f4f3ef;--border:#dedcd5;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#7a7873;--grid:#e6e4de;
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--surface:#1a1a19;--panel:#232321;--border:#3a3936;--text-primary:#f0efec;
--text-secondary:#c3c2b7;--text-muted:#94928a;--grid:#302f2c;
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--accent:#3987e5}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--panel:#232321;
--border:#3a3936;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#94928a;
--grid:#302f2c;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--accent:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text-primary);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 10px;
padding-top:12px;border-top:1px solid var(--border)}h3{font-size:15px;margin:22px 0 6px}
p,li{color:var(--text-secondary)}.sub{color:var(--text-muted);margin:0 0 20px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:18px 0}
.kpi{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px}
.kpi b{display:block;font-size:24px;font-variant-numeric:tabular-nums}
.kpi span{font-size:12.5px;color:var(--text-muted)}
.box{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px 16px}
.tw{overflow-x:auto;margin:8px 0}table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:6px 9px;border-bottom:1px solid var(--border);text-align:right;
font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--text-muted);font-weight:600}
tr.main td{font-weight:650}td.l,th.l{text-align:left;white-space:normal;min-width:120px}code{font-size:12.5px;background:var(--panel);padding:1px 4px;
border-radius:4px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;
margin-right:6px;vertical-align:middle}.legend{display:flex;flex-wrap:wrap;gap:16px;
font-size:13px;color:var(--text-secondary);margin:6px 0}
svg text{fill:var(--text-secondary);font:12px system-ui,sans-serif}
svg .t{fill:var(--text-primary);font-weight:600}svg .m{fill:var(--text-muted);font-size:11px}
#tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--border);
border-radius:6px;padding:6px 9px;font-size:12.5px;display:none;color:var(--text-primary);
box-shadow:0 2px 8px rgba(0,0,0,.15)}
.hit{fill:transparent;cursor:default}
"""


def esc(x):
    return html.escape(str(x))


def table(df, fmt=None, main_rows=()):
    fmt = fmt or {}
    txt = {c for c in df.columns if df[c].dtype == object and not
           df[c].astype(str).str.match(r"^[\s+\-−\d.,±\[\]%]*$").all()}
    h = "<div class='tw'><table><tr>" + "".join(
        f"<th{' class=l' if c in txt else ''}>{esc(c)}</th>" for c in df.columns)
    h += "</tr>"
    for _, r in df.iterrows():
        cls = " class='main'" if r.iloc[0] in main_rows else ""
        cells = []
        for c in df.columns:
            v = r[c]
            cells.append(f"<td{' class=l' if c in txt else ''}>{esc(fmt[c](v) if c in fmt and pd.notna(v) else ('' if pd.isna(v) else v))}</td>")
        h += f"<tr{cls}>" + "".join(cells) + "</tr>"
    return h + "</table></div>"


def f3(v):
    return f"{v:.3f}"


def dot_chart(summ):
    """Horizontal dot plot: macro-F1 mean ± SD over repeats, one row per model."""
    rows = [m for m in FAMILY if m in summ.index]
    W, L, R, rh, top = 900, 330, 30, 26, 30
    H = top + rh * len(rows) + 34
    x0, x1 = 0.3, 1.0
    X = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    s = [f"<svg viewBox='0 0 {W} {H}' width='100%' role='img' "
         f"aria-label='Macro-F1 by model, mean and SD over 10 repeats'>"]
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        s.append(f"<line x1='{X(t):.1f}' x2='{X(t):.1f}' y1='{top - 8}' y2='{H - 30}' "
                 f"stroke='var(--grid)' stroke-width='1'/>"
                 f"<text x='{X(t):.1f}' y='{H - 14}' text-anchor='middle' class='m'>{t:.1f}</text>")
    s.append(f"<text x='{L}' y='14' class='m'>macro-F1 (mean ± SD over repeats) — "
             f"0.5 ≈ chance for 2 balanced classes</text>")
    for i, m in enumerate(rows):
        fam, lab = FAMILY[m]
        y = top + i * rh + rh / 2
        mu, sd = summ.loc[m, "macro_f1_mean"], summ.loc[m, "macro_f1_std"]
        col = f"var({FAM_VAR[fam]})"
        weight = " class='t'" if m == "pt_fusion_concat" else ""
        s.append(f"<text x='{L - 10}' y='{y + 4:.1f}' text-anchor='end'{weight}>{esc(lab)}</text>")
        s.append(f"<line x1='{X(mu - sd):.1f}' x2='{X(mu + sd):.1f}' y1='{y:.1f}' y2='{y:.1f}' "
                 f"stroke='{col}' stroke-width='2' stroke-linecap='round'/>")
        s.append(f"<circle cx='{X(mu):.1f}' cy='{y:.1f}' r='5' fill='{col}' "
                 f"stroke='var(--surface)' stroke-width='2'/>")
        tip = (f"{lab}|macro-F1 {mu:.3f} ± {sd:.3f}|bal. acc {summ.loc[m, 'balanced_accuracy_mean']:.3f}"
               f"|ROC AUC {summ.loc[m, 'roc_auc_mean']:.3f}|{FAM_NAME[fam]}")
        s.append(f"<rect class='hit' x='0' y='{y - rh / 2:.1f}' width='{W}' height='{rh}' "
                 f"data-tip='{esc(tip)}'/>")
    s.append("</svg>")
    legend = "<div class='legend'>" + "".join(
        f"<span><span class='dot' style='background:var({FAM_VAR[f]})'></span>{esc(n)}</span>"
        for f, n in FAM_NAME.items()) + "</div>"
    return legend + "".join(s)


def arch_svg(d_eeg, d_mri, hid, emb, n_eeg, n_mri):
    """Two-stage architecture diagram."""
    DASH = ' stroke-dasharray="5 4"'
    box = lambda x, y, w, h, t, sub="", dash=False: (
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='var(--panel)' "
        f"stroke='var(--border)' stroke-width='1.5'{DASH if dash else ''}/>"
        f"<text x='{x + w / 2}' y='{y + h / 2 - (6 if sub else -4)}' text-anchor='middle' class='t'>{t}</text>"
        + (f"<text x='{x + w / 2}' y='{y + h / 2 + 12}' text-anchor='middle' class='m'>{sub}</text>" if sub else ""))
    arr = lambda x1, y1, x2, y2: (f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='var(--text-muted)' "
                                  f"stroke-width='1.5' marker-end='url(#ah)'/>")
    s = ["<svg viewBox='0 0 940 430' width='100%' role='img' aria-label='Model architecture'>",
         "<defs><marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' "
         "markerHeight='7' orient='auto'><path d='M0,0L10,5L0,10z' fill='var(--text-muted)'/></marker></defs>",
         "<text x='10' y='18' class='t'>Stage 1 · pretraining (per outer split, test participants removed from both pools)</text>",
         "<text x='10' y='262' class='t'>Stage 2 · fusion on the paired cohort (encoders frozen)</text>"]
    for y, name, d, n in ((40, "EEG", d_eeg, n_eeg), (140, "MRI", d_mri, n_mri)):
        s.append(box(10, y, 170, 70, f"{name} pool", f"n ≈ {n} participants"))
        s.append(arr(180, y + 35, 215, y + 35))
        s.append(box(215, y, 150, 70, "z-score", f"{d} features"))
        s.append(arr(365, y + 35, 400, y + 35))
        s.append(box(400, y, 250, 70, f"{name} encoder  {d}→{hid}→{emb}",
                     "Linear · ReLU · Dropout 0.3 · Linear · ReLU"))
        s.append(arr(650, y + 35, 685, y + 35))
        s.append(box(685, y, 120, 70, "head", f"Linear {emb}→2"))
        s.append(arr(805, y + 35, 835, y + 35))
        s.append(box(835, y, 95, 70, "CN / AD", "balanced CE"))
    # stage 2
    s.append(box(10, 285, 170, 120, "Paired cohort", "25 participants · EEG + MRI"))
    s.append(arr(180, 320, 400, 320))
    s.append(arr(180, 370, 400, 370))
    s.append(box(400, 290, 250, 50, "frozen EEG encoder", f"→ z_EEG ∈ ℝ^{emb}", dash=True))
    s.append(box(400, 350, 250, 50, "frozen MRI encoder", f"→ z_MRI ∈ ℝ^{emb}", dash=True))
    s.append(arr(650, 315, 685, 340))
    s.append(arr(650, 375, 685, 350))
    s.append(box(685, 310, 120, 70, "concat", f"{2 * emb} dims"))
    s.append(arr(805, 345, 835, 345))
    s.append(box(835, 300, 95, 90, "LR head", "z-score · L2 LR"))
    s.append("</svg>")
    return "".join(s)


def n_params(d, hid, emb):
    return d * hid + hid + hid * emb + emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_config.yaml")
    ap.add_argument("--findings", default=None, help="optional HTML snippet with the findings")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    res = cfg["paths"]["results"]
    out_dir = cfg["paths"]["report_dir"]
    H = cfg["pretrain"]
    hid, emb = H["hidden"], H["emb_dim"]
    info = json.load(open(os.path.join(res, "run_info.json")))
    summ = pd.read_csv(os.path.join(res, "metrics_summary.csv"), index_col=0)
    comp = pd.read_csv(os.path.join(res, "comparisons.csv"))
    enc = pd.read_csv(os.path.join(res, "pretrain_encoders.csv"))
    counts = pd.read_csv(os.path.join(res, "pool_counts.csv"))
    d_eeg, d_mri = info["n_eeg_features"], info["n_mri_features"]
    n_eeg, n_mri = info["eeg_pool"]["n"], info["mri_pool"]["n"]

    main_m = summ.loc["pt_fusion_concat"]
    best_po = summ.loc[[m for m in ("fusion_concat", "mlp_mri", "logreg_concat") if m in summ.index],
                       "macro_f1_mean"].idxmax()
    kpis = [(f"{main_m.macro_f1_mean:.3f}", "macro-F1, pretrain-then-fuse (main)"),
            (f"{summ.loc[best_po, 'macro_f1_mean']:.3f}", f"macro-F1, best paired-only ({best_po})"),
            (f"{main_m.roc_auc_mean:.3f}", "ROC AUC, pretrain-then-fuse"),
            (f"{n_mri} / {n_eeg}", "MRI / EEG pretraining pools"),
            ("25", "paired test participants (15 AD, 10 CN)")]

    piv = counts.pivot_table(index=["modality", "site"], columns="dx", values="n",
                             fill_value=0).reset_index()
    piv["paired-cohort site"] = piv["site"].isin(["CLB", "CL"]).map({True: "yes", False: ""})

    # parameter table
    pt = pd.DataFrame([
        ["EEG encoder", f"Linear({d_eeg},{hid}) → ReLU → Dropout(0.3) → Linear({hid},{emb}) → ReLU",
         n_params(d_eeg, hid, emb), f"EEG pool (≈{n_eeg})"],
        ["EEG pretraining head", f"Linear({emb},2)", emb * 2 + 2, "EEG pool; discarded for concat"],
        ["MRI encoder", f"Linear({d_mri},{hid}) → ReLU → Dropout(0.3) → Linear({hid},{emb}) → ReLU",
         n_params(d_mri, hid, emb), f"MRI pool (≈{n_mri})"],
        ["MRI pretraining head", f"Linear({emb},2)", emb * 2 + 2, "MRI pool; discarded for concat"],
        ["Fusion head", f"StandardScaler → LogisticRegression(L2, balanced) on {2 * emb} dims",
         2 * emb + 1, "paired training fold (≈20 participants)"],
    ], columns=["Component", "Structure", "Trainable parameters", "Fitted on"])

    rows = [m for m in FAMILY if m in summ.index]
    mt = pd.DataFrame({"Model": [FAMILY[m][1] for m in rows],
                       "Family": [FAM_NAME[FAMILY[m][0]] for m in rows],
                       "macro-F1": [f"{summ.loc[m, 'macro_f1_mean']:.3f} ± {summ.loc[m, 'macro_f1_std']:.3f}" for m in rows],
                       "bal. acc": [f"{summ.loc[m, 'balanced_accuracy_mean']:.3f}" for m in rows],
                       "sens. CN": [f"{summ.loc[m, 'sensitivity_CN_mean']:.2f}" for m in rows],
                       "sens. AD": [f"{summ.loc[m, 'sensitivity_AD_mean']:.2f}" for m in rows],
                       "ROC AUC": [f"{summ.loc[m, 'roc_auc_mean']:.3f}" for m in rows]})

    lab = lambda m: FAMILY.get(m, ("", m))[1]
    c = comp.copy()
    c = c[c["metric"] == "macro_f1"]
    ct = pd.DataFrame({"A": c["model_a"].map(lab), "B": c["model_b"].map(lab),
                       "Δ macro-F1 (A − B)": c["diff_a_minus_b"].map(lambda v: f"{v:+.3f}"),
                       "95% CI": [f"[{a:+.3f}, {b:+.3f}]" for a, b in zip(c["ci95_low"], c["ci95_high"])],
                       "P(Δ>0) bootstrap": c["boot_frac_gt0"].map(lambda v: f"{v:.2f}")})

    enc = enc.rename(columns={"encoder": "Encoder", "n_pool_median": "pool n (median)",
                              "val_f1_mean": "pool val macro-F1", "val_f1_std": "SD"})
    findings = open(args.findings).read() if args.findings else ""

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pretrain-then-Fuse Report</title><style>{CSS}</style></head><body><main>
<h1>Pretrain-then-fuse EEG + MRI model</h1>
<p class="sub">CROSS-AD · BrainLat · CN vs AD · model 1 (unpaired pretraining, paired fusion) ·
{info['repeats']} repeats × {info['outer_folds']}-fold CV on the 25 paired participants</p>
<div class="kpis">{''.join(f"<div class='kpi'><b>{a}</b><span>{esc(b)}</span></div>" for a, b in kpis)}</div>
{findings}

<h2>1. Why this design</h2>
<p>Only 25 BrainLat participants have both a usable EEG recording and a T1 MRI, all from one
site (CLB). Concatenating EEG and MRI representations needs both modalities for the same person,
so the fused model itself can only be fitted and tested on those 25. What the unpaired data can do is
train the <em>encoders</em>: each modality's encoder is pretrained on every CN/AD participant
that has that modality, then frozen, and only a {2 * emb + 1}-parameter logistic-regression head is
fitted on the paired training fold.</p>

<h2>2. Cohorts</h2>
<div class="tw">{table(piv)}</div>
<p>MRI pool: CN/AD participants with a T1 in the Synapse release ({n_mri} after near-duplicate
screening). EEG pool: resting-state recordings in the AD and HC folders with a readable signal
({n_eeg}; recordings whose .fdt is missing on Synapse are excluded). Both pools contain the 25 paired
participants, who are removed per split as described below. Several MRI sites contribute only one
diagnosis (AR, COA, MXA: CN only; CLA, MXB: AD only), so site and diagnosis are confounded in the
full MRI pool; the <code>mixed_sites</code> and <code>clb_only</code> pool variants check how much that matters.</p>

<h2>3. Model structure</h2>
<div class="box">{arch_svg(d_eeg, d_mri, hid, emb, n_eeg, n_mri)}</div>
<h3>Components</h3>{table(pt)}
<h3>Inputs</h3>
<ul>
<li><b>EEG ({d_eeg} features)</b>: 1–30 Hz, 256 Hz, average reference, ≤300 s, 2 s epochs with
bad-channel and bad-epoch rejection; per epoch, relative delta/theta/alpha/beta power and
theta/alpha ratio in frontal, central, temporal, posterior and global regions, median over epochs, plus
peak alpha frequency. Identical to the paired run (checked: max |difference| =
{info['paired_feature_max_abs_diff']['eeg']:.1e}).</li>
<li><b>MRI ({d_mri} features)</b>: FreeSurfer 8.2 SynthSeg (<code>--parc --robust</code>) on the T1;
16 AD-signature volumes (hippocampus, amygdala, ventricles, entorhinal, parahippocampal, temporal,
precuneus, cingulate, parietal, total cortex and white matter), left+right summed and divided by
intracranial volume. Identical to the paired run (max |difference| =
{info['paired_feature_max_abs_diff']['mri']:.1e}).</li>
</ul>

<h2>4. Training protocol and leakage control</h2>
<ol>
<li>Outer splits: the same 10 × 5-fold stratified splits as the paired-only run (verified fold by fold), so
every model below is scored on identical test folds.</li>
<li>For each outer split, the test participants are removed from <b>both</b> pools. The z-scoring,
encoders, pretraining heads and fusion head never see a test participant's EEG or MRI.</li>
<li>Pretraining: full-batch AdamW (lr {H['lr']}, {H['epochs']} epochs), class-balanced cross-entropy.
Weight decay ∈ {H['weight_decay']} chosen on a stratified {int(H['val_fraction'] * 100)}% validation
split of the pool (ties → stronger decay), then the encoder is refitted on the whole pool.</li>
<li>Fusion: encoders frozen; the {2 * emb}-dim concatenated embedding of each paired participant is
z-scored and fed to a class-balanced L2 logistic regression, C ∈ {cfg['fusion']['logreg_C']} tuned by
inner stratified folds on the paired training fold.</li>
<li>Uncertainty: metrics per repeat from out-of-fold predictions (mean ± SD over repeats);
model differences by a participant-level stratified bootstrap (2,000 resamples) of the
repeat-averaged difference.</li>
</ol>

<h2>5. Results</h2>
{dot_chart(summ)}
<h3>Table view</h3>
{table(mt, main_rows=(FAMILY['pt_fusion_concat'][1],))}

<h3>Pairwise comparisons (macro-F1)</h3>
{table(ct)}
<p>A 95% interval that includes 0 means the 25-participant test set cannot tell the two models apart.</p>

<h3>Pretrained encoders on their own pools</h3>
{table(enc, fmt={'pool val macro-F1': f3, 'SD': f3, 'pool n (median)': lambda v: f'{v:.0f}'})}
<p>Validation macro-F1 on the held-out 20% of each pool (mixed sites for the full MRI pool, so it is
inflated by site–diagnosis confounding).</p>

<h2>6. Limitations</h2>
<ul>
<li>The final numbers still come from 25 participants (15 AD, 10 CN) at one site; confidence
intervals on differences are wide.</li>
<li>The EEG pool is small ({n_eeg}), so the EEG encoder gains much less from pretraining than the MRI encoder.</li>
<li>The full MRI pool mixes sites whose diagnosis mix differs; an encoder can partly learn site
instead of disease. The site-restricted pools are the check for this.</li>
<li>The fusion head's C is tuned on inner folds whose validation participants were in the encoder
pretraining pool. This can only bias the choice of C, never the test fold.</li>
<li>Pretraining does not create new paired data. This design improves the representations, not the
amount of evidence about EEG–MRI complementarity.</li>
</ul>
<p class="sub">Generated by <code>make_pretrain_report.py</code> from aggregate tables in
<code>{esc(res)}</code>. Runtime {info['runtime_s'] / 60:.1f} min; Python {info['python']}, torch {info['torch']}.</p>
</main><div id="tip"></div>
<script>
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{{
 el.addEventListener('mousemove',e=>{{const p=el.dataset.tip.split('|');
  tip.innerHTML='<b>'+p[0]+'</b><br>'+p.slice(1).join('<br>');tip.style.display='block';
  tip.style.left=Math.min(e.clientX+14,innerWidth-260)+'px';tip.style.top=(e.clientY+14)+'px';}});
 el.addEventListener('mouseleave',()=>tip.style.display='none');}});
</script></body></html>"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pretrain_fusion_report.html")
    with open(path, "w") as f:
        f.write(page)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
