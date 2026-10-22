"""
Build the data-introduction report for the AD vs CN EEG and MRI data.

Reads the processed outputs (no raw data is re-read except one header per EEG dataset,
for the channel-to-region counts) and writes
  reports/data_introduction.html   self-contained report (figures embedded)
  reports/data_introduction.pdf    PDF copy (needs `module load wkhtmltopdf`)
  reports/figures/*.png            the same figures as separate files

Usage:  python make_data_report.py [--base /project/mlresearch/Runkun]
"""
import argparse
import base64
import glob
import html
import io
import os
import shutil
import subprocess
from datetime import date

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from prepare_brainlat_eeg import BANDS, ensure_positions, region_of

mne.set_log_level("ERROR")

# validated categorical slots 1-2 (light surface); see dataviz palette
COL = {"CN": "#2a78d6", "AD": "#eb6834"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#104281", "#5598e7", "#f0efec", "#e98585", "#b83232"])
DX = ["CN", "AD"]
REGIONS = ["global", "frontal", "central", "temporal", "posterior"]
BRAINLAT_EEG = "/projectnb/mlresearch/Runkun/brainlat_eeg_full"

# required by the ADNI Data Use Agreement in anything published; also cites the EEG datasets
ACKNOWLEDGMENTS_HTML = """
<div class="keep"><h2>Acknowledgments and data sources</h2>
<p class="note">Data collection and sharing for this project was funded by the Alzheimer's Disease
Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI
(Department of Defense award number W81XWH-12-2-0012). Data used in preparation of this work were
obtained from the ADNI database (adni.loni.usc.edu). As such, the investigators within the ADNI
contributed to the design and implementation of ADNI and/or provided data but did not participate in
analysis or writing of this report. A complete listing of ADNI investigators can be found at
http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf</p>
<p class="note">BrainLat EEG: Prado P. et al., The BrainLat project, a multimodal neuroimaging dataset
of neurodegeneration from underrepresented backgrounds, Scientific Data 10, 889 (2023);
Synapse syn51549340.</p>
<p class="note">ds004504 EEG: Miltiadous A. et al., A dataset of scalp EEG recordings of Alzheimer's
disease, frontotemporal dementia and healthy subjects from routine EEG, Data 8(6), 95 (2023);
OpenNeuro ds004504 (CC0).</p></div>
"""
DS_EEG = "/projectnb/mlresearch/Runkun/ds004504"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


# ------------------------------------------------------------------ helpers
def cohens_d(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    sd = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sd


def mwu_p(a, b):
    return stats.mannwhitneyu(pd.Series(a).dropna(), pd.Series(b).dropna()).pvalue


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def strip(ax, df, col, groups=DX, label_n=True, log=False):
    """Jittered dots + thin IQR box + median bar per group; identity also on the x labels."""
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        v = df.loc[df["dx"] == g, col].dropna().to_numpy()
        if not len(v):
            continue
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=16, color=COL[g], alpha=0.75,
                   edgecolor=SURFACE, linewidth=0.6, zorder=3)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(plt.Rectangle((i - 0.3, q1), 0.6, q3 - q1, fill=False, edgecolor=INK2,
                                   linewidth=1, zorder=4))
        ax.plot([i - 0.3, i + 0.3], [med, med], color=INK, linewidth=2, zorder=5,
                solid_capstyle="round")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\nn={int((df['dx'] == g).sum())}" if label_n else g
                        for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 0.2, 0.5, 1, 2, 5]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def dx_legend(fig, y=1.0):
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=COL[g],
                          label={"CN": "CN (cognitively normal)", "AD": "AD"}[g]) for g in DX]
    # sits above the axes area so it never collides with panel titles
    fig.legend(handles=handles, loc="lower right", ncol=2, bbox_to_anchor=(1, y),
               fontsize=9, handletextpad=0.3)


def save(fig, name, figdir):
    path = os.path.join(figdir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def table(df, index=False, cls=""):
    return df.to_html(index=index, escape=True, border=0, classes=f"tbl {cls}",
                      na_rep="–", float_format=lambda x: f"{x:,.2f}")


def dir_bytes(paths):
    return sum(os.path.getsize(p) for p in paths if os.path.exists(p))


def region_counts(set_path, montage=None):
    raw = mne.io.read_raw_eeglab(set_path, preload=False)
    raw.pick("eeg")
    if montage:
        raw.set_montage(mne.channels.make_standard_montage(montage), on_missing="ignore")
    ensure_positions(raw)
    pos = raw.get_montage().get_positions()["ch_pos"]
    labels = pd.Series([region_of(pos[ch]) for ch in raw.ch_names if ch in pos])
    return labels.value_counts().reindex(REGIONS[1:]).fillna(0).astype(int)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/project/mlresearch/Runkun")
    args = ap.parse_args()
    B = args.base
    out = os.path.join(B, "reports")
    figdir = os.path.join(out, "figures")
    os.makedirs(figdir, exist_ok=True)

    # ---------------- load
    mri = pd.read_csv(os.path.join(B, "processed/mri/adni_mri_cohort.csv"), low_memory=False)
    mri["hippo_icv"] = (mri.LeftHippocampus_SV + mri.RightHippocampus_SV) / mri.Icv_CV * 1e3
    mri["vent_icv"] = (mri.LeftLateralVentricle_SV + mri.RightLateralVentricle_SV) / mri.Icv_CV * 1e3
    mri["ento_thick"] = (mri.LeftEntorhinal_TA + mri.RightEntorhinal_TA) / 2
    n_fs = sum(c.endswith(("_CV", "_SA", "_TA", "_TS", "_SV")) for c in mri.columns)

    bl = pd.read_csv(os.path.join(B, "processed/eeg_full/eeg_features.csv"))
    bl_inv = pd.read_csv(os.path.join(B, "processed/eeg_full/eeg_inventory.csv"))
    bl = bl.merge(bl_inv[["file", "orig_seconds", "orig_nchan"]], on="file", how="left")
    bl_miss = pd.read_csv(os.path.join(B, "processed/eeg_full/eeg_missing_subjects.csv"))
    ds = pd.read_csv(os.path.join(B, "processed/ds004504/eeg_features.csv"))
    ds_inv = pd.read_csv(os.path.join(B, "processed/ds004504/eeg_inventory.csv"))
    ds = ds.merge(ds_inv[["file", "orig_seconds", "orig_nchan"]], on="file", how="left")
    part = pd.read_csv(os.path.join(DS_EEG, "participants.tsv"), sep="\t")

    eeg = {"BrainLat": bl, "ds004504": ds}
    sets = {"ADNI MRI": mri, "BrainLat EEG": bl, "ds004504 EEG": ds}

    # sizes of the data actually used
    bl_used = [os.path.join(BRAINLAT_EEG, f) for f in bl["file"]]
    bl_used += [p[:-4] + ".fdt" for p in bl_used]
    ds_used = [os.path.join(DS_EEG, f.split("_")[0], "eeg", f) for f in ds["file"]]
    mri_tables = glob.glob(os.path.join(B, "MRI", "*.csv"))
    gb = lambda x: f"{x / 1e9:.1f} GB" if x > 1e9 else f"{x / 1e6:.0f} MB"

    # ---------------- tables
    overview = pd.DataFrame([
        {"Dataset": "ADNI (UCSF FreeSurfer 7, UCSFFSX7)", "Modality": "Structural MRI, 3T T1",
         "Data form": f"Tabular FreeSurfer measures ({n_fs} per subject)",
         "Location": f"{B}/MRI → processed/mri", "Size used": gb(dir_bytes(mri_tables)),
         "AD": int((mri.dx == "AD").sum()), "CN": int((mri.dx == "CN").sum())},
        {"Dataset": "BrainLat (Synapse syn51549340)", "Modality": "Resting-state EEG",
         "Data form": "EEGLAB .set/.fdt, 128-ch BioSemi, 512 Hz",
         "Location": BRAINLAT_EEG, "Size used": gb(dir_bytes(bl_used)),
         "AD": int((bl.dx == "AD").sum()), "CN": int((bl.dx == "CN").sum())},
        {"Dataset": "OpenNeuro ds004504", "Modality": "Resting-state EEG, eyes closed",
         "Data form": "BIDS, EEGLAB .set, 19-ch 10–20, 500 Hz",
         "Location": DS_EEG, "Size used": gb(dir_bytes(ds_used)),
         "AD": int((ds.dx == "AD").sum()), "CN": int((ds.dx == "CN").sum())},
    ])

    demo_rows = []
    for name, df in sets.items():
        edu = "pteducat" if "pteducat" in df else "years_education"
        for g in DX:
            s = df[df.dx == g]
            demo_rows.append({
                "Dataset": name, "Group": g, "n": len(s),
                "Age, mean (SD)": f"{s.age.mean():.1f} ({s.age.std():.1f})",
                "Female, n (%)": f"{(s.sex == 'F').sum()} ({(s.sex == 'F').mean() * 100:.0f}%)",
                "Education, y": (f"{s[edu].mean():.1f} ({s[edu].std():.1f})"
                                 if s[edu].notna().any() else "not recorded"),
                "MMSE": (f"{s.mmse.mean():.1f} ({s.mmse.std():.1f})"
                         if "mmse" in s and s.mmse.notna().any() else "–"),
            })
        a, c = df[df.dx == "AD"], df[df.dx == "CN"]
        demo_rows[-1]["p (age)"] = fmt_p(mwu_p(a.age, c.age))
        demo_rows[-1]["p (sex)"] = fmt_p(stats.chi2_contingency(
            pd.crosstab(df.dx, df.sex))[1])
        demo_rows[-1]["p (education)"] = (fmt_p(mwu_p(a[edu], c[edu]))
                                          if a[edu].notna().any() else "–")
    demo = pd.DataFrame(demo_rows).fillna("")

    bl_records = {"AD": 35, "CN": 46}
    attrition = pd.DataFrame([
        {"Dataset": "BrainLat", "Step": "Subjects in Records CSVs", "AD": 35, "CN": 46},
        {"Dataset": "BrainLat", "Step": "Signal file (.fdt / embedded data) on Synapse",
         "AD": 35, "CN": 46 - int((bl_miss.dx == "CN").sum())},
        {"Dataset": "BrainLat", "Step": "Loads, harmonized, features extracted",
         "AD": int((bl.dx == "AD").sum()), "CN": int((bl.dx == "CN").sum())},
        {"Dataset": "ds004504", "Step": "Participants (Group A / C; 23 FTD excluded)",
         "AD": int((part.Group == "A").sum()), "CN": int((part.Group == "C").sum())},
        {"Dataset": "ds004504", "Step": "Loads, harmonized, features extracted",
         "AD": int((ds.dx == "AD").sum()), "CN": int((ds.dx == "CN").sum())},
        {"Dataset": "ADNI MRI", "Step": "AD/CN baseline 3T scans, CN age/sex-matched 2:1",
         "AD": int((mri.dx == "AD").sum()), "CN": int((mri.dx == "CN").sum())},
    ])
    assert bl_records == {"AD": 35, "CN": 46}

    try:
        bl_reg = region_counts(os.path.join(BRAINLAT_EEG, bl["file"].iloc[0]))
        ds_reg = region_counts(ds_used[0], montage="standard_1020")
        regions_tbl = pd.DataFrame({"BrainLat (BioSemi 128)": bl_reg,
                                    "ds004504 (10–20, 19 ch)": ds_reg}).T
        regions_tbl.insert(0, "Total", regions_tbl.sum(axis=1))
        regions_html = table(regions_tbl, index=True)
    except Exception as e:  # raw data not reachable: skip, the rest of the report still builds
        regions_html = f"<p class='note'>Region counts unavailable ({html.escape(str(e))}).</p>"

    eff_rows = []
    for name, df in eeg.items():
        for feat in ["relpow_delta_global", "relpow_theta_global", "relpow_alpha_global",
                     "relpow_beta_global", "theta_alpha_ratio_global", "peak_alpha_freq"]:
            a, c = df[df.dx == "AD"][feat], df[df.dx == "CN"][feat]
            eff_rows.append({"Dataset": name, "Feature": feat,
                             "AD median": a.median(), "CN median": c.median(),
                             "Cohen's d (AD−CN)": cohens_d(a, c),
                             "Mann–Whitney p": fmt_p(mwu_p(a, c))})
    effects = pd.DataFrame(eff_rows)

    # ---------------- figures
    F = {}

    # 1 cohort sizes
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    names = list(sets)[::-1]
    h = 0.36
    for j, g in enumerate(DX):
        vals = [int((sets[n].dx == g).sum()) for n in names]
        y = np.arange(len(names)) + (j - 0.5) * (h + 0.04)
        ax.barh(y, vals, height=h, color=COL[g], label=g, zorder=3)
        for yy, v in zip(y, vals):
            ax.text(v + 6, yy, f"{g} {v}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Subjects")
    ax.set_xlim(0, max(len(d) for d in sets.values()) * 0.8)
    ax.set_title("Subjects per dataset and diagnosis")
    dx_legend(fig, 0.99)
    F["cohorts"] = save(fig, "fig1_cohort_sizes.png", figdir)

    # 2 age
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2), sharey=True)
    for ax, (name, df) in zip(axs, sets.items()):
        strip(ax, df, "age")
        ax.set_title(name)
    axs[0].set_ylabel("Age (years)")
    dx_legend(fig, 0.99)
    F["age"] = save(fig, "fig2_age.png", figdir)

    # 3 MRI markers
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, (col, lab) in zip(axs, [("hippo_icv", "Hippocampus / ICV (×10⁻³)"),
                                    ("vent_icv", "Lateral ventricles / ICV (×10⁻³)"),
                                    ("ento_thick", "Entorhinal thickness (mm)")]):
        strip(ax, mri, col)
        d = cohens_d(mri[mri.dx == "AD"][col], mri[mri.dx == "CN"][col])
        ax.set_title(f"{lab.split(' (')[0]}  (d = {d:+.2f})", fontsize=10)
        ax.set_ylabel(lab)
    dx_legend(fig, 0.99)
    fig.tight_layout()
    F["mri"] = save(fig, "fig3_mri_markers.png", figdir)

    # 4 ADNI phase
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    phases = ["ADNI1", "ADNIGO", "ADNI2", "ADNI3", "ADNI4"]
    ct = pd.crosstab(mri.phase, mri.dx).reindex(phases).fillna(0)
    x = np.arange(len(phases))
    for j, g in enumerate(DX):
        ax.bar(x + (j - 0.5) * 0.38, ct[g], width=0.36, color=COL[g], zorder=3, label=g)
        for xx, v in zip(x + (j - 0.5) * 0.38, ct[g]):
            ax.text(xx, v + 4, int(v), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(phases)
    ax.set_ylabel("Subjects")
    ax.set_title("ADNI MRI cohort by study phase")
    dx_legend(fig, 0.99)
    F["phase"] = save(fig, "fig4_adni_phase.png", figdir)

    # 5 EEG duration
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, (name, df) in zip(axs, eeg.items()):
        strip(ax, df, "orig_seconds")
        ax.axhline(300, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.text(-0.55, 292, "300 s cap", va="top", ha="left", fontsize=8, color=INK2)
        ax.set_title(name)
    axs[0].set_ylabel("Raw recording length (s)")
    dx_legend(fig, 0.99)
    F["duration"] = save(fig, "fig5_eeg_duration.png", figdir)

    # 6 global relative band power
    fig, axs = plt.subplots(2, 4, figsize=(9.5, 5.6), sharey="col")
    for r, (name, df) in enumerate(eeg.items()):
        for c, band in enumerate(BANDS):
            ax = axs[r, c]
            strip(ax, df, f"relpow_{band}_global", label_n=(r == 1))
            lo, hi = BANDS[band]
            if r == 0:
                ax.set_title(f"{band.capitalize()} {lo}–{hi} Hz")
        axs[r, 0].set_ylabel(f"{name}\nrelative power")
    dx_legend(fig, 0.99)
    fig.tight_layout()
    F["bands"] = save(fig, "fig6_eeg_bandpower.png", figdir)

    # 7 theta/alpha + peak alpha
    fig, axs = plt.subplots(1, 4, figsize=(9.5, 3.2))
    for k, (name, df) in enumerate(eeg.items()):
        strip(axs[k], df, "theta_alpha_ratio_global", log=True)
        axs[k].set_title(f"{name}: θ/α ratio", fontsize=10)
        strip(axs[k + 2], df, "peak_alpha_freq")
        axs[k + 2].set_title(f"{name}: peak α", fontsize=10)
    axs[0].set_ylabel("Theta / alpha (log scale)")
    axs[2].set_ylabel("Peak alpha frequency (Hz)")
    for ax in axs[2:]:
        ax.set_ylim(6.8, 13.2)
    dx_legend(fig, 0.99)
    fig.tight_layout()
    F["markers"] = save(fig, "fig7_eeg_markers.png", figdir)

    # 8 regional effect sizes
    rows = [f"relpow_{b}" for b in BANDS] + ["theta_alpha_ratio"]
    rlab = [b.capitalize() for b in BANDS] + ["θ/α ratio"]
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 3.4), gridspec_kw={"wspace": 0.32})
    for ax, (name, df) in zip(axs, eeg.items()):
        M = np.array([[cohens_d(df[df.dx == "AD"][f"{f}_{r}"], df[df.dx == "CN"][f"{f}_{r}"])
                       for r in REGIONS] for f in rows])
        im = ax.imshow(M, cmap=DIVERGING, vmin=-1.5, vmax=1.5, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                        color=INK if abs(M[i, j]) < 0.9 else "#ffffff")
        ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels(REGIONS, fontsize=9)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rlab)
        ax.grid(False)
        ax.set_title(name)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label("Cohen's d (AD − CN)", color=INK2)
    cb.outline.set_visible(False)
    F["effects"] = save(fig, "fig8_eeg_effect_sizes.png", figdir)

    # ---------------- HTML
    def fig_html(key, caption, alt):
        return (f'<figure><img src="data:image/png;base64,{F[key]}" alt="{html.escape(alt)}">'
                f"<figcaption>{caption}</figcaption></figure>")

    bl_ad, bl_cn = bl[bl.dx == "AD"], bl[bl.dx == "CN"]
    ds_ad, ds_cn = ds[ds.dx == "AD"], ds[ds.dx == "CN"]
    short_ad = int((bl_ad.orig_seconds < 300).sum())

    body = f"""
<header>
  <p class="eyebrow">Cross-modal AD vs CN project · Runkun</p>
  <h1>Data introduction: EEG and MRI</h1>
  <p class="meta">Generated {date.today():%d %B %Y} by <code>make_data_report.py</code> from
  <code>processed/</code>. Figures are also saved in <code>reports/figures/</code>.</p>
</header>

<section class="summary">
  <div class="tile"><span class="num">{len(mri)}</span><span class="lab">ADNI MRI subjects<br>
    {int((mri.dx=='AD').sum())} AD · {int((mri.dx=='CN').sum())} CN</span></div>
  <div class="tile"><span class="num">{len(bl)}</span><span class="lab">BrainLat EEG subjects<br>
    {len(bl_ad)} AD · {len(bl_cn)} CN</span></div>
  <div class="tile"><span class="num">{len(ds)}</span><span class="lab">ds004504 EEG subjects<br>
    {len(ds_ad)} AD · {len(ds_cn)} CN</span></div>
</section>

<h2>1. Overview</h2>
<p>The project compares Alzheimer's disease (AD) with cognitively normal controls (CN) in two
modalities. Structural MRI comes from ADNI as FreeSurfer-derived regional measures (no images
are used yet). Resting-state EEG comes from two independent public datasets, BrainLat (Latin
America, two sites) and OpenNeuro ds004504 (Greece, one site), processed with the same pipeline.
The labels are consistent across all three: <b>dx = AD / CN</b>, <b>sex = M / F</b>.</p>
{table(overview)}
{fig_html("cohorts", "Figure 1. Subjects per dataset. ADNI CN subjects are age- and sex-matched 2:1 to AD.", "Bar chart of AD and CN subject counts for the three datasets")}

<div class="keep"><h3>How each cohort was selected</h3>
{table(attrition)}</div>
<p class="note">BrainLat: 14 HC recordings are header-only on Synapse (the .fdt signal file does not
exist anywhere in the project), so CN drops from 46 to {len(bl_cn)}. All 35 AD recordings are usable.</p>

<h2>2. Demographics</h2>
{table(demo)}
<p class="note">p values: Mann–Whitney U for age and education, χ² for sex (AD vs CN within each dataset).
MMSE is only available for ds004504, where every CN subject scores 30.</p>
{fig_html("age", "Figure 2. Age by diagnosis. Box = interquartile range, dark bar = median, dots = subjects.", "Strip plots of age for AD and CN in each dataset")}

<h2 class="newpage">3. MRI: ADNI structural measures</h2>
<p>Source tables: <code>All_Subjects_UCSFFSX7</code> (UCSF FreeSurfer 7 cross-sectional runs),
<code>DXSUM</code> (diagnosis), <code>PTDEMOG</code> (demographics), <code>DATADIC</code> and
<code>Key_MRI</code>. One scan per subject (the earliest that passes QC), 3T only, diagnosis from the closest
DXSUM visit (median gap {mri.dx_gap_days.median():.0f} days). Each subject has {n_fs} measures:
cortical volume (_CV), surface area (_SA), thickness average (_TA) and SD (_TS), and subcortical volume (_SV).</p>
<ul>
  <li>All {len(mri)} scans are 3T; FreeSurfer {", ".join(f"{v} (n={n})" for v, n in mri.fsver.value_counts().items())}.</li>
  <li>Only {int(mri.qc_rated.sum())} runs carry a UCSF QC rating; the rest are unrated (<code>qc_rated = False</code>) but kept.</li>
  <li>Scanner make is known for {int(mri.scanner_manufacturer.notna().sum())} subjects
      ({", ".join(f"{k} {v}" for k, v in mri.scanner_manufacturer.value_counts().items())}).</li>
  <li>Matching worked: age {mri[mri.dx=='AD'].age.mean():.1f} vs {mri[mri.dx=='CN'].age.mean():.1f} years, ~50% female in both groups.
      Education still differs ({mri[mri.dx=='AD'].pteducat.mean():.1f} vs {mri[mri.dx=='CN'].pteducat.mean():.1f} years).</li>
</ul>
{fig_html("mri", "Figure 3. Classic AD atrophy markers: hippocampal and lateral-ventricle volume normalized to intracranial volume (ICV), and mean entorhinal cortical thickness. d = Cohen's d (AD − CN).", "Strip plots of hippocampal volume, ventricle volume and entorhinal thickness by diagnosis")}
{fig_html("phase", "Figure 4. ADNI phase of each subject's scan. The AD:CN ratio varies by phase (CN were matched on age and sex, not phase), so phase/protocol should be a covariate.", "Bar chart of subjects per ADNI phase by diagnosis")}

<h2 class="newpage">4. EEG: BrainLat and ds004504</h2>
<p>Both EEG datasets run through the same code (<code>prepare_brainlat_eeg.py</code>;
<code>prepare_ds004504_eeg.py</code> imports its functions): EEG channels only, average reference,
1–30 Hz band-pass, resample to 256 Hz, first ≤300 s. Features from Welch PSD (2 s windows, 50% overlap):
relative power in delta (1–4 Hz), theta (4–8), alpha (8–13) and beta (13–30), globally and per region,
theta/alpha ratio, and peak alpha frequency (7–13 Hz).</p>
<table class="tbl">
<thead><tr><th></th><th>BrainLat</th><th>ds004504</th></tr></thead>
<tbody>
<tr><td>Recording</td><td>AD: <code>rs-HEP</code> files · CN: <code>rs</code> files</td><td>Eyes-closed rest (<code>task-eyesclosed</code>)</td></tr>
<tr><td>Amplifier / montage</td><td>BioSemi, 128 channels</td><td>Nihon Kohden, 19 channels (10–20; T3/T4/T5/T6 names)</td></tr>
<tr><td>Sampling rate</td><td>512 Hz (all)</td><td>500 Hz (all)</td></tr>
<tr><td>Sites</td><td>AR (Argentina) and CL (Chile)</td><td>Single site</td></tr>
<tr><td>Raw length, median</td><td>AD {bl_ad.orig_seconds.median():.0f} s · CN {bl_cn.orig_seconds.median():.0f} s</td><td>AD {ds_ad.orig_seconds.median():.0f} s · CN {ds_cn.orig_seconds.median():.0f} s</td></tr>
<tr><td>Files used</td><td><code>s6_sub-*_rs[-HEP]_eeg.set</code> + .fdt</td><td><code>sub-*/eeg/*_eeg.set</code> (raw, not derivatives)</td></tr>
</tbody></table>

<div class="keep"><h3>Channels per scalp region</h3>
<p>Regions come from 3-D head coordinates (front &gt; 3 cm, back &lt; −4 cm, lateral &gt; 5 cm = temporal,
otherwise central). The two montages differ a lot in density: a BrainLat region averages 19–44 channels,
a ds004504 region 1–7. With this rule C3/C4 fall in "temporal", so ds004504 "central" is Cz alone and its
central features are noisier than the others.</p>
{regions_html}</div>
{fig_html("duration", f"Figure 5. Raw recording length before the 300 s cap. {short_ad} BrainLat AD recordings are shorter than 300 s (minimum {bl_ad.orig_seconds.min():.0f} s), so those subjects contribute less data.", "Strip plots of raw EEG recording length by diagnosis for both datasets")}
{fig_html("bands", "Figure 6. Global relative band power (fraction of 1–30 Hz power). Top row BrainLat, bottom row ds004504; each column shares its y-axis.", "Grid of strip plots of relative delta, theta, alpha and beta power by diagnosis")}
{fig_html("markers", "Figure 7. Theta/alpha ratio (log scale) and peak alpha frequency. AD shows EEG slowing in both datasets: higher θ/α and lower peak alpha. Peak alpha has 0.5 Hz resolution (2 s Welch windows) and is searched in 7–13 Hz, so values at 7 Hz (the ds004504 AD median) mean the spectral peak is at or below the search floor.", "Strip plots of theta/alpha ratio and peak alpha frequency by diagnosis")}
{fig_html("effects", "Figure 8. Effect size (Cohen's d, AD − CN) for each band and region. Red = higher in AD, blue = lower in AD.", "Heatmaps of AD minus CN effect sizes per band and region for both datasets")}
<div class="keep"><h3>Global EEG features: AD vs CN</h3>
{table(effects)}</div>

<h2 class="newpage">5. Confounds and caveats</h2>
<table class="tbl confound">
<thead><tr><th>Issue</th><th>Dataset</th><th>Details</th><th>What to do</th></tr></thead>
<tbody>
<tr><td><b>Recording type = diagnosis</b></td><td>BrainLat</td><td>All AD files are <code>rs-HEP</code>; all CN files are <code>rs</code>. Any AD/CN difference may partly reflect the recording protocol.</td><td>Confirm that rs-HEP is the same eyes-closed rest protocol; otherwise treat BrainLat results as exploratory and use ds004504 for replication.</td></tr>
<tr><td><b>Education</b></td><td>BrainLat, ADNI</td><td>BrainLat AD {bl_ad.years_education.mean():.1f} vs CN {bl_cn.years_education.mean():.1f} years; ADNI {mri[mri.dx=='AD'].pteducat.mean():.1f} vs {mri[mri.dx=='CN'].pteducat.mean():.1f}. Not recorded in ds004504.</td><td>Covariate where available.</td></tr>
<tr><td><b>Age</b></td><td>BrainLat</td><td>AD {bl_ad.age.mean():.1f} vs CN {bl_cn.age.mean():.1f} years. Matched in ADNI and roughly similar in ds004504.</td><td>Covariate, or age-matched subset.</td></tr>
<tr><td><b>Sex</b></td><td>ds004504</td><td>AD {(ds_ad.sex=='F').mean()*100:.0f}% female vs CN {(ds_cn.sex=='F').mean()*100:.0f}%.</td><td>Covariate.</td></tr>
<tr><td><b>Recording length</b></td><td>BrainLat</td><td>{short_ad} AD recordings &lt; 300 s (as short as {bl_ad.orig_seconds.min():.0f} s); 2 CN &lt; 300 s.</td><td>Sensitivity analysis at a common shorter cap (e.g. 140 s).</td></tr>
<tr><td><b>MMSE ceiling</b></td><td>ds004504</td><td>Every CN subject scores 30; MMSE is only available here.</td><td>Use only as an AD severity measure within AD.</td></tr>
<tr><td><b>Montage density</b></td><td>EEG, across datasets</td><td>128 vs 19 channels; region definitions are the same, but the electrodes behind them are not.</td><td>Prefer global features for cross-dataset comparison.</td></tr>
<tr><td><b>ADNI phase / QC</b></td><td>ADNI</td><td>AD:CN ratio varies by phase; {int((~mri.qc_rated).sum())} of {len(mri)} FreeSurfer runs are not QC-rated.</td><td>Phase as covariate; check that results hold in the QC-rated subset.</td></tr>
<tr><td><b>Not the same subjects</b></td><td>All</td><td>EEG and MRI come from different cohorts (no subject has both).</td><td>Cross-modal analyses compare group-level effects, not paired data.</td></tr>
</tbody></table>

<div class="keep"><h2>6. Files</h2>
<table class="tbl">
<thead><tr><th>File</th><th>Contents</th></tr></thead><tbody>
<tr><td><code>processed/mri/adni_mri_cohort.csv</code></td><td>One row per ADNI subject: IDs, dx, scan info, demographics, {n_fs} FreeSurfer measures</td></tr>
<tr><td><code>processed/eeg_full/eeg_features.csv</code></td><td>BrainLat AD/CN EEG features (one row per subject)</td></tr>
<tr><td><code>processed/eeg_full/eeg_inventory.csv</code>, <code>eeg_missing_subjects.csv</code></td><td>Status of every BrainLat .set file; subjects without usable data</td></tr>
<tr><td><code>processed/ds004504/eeg_features.csv</code></td><td>ds004504 AD/CN EEG features, same columns as BrainLat plus <code>mmse</code></td></tr>
<tr><td><code>processed/eeg/brainlat_verification.csv</code>, <code>processed/ds004504_verification.csv</code></td><td>Download checks (file present, size, loads, sfreq, channels, duration)</td></tr>
<tr><td><code>processed/eeg/synapse_ad_search.csv</code></td><td>Every AD file on BrainLat Synapse (metadata)</td></tr>
</tbody></table></div>
{ACKNOWLEDGMENTS_HTML}
"""

    css = f"""
:root {{ --bg:#f6f5f2; --card:{SURFACE}; --ink:{INK}; --ink2:{INK2}; --line:{GRID};
        --accent:{COL['CN']}; color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
       font: 15px/1.55 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
main {{ max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }}
header h1 {{ font-size: 30px; margin: 4px 0 6px; letter-spacing: -0.01em; }}
.eyebrow {{ text-transform: uppercase; letter-spacing: .08em; font-size: 12px; color: var(--ink2); margin:0; }}
.meta, .note {{ color: var(--ink2); font-size: 13px; }}
h2 {{ font-size: 21px; margin: 40px 0 10px; padding-top: 14px; border-top: 1px solid var(--line); }}
h3 {{ font-size: 16px; margin: 24px 0 8px; }}
code {{ font: 12.5px ui-monospace, SFMono-Regular, Menlo, monospace; background: #ecebe6;
        padding: 1px 4px; border-radius: 4px; }}
.summary {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap:12px; margin: 22px 0 8px; }}
.tile {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }}
.tile .num {{ display:block; font-size: 34px; font-weight: 700; line-height: 1.1; }}
.tile .lab {{ color: var(--ink2); font-size: 13px; }}
.tbl {{ width:100%; border-collapse: collapse; background: var(--card); font-size: 13px;
        border: 1px solid var(--line); border-radius: 8px; margin: 10px 0; display:block; overflow-x:auto; }}
.tbl th, .tbl td {{ text-align:left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
.tbl th {{ background: #efeee9; font-weight: 600; white-space: nowrap; }}
.tbl tr:last-child td {{ border-bottom: 0; }}
figure {{ margin: 18px 0; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }}
figure img {{ width: 100%; height: auto; display: block; }}
figcaption {{ color: var(--ink2); font-size: 13px; margin-top: 8px; }}
ul {{ padding-left: 20px; }}
/* PDF (wkhtmltopdf, old WebKit: no grid, weak break control) */
@media print {{
  body {{ background: #fff; font-size: 12.5px; }}
  main {{ max-width: none; padding: 0; }}
  .summary {{ display: block; }}
  .tile {{ display: inline-block; width: 31.5%; margin-right: 1%; vertical-align: top; }}
  .tbl {{ display: table; font-size: 11px; }}
  .tbl td code {{ word-break: break-all; }}
  .tbl td {{ word-break: normal; }}
  h2 {{ page-break-after: avoid; }}
  h2.newpage {{ page-break-before: always; border-top: 0; margin-top: 0; }}
  figure, tr, .keep, .summary {{ page-break-inside: avoid; }}
}}
"""
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>EEG and MRI Data</title><style>{css}</style></head>"
            f"<body><main>{body}</main></body></html>")
    path = os.path.join(out, "data_introduction.html")
    with open(path, "w") as f:
        f.write(page)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB) and {len(F)} figures in {figdir}")

    # PDF copy (SCC: `module load wkhtmltopdf` first); skipped if the tool is not on PATH
    if shutil.which("wkhtmltopdf"):
        pdf = path[:-5] + ".pdf"
        subprocess.run(["wkhtmltopdf", "--quiet", "--print-media-type", "-s", "Letter",
                        "-T", "14mm", "-B", "14mm", "-L", "12mm", "-R", "12mm",
                        "--footer-center", "[page] / [topage]", "--footer-font-size", "8",
                        path, pdf], check=True)
        print(f"wrote {pdf}")
    else:
        print("wkhtmltopdf not found: PDF skipped (module load wkhtmltopdf)")
    print(demo.to_string(index=False))
    print(effects.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
