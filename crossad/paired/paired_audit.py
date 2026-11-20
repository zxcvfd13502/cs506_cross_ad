"""
CROSS-AD paired cohort audit: BrainLat EEG + structural MRI.

What it checks
  * the diagnosis table (BrainLat_Demographic_MRI.csv): labels, missing labels,
    duplicated MRI_ID / EEG_ID rows, label agreement with the EEG Records tables
  * which MRI participants are in the Synapse release, which have a T1-weighted volume,
    and whether the BrainLat_dataset_NewIDs copy is identical to the site-folder copy
  * which documented EEG_IDs have a resting-state EEGLAB file with signal
  * MRI formats (T1w / FLAIR / bold / dwi; regional measures are not distributed) and
    EEG formats (.set embedded / .set + .fdt)
  * the target task: CN/MCI/AD if MCI is documented with enough participants, else CN vs AD

Pairing rule (documented in configs/paired_config.yaml): MRI_ID <-> EEG_ID from the demographics
table. BrainLat has one assessment per participant (no session key), so the visit is the
participant's single BrainLat visit. Acquisition dates are anonymised, so no date window
is applied.

Outputs
  <work>/pairing_manifest.csv      every MRI participant: ids, dx, files, status, reason
  <work>/paired_cohort.csv         included pairs only (input to the feature scripts)
  <results>/audit_counts.csv       aggregate counts by diagnosis and stage (git-safe)
  <results>/audit_summary.json     task choice, duplicates, exclusions, formats (git-safe)

  python -m crossad.paired.paired_audit --config configs/paired_config.yaml
"""
import argparse
import collections
import glob
import json
import os
import re

import pandas as pd
import yaml

from crossad.preprocess.prepare_brainlat_eeg import data_status, subject_key


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def read_table(path):
    df = pd.read_csv(path, encoding="latin1", dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    return df.apply(lambda s: s.str.strip())


# ------------------------------------------------------------------ MRI files
def t1_candidates(mri_root, dirs, mri_id):
    hits = []
    for d in dirs:
        hits += sorted(glob.glob(os.path.join(mri_root, d, mri_id, "anat", "*T1w*.nii*")))
    return hits


def choose_t1(paths, mri_id):
    """Plain sub-X_T1w.nii(.gz) first; otherwise the first variant (run-1, acq-*, part-*)."""
    plain = [p for p in paths if re.fullmatch(rf"{mri_id}_T1w\.nii(\.gz)?", os.path.basename(p))]
    return (plain or paths or [None])[0]


def mri_modalities(mri_root, dirs):
    c = collections.Counter()
    for d in dirs:
        for p in glob.glob(os.path.join(mri_root, d, "sub-*", "*", "*.nii*")):
            name = re.sub(r"^sub-[A-Za-z]+\d+_", "", os.path.basename(p))
            c[re.sub(r"\.nii(\.gz)?$", "", name)] += 1
    return dict(c.most_common())


# ------------------------------------------------------------------ EEG files
def eeg_index(eeg_root, exclude):
    """subject key -> resting-state .set files in the EEG tree (task/rejected files dropped)."""
    idx = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(eeg_root, "**", "*.set"), recursive=True)):
        stem = os.path.splitext(os.path.basename(p))[0]
        toks = [t.lower() for t in re.split(r"[_\-\s.]+", stem)]
        if any(e.lower() in toks for e in exclude):
            continue
        k = subject_key(stem)
        if k:
            idx[k].append(p)
    return idx


def eeg_records(eeg_root):
    """EEG Records tables (per group folder): subject key -> diagnosis."""
    out = {}
    for p in glob.glob(os.path.join(eeg_root, "*", "Records_*_EEG_data.csv")):
        try:
            df = read_table(p)
        except Exception:
            continue                    # e.g. PD tables that are Drive HTML pages
        if "id_EEG" not in df.columns or "diagnosis" not in df.columns:
            continue
        for r in df.itertuples():
            out[subject_key(r.id_EEG)] = r.diagnosis
    return out


def signal_ok(path):
    st, _ = data_status(path)
    return st in ("embedded", "fdt present", "v7.3 (unchecked)"), st


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paired_config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    D, P, T = cfg["data"], cfg["paths"], cfg["task"]
    root = D["root"]
    mri_root, eeg_root = os.path.join(root, D["mri_dir"]), os.path.join(root, D["eeg_dir"])
    work, res = P["work"], P["results"]
    os.makedirs(work, exist_ok=True)
    os.makedirs(res, exist_ok=True)

    demo = read_table(os.path.join(root, D["demographics"]))
    need = {"MRI_ID", "EEG_ID", "diagnosis"}
    if not need <= set(demo.columns):
        raise SystemExit(f"{D['demographics']} lacks columns {need - set(demo.columns)}")
    print(f"Diagnosis table: {len(demo)} rows, columns {list(demo.columns)}")

    # ---------------- task definition from the documented labels
    labels = collections.Counter(demo["diagnosis"])
    pref = T["preferred_classes"]
    if all(labels.get(c, 0) >= T["min_participants_per_class"] for c in pref):
        classes, why = pref, "all preferred classes documented with enough participants"
    else:
        classes = T["fallback_classes"]
        missing = [c for c in pref if labels.get(c, 0) == 0]
        why = (f"preferred classes {pref} not all documented (absent: {missing}); "
               f"labels present: {dict(labels)}. Using {classes}. FTD/PD/MS are distinct "
               f"diagnoses and are not merged into AD or CN.")
    print(f"Task: {' vs '.join(classes)} -- {why}")

    # ---------------- duplicates in the diagnosis table
    dup_mri = demo[demo.duplicated("MRI_ID", keep=False)]
    with_eeg = demo[demo["EEG_ID"] != ""]
    dup_eeg = with_eeg[with_eeg.duplicated("EEG_ID", keep=False)]
    ambiguous_eeg = set(dup_eeg["EEG_ID"])
    # duplicated MRI_ID: keep the row carrying an EEG_ID if exactly one does
    rows = []
    for mid, g in demo.groupby("MRI_ID", sort=False):
        if len(g) == 1:
            rows.append(g.iloc[0].to_dict() | {"dup_note": ""})
            continue
        g_e = g[g["EEG_ID"] != ""]
        keep = g_e.iloc[0] if len(g_e) == 1 else g.iloc[0]
        conflict = g.drop(columns=["EEG_ID"]).drop_duplicates().shape[0] > 1
        rows.append(keep.to_dict() | {"dup_note": f"{len(g)} rows for MRI_ID"
                                      + ("; fields differ" if conflict else "")})
    demo = pd.DataFrame(rows)

    # ---------------- file indices
    site_dirs, fb_dirs = D["mri_sites"], D["mri_fallback_dirs"]
    present = {os.path.basename(p) for d in site_dirs + fb_dirs
               for p in glob.glob(os.path.join(mri_root, d, "sub-*"))}
    eidx = eeg_index(eeg_root, cfg["eeg"]["exclude_name_parts"])
    erec = eeg_records(eeg_root)

    out, copy_checks = [], collections.Counter()
    for r in demo.to_dict("records"):
        mid, eid, dx = r["MRI_ID"], r["EEG_ID"], r["diagnosis"]
        row = {"mri_id": mid, "eeg_id": eid, "dx": dx, "site": re.sub(r"^sub-([A-Z]+).*", r"\1", mid),
               "age": r.get("Age", ""), "sex": r.get("sex", ""),
               "years_education": r.get("years_education", ""), "dup_note": r["dup_note"],
               "flags": []}
        # MRI
        t1s = t1_candidates(mri_root, site_dirs, mid)
        t1fb = t1_candidates(mri_root, fb_dirs, mid)
        row["mri_in_release"] = mid in present
        row["t1_path"] = choose_t1(t1s or t1fb, mid)
        row["n_t1_variants"] = len(t1s)
        if t1s and t1fb:
            same = {os.path.basename(a): os.path.getsize(a) for a in t1fb} == \
                   {os.path.basename(a): os.path.getsize(a) for a in t1s}
            copy_checks["identical" if same else "differs"] += 1
            if not same:
                row["flags"].append("T1 copies differ between site folder and NewIDs")
        # EEG
        eeg_files, layout = [], glob.glob(os.path.join(mri_root, "*", mid, "eeg", "*.set"))
        if eid:
            eeg_files = eidx.get(subject_key(eid), [])
        cands = eeg_files + sorted(layout, key=lambda p: [d in p for d in fb_dirs])
        row["eeg_path"], row["eeg_status"] = None, "no file"
        for p in cands:
            ok, st = signal_ok(p)
            row["eeg_status"] = st
            if ok:
                row["eeg_path"] = p
                break
        row["eeg_source"] = ("eeg_tree" if row["eeg_path"] in eeg_files else
                             "mri_tree_bids" if row["eeg_path"] else "")
        rec_dx = erec.get(subject_key(eid)) if eid else None
        row["eeg_records_dx"] = rec_dx or ""

        # status (first failing check)
        if dx == "":
            reason = "missing diagnosis"
        elif dx not in classes:
            reason = f"diagnosis {dx} not in task"
        elif eid and eid in ambiguous_eeg:
            reason = "EEG_ID linked to more than one MRI_ID (ambiguous)"
        elif not eid and not (layout and cfg["pairing"]["allow_bids_layout_only_pairs"]):
            reason = ("no documented EEG_ID (EEG only via BIDS layout)" if layout
                      else "no documented EEG_ID")
        elif rec_dx and rec_dx != dx:
            reason = f"label mismatch: MRI table {dx}, EEG Records {rec_dx}"
        elif not row["mri_in_release"]:
            reason = "MRI participant not in Synapse release"
        elif row["t1_path"] is None:
            reason = "no T1w volume (other MRI only)"
        elif row["eeg_path"] is None:
            reason = f"no EEG signal ({row['eeg_status']})"
        else:
            reason = ""
        if not eid and layout:
            row["flags"].append("EEG present via BIDS layout without documented EEG_ID")
        if r["dup_note"]:
            row["flags"].append(r["dup_note"])
        row["included"] = reason == ""
        row["exclusion_reason"] = reason
        row["flags"] = "; ".join(row["flags"])
        out.append(row)

    man = pd.DataFrame(out)
    man.to_csv(os.path.join(work, "pairing_manifest.csv"), index=False)
    cohort = man[man["included"]].copy()
    if cohort["mri_id"].duplicated().any() or cohort["eeg_id"].duplicated().any():
        raise SystemExit("pairing is not one-to-one; inspect pairing_manifest.csv")
    cohort.to_csv(os.path.join(work, "paired_cohort.csv"), index=False)

    # ---------------- aggregate counts
    stages = [
        ("1 rows in diagnosis table (unique MRI_ID)", man),
        ("2 documented EEG_ID", man[man["eeg_id"] != ""]),
        ("3 + MRI participant in Synapse release", man[(man["eeg_id"] != "") & man["mri_in_release"]]),
        ("4 + T1w volume", man[(man["eeg_id"] != "") & man["t1_path"].notna()]),
        ("5 + EEG file with signal", man[(man["eeg_id"] != "") & man["t1_path"].notna()
                                         & man["eeg_path"].notna()]),
        ("6 included (task classes, unambiguous, labels agree)", cohort),
    ]
    cnt = pd.DataFrame({name: df["dx"].value_counts() for name, df in stages}).T.fillna(0).astype(int)
    cnt["total"] = cnt.sum(axis=1)
    cnt.index.name = "stage"
    cnt.to_csv(os.path.join(res, "audit_counts.csv"))
    excl = (man[~man["included"] & man["dx"].isin(classes)]
            .groupby(["exclusion_reason", "dx"]).size().unstack(fill_value=0))
    excl.to_csv(os.path.join(res, "audit_exclusions.csv"))

    # EEG-side participants (Records tables) without an MRI link
    linked = {subject_key(e) for e in man["eeg_id"] if e}
    eeg_only = collections.Counter(dx for k, dx in erec.items() if k not in linked)

    eeg_fmt = collections.Counter()
    for files in eidx.values():
        for p in files:
            eeg_fmt[signal_ok(p)[1]] += 1
    summary = {
        "task_classes": classes, "task_reason": why,
        "labels_in_table": dict(labels),
        "missing_labels": int((man["dx"] == "").sum()),
        "duplicated_mri_id_rows": dup_mri[["MRI_ID", "EEG_ID", "diagnosis"]].shape[0],
        "duplicated_eeg_id_rows": dup_eeg[["MRI_ID", "EEG_ID", "diagnosis"]].shape[0],
        "duplicated_mri_id_dx": dict(collections.Counter(dup_mri["diagnosis"])),
        "duplicated_eeg_id_dx": dict(collections.Counter(dup_eeg["diagnosis"])),
        "visits_per_participant": 1,
        "session_key": "none (BrainLat is cross-sectional; one visit per participant)",
        "pairing_rule": "MRI_ID <-> EEG_ID in " + D["demographics"],
        "date_window_days": cfg["pairing"]["max_days_apart"],
        "mri_participants_in_release": int(man["mri_in_release"].sum()),
        "mri_participants_with_t1": int(man["t1_path"].notna().sum()),
        "t1_copy_check_site_vs_newids": dict(copy_checks),
        "mri_modalities_file_counts": mri_modalities(mri_root, site_dirs),
        "mri_regional_measures_distributed": False,
        "eeg_resting_files_by_data_status": dict(eeg_fmt),
        "eeg_records_participants_without_mri_link": dict(eeg_only),
        "included_by_dx": dict(collections.Counter(cohort["dx"])),
        "included_by_site": dict(collections.Counter(cohort["site"])),
        "included_eeg_source": dict(collections.Counter(cohort["eeg_source"])),
        "flagged_included": int((cohort["flags"] != "").sum()),
    }
    with open(os.path.join(res, "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nCounts by stage and diagnosis:\n" + cnt.to_string())
    print("\nExcluded (task classes) by reason:\n" + excl.to_string())
    print("\n" + json.dumps({k: summary[k] for k in summary
                             if k not in ("mri_modalities_file_counts",)}, indent=2))
    print(f"\nwrote {work}/pairing_manifest.csv, {work}/paired_cohort.csv (participant-level, local)")
    print(f"wrote {res}/audit_counts.csv, audit_exclusions.csv, audit_summary.json (aggregate)")


if __name__ == "__main__":
    main()
