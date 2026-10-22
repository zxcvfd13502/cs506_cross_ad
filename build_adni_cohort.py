"""
Build an ADNI AD-vs-CN baseline structural MRI cohort from IDA table exports.

Inputs (CSV exports from the ADNI IDA "Tables" download):
  --fs       All_Subjects_UCSFFSX7_*.csv   UCSF cross-sectional FreeSurfer 7 (required)
  --dxsum    DXSUM*.csv                    Diagnostic Summary (required)
  --ptdemog  PTDEMOG*.csv                  demographics: sex, birth year, education (optional)
  --datadic  DATADIC_*.csv                 readable feature names + region-level QC masking (optional)
  --keymri   All_Subjects_Key_MRI_*.csv    adds scanner make/model where image IDs match (optional).
             ADNI1/GO/2 FreeSurfer runs used pre-processed image IDs that are not in Key MRI,
             so the cohort is built from UCSFFSX7 itself, not from Key MRI.

Logic:
  1. Candidate scans = UCSF FreeSurfer runs, minus runs rated OVERALLQC = Fail or
     "Hippocampus Only". Most runs (STATUS = partial) were never QC-rated; they are
     kept and flagged (qc_rated = False). Use --qc-rated-only to drop them.
  2. In runs rated overall "Partial", features of regions whose lobe-level QC failed
     are set to NaN (needs --datadic).
  3. One scan per subject: the EARLIEST candidate scan (no subject appears twice).
  4. Diagnosis from the DXSUM visit closest in date (within --max-days).
  5. Keep AD (dementia) and CN only.

Outputs (in --out):
  adni_mri_cohort.csv   one row per subject: IDs, diagnosis, scan info, demographics, FS features
  adni_image_ids.txt    comma-separated IMAGEUIDs (for IDA Advanced Search > Image ID)

Example:
  python build_adni_cohort.py \\
      --fs All_Subjects_UCSFFSX7_21Oct2026.csv --dxsum DXSUM_21Oct2026.csv \\
      --ptdemog PTDEMOG_21Oct2026.csv --datadic DATADIC_21Oct2026.csv \\
      --keymri All_Subjects_Key_MRI_21Oct2026.csv --out adni_out
"""
import argparse
import os
import re

import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path, low_memory=False, encoding="latin1")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def pick(df, options, what):
    for c in options:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find a {what} column; looked for {options}. "
                   f"Available: {list(df.columns)[:30]}...")


# ---------------------------------------------------------------- diagnosis
def norm_code(v):
    """'1', 1, 1.0 -> '1'; text -> lowercase text; missing -> None."""
    if pd.isna(v) or str(v).strip() == "":
        return None
    s = str(v).strip()
    try:
        return str(int(float(s)))
    except ValueError:
        return s.lower()


def text_to_dx(s):
    """Map a text label ('CN', 'Dementia', 'Stable: NL to NL', ...) to CN/MCI/AD."""
    if not isinstance(s, str):
        return None
    s = (s.split(" to ")[-1] if " to " in s else s.split(":")[-1]).strip()
    if s in ("cn", "nl", "normal"):
        return "CN"
    if "mci" in s:
        return "MCI"
    if "dementia" in s or s == "ad":
        return "AD"
    return None


DIAGNOSIS_MAP = {"1": "CN", "2": "MCI", "3": "AD"}            # ADNI3/4: DIAGNOSIS
DXCURREN_MAP = {"1": "CN", "2": "MCI", "3": "AD"}             # ADNI1:   DXCURREN
DXCHANGE_MAP = {"1": "CN", "7": "CN", "9": "CN",              # ADNIGO/2: DXCHANGE
                "2": "MCI", "4": "MCI", "8": "MCI",
                "3": "AD", "5": "AD", "6": "AD"}


def harmonize_dx(dx):
    """One CN/MCI/AD column across ADNI phases."""
    out = pd.Series([None] * len(dx), index=dx.index, dtype=object)
    for col, cmap in (("diagnosis", DIAGNOSIS_MAP),
                      ("dxcurren", DXCURREN_MAP),
                      ("dxchange", DXCHANGE_MAP)):
        if col not in dx.columns:
            continue
        codes = dx[col].map(norm_code)
        mapped = codes.map(lambda c: None if not isinstance(c, str)
                           else cmap.get(c) if c.isdigit() else text_to_dx(c))
        out = out.fillna(mapped)
    return out


# ---------------------------------------------------------------- data dictionary helpers
def fs_dictionary(datadic_path):
    """Return (rename map ST###xx -> Region_xx, {qc_col: [region codes]})."""
    dd = load(datadic_path)
    dd = dd[dd["tblname"].astype(str).str.upper() == "UCSFFSX7"]
    rename, qc_regions = {}, {}
    for fld, text in zip(dd["fldname"].astype(str), dd["text"].astype(str)):
        m = re.search(r"of\s+(\w+)\s*$", text)
        if re.fullmatch(r"ST\d+(CV|SV|SA|TA|TS)", fld) and m:
            rename[fld.lower()] = f"{m.group(1)}_{fld[-2:]}"
        if fld.upper().endswith("QC") and fld.upper() != "OVERALLQC":
            qc_regions[fld.lower()] = [c.lower() for c in re.findall(r"\((ST\d+)\)", text)]
    return rename, qc_regions


def mask_failed_regions(fs, qc_regions):
    """Set features of regions with failed lobe-level QC to NaN."""
    n = 0
    for qc_col, codes in qc_regions.items():
        if qc_col not in fs.columns:
            continue
        failed = fs[qc_col].astype(str).str.lower().eq("fail")
        cols = [c for c in fs.columns if any(re.fullmatch(code + r"(cv|sv|sa|ta|ts)", c)
                                             for code in codes)]
        fs.loc[failed, cols] = np.nan
        n += int(failed.sum()) * len(cols)
    return n


# ---------------------------------------------------------------- matching
def match_controls(cohort, ratio, caliper):
    """Greedy 1:ratio matching of CN to AD on sex and age (|age diff| <= caliper)."""
    ad = cohort[cohort["dx"] == "AD"].dropna(subset=["age", "sex"])
    cn = cohort[cohort["dx"] == "CN"].dropna(subset=["age", "sex"])
    used, keep_ad, keep_cn = set(), [], []
    for _, row in ad.sort_values("age").iterrows():
        pool = cn[(cn["sex"] == row["sex"]) & (~cn.index.isin(used))]
        pool = pool.assign(gap=(pool["age"] - row["age"]).abs())
        pool = pool[pool["gap"] <= caliper].nsmallest(ratio, "gap")
        if len(pool):
            keep_ad.append(row.name)
            keep_cn.extend(pool.index)
            used.update(pool.index)
    return cohort.loc[keep_ad + keep_cn]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", required=True)
    ap.add_argument("--dxsum", required=True)
    ap.add_argument("--ptdemog")
    ap.add_argument("--datadic")
    ap.add_argument("--keymri")
    ap.add_argument("--out", default="adni_out")
    ap.add_argument("--max-days", type=int, default=180,
                    help="max gap between scan date and diagnosis visit (days)")
    ap.add_argument("--field-strength", type=float, default=None,
                    help="optionally keep only 1.5 or 3.0 T scans")
    ap.add_argument("--qc-rated-only", action="store_true",
                    help="drop FreeSurfer runs that were never QC-rated")
    ap.add_argument("--match-ratio", type=int, default=0,
                    help="match k CN per AD on sex and age (0 = no matching)")
    ap.add_argument("--caliper", type=float, default=2.0,
                    help="max age difference (years) for matching")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1. FreeSurfer runs ---------------------------------------------------------
    fs = load(args.fs)
    fs = fs.rename(columns={"imageuid": "image_id"}).copy()
    fs["scan_date"] = pd.to_datetime(fs["examdate"], errors="coerce")
    fs["field_strength_t"] = (fs["field_strength"].astype(str)
                              .str.extract(r"([\d.]+)")[0].astype(float))
    qc = fs["overallqc"].astype(str).str.lower()
    fs["qc_rated"] = fs["overallqc"].notna()
    print(f"FreeSurfer runs: {len(fs):,} ({fs['ptid'].nunique():,} subjects); "
          f"QC-rated: {fs['qc_rated'].sum():,}")

    keep = ~qc.isin(["fail", "hippocampus only"])
    if args.qc_rated_only:
        keep &= fs["qc_rated"]
    if args.field_strength:
        keep &= fs["field_strength_t"].eq(args.field_strength)
    fs = fs[keep].copy()

    rename = {}
    if args.datadic:
        rename, qc_regions = fs_dictionary(args.datadic)
        n = mask_failed_regions(fs, qc_regions)
        print(f"Masked {n:,} feature values in regions that failed lobe-level QC")

    # 2. Earliest usable scan per subject ---------------------------------------
    base = (fs.dropna(subset=["scan_date"])
              .sort_values(["ptid", "scan_date", "image_id"])
              .groupby("ptid", as_index=False).head(1))
    print(f"Subjects with a usable baseline scan: {len(base):,}")

    # 3. Diagnosis closest in time to the scan ----------------------------------
    dx = load(args.dxsum)
    dx = dx.rename(columns={pick(dx, ["ptid", "subject_id"], "subject ID"): "ptid",
                            pick(dx, ["examdate", "exam_date", "userdate"], "exam date"): "dx_date"})
    dx["dx"] = harmonize_dx(dx)
    dx["dx_date"] = pd.to_datetime(dx["dx_date"], errors="coerce")
    dx = dx.dropna(subset=["dx", "dx_date"])[["ptid", "dx_date", "dx"]].sort_values("dx_date")

    base = pd.merge_asof(base.sort_values("scan_date"), dx,
                         left_on="scan_date", right_on="dx_date", by="ptid",
                         direction="nearest", tolerance=pd.Timedelta(days=args.max_days))
    base["dx_gap_days"] = (base["scan_date"] - base["dx_date"]).dt.days.abs()
    print("Diagnosis at baseline scan:\n" +
          base["dx"].value_counts(dropna=False).to_string())

    cohort = base[base["dx"].isin(["CN", "AD"])].copy()

    # 4. Optional extras ----------------------------------------------------------
    if args.ptdemog:
        dm = load(args.ptdemog)
        dm = dm.rename(columns={pick(dm, ["ptid", "subject_id"], "subject ID"): "ptid"})
        if "ptdobyy" in dm.columns:   # may be a year or a date like '1935-01-01'
            dm["ptdobyy"] = pd.to_numeric(dm["ptdobyy"].astype(str).str.extract(r"(\d{4})")[0],
                                          errors="coerce")
        elif "ptdob" in dm.columns:
            dm["ptdobyy"] = pd.to_numeric(dm["ptdob"].astype(str).str.extract(r"(\d{4})")[0],
                                          errors="coerce")
        cols = [c for c in ("ptgender", "ptdobyy", "pteducat") if c in dm.columns]
        for c in cols:   # ADNI codes missing values as negative numbers
            dm[c] = pd.to_numeric(dm[c], errors="coerce").where(lambda s: s >= 0)
        cohort = cohort.merge(dm.groupby("ptid")[cols].first().reset_index(),
                              on="ptid", how="left")
        if "ptdobyy" in cohort.columns:
            cohort["age"] = cohort["scan_date"].dt.year - cohort["ptdobyy"]
        if "ptgender" in cohort.columns:
            cohort["sex"] = cohort["ptgender"].map({1: "M", 2: "F"})

    if args.keymri:
        km = load(args.keymri)[["image_id", "scanner_manufacturer", "scanner_model",
                                "series_description"]]
        cohort = cohort.merge(km.drop_duplicates("image_id"), on="image_id", how="left")

    if args.match_ratio > 0:
        if not {"age", "sex"} <= set(cohort.columns):
            raise SystemExit("--match-ratio needs --ptdemog (for age and sex)")
        cohort = match_controls(cohort, args.match_ratio, args.caliper)
        print(f"\nAge/sex matched (1:{args.match_ratio}, caliper {args.caliper} y): "
              f"{len(cohort)} subjects")

    cohort = cohort.rename(columns=rename).sort_values(["dx", "ptid"])

    # 5. Save + summary ----------------------------------------------------------
    out_csv = os.path.join(args.out, "adni_mri_cohort.csv")
    cohort.to_csv(out_csv, index=False)
    with open(os.path.join(args.out, "adni_image_ids.txt"), "w") as f:
        f.write(",".join(str(int(i)) for i in cohort["image_id"]))

    print(f"\nFinal AD/CN cohort: {len(cohort)} subjects -> {out_csv}")
    print(pd.crosstab(cohort["dx"], cohort["field_strength_t"], margins=True))
    print(pd.crosstab(cohort["dx"], cohort["phase"], margins=True))
    print(pd.crosstab(cohort["dx"], cohort["qc_rated"], margins=True))
    if "age" in cohort.columns:
        print(cohort.groupby("dx")["age"].describe()[["count", "mean", "std"]].round(1))
    if "sex" in cohort.columns:
        print(pd.crosstab(cohort["dx"], cohort["sex"]))


if __name__ == "__main__":
    main()
