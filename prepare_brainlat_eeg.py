"""
Select AD and healthy-control resting-state EEG from BrainLat, harmonize it,
and extract interpretable spectral features (one row per subject).

Steps
  1. Read the Records / Demographics CSVs for every group (AD, HC, bvFTD, MS) so each
     .set file can be assigned to a subject and group; only AD and HC are processed.
  2. Match every .set file to a subject by its parsed ID (sub-30011, suj_509 -> sub_509,
     s40001_AR_PD -> PD, which has no Records file).
  3. Check that the signal is present: many BrainLat .set files are header-only and
     point to a .fdt file that is not part of the Synapse download.
  4. Drop task recordings (filenames containing 'extero' or 'MW' by default).
  5. Harmonize every recording the same way: EEG channels only, average reference,
     1-30 Hz band-pass, resample to 256 Hz, at most --max-seconds of data.
  6. Features: relative band power (delta, theta, alpha, beta) per scalp region and
     globally, theta/alpha ratio, and peak alpha frequency.

Outputs (in --out)
  eeg_inventory.csv          every .set file: subject, group, data status, sampling rate, status
  eeg_missing_subjects.csv   AD/HC subjects in Records without a usable recording, with the reason
  eeg_features.csv           one row per AD/CN subject: dx, site, age, sex, features
  check the "name_pattern", "orig_sfreq" and "site" columns against dx for recording-setup confounds

Example
  python prepare_brainlat_eeg.py --eeg-dir /project/mlresearch/Runkun/EEG \
      --out /projectnb/mlresearch/Runkun/eeg_out
"""
import argparse
import glob
import os
import re

import mne
import numpy as np
import pandas as pd
import scipy.io

mne.set_log_level("ERROR")

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
GROUPS = {"AD": "AD", "HC": "CN", "bvFTD": "FTD", "MS": "MS"}   # Records file -> group label
KEEP = ("AD", "CN")                                             # labels used for ADNI too


# ------------------------------------------------------------------ CSV handling
def read_csv(path):
    df = pd.read_csv(path, low_memory=False, encoding="latin1")
    # "id EEG" (Demographics) and "id_EEG" (Records) both become "id_eeg"
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    return df


def find_csv(eeg_dir, kind, group):
    hits = glob.glob(os.path.join(eeg_dir, "**", f"{kind}_{group}_EEG_data.csv"), recursive=True)
    return hits[0] if hits else None


def subject_key(s):
    """Canonical subject key from an ID or filename: sub-30011 / s6_sub-30011_rs-HEP_eeg -> 30011,
    sub_509 / suj_509_rs_eeg / Suj_502_extero -> 509, s40001_AR_PD_reject -> 40001."""
    s = str(s)
    m = re.search(r"(?:sub|suj)[-_](\d+)", s, re.I) or re.match(r"s(\d{4,})", s, re.I)
    return str(int(m.group(1))) if m else None


def build_subject_table(eeg_dir):
    rows = []
    for group, label in GROUPS.items():
        rec_path = find_csv(eeg_dir, "Records", group)
        dem_path = find_csv(eeg_dir, "Demographics", group)
        if rec_path is None:
            raise SystemExit(f"Records_{group}_EEG_data.csv not found under {eeg_dir}")
        rec = read_csv(rec_path)
        print(f"\n{os.path.basename(rec_path)}: {rec.shape}\n  columns: {list(rec.columns)}")
        rec["site"] = rec["path"].astype(str).str.split("/").str[-1]     # e.g. 1_AD/AR -> AR
        if dem_path:
            dem = read_csv(dem_path)
            extra = [c for c in dem.columns if c not in rec.columns or c == "id_eeg"]
            rec = rec.merge(dem[extra].drop_duplicates("id_eeg"), on="id_eeg", how="left")
        rec["dx"] = label
        rows.append(rec)
    subj = pd.concat(rows, ignore_index=True)
    subj["key"] = subj["id_eeg"].map(subject_key)
    dup = subj[subj.duplicated("key", keep=False)]
    if len(dup):
        print(f"\nWarning: duplicated subject IDs in Records (first row kept):\n"
              f"{dup[['id_eeg', 'dx', 'site']].to_string()}")
    return subj.drop_duplicates("key").set_index("key")


def file_tokens(stem):
    return [t.lower() for t in re.split(r"[_\-\s.]+", stem) if t]


# ------------------------------------------------------------------ EEG handling
def data_status(path):
    """'embedded', 'fdt present' or 'fdt missing' -- read from the .set header only."""
    try:
        m = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False,
                             variable_names=["EEG", "data"])
    except NotImplementedError:          # MATLAB v7.3 (HDF5): let MNE handle it
        return "v7.3 (unchecked)", None
    d = getattr(m["EEG"], "data", None) if "EEG" in m else m.get("data")
    if not isinstance(d, str):
        return "embedded", None
    folder = os.path.dirname(path)
    candidates = [os.path.join(folder, d), os.path.splitext(path)[0] + ".fdt"]
    return ("fdt present" if any(os.path.exists(c) for c in candidates) else "fdt missing"), d


def load_continuous(path):
    """Load an EEGLAB .set file as Raw; epoched files are concatenated."""
    try:
        return mne.io.read_raw_eeglab(path, preload=True), "continuous"
    except Exception as raw_err:
        try:
            ep = mne.read_epochs_eeglab(path)
        except Exception:
            raise raw_err                # the continuous-reader error is the informative one
        data = np.concatenate(list(ep.get_data()), axis=-1)
        raw = mne.io.RawArray(data, ep.info)
        return raw, f"epoched({len(ep)})"


def ensure_positions(raw):
    """Make sure channels have 3-D positions (needed for region features)."""
    mont = raw.get_montage()
    if mont is not None:
        pos = np.array(list(mont.get_positions()["ch_pos"].values()))
        valid = np.isfinite(pos).all(axis=1) & (np.abs(pos).sum(axis=1) > 0)
        if valid.sum() >= 0.8 * len(raw.ch_names):
            return True
    for name in ("biosemi128", "standard_1005"):
        std = mne.channels.make_standard_montage(name)
        if sum(ch in std.ch_names for ch in raw.ch_names) >= 0.8 * len(raw.ch_names):
            raw.set_montage(std, on_missing="ignore")
            return True
    return False


def region_of(pos):
    """Coarse scalp region from head coordinates (meters, +y = nose, +x = right)."""
    x, y, _ = pos
    if y > 0.03:
        return "frontal"
    if y < -0.04:
        return "posterior"
    if abs(x) > 0.05:
        return "temporal"
    return "central"


def harmonize(raw, max_seconds):
    raw.pick("eeg")
    drop = [ch for ch in raw.ch_names if re.match(r"(EXG|EOG|ECG|EMG|M[12]$)", ch, re.I)]
    raw.drop_channels(drop)
    raw.set_eeg_reference("average", verbose=False)
    raw.filter(1.0, 30.0, verbose=False)
    if raw.info["sfreq"] != 256:
        raw.resample(256, verbose=False)
    if raw.times[-1] > max_seconds:
        raw.crop(0, max_seconds)
    return raw


def features(raw):
    psd = raw.compute_psd(method="welch", fmin=1, fmax=30, n_fft=512,
                          n_overlap=256, reject_by_annotation=True, verbose=False)
    p, f = psd.get_data(return_freqs=True)                 # (n_ch, n_freq)
    rel = {}
    total = p.sum(axis=1, keepdims=True)
    band_rel = {b: p[:, (f >= lo) & (f < hi)].sum(axis=1) / total[:, 0]
                for b, (lo, hi) in BANDS.items()}

    regions = {"global": np.arange(len(raw.ch_names))}
    if ensure_positions(raw):
        pos = raw.get_montage().get_positions()["ch_pos"]
        labels = np.array([region_of(pos[ch]) if ch in pos and np.isfinite(pos[ch]).all()
                           else "none" for ch in raw.ch_names])
        for r in ("frontal", "central", "temporal", "posterior"):
            if (labels == r).any():
                regions[r] = np.where(labels == r)[0]

    for r, idx in regions.items():
        for b in BANDS:
            rel[f"relpow_{b}_{r}"] = float(band_rel[b][idx].mean())
        rel[f"theta_alpha_ratio_{r}"] = rel[f"relpow_theta_{r}"] / rel[f"relpow_alpha_{r}"]

    mean_spec = p.mean(axis=0)
    a = (f >= 7) & (f <= 13)
    rel["peak_alpha_freq"] = float(f[a][np.argmax(mean_spec[a])])
    rel["n_channels_used"] = len(raw.ch_names)
    rel["seconds_used"] = round(float(raw.times[-1]), 1)
    return rel


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-dir", required=True, help="folder with the unzipped BrainLat EEG")
    ap.add_argument("--out", default="eeg_out")
    ap.add_argument("--exclude", default="extero,MW",
                    help="comma-separated filename parts marking task recordings to drop")
    ap.add_argument("--other-groups", default="PD",
                    help="filename tags of groups without a Records file (labelled, not processed)")
    ap.add_argument("--max-seconds", type=float, default=300)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    subj = build_subject_table(args.eeg_dir)
    files = sorted(glob.glob(os.path.join(args.eeg_dir, "**", "*.set"), recursive=True))
    excl = [e.strip().lower() for e in args.exclude.split(",") if e.strip()]
    other = [g.strip().lower() for g in args.other_groups.split(",") if g.strip()]
    print(f"\nFound {len(files)} .set files; subjects in records: "
          f"{subj['dx'].value_counts().to_dict()}")

    inv, feats = [], []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        toks = file_tokens(stem)
        key = subject_key(stem)
        row = {"file": os.path.basename(path), "name_pattern": re.sub(r"\d+", "#", stem),
               "subject_key": key}
        row["data"], row["fdt_name"] = data_status(path)
        tag = next((g.upper() for g in other if g in toks), None)
        if key in subj.index:
            s = subj.loc[key]
            row.update(id_eeg=s["id_eeg"], dx=s["dx"], site=s["site"])
        else:
            row["dx"] = tag or "unknown"
            row["site"] = next((t.upper() for t in toks if t in ("ar", "ch", "cl")), None)
        if row["dx"] not in KEEP:
            row["status"] = f"skipped: group {row['dx']}"
        elif any(e in toks for e in excl):
            row["status"] = "excluded (task recording)"
        elif row["data"] == "fdt missing":
            row["status"] = f"no signal: header only, {row['fdt_name']} not downloaded"
        else:
            try:
                raw, kind = load_continuous(path)
                row.update(kind=kind, orig_sfreq=raw.info["sfreq"],
                           orig_nchan=len(raw.ch_names), orig_seconds=round(raw.times[-1], 1))
                raw = harmonize(raw, args.max_seconds)
                f = features(raw)
                row["status"] = "ok"
                demo = {k: s[k] for k in s.index
                        if k in ("id", "age", "sex", "years_education", "diagnosis")}
                feats.append({"file": row["file"], "dx": s["dx"], "subject_key": key,
                              "id_eeg": s["id_eeg"], "site": s["site"],
                              "name_pattern": row["name_pattern"],
                              "orig_sfreq": row["orig_sfreq"], **demo, **f})
            except Exception as e:
                row["status"] = f"error: {type(e).__name__}: {str(e)[:150]}"
        inv.append(row)
        if row["dx"] in KEEP:
            print(f"  {row['file']}: {row['status']}")

    inv = pd.DataFrame(inv)
    inv.to_csv(os.path.join(args.out, "eeg_inventory.csv"), index=False)

    # AD/HC subjects listed in Records without a usable recording
    ok_keys = set(inv.loc[inv["status"] == "ok", "subject_key"])
    missing = []
    for key, s in subj[subj["dx"].isin(KEEP)].iterrows():
        if key in ok_keys:
            continue
        have = inv[inv["subject_key"] == key]
        reason = ("no .set file in download" if have.empty
                  else "; ".join(f"{r.file}: {r.status}" for r in have.itertuples()))
        missing.append({"id_eeg": s["id_eeg"], "dx": s["dx"], "site": s["site"],
                        "eeg_flag": s.get("eeg"), "reason": reason})
    missing = pd.DataFrame(missing)
    missing.to_csv(os.path.join(args.out, "eeg_missing_subjects.csv"), index=False)

    print("\nFiles per group and data status:")
    print(pd.crosstab(inv["dx"], inv["data"], margins=True))
    print("\nAD/CN file status:\n" + inv.loc[inv["dx"].isin(KEEP), "status"]
          .str.replace(r"s6_sub-\d+_rs_eeg\.fdt", "<.fdt>", regex=True).value_counts().to_string())
    if len(missing):
        print("\nAD/CN subjects without a usable recording: "
              f"{missing['dx'].value_counts().to_dict()} (see eeg_missing_subjects.csv)")

    feats = pd.DataFrame(feats)
    if feats.empty:
        raise SystemExit("No AD/HC recordings were processed; check eeg_inventory.csv")

    # one recording per subject (keep the longest)
    feats = (feats.sort_values("seconds_used", ascending=False)
                  .drop_duplicates("subject_key").sort_values(["dx", "subject_key"]))
    if "sex" in feats.columns:   # BrainLat: 0 = female, 1 = male (same labels as ADNI file)
        feats["sex"] = pd.to_numeric(feats["sex"], errors="coerce").map({0: "F", 1: "M"})
    feats.to_csv(os.path.join(args.out, "eeg_features.csv"), index=False)

    print(f"\nSubjects with features: {feats['dx'].value_counts().to_dict()}")
    print("\nConfound check (recording setup vs diagnosis):")
    for col in ("name_pattern", "orig_sfreq", "site"):
        print(pd.crosstab(feats[col], feats["dx"]), "\n")


if __name__ == "__main__":
    main()
