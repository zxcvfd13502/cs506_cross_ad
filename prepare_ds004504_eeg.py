"""
Select AD and healthy-control resting-state EEG from OpenNeuro ds004504, harmonize it,
and extract the same spectral features as prepare_brainlat_eeg.py (one row per subject).

Steps
  1. Read participants.tsv: Group (A = AD, F = FTD, C = CN), Age, Gender, MMSE.
     Only A and C are processed, relabelled AD and CN.
  2. Use the raw recordings in sub-*/eeg/ (not derivatives/), so preprocessing is ours.
  3. Set the standard_1020 montage (19 channels, 10-20 names incl. T3/T4/T5/T6).
  4. Harmonize and extract features with the functions from prepare_brainlat_eeg.py:
     average reference, 1-30 Hz, 256 Hz, at most --max-seconds; relative band power
     globally and per region, theta/alpha ratio, peak alpha frequency.

Outputs (in --out)
  eeg_inventory.csv   every subject: group, file, status, original sfreq / channels / seconds
  eeg_features.csv    one row per AD/CN subject, same columns as the BrainLat features
                      (dx = AD/CN, sex = M/F) plus mmse

Example
  python prepare_ds004504_eeg.py --bids-dir /projectnb/mlresearch/Runkun/ds004504 \
      --out processed/ds004504
"""
import argparse
import glob
import os
import re

import mne
import pandas as pd

from prepare_brainlat_eeg import features, harmonize, load_continuous

mne.set_log_level("ERROR")

GROUPS = {"A": "AD", "F": "FTD", "C": "CN"}
KEEP = ("AD", "CN")
# column order of the BrainLat eeg_features.csv (features follow)
ID_COLS = ["file", "dx", "subject_key", "id_eeg", "site", "name_pattern", "orig_sfreq",
           "diagnosis", "sex", "age", "years_education", "id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-dir", required=True, help="root of the ds004504 download")
    ap.add_argument("--out", default="processed/ds004504")
    ap.add_argument("--max-seconds", type=float, default=300)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    part = pd.read_csv(os.path.join(args.bids_dir, "participants.tsv"), sep="\t")
    part.columns = [c.strip().lower() for c in part.columns]
    part["dx"] = part["group"].map(GROUPS)
    print(f"participants.tsv: {part.shape}; groups {part['group'].value_counts().to_dict()}")

    montage = mne.channels.make_standard_montage("standard_1020")
    inv, feats = [], []
    for s in part.itertuples():
        row = {"participant_id": s.participant_id, "group": s.group, "dx": s.dx}
        files = sorted(glob.glob(os.path.join(args.bids_dir, s.participant_id, "eeg",
                                              f"{s.participant_id}_task-*_eeg.set")))
        row["file"] = os.path.basename(files[0]) if files else None
        if s.dx not in KEEP:
            row["status"] = f"skipped: group {s.dx}"
        elif not files:
            row["status"] = "no raw .set in sub-*/eeg/"
        else:
            try:
                raw, kind = load_continuous(files[0])
                row.update(kind=kind, orig_sfreq=raw.info["sfreq"],
                           orig_nchan=len(raw.ch_names), orig_seconds=round(raw.times[-1], 1))
                raw.set_montage(montage, on_missing="warn")
                raw = harmonize(raw, args.max_seconds)
                f = features(raw)
                row["status"] = "ok"
                stem = os.path.splitext(row["file"])[0]
                feats.append({"file": row["file"], "dx": s.dx,
                              "subject_key": s.participant_id.replace("sub-", ""),
                              "id_eeg": s.participant_id, "site": "ds004504",
                              "name_pattern": re.sub(r"\d+", "#", stem),
                              "orig_sfreq": row["orig_sfreq"], "diagnosis": s.group,
                              "sex": s.gender, "age": s.age, "years_education": None,
                              "id": s.participant_id, **f, "mmse": s.mmse})
            except Exception as e:
                row["status"] = f"error: {type(e).__name__}: {str(e)[:150]}"
        inv.append(row)
        if s.dx in KEEP:
            print(f"  {s.participant_id} ({s.dx}): {row['status']}")

    inv = pd.DataFrame(inv)
    inv.to_csv(os.path.join(args.out, "eeg_inventory.csv"), index=False)
    print("\nStatus by group:\n" + pd.crosstab(inv["status"], inv["dx"]).to_string())

    feats = pd.DataFrame(feats)
    if feats.empty:
        raise SystemExit("No AD/CN recordings were processed; check eeg_inventory.csv")
    feats["sex"] = feats["sex"].str.strip().str.upper()
    feats = feats[ID_COLS + [c for c in feats.columns if c not in ID_COLS]]
    feats = feats.sort_values(["dx", "subject_key"])
    feats.to_csv(os.path.join(args.out, "eeg_features.csv"), index=False)

    print(f"\nSubjects with features: {feats['dx'].value_counts().to_dict()}")
    print("\nConfound check (recording setup vs diagnosis):")
    for col in ("name_pattern", "orig_sfreq", "n_channels_used"):
        print(pd.crosstab(feats[col], feats["dx"]), "\n")
    print(feats.groupby("dx")[["age", "mmse", "seconds_used"]].describe().T.to_string())
    print("\n" + pd.crosstab(feats["sex"], feats["dx"]).to_string())


if __name__ == "__main__":
    main()
