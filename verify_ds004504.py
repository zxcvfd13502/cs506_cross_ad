"""
Verify the OpenNeuro ds004504 download: participants.tsv, subject counts by Group
(A = AD, F = FTD, C = CN), EEG file + sidecars in every sub-*/eeg/, derivatives/ presence,
and an MNE load test of every AD and CN raw recording (sfreq, channels, duration).

Usage:  python verify_ds004504.py --bids-dir /projectnb/mlresearch/Runkun/ds004504 \
            --out processed/ds004504_verification.csv
"""
import argparse
import glob
import os

import mne
import pandas as pd

mne.set_log_level("ERROR")
GROUPS = {"A": "AD", "F": "FTD", "C": "CN"}
SIDECARS = ("eeg.json", "channels.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    part = pd.read_csv(os.path.join(args.bids_dir, "participants.tsv"), sep="\t")
    print(f"participants.tsv: {len(part)} rows\n" +
          part["Group"].map(GROUPS).value_counts().to_string())
    dirs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(args.bids_dir, "sub-*")))
    print(f"sub-* folders: {len(dirs)}; not in participants.tsv: "
          f"{sorted(set(dirs) - set(part['participant_id']))}")

    rows = []
    for s in part.itertuples():
        eeg = os.path.join(args.bids_dir, s.participant_id, "eeg")
        sets = sorted(glob.glob(os.path.join(eeg, "*_eeg.set")))
        row = {"participant_id": s.participant_id, "group": GROUPS.get(s.Group, s.Group),
               "eeg_file": os.path.basename(sets[0]) if sets else None,
               "eeg_bytes": os.path.getsize(sets[0]) if sets else None, "loads": None,
               "sfreq": None, "nchan": None, "duration_s": None, "error": None}
        stem = sets[0][:-len("eeg.set")] if sets else None
        for sc in SIDECARS:
            row[f"has_{sc.replace('.', '_')}"] = bool(stem and os.path.exists(stem + sc))
        deriv = glob.glob(os.path.join(args.bids_dir, "derivatives", s.participant_id,
                                       "eeg", "*_eeg.set"))
        row["has_derivative"] = bool(deriv)
        errs = [] if sets else ["no raw *_eeg.set"]
        errs += [f"missing {sc}" for sc in SIDECARS if not row[f"has_{sc.replace('.', '_')}"]]
        if sets and row["group"] in ("AD", "CN"):
            try:
                raw = mne.io.read_raw_eeglab(sets[0], preload=True)
                row.update(loads=True, sfreq=raw.info["sfreq"], nchan=len(raw.ch_names),
                           duration_s=round(raw.n_times / raw.info["sfreq"], 1))
            except Exception as e:
                row["loads"] = False
                errs.append(f"load: {type(e).__name__}: {str(e)[:200]}")
        row["error"] = "; ".join(errs) or None
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")
    print(f"derivatives/ present: {os.path.isdir(os.path.join(args.bids_dir, 'derivatives'))}; "
          f"subjects with a derivative .set: {int(df['has_derivative'].sum())}/{len(df)}")
    print(df.groupby("group")[["has_eeg_json", "has_channels_tsv", "has_derivative"]].sum())
    t = df[df["loads"].notna()]
    print("\nload test:\n", pd.crosstab(t["group"], t["loads"]))
    print(t.groupby("group")[["sfreq", "nchan", "duration_s"]].describe().T.to_string())
    bad = df[df["error"].notna()]
    print("\nProblems:\n" + (bad[["participant_id", "group", "error"]].to_string(index=False)
                             if len(bad) else "none"))


if __name__ == "__main__":
    main()
