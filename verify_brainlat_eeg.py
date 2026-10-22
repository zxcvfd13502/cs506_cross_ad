"""
Verify the BrainLat EEG download in --eeg-dir:
  * every .set whose header points to a .fdt has that .fdt, with size
    nbchan x pnts x trials x 4 bytes (float32) as stated in the header;
  * every HC resting-state file (s6_sub-1xxxx_rs_eeg.set) loads with
    mne.io.read_raw_eeglab (sfreq, channels, duration reported).
AD files are load-tested too, since they feed the AD vs CN comparison.

Usage:  python verify_brainlat_eeg.py --eeg-dir /projectnb/mlresearch/Runkun/brainlat_eeg_full \
            --out processed/eeg/brainlat_verification.csv
"""
import argparse
import glob
import os
import re

import mne
import pandas as pd
import scipy.io

from prepare_brainlat_eeg import build_subject_table, file_tokens, subject_key

mne.set_log_level("ERROR")
HC_RS = re.compile(r"^s6_sub-1\d+_rs_eeg\.set$")


def header(path):
    """(fdt name or None, nbchan, pnts, trials) from the .set header."""
    m = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    eeg = m.get("EEG")
    get = (lambda k: getattr(eeg, k, None)) if eeg is not None else m.get
    d = get("data")
    return (d if isinstance(d, str) else None,
            int(get("nbchan")), int(get("pnts")), int(get("trials") or 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    subj = build_subject_table(args.eeg_dir)
    rows = []
    for path in sorted(glob.glob(os.path.join(args.eeg_dir, "**", "*.set"), recursive=True)):
        name = os.path.basename(path)
        key = subject_key(name)
        group = subj.loc[key, "dx"] if key in subj.index else (
            "PD" if "pd" in file_tokens(name) else "unknown")
        row = {"file": name, "group": group, "fdt_present": None, "size_ok": None,
               "loads": None, "sfreq": None, "nchan": None, "duration_s": None, "error": None}
        errs = []
        try:
            fdt, nbchan, pnts, trials = header(path)
            row.update(header_nchan=nbchan, header_pnts=pnts, header_trials=trials,
                       fdt_name=fdt)
            if fdt is None:
                row["fdt_present"] = "embedded"
            else:
                cands = [os.path.join(os.path.dirname(path), fdt),
                         os.path.splitext(path)[0] + ".fdt"]
                hit = next((c for c in cands if os.path.exists(c)), None)
                # several MS/PD headers name another subject's (HC) .fdt; flag them
                row["fdt_is_own"] = os.path.splitext(fdt)[0] == os.path.splitext(name)[0]
                if not row["fdt_is_own"]:
                    errs.append(f"header points to another file's data: {fdt}")
                row["fdt_present"] = hit is not None
                if hit:
                    expected = nbchan * pnts * trials * 4
                    actual = os.path.getsize(hit)
                    row.update(fdt_bytes=actual, expected_bytes=expected,
                               size_ok=actual == expected)
                    if actual != expected:
                        errs.append(f"fdt size {actual} != expected {expected}")
                else:
                    errs.append(f"{fdt} missing")
        except Exception as e:
            errs.append(f"header: {type(e).__name__}: {e}")

        if HC_RS.match(name) or group == "AD":
            try:
                raw = mne.io.read_raw_eeglab(path, preload=True)
                row.update(loads=True, sfreq=raw.info["sfreq"], nchan=len(raw.ch_names),
                           duration_s=round(raw.n_times / raw.info["sfreq"], 1))
            except Exception as e:
                row["loads"] = False
                errs.append(f"load: {type(e).__name__}: {str(e)[:200]}")
        row["error"] = "; ".join(errs) or None
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} .set files -> {args.out}")
    print(pd.crosstab(df["group"], df["fdt_present"].astype(str), margins=True))
    print("\nsize_ok:\n", df["size_ok"].value_counts(dropna=False).to_string())
    t = df[df["loads"].notna()]
    print("\nload test:\n", pd.crosstab(t["group"], t["loads"]))
    print("\n", t.groupby("group")[["sfreq", "nchan", "duration_s"]].describe().T.to_string())
    bad = df[df["error"].notna()]
    if len(bad):
        print("\nProblems:\n" + bad[["file", "group", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
