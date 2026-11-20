"""
EEG branch features for the paired cohort: clean epochs -> region-level spectral power ->
one vector per visit.

Per recording (harmonisation shared with prepare_brainlat_eeg.py):
  EEG channels only, average reference, band-pass, resample, crop to max_seconds
  -> fixed-length epochs
  -> bad channels: flat (<0.5 uV median peak-to-peak) or median peak-to-peak above
     median + 5 MAD across channels; dropped, then re-referenced
  -> bad epochs: max channel peak-to-peak above max(reject_abs_uv, median + k MAD)
  -> per clean epoch: Welch PSD per channel, relative band power (delta/theta/alpha/beta)
     averaged within scalp regions (frontal/central/temporal/posterior/global),
     theta/alpha ratio per region
  -> visit vector = median over clean epochs; peak alpha frequency from the median
     spectrum. Epochs never leave the visit, so a participant's epochs cannot be split.

Outputs (participant-level, local): <work>/eeg_features.csv, <work>/eeg_epochs/<mri_id>.npz
Aggregate QC (git-safe):           <results>/eeg_qc_summary.csv

  python -m crossad.paired.paired_eeg_features --config configs/paired_config.yaml [--limit 2]
"""
import argparse
import os

import mne
import numpy as np
import pandas as pd
import yaml
from mne.time_frequency import psd_array_welch

from crossad.preprocess.prepare_brainlat_eeg import BANDS, ensure_positions, harmonize, load_continuous, region_of

mne.set_log_level("ERROR")
REGIONS = ("frontal", "central", "temporal", "posterior")


def mad(x):
    return np.median(np.abs(x - np.median(x)))


def clean_epochs(raw, E):
    ep = mne.make_fixed_length_epochs(raw, duration=E["epoch_seconds"], preload=True,
                                      reject_by_annotation=True)
    X = ep.get_data() * 1e6                                       # (n_ep, n_ch, n_t), uV
    ptp = X.max(axis=2) - X.min(axis=2)                           # (n_ep, n_ch)
    ch_med = np.median(ptp, axis=0)
    bad_ch = (ch_med < 0.5) | (ch_med > np.median(ch_med) + 5 * max(mad(ch_med), 1e-6))
    keep_ch = np.where(~bad_ch)[0]
    X = X[:, keep_ch] - X[:, keep_ch].mean(axis=1, keepdims=True)   # re-reference
    names = [ep.ch_names[i] for i in keep_ch]
    ep_ptp = (X.max(axis=2) - X.min(axis=2)).max(axis=1)
    thr = max(E["reject_abs_uv"], np.median(ep_ptp) + E["reject_mad_k"] * mad(ep_ptp))
    good = ep_ptp <= thr
    return X[good], names, dict(n_epochs=len(X), n_clean_epochs=int(good.sum()),
                                n_bad_channels=int(bad_ch.sum()), ptp_threshold_uv=round(thr, 1))


def region_index(raw, names):
    idx = {"global": np.arange(len(names))}
    if ensure_positions(raw):
        pos = raw.get_montage().get_positions()["ch_pos"]
        lab = np.array([region_of(pos[c]) if c in pos and np.isfinite(pos[c]).all() else "none"
                        for c in names])
        for r in REGIONS:
            if (lab == r).any():
                idx[r] = np.where(lab == r)[0]
    return idx


def epoch_features(X, sfreq, regions, E):
    n_fft = int(E["epoch_seconds"] * sfreq)
    p, f = psd_array_welch(X, sfreq, fmin=E["l_freq"], fmax=E["h_freq"], n_fft=n_fft,
                           verbose=False)                         # (n_ep, n_ch, n_f)
    tot = p.sum(axis=2)
    rel = {b: p[:, :, (f >= lo) & (f < hi)].sum(axis=2) / tot for b, (lo, hi) in BANDS.items()}
    cols, F = [], []
    for r, ix in regions.items():
        for b in BANDS:
            cols.append(f"relpow_{b}_{r}")
            F.append(rel[b][:, ix].mean(axis=1))
        cols.append(f"theta_alpha_ratio_{r}")
        F.append(rel["theta"][:, ix].mean(axis=1) / rel["alpha"][:, ix].mean(axis=1))
    F = np.stack(F, axis=1)                                       # (n_ep, n_feat)
    med_spec = np.median(p.mean(axis=1), axis=0)
    a = (f >= 7) & (f <= 13)
    paf = float(f[a][np.argmax(med_spec[a])])
    return F, cols, paf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paired_config.yaml")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N (smoke test)")
    ap.add_argument("--out", default=None, help="override output CSV")
    ap.add_argument("--cohort", default=None, help="cohort CSV (default <work>/paired_cohort.csv)")
    ap.add_argument("--work", default=None, help="output folder (default paths.work)")
    ap.add_argument("--id-col", default="mri_id", help="participant ID column of the cohort")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    E, work, res = cfg["eeg"], cfg["paths"]["work"], cfg["paths"]["results"]
    cohort = pd.read_csv(args.cohort or os.path.join(work, "paired_cohort.csv"))
    cohort["mri_id"] = cohort[args.id_col]          # rows are keyed by this ID below
    if args.work:
        work, res = args.work, os.path.join(args.work, "qc")
    if args.limit:
        cohort = cohort.groupby("dx").head(max(1, args.limit // 2)).head(args.limit)
    os.makedirs(os.path.join(work, "eeg_epochs"), exist_ok=True)

    rows, qc = [], []
    for r in cohort.itertuples():
        q = {"mri_id": r.mri_id, "dx": r.dx}
        try:
            raw, kind = load_continuous(r.eeg_path)
            q.update(orig_sfreq=raw.info["sfreq"], orig_nchan=len(raw.ch_names),
                     orig_seconds=round(raw.times[-1], 1), kind=kind)
            raw = harmonize(raw, E["max_seconds"])
            if raw.info["sfreq"] != E["sfreq"]:
                raw.resample(E["sfreq"])
            X, names, cq = clean_epochs(raw, E)
            q.update(cq)
            if cq["n_clean_epochs"] < E["min_clean_epochs"]:
                q["status"] = f"excluded: {cq['n_clean_epochs']} clean epochs"
                qc.append(q)
                continue
            F, cols, paf = epoch_features(X, E["sfreq"], region_index(raw, names), E)
            np.savez_compressed(os.path.join(work, "eeg_epochs", f"{r.mri_id}.npz"),
                                features=F.astype(np.float32), columns=np.array(cols))
            vec = dict(zip(cols, np.median(F, axis=0)))
            vec["peak_alpha_freq"] = paf
            rows.append({"mri_id": r.mri_id, "dx": r.dx, **vec})
            q["status"] = "ok"
        except Exception as e:
            q["status"] = f"error: {type(e).__name__}: {str(e)[:120]}"
        qc.append(q)
        print(f"  {r.mri_id} {r.dx}: {q['status']} "
              f"({q.get('n_clean_epochs', '?')}/{q.get('n_epochs', '?')} epochs, "
              f"{q.get('n_bad_channels', '?')} bad ch)", flush=True)

    feats, qc = pd.DataFrame(rows), pd.DataFrame(qc)
    out = args.out or os.path.join(work, "eeg_features.csv")
    feats.to_csv(out, index=False)
    qc.to_csv(os.path.join(work, "eeg_qc.csv"), index=False)
    if not args.limit:
        os.makedirs(res, exist_ok=True)
        num = ["orig_sfreq", "orig_nchan", "orig_seconds", "n_epochs", "n_clean_epochs",
               "n_bad_channels"]
        agg = qc.groupby("dx")[[c for c in num if c in qc]].agg(["median", "min", "max"])
        agg[("n", "ok")] = qc[qc["status"] == "ok"].groupby("dx").size()
        agg.to_csv(os.path.join(res, "eeg_qc_summary.csv"))
        print(agg.to_string())
    print(f"wrote {out}: {feats.shape}")


if __name__ == "__main__":
    main()
