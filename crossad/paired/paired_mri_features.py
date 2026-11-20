"""
MRI branch features for the paired cohort.

BrainLat distributes raw T1-weighted volumes only (no regional measures), and every
included pair is a CLB 3T scan, mostly 1 mm isotropic with one 1x1x2 mm scan. With ~25
participants a 3D CNN cannot be trained reliably, so the T1s go through an existing
regional pipeline: FreeSurfer SynthSeg (--parc --robust), which is contrast- and
resolution-agnostic, needs no site-specific tuning, and runs in ~2-3 min per scan on CPU.

Features: a priori AD-signature regions (left + right summed) divided by total
intracranial volume. With mri.feature_set: all, every SynthSeg volume is used (/ICV)
and any feature selection happens inside the training folds (paired_train.py).
SynthSeg QC scores below mri.min_qc are flagged (not silently dropped).

Outputs (participant-level, local): <work>/synthseg/, <work>/mri_features.csv
Aggregate QC (git-safe):           <results>/mri_qc_summary.csv

  python -m crossad.paired.paired_mri_features --config configs/paired_config.yaml            # segment + collect
  python -m crossad.paired.paired_mri_features --config configs/paired_config.yaml --collect-only
"""
import argparse
import os
import subprocess

import pandas as pd
import yaml

AD_SIGNATURE = {
    "hippocampus": ["hippocampus"], "amygdala": ["amygdala"],
    "inf_lat_ventricle": ["inferior lateral ventricle"], "lateral_ventricle": ["lateral ventricle"],
    "entorhinal": ["entorhinal"], "parahippocampal": ["parahippocampal"],
    "fusiform": ["fusiform"], "inferior_temporal": ["inferiortemporal"],
    "middle_temporal": ["middletemporal"], "temporal_pole": ["temporalpole"],
    "precuneus": ["precuneus"], "isthmus_cingulate": ["isthmuscingulate"],
    "posterior_cingulate": ["posteriorcingulate"], "inferior_parietal": ["inferiorparietal"],
    "cerebral_cortex": ["cerebral cortex"], "cerebral_white_matter": ["cerebral white matter"],
}


def side_cols(cols, key):
    """SynthSeg column names for one region in both hemispheres."""
    hits = [c for c in cols if c in (f"left {key}", f"right {key}",
                                     f"ctx-lh-{key}", f"ctx-rh-{key}")]
    if len(hits) != 2:
        raise KeyError(f"expected 2 hemisphere columns for {key!r}, got {hits}")
    return hits


def segment(cohort, out_dir, M):
    env = dict(os.environ, FREESURFER_HOME=M["freesurfer_home"],
               PATH=os.path.join(M["freesurfer_home"], "bin") + os.pathsep + os.environ["PATH"])
    todo = [r for r in cohort.itertuples()
            if not os.path.exists(os.path.join(out_dir, f"{r.mri_id}_vol.csv"))]
    print(f"SynthSeg: {len(cohort) - len(todo)} done, {len(todo)} to run")
    for r in todo:
        base = os.path.join(out_dir, r.mri_id)
        cmd = ["mri_synthseg", "--i", r.t1_path, "--o", f"{base}_seg.nii.gz", "--parc",
               "--robust", "--vol", f"{base}_vol.csv", "--qc", f"{base}_qc.csv",
               "--threads", str(M["threads"]), "--cpu"]
        print("  " + r.mri_id, flush=True)
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)


def collect(cohort, out_dir, M):
    rows, qc = [], []
    for r in cohort.itertuples():
        v = pd.read_csv(os.path.join(out_dir, f"{r.mri_id}_vol.csv")).iloc[0]
        q = pd.read_csv(os.path.join(out_dir, f"{r.mri_id}_qc.csv")).iloc[0]
        icv = float(v["total intracranial"])
        row = {"mri_id": r.mri_id, "dx": r.dx, "icv_ml": icv / 1000}
        if M["feature_set"] == "ad_signature":
            for name, (key,) in AD_SIGNATURE.items():
                row[f"vol_{name}"] = sum(float(v[c]) for c in side_cols(v.index, key)) / icv
        else:
            for c in v.index[1:]:
                row["vol_" + c.replace(" ", "_")] = float(v[c]) / icv
        qv = q.drop(labels=["subject"]).astype(float)
        row["synthseg_qc_min"] = float(qv.min())
        rows.append(row)
        qc.append({"mri_id": r.mri_id, "dx": r.dx, "qc_min": float(qv.min()),
                   "qc_flag": bool(qv.min() < M["min_qc"]), "icv_ml": icv / 1000})
    return pd.DataFrame(rows), pd.DataFrame(qc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paired_config.yaml")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cohort", default=None, help="cohort CSV (default <work>/paired_cohort.csv)")
    ap.add_argument("--work", default=None, help="folder for features/QC (default paths.work); "
                    "SynthSeg outputs always go to <paths.work>/synthseg so scans are reused")
    ap.add_argument("--shard", default=None, help="i/n: segment only every n-th scan from i "
                    "(array jobs); implies no collect")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    M, work, res = cfg["mri"], cfg["paths"]["work"], cfg["paths"]["results"]
    if M["method"] != "synthseg":
        raise SystemExit(f"mri.method {M['method']!r} not implemented")
    cohort = pd.read_csv(args.cohort or os.path.join(work, "paired_cohort.csv"))
    if args.limit:
        cohort = cohort.head(args.limit)
    out_dir = os.path.join(work, "synthseg")
    os.makedirs(out_dir, exist_ok=True)
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        segment(cohort.iloc[i::n], out_dir, M)
        return
    if args.work:
        work, res = args.work, os.path.join(args.work, "qc")
    if not args.collect_only:
        segment(cohort, out_dir, M)
    feats, qc = collect(cohort, out_dir, M)
    feats.to_csv(os.path.join(work, "mri_features.csv"), index=False)
    qc.to_csv(os.path.join(work, "mri_qc.csv"), index=False)
    os.makedirs(res, exist_ok=True)
    agg = qc.groupby("dx").agg(n=("mri_id", "size"), qc_min_median=("qc_min", "median"),
                               qc_min_lowest=("qc_min", "min"), n_qc_flag=("qc_flag", "sum"),
                               icv_ml_median=("icv_ml", "median"))
    if not args.limit:
        agg.to_csv(os.path.join(res, "mri_qc_summary.csv"))
    print(agg.to_string())
    print(f"wrote {work}/mri_features.csv: {feats.shape}")


if __name__ == "__main__":
    main()
