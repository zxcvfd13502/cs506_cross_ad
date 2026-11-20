"""
CROSS-AD pretrain-then-fuse: build the unpaired single-modality pools.

MRI pool  every BrainLat MRI participant with diagnosis CN or AD and a T1w volume in the
          Synapse release (taken from pairing_manifest.csv, so T1 choice and duplicate
          handling match the paired audit). Paired participants are flagged.
EEG pool  every resting-state recording in the BrainLat EEG tree under the AD and HC
          group folders (1_AD, 5_HC) whose signal is readable. Recordings that belong to
          a paired participant are mapped to that participant's MRI_ID via the EEG ID.

Leakage checks written to the summary:
  * every paired participant must appear in both pools (so it can be excluded by ID);
  * EEG recordings with the same sample count + channel count + sampling rate as a
    paired participant's recording but a different ID are flagged as likely duplicates
    (a case seen in BrainLat: two IDs sharing one recording). Flagged recordings are dropped from the pool.

Outputs (participant-level, local):  <pretrain_work>/mri_pool.csv, eeg_pool.csv
Aggregate (git-safe):                <pretrain_results>/pool_counts.csv, pool_summary.json

  python -m crossad.pretrain.pretrain_cohorts --config configs/pretrain_config.yaml
"""
import argparse
import glob
import json
import os

import pandas as pd
import scipy.io
import yaml

from crossad.paired.paired_audit import eeg_index
from crossad.preprocess.prepare_brainlat_eeg import data_status, subject_key

GROUP_DX = {"1_AD": "AD", "5_HC": "CN"}


def load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    with open(cfg["paired_config"]) as f:
        cfg["paired"] = yaml.safe_load(f)
    return cfg


def eeg_header(path):
    """(n_samples, n_channels, srate) from the EEGLAB header, for duplicate detection."""
    m = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False,
                         variable_names=["EEG", "pnts", "nbchan", "srate", "trials"])
    E = m.get("EEG")
    get = (lambda k: getattr(E, k, None)) if E is not None else m.get
    return int(get("pnts")) * int(get("trials") or 1), int(get("nbchan")), float(get("srate"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    P = cfg["paired"]
    pwork, work, res = P["paths"]["work"], cfg["paths"]["work"], cfg["paths"]["results"]
    os.makedirs(work, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    classes = cfg["classes"]

    man = pd.read_csv(os.path.join(pwork, "pairing_manifest.csv"))
    paired = pd.read_csv(os.path.join(pwork, "paired_cohort.csv"))
    paired_ids = set(paired["mri_id"])

    # ---------------- MRI pool
    mri = man[man["dx"].isin(classes) & man["mri_in_release"] & man["t1_path"].notna()].copy()
    mri = mri[["mri_id", "dx", "site", "age", "sex", "t1_path"]]
    mri["paired"] = mri["mri_id"].isin(paired_ids)
    assert paired_ids <= set(mri["mri_id"]), "paired participant missing from MRI pool"
    assert mri["mri_id"].is_unique

    # ---------------- EEG pool
    root = os.path.join(P["data"]["root"], P["data"]["eeg_dir"])
    eidx = eeg_index(root, P["eeg"]["exclude_name_parts"])
    # paired participants are matched by EEG ID; their pool row always uses the exact file
    # the paired run used (for 5 controls that is the MRI-tree copy, the EEG-tree copy
    # lacks its .fdt), so their features are identical in both experiments
    key2pair = {subject_key(e): (m, p) for e, m, p in
                zip(paired["eeg_id"], paired["mri_id"], paired["eeg_path"])}
    rows = []
    for key, files in eidx.items():
        for p in files:
            rel = os.path.relpath(p, root).split(os.sep)
            if rel[0] not in GROUP_DX:
                continue
            mid, ppath = key2pair.get(key, ("", None))
            p = ppath or p
            st, _ = data_status(p)
            row = {"eeg_key": key, "eeg_id": f"sub-{key}", "dx": GROUP_DX[rel[0]],
                   "site": rel[1], "eeg_path": p, "eeg_status": st,
                   "mri_id": mid, "paired": bool(mid)}
            if st in ("embedded", "fdt present", "v7.3 (unchecked)"):
                try:
                    row["n_samples"], row["n_chan"], row["srate"] = eeg_header(p)
                except Exception as e:
                    row["eeg_status"] = f"header error: {type(e).__name__}"
            rows.append(row)
    eeg = pd.DataFrame(rows)
    n_files = len(eeg)
    eeg["usable"] = eeg["eeg_status"].isin(["embedded", "fdt present", "v7.3 (unchecked)"])
    multi = eeg[eeg.duplicated("eeg_key", keep=False)]
    assert not len(multi), f"several resting files for one EEG ID:\n{multi}"

    # duplicate recordings of a paired participant under another ID
    sig = ["n_samples", "n_chan", "srate"]
    ref = eeg[eeg["paired"]][sig + ["eeg_id"]].rename(columns={"eeg_id": "paired_eeg_id"})
    cand = eeg[~eeg["paired"] & eeg["usable"]].merge(ref, on=sig)
    eeg["dup_of_paired"] = eeg["eeg_id"].map(dict(zip(cand["eeg_id"], cand["paired_eeg_id"])))
    eeg["dup_of_paired"] = eeg["dup_of_paired"].fillna("")
    missing = paired_ids - set(eeg.loc[eeg["usable"], "mri_id"])
    assert not missing, f"paired participants without a usable pool EEG: {missing}"

    eeg["exclusion_reason"] = ""
    eeg.loc[~eeg["usable"], "exclusion_reason"] = "no signal (" + eeg["eeg_status"] + ")"
    eeg.loc[eeg["usable"] & (eeg["dup_of_paired"] != ""), "exclusion_reason"] = \
        "same length/channels/srate as a paired recording"
    eeg_pool = eeg[eeg["exclusion_reason"] == ""].drop(columns="usable")
    eeg_pool = eeg_pool.rename(columns={"eeg_id": "pool_id"})
    # participant ID used everywhere downstream: MRI_ID for paired, EEG ID otherwise
    eeg_pool["participant"] = eeg_pool["mri_id"].where(eeg_pool["paired"], eeg_pool["pool_id"])
    mri["participant"] = mri["mri_id"]

    mri.to_csv(os.path.join(work, "mri_pool.csv"), index=False)
    eeg_pool.to_csv(os.path.join(work, "eeg_pool.csv"), index=False)
    eeg.to_csv(os.path.join(work, "eeg_pool_inventory.csv"), index=False)

    counts = pd.concat([
        mri.assign(modality="MRI").groupby(["modality", "site", "dx"]).size(),
        eeg_pool.assign(modality="EEG").groupby(["modality", "site", "dx"]).size()]) \
        .rename("n").reset_index()
    counts.to_csv(os.path.join(res, "pool_counts.csv"), index=False)
    summ = {"classes": classes,
            "mri_pool": {"n": len(mri), "by_dx": mri["dx"].value_counts().to_dict(),
                         "n_paired": int(mri["paired"].sum())},
            "eeg_files_in_AD_HC_folders": n_files,
            "eeg_excluded": eeg["exclusion_reason"].value_counts().drop("", errors="ignore")
                                                    .to_dict(),
            "eeg_pool": {"n": len(eeg_pool), "by_dx": eeg_pool["dx"].value_counts().to_dict(),
                         "n_paired": int(eeg_pool["paired"].sum())}}
    with open(os.path.join(res, "pool_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(counts.pivot_table(index=["modality", "site"], columns="dx", values="n",
                             fill_value=0).to_string())
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
