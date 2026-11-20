"""
Download the whole BrainLat Synapse project (EEG + MRI) into --dest, keeping the
Synapse folder tree (BrainLat_dataset/EEG data/..., BrainLat_dataset/MRI data/...).
Files already on disk are kept (ifcollision="keep.local"), so the job can be re-run
to resume after an interruption. synapseclient checks each download's MD5.

Usage:  python -m crossad.data.fetch_brainlat_all --dest /projectnb/mlresearch/Runkun/brainlat_synapse
"""
import argparse

import synapseclient
import synapseutils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="syn51549340")
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    syn = synapseclient.login(silent=True)
    files = synapseutils.syncFromSynapse(syn, args.project, path=args.dest,
                                         ifcollision="keep.local", followLink=False)
    print(f"{len(files)} files synced to {args.dest}")


if __name__ == "__main__":
    main()
