"""
Download the BrainLat AD resting-state .set files listed in synapse_ad_search.csv
(from search_synapse_ad.py) that are not already in --dest. Files already on disk are
never overwritten; synapseclient checks each download's MD5.

Usage:  python fetch_brainlat_ad.py --search processed/eeg/synapse_ad_search.csv \
            --dest /projectnb/mlresearch/Runkun/brainlat_eeg_full
"""
import argparse
import os

import pandas as pd
import synapseclient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    hits = pd.read_csv(args.search)
    want = hits[hits["name"].str.endswith(".set")]
    todo = want[~want["name"].map(lambda n: os.path.exists(os.path.join(args.dest, n)))]
    print(f"{len(want)} AD .set files on Synapse; {len(want) - len(todo)} already on disk; "
          f"downloading {len(todo)} ({todo['size_bytes'].sum() / 1e9:.1f} GB)")

    syn = synapseclient.login(silent=True)
    for i, r in enumerate(todo.itertuples(), 1):
        ent = syn.get(r.id, downloadLocation=args.dest, ifcollision="keep.local")
        ok = os.path.getsize(ent.path) == r.size_bytes
        print(f"[{i}/{len(todo)}] {r.name} {'ok' if ok else 'SIZE MISMATCH'}")


if __name__ == "__main__":
    main()
