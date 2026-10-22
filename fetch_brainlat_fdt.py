"""
Download the EEGLAB .fdt data files that are missing next to BrainLat .set headers.

Reads manifest.csv (the Synapse manifest from the EEG download), visits each parent
folder, and downloads every file there that is not already on disk (the .fdt files,
plus anything else the zip export left out). Prints what each folder contains, so you
can also see whether more AD recordings exist on Synapse.

Needs:  pip install --user synapseclient   and a token in ~/.synapseConfig
Usage:  python fetch_brainlat_fdt.py --eeg-dir /project/mlresearch/Runkun/EEG \
            --dest /projectnb/mlresearch/Runkun/brainlat_eeg_full [--dry-run]
"""
import argparse
import glob
import os
import shutil

import pandas as pd
import synapseclient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-dir", required=True)
    ap.add_argument("--dest", required=True, help="where to put downloaded files")
    ap.add_argument("--dry-run", action="store_true", help="list only, download nothing")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)

    man_path = glob.glob(os.path.join(args.eeg_dir, "**", "manifest.csv"), recursive=True)[0]
    man = pd.read_csv(man_path)
    parent_col = next(c for c in man.columns if "parent" in c.lower())
    parents = man[parent_col].dropna().astype(str).unique()
    have = {os.path.basename(p) for p in glob.glob(os.path.join(args.eeg_dir, "**", "*"),
                                                  recursive=True)}
    have |= set(os.listdir(args.dest))
    print(f"{len(parents)} Synapse folders in manifest; {len(have)} files already on disk")

    syn = synapseclient.login(silent=True)
    todo, rows = [], []
    for pid in parents:
        for child in syn.getChildren(pid, includeTypes=["file", "folder"]):
            rows.append({"parent": pid, "name": child["name"], "id": child["id"],
                         "type": child["type"].split(".")[-1]})
            if child["type"].endswith("FileEntity") and child["name"] not in have:
                todo.append(child)

    listing = pd.DataFrame(rows)
    listing.to_csv(os.path.join(args.dest, "synapse_folder_listing.csv"), index=False)
    print(f"Listed {len(listing)} entries -> synapse_folder_listing.csv")
    print(f"Missing files to download: {len(todo)} "
          f"({sum(c['name'].endswith('.fdt') for c in todo)} .fdt)")
    for c in todo[:20]:
        print("  ", c["name"])
    if args.dry_run:
        return

    for i, c in enumerate(todo, 1):
        ent = syn.get(c["id"], downloadLocation=args.dest, ifcollision="keep.local")
        print(f"[{i}/{len(todo)}] {os.path.basename(ent.path)}")

    # put the .set headers (and the CSVs) next to the .fdt files so everything loads together
    for s in (glob.glob(os.path.join(args.eeg_dir, "**", "*.set"), recursive=True) +
              glob.glob(os.path.join(args.eeg_dir, "**", "*.csv"), recursive=True)):
        target = os.path.join(args.dest, os.path.basename(s))
        if not os.path.exists(target):
            shutil.copy2(s, target)
    print(f"Done. Point prepare_brainlat_eeg.py at --eeg-dir {args.dest}")


if __name__ == "__main__":
    main()
