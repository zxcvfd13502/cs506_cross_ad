"""
Walk the whole BrainLat Synapse project and list every file that belongs to an AD EEG
subject: the file name contains an AD EEG ID (sub-300xx / 300xx), or the file sits in the
AD EEG folder (syn53222482) or in a folder whose name contains such an ID.
Metadata only -- nothing is downloaded.

Usage:  python search_synapse_ad.py --project syn51549340 \
            --out processed/eeg/synapse_ad_search.csv [--all-out listing.csv]
"""
import argparse
import re

import pandas as pd
import synapseclient
import synapseutils

AD_ID = re.compile(r"(?:sub[-_]?)?(300\d\d)(?!\d)", re.I)
AD_EEG_FOLDER = "syn53222482"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="syn51549340")
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-out", help="optional: full listing of every file in the project")
    args = ap.parse_args()

    syn = synapseclient.login(silent=True)
    rows = []
    ad_folders = {AD_EEG_FOLDER}      # folder ids inside the AD EEG tree, filled while walking
    for (dirpath, dirid), dirnames, filenames in synapseutils.walk(syn, args.project):
        in_ad = dirid in ad_folders or AD_ID.search(dirpath.split("/")[-1]) is not None
        if in_ad:
            ad_folders.update(i for _, i in dirnames)
        for name, fid in filenames:
            m = AD_ID.search(name) or AD_ID.search(dirpath)
            rows.append({"path": dirpath, "folder_id": dirid, "name": name, "id": fid,
                         "name_has_ad_id": AD_ID.search(name) is not None,
                         "in_ad_folder": in_ad,
                         "ad_id": f"sub-{m.group(1)}" if m else None})
    allf = pd.DataFrame(rows)
    print(f"Walked {len(allf)} files in {allf['folder_id'].nunique()} folders")
    if args.all_out:
        allf.to_csv(args.all_out, index=False)

    hits = allf[allf["name_has_ad_id"] | allf["in_ad_folder"]].copy()
    sizes = []
    for fid in hits["id"]:
        ent = syn.get(fid, downloadFile=False)
        fh = ent._file_handle or {}
        sizes.append({"id": fid, "size_bytes": fh.get("contentSize"),
                      "content_type": fh.get("contentType"), "modified": ent.modifiedOn})
    hits = hits.merge(pd.DataFrame(sizes, columns=["id", "size_bytes", "content_type",
                                                    "modified"]), on="id", how="left")
    hits["ext"] = hits["name"].str.extract(r"(\.[^.]+)$")[0].str.lower()
    hits["modality"] = hits["path"].str.contains("EEG", case=False).map(
        {True: "EEG path", False: "non-EEG path"})
    hits = hits.sort_values(["ad_id", "path", "name"])
    hits.to_csv(args.out, index=False)
    print(f"{len(hits)} AD-related files -> {args.out}")
    print(hits.groupby(["modality", "ext"]).size().to_string())


if __name__ == "__main__":
    main()
