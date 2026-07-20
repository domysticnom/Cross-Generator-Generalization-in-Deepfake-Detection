"""
Re-tag an existing crops.parquet's official_split column from the official FF++
train/val/test splits, in place -- NO re-cropping.

Use this to repair a manifest that was built by preprocess.py WITHOUT
--official-splits: in that case every clip defaults to official_split="train", so
make_splits.py produces empty val folds (and test with no real), and training
reports val_acc 0.0000. Fresh preprocess runs pass --official-splits and do not
need this; it exists to fix crops already on disk.

Usage (from the repo root):
    python data/tag_official_splits.py
    python data/make_splits.py --manifest data/manifests/crops.parquet --out data/splits
"""

import argparse
import os
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_official_split  # noqa: E402

BASE = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits"


def ensure_splits(splits_dir):
    # the official FF++ split json files are public id lists (no dataset access needed)
    os.makedirs(splits_dir, exist_ok=True)
    for role in ("train", "val", "test"):
        dest = os.path.join(splits_dir, role + ".json")
        if not os.path.exists(dest):
            print("downloading", role + ".json")
            urllib.request.urlretrieve(f"{BASE}/{role}.json", dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/manifests/crops.parquet")
    ap.add_argument("--splits-dir", default="data/ffpp_splits")
    args = ap.parse_args()

    ensure_splits(args.splits_dir)
    mapping = load_official_split(args.splits_dir)
    if not mapping:
        raise SystemExit(f"no official split json found in {args.splits_dir}")

    m = pd.read_parquet(args.manifest)
    sid = m["source_id"].astype(str)
    # official json ids are 3-digit zero-padded (e.g. "035"); match directly, then
    # fall back to zero-padded in case the manifest stored unpadded ids
    match = sid.isin(mapping).mean()
    if match < 0.5:
        sid = sid.str.zfill(3)
        match = sid.isin(mapping).mean()
    print(f"source_ids matched to official split: {match:.1%}")
    if match < 0.5:
        raise SystemExit("source_ids do not match the official split ids; inspect "
                         "crops.parquet source_id vs the json ids before proceeding.")

    m["official_split"] = sid.map(mapping).fillna("train")
    print("official_split now:", m["official_split"].value_counts().to_dict())
    m.to_parquet(args.manifest, index=False)
    print("re-tagged", args.manifest)


if __name__ == "__main__":
    main()
