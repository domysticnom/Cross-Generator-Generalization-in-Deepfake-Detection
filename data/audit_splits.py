"""Check the cross-generator splits are identity-disjoint (DATA-05).

    python data/audit_splits.py                 # every fold
    python data/audit_splits.py --fold FaceSwap # one fold

Exit 0 if every fold is clean, 1 if any fold leaks.
"""

import argparse
import os
import sys

import pandas as pd

FFPP_METHODS = ["DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]


# Parse the ids out of clip_id rather than using manifest source_id: an FF++ fake
# is <target>_<donor> (000_003) and source_id keeps only the target, so it drops the
# donor -- who is the face you actually see in a swap. Auditing on source_id alone
# under-reports leakage. SimSwap is <target>_to_<donor> with source_id = the donor.
def identities(clip_id):
    c = str(clip_id)
    if "_to_" in c:
        return set(c.split("_to_"))
    return set(c.split("_"))        # "000" -> {000}; "000_003" -> {000, 003}


def audit_fold(manifest, split_path):
    split = pd.read_csv(split_path)
    j = manifest.merge(split, on="crop_id", how="inner")
    if j.empty:
        return None

    out = {}
    for label, roles in (("train/val", ["train", "val"]), ("test", ["test"])):
        sub = j[j["role"].isin(roles)]
        ids = set()
        for c in sub["clip_id"].unique():
            ids |= identities(c)
        out[label] = ids
        out[label + "_naive"] = set(sub["source_id"].astype(str).unique())
        out[label + "_clips"] = sub["clip_id"].nunique()

    tr, te = out["train/val"], out["test"]
    out["overlap"] = tr & te
    out["overlap_naive"] = out["train/val_naive"] & out["test_naive"]
    # test identities that are genuinely unseen -- the only rows a clean
    # cross-generator number could be computed on without retraining
    out["clean_test_ids"] = te - tr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/crops.parquet")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--fold", default=None, help="audit one held-out method only")
    args = ap.parse_args()

    manifest = pd.read_parquet(args.manifest)
    folds = [args.fold] if args.fold else FFPP_METHODS

    failed = []
    print(f"{'fold':<17}{'train ids':>10}{'test ids':>10}{'OVERLAP':>9}"
          f"{'clean test':>12}{'naive overlap':>15}")
    for held in folds:
        p = os.path.join(args.splits, f"holdout-{held.lower()}.csv")
        if not os.path.exists(p):
            print(f"{held:<17} (no split file at {p})")
            continue
        r = audit_fold(manifest, p)
        if r is None:
            print(f"{held:<17} (split matched 0 crops in this manifest)")
            continue
        n_over = len(r["overlap"])
        if n_over:
            failed.append(held)
        print(f"{held:<17}{len(r['train/val']):>10}{len(r['test']):>10}{n_over:>9}"
              f"{len(r['clean_test_ids']):>12}{len(r['overlap_naive']):>15}")

    print()
    if failed:
        print(f"FAIL: {len(failed)} fold(s) are NOT identity-disjoint: {failed}")
        print("Every unseen-generator number from these folds is inflated by identity")
        print("leakage. Quantify with experiments/leakage_impact.py before reporting.")
        return 1
    print("PASS: every fold is identity-disjoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
