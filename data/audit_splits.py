"""DATA-05 audit: are the cross-generator splits identity-disjoint?

INTERFACES.md Contract 2 calls identity-disjointness a hard invariant "checked by
an audit script before any training". This is that script; it did not exist while
the eight runs were produced.

    python data/audit_splits.py                 # audit every fold
    python data/audit_splits.py --fold FaceSwap # one fold

Exit code 0 if every fold is identity-disjoint, 1 otherwise.

Why identities are parsed and not read from `source_id`
------------------------------------------------------
An FF++ fake clip is named `<target>_<donor>` (e.g. 000_003): the video is target
000's footage carrying donor 003's face. `source_id` in the manifest is only the
FIRST component, so it records the target and silently drops the donor. For the
swap methods the donor is the identity you actually see, so an audit keyed on
`source_id` alone would under-report leakage. SimSwap uses a third convention
(`<target>_to_<donor>`, with source_id set to the DONOR). This script therefore
derives the identity set from clip_id directly:

    real            "000"           -> {000}
    FF++ fake       "000_003"       -> {000, 003}
    SimSwap         "783_to_632"    -> {783, 632}

Leakage matters because a model that has seen identity 003's face in training as
a REAL video can recognize that face in test as a FAKE for reasons that have
nothing to do with manipulation artifacts. That inflates every unseen number,
which is the project's headline result.
"""

import argparse
import os
import sys

import pandas as pd

FFPP_METHODS = ["DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def identities(clip_id):
    """All identities appearing in a clip, from its id."""
    c = str(clip_id)
    if "_to_" in c:                 # SimSwap: <target>_to_<donor>
        return set(c.split("_to_"))
    parts = c.split("_")
    return set(parts)               # "000" -> {000}; "000_003" -> {000, 003}


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
