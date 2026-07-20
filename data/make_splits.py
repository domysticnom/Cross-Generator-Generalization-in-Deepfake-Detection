import argparse
import os

import pandas as pd

FFPP_METHODS = ["DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def build_fold(manifest, held):
    trained = [m for m in FFPP_METHODS if m != held]
    rows = []
    for _, r in manifest.iterrows():
        method = r["method"]
        osplit = r["official_split"]
        if method == held:
            role = "test"
        elif method == "real":
            role = osplit
        elif method in trained:
            if osplit == "test":
                continue  # trained methods only feed train/val
            role = osplit
        else:
            continue
        rows.append((r["crop_id"], role))
    return pd.DataFrame(rows, columns=["crop_id", "role"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/crops.parquet")
    ap.add_argument("--out", default="data/splits")
    args = ap.parse_args()

    manifest = pd.read_parquet(args.manifest)
    os.makedirs(args.out, exist_ok=True)
    for held in FFPP_METHODS:
        df = build_fold(manifest, held)
        name = f"holdout-{held.lower()}.csv"
        counts = df["role"].value_counts()
        # fail loud: an empty val (or test) fold means the manifest's official_split
        # is all "train" -- preprocess was run without --official-splits. Silently
        # writing it makes training report val_acc 0.0000 and breaks evaluation.
        for role in ("train", "val", "test"):
            if counts.get(role, 0) == 0:
                raise SystemExit(
                    f"{name}: 0 {role} rows (roles: {counts.to_dict()}). The manifest's "
                    "official_split is likely all 'train' -- preprocess ran without "
                    "--official-splits. Re-tag crops.parquet with the official FF++ splits "
                    "(data/ffpp_splits) and regenerate.")
        df.to_csv(os.path.join(args.out, name), index=False)
        print(name, len(df), "rows", counts.to_dict())


if __name__ == "__main__":
    main()
