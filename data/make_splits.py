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
        df.to_csv(os.path.join(args.out, name), index=False)
        print(name, len(df), "rows")


if __name__ == "__main__":
    main()
