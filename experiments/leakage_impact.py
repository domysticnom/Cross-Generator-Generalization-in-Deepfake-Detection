"""Measure what the identity leakage is worth in AUC, without retraining.

    python experiments/leakage_impact.py --config configs/<run>.yaml --subset clean
    python experiments/leakage_impact.py --config configs/<run>.yaml --subset full

Scores the held-out method on every test clip (full) or only on clips whose
identities never appear in training (clean); the gap is the leakage. Appends to
experiments/results/identity_leakage.csv and never touches the per-run JSONs.
"""

import argparse
import csv
import os
import sys

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.detector import build_model                              # noqa: E402
from data.dataset import CropDataset                                 # noqa: E402
from data.audit_splits import identities                             # noqa: E402
from experiments.evaluate import metrics, to_video, resolve_device   # noqa: E402

OUT = "experiments/results/identity_leakage.csv"
FIELDS = ["run_name", "backbone", "held_out", "subset", "clips", "crops",
          "n_real", "n_fake", "auc", "acc", "precision", "recall", "f1"]


def build_subset(cfg, held, subset):
    """Returns a CropDataset restricted to the held-out method + real, optionally
    filtered to identity-clean test clips."""
    manifest = pd.read_parquet(cfg["manifest"])
    split = pd.read_csv(cfg["split"])
    j = manifest.merge(split, on="crop_id")

    train_ids = set()
    for c in j[j["role"].isin(["train", "val"])]["clip_id"].unique():
        train_ids |= identities(c)

    test = j[(j["role"] == "test") & (j["method"].isin(["real", held]))]
    if subset == "clean":
        keep = [not (identities(c) & train_ids) for c in test["clip_id"]]
        test = test[keep]

    ds = CropDataset(cfg["manifest"], cfg["split"], "test", cfg["input_size"], ["real", held])
    ds.df = ds.df[ds.df["crop_id"].isin(set(test["crop_id"]))].reset_index(drop=True)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset", choices=["clean", "full"], default="clean")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    held = cfg["held_out_method"]
    device = resolve_device()

    ds = build_subset(cfg, held, args.subset)
    if len(ds) == 0:
        sys.exit(f"0 crops in the {args.subset} subset -- nothing to score")
    n_real = int((ds.df["label"] == 0).sum())
    n_fake = int((ds.df["label"] == 1).sum())
    if n_real == 0 or n_fake == 0:
        sys.exit(f"{args.subset} subset has one class only (real={n_real} fake={n_fake}); AUC undefined")
    print(f"{cfg['run_name']} [{args.subset}]: {len(ds)} crops / {ds.df['clip_id'].nunique()} clips "
          f"(real={n_real} fake={n_fake}) device={device}", flush=True)

    model = build_model(cfg["backbone"], pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg["checkpoint_dir"], "model.pt"),
                                     map_location=device)["model"])
    model.eval()

    probs, labels, clips = [], [], []
    with torch.no_grad():
        for i, (img, label, _, clip) in enumerate(
                DataLoader(ds, batch_size=cfg["batch_size"], num_workers=args.workers)):
            probs.extend(torch.softmax(model(img.to(device)), 1)[:, 1].cpu().numpy())
            labels.extend(label.numpy())
            clips.extend(clip)
            if i % 25 == 0:
                print(f"    {len(probs)}/{len(ds)}", flush=True)

    import numpy as np
    p, l, c = np.array(probs), np.array(labels), np.array(clips)
    if cfg.get("video_level"):
        p, l = to_video(p, l, c)
    m = metrics(p, l)
    print("  ->", m)

    row = dict(run_name=cfg["run_name"], backbone=cfg["backbone"], held_out=held,
               subset=args.subset, clips=int(ds.df["clip_id"].nunique()), crops=len(ds),
               n_real=n_real, n_fake=n_fake, **m)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print("  appended to", OUT)


if __name__ == "__main__":
    main()
