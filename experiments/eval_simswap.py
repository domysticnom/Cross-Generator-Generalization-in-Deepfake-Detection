"""Re-score ONLY the SimSwap column of an existing run, in place.

Why this exists instead of just re-running evaluate.py: evaluate.py recomputes
every column and rewrites the whole results JSON. The FF++ columns in some runs
were produced by other teammates on their own machines, and re-running them here
would silently replace their numbers with ours. This script touches exactly one
row -- tested_on == "SimSwap" -- and leaves every other row byte-identical.

The point is to make the SimSwap column comparable across runs. Right now
different rows of that column were scored against different, non-overlapping
SimSwap sets, so reading down the column compares test sets, not models.

    python experiments/eval_simswap.py --config configs/<run>.yaml
    python experiments/eval_simswap.py --config configs/<run>.yaml --limit 256   # smoke test
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.detector import build_model            # noqa: E402
from data.dataset import CropDataset               # noqa: E402
from experiments.evaluate import metrics, to_video, resolve_device  # noqa: E402


def score(model, ds, device, batch_size, video_level, workers):
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers)
    probs, labels, clips = [], [], []
    model.eval()
    with torch.no_grad():
        for i, (img, label, _, clip) in enumerate(dl):
            p = torch.softmax(model(img.to(device)), 1)[:, 1].cpu().numpy()
            probs.extend(p)
            labels.extend(label.numpy())
            clips.extend(clip)
            if i % 20 == 0:
                print(f"    {len(probs)}/{len(ds)}", flush=True)
    probs, labels, clips = np.array(probs), np.array(labels), np.array(clips)
    if video_level:
        probs, labels = to_video(probs, labels, clips)
    return metrics(probs, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0, help="score only N crops (smoke test; does not write)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--results-dir", default="experiments/results")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    extras = [e for e in cfg.get("extra_test_sets", []) if e["name"] == "SimSwap"]
    if not extras:
        sys.exit(f"{args.config} has no SimSwap extra_test_set")
    split_path = extras[0]["split"]
    if not os.path.exists(split_path):
        sys.exit(f"missing split {split_path}")

    # Validate the destination BEFORE inference. Scoring a run takes ~20 minutes on
    # CPU, and the results JSON is not touched until the very end -- without this
    # check a wrong --results-dir costs a full run and then dies on FileNotFoundError.
    out_path = os.path.join(args.results_dir, f"{cfg['run_name']}.json")
    if not args.limit:
        if not os.path.exists(out_path):
            sys.exit(f"no results file at {out_path} -- run experiments/evaluate.py for this "
                     "run first; this script only rewrites an existing SimSwap row")
        try:
            json.load(open(out_path, encoding="utf-8"))
        except Exception as e:
            sys.exit(f"{out_path} is not readable JSON ({e}) -- refusing to overwrite it")

    device = resolve_device()
    ds = CropDataset(cfg["manifest"], split_path, "test", cfg["input_size"], ["real", "SimSwap"])
    if len(ds) == 0:
        sys.exit("0 crops matched -- the split's crop_ids are not in this machine's manifest")

    labels = ds.df["label"].values
    print(f"{cfg['run_name']}: device={device} crops={len(ds)} "
          f"(fake={int((labels == 1).sum())} real={int((labels == 0).sum())}) split={split_path}")

    if args.limit:
        ds = Subset(ds, range(min(args.limit, len(ds))))

    model = build_model(cfg["backbone"], pretrained=False).to(device)
    ckpt = torch.load(os.path.join(cfg["checkpoint_dir"], "model.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])

    m = score(model, ds, device, cfg["batch_size"], cfg.get("video_level"), args.workers)
    print("  SimSwap ->", m)

    if args.limit:
        print("  (--limit set: smoke test only, results NOT written)")
        return

    doc = json.load(open(out_path, encoding="utf-8"))
    m.update(tested_on="SimSwap", seen=False)
    rows = [r for r in doc["results"] if r.get("tested_on") != "SimSwap"]
    rows.append(m)
    # keep the original column order: FF++ methods as they were, SimSwap last
    doc["results"] = rows
    doc["simswap_split"] = split_path          # provenance: which set produced this column
    json.dump(doc, open(out_path, "w", encoding="utf-8"), indent=2)
    print("  rewrote SimSwap row in", out_path)


if __name__ == "__main__":
    main()
