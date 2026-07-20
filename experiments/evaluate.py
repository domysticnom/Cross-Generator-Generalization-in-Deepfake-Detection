import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.detector import build_model
from data.dataset import CropDataset

FFPP_METHODS = ["DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def resolve_device():
    # cuda on the GPU boxes, mps so Apple-silicon laptops are not stuck on cpu
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict(model, ds, device, batch_size):
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4)
    probs, labels, clips = [], [], []
    model.eval()
    with torch.no_grad():
        for img, label, _, clip in dl:
            p = torch.softmax(model(img.to(device)), 1)[:, 1].cpu().numpy()
            probs.extend(p)
            labels.extend(label.numpy())
            clips.extend(clip)
    return np.array(probs), np.array(labels), np.array(clips)


def to_video(probs, labels, clips):
    df = pd.DataFrame({"prob": probs, "label": labels, "clip": clips})
    g = df.groupby("clip").agg({"prob": "mean", "label": "first"})
    return g["prob"].values, g["label"].values


def metrics(probs, labels):
    pred = (probs >= 0.5).astype(int)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    return dict(auc=round(float(auc), 4), acc=round(float(accuracy_score(labels, pred)), 4),
                precision=round(float(p), 4), recall=round(float(r), 4), f1=round(float(f1), 4))


def eval_set(model, cfg, device, method, split_path, role):
    ds = CropDataset(cfg["manifest"], split_path, role, cfg["input_size"], ["real", method])
    if len(ds) == 0:
        return None
    probs, labels, clips = predict(model, ds, device, cfg["batch_size"])
    if cfg.get("video_level"):
        probs, labels = to_video(probs, labels, clips)
    return metrics(probs, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = resolve_device()
    print("device", device)

    ckpt = torch.load(os.path.join(cfg["checkpoint_dir"], "model.pt"), map_location=device)
    model = build_model(cfg["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    held = cfg["held_out_method"]
    results = []
    for method in FFPP_METHODS:
        role = "test" if method == held else "val"
        m = eval_set(model, cfg, device, method, cfg["split"], role)
        if m is None:
            continue
        m.update(tested_on=method, seen=(method != held))
        results.append(m)

    for extra in cfg.get("extra_test_sets", []):
        if not os.path.exists(extra["split"]):
            print("skip", extra["name"], "(no split file yet)")
            continue
        m = eval_set(model, cfg, device, extra["name"], extra["split"], "test")
        if m is None:
            continue
        m.update(tested_on=extra["name"], seen=False)
        results.append(m)

    out = dict(run_name=cfg["run_name"], backbone=cfg["backbone"], held_out_method=held,
               seed=cfg["seed"], level="video" if cfg.get("video_level") else "frame",
               results=results)

    os.makedirs("experiments/results", exist_ok=True)
    path = f"experiments/results/{cfg['run_name']}.json"
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)
    for r in results:
        print(f"{r['tested_on']:16s} seen={r['seen']} auc={r['auc']} acc={r['acc']}")


if __name__ == "__main__":
    main()
