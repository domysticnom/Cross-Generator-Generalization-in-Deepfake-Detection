import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.detector import build_model
from data.dataset import CropDataset

FFPP_METHODS = ["DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device():
    # cuda on the GPU boxes, mps so Apple-silicon laptops are not stuck on cpu
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the checkpoint in checkpoint_dir if one exists "
                         "(default: retrain from scratch, as before)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])

    device = resolve_device()
    print("device", device)
    train_methods = [m for m in FFPP_METHODS if m != cfg["held_out_method"]] + ["real"]

    train_ds = CropDataset(cfg["manifest"], cfg["split"], "train", cfg["input_size"], train_methods)
    val_ds = CropDataset(cfg["manifest"], cfg["split"], "val", cfg["input_size"], train_methods)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], num_workers=4)

    model = build_model(cfg["backbone"], cfg["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()

    amp_dtype = torch.bfloat16 if cfg.get("amp") == "bf16" else torch.float16
    use_amp = cfg.get("amp") is not None and device == "cuda"

    # checkpoint every epoch, not just at the end: a 15-epoch run is 1-2 hours and a
    # crash / sleep / driver reset at epoch 14 would otherwise cost the whole thing.
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    path = os.path.join(cfg["checkpoint_dir"], "model.pt")

    def save(epoch):
        # write-then-replace so a kill mid-write cannot truncate an existing model.pt.
        # "model" stays the top-level key evaluate.py reads; the rest is additive.
        tmp = path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "epoch": epoch, "config": cfg}, tmp)
        os.replace(tmp, path)

    # resume is OPT-IN so the default behaviour is unchanged for everyone else:
    # without --resume this still retrains from scratch exactly as before.
    start_epoch = 0
    if args.resume and os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", -1) + 1
        if start_epoch >= cfg["epochs"]:
            print(f"resume: {path} already has all {cfg['epochs']} epochs, nothing to do")
            return
        print(f"resume: continuing from epoch {start_epoch + 1}/{cfg['epochs']}")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        running = 0.0
        for img, label, _, _ in train_dl:
            img, label = img.to(device), label.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda" if use_amp else "cpu",
                                dtype=amp_dtype, enabled=use_amp):
                loss = loss_fn(model(img), label)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"epoch {epoch + 1}/{cfg['epochs']}  train_loss {running / len(train_dl):.4f}")

        model.eval()
        correct = n = 0
        with torch.no_grad():
            for img, label, _, _ in val_dl:
                pred = model(img.to(device)).argmax(1).cpu()
                correct += (pred == label).sum().item()
                n += len(label)
        print(f"           val_acc {correct / max(n, 1):.4f}")

        save(epoch)

    print("saved", path)


if __name__ == "__main__":
    main()
