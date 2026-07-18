import argparse
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
import yaml

# expects a normalized layout: raw/<method>/<clip>.mp4
METHODS = ["real", "DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]

# the committed data/manifests/crops.parquet schema (docs/INTERFACES.md Contract 1).
# data/dataset.py and data/make_splits.py consume these exact columns in this
# order; changing it breaks Phase 4 and Phase 5, so do not reorder or rename.
CROP_COLUMNS = ["crop_id", "clip_id", "source_id", "method", "label",
                "official_split", "frame_idx", "compression", "path"]

# documented defaults (Phase 2 EDA placeholders; a later EDA pass may revise
# these by editing configs/preprocess.yaml rather than this code)
DEFAULT_FRAMES = 20
DEFAULT_SIZE = 256
DEFAULT_MARGIN = 0.3
DEFAULT_CONFIDENCE = 0.5
DEFAULT_SEED = 1337
DEFAULT_COMPRESSION = "c23"

# the config knobs the run reads; also the keys allowed in configs/preprocess.yaml
CONFIG_KEYS = ["frames", "size", "margin", "confidence", "seed", "compression"]


def default_config():
    return {
        "frames": DEFAULT_FRAMES,
        "size": DEFAULT_SIZE,
        "margin": DEFAULT_MARGIN,
        "confidence": DEFAULT_CONFIDENCE,
        "seed": DEFAULT_SEED,
        "compression": DEFAULT_COMPRESSION,
    }


def load_config(config_path):
    # start from the documented defaults, then overlay any keys the YAML sets
    cfg = default_config()
    if config_path and os.path.isfile(config_path):
        loaded = yaml.safe_load(open(config_path)) or {}
        for k, v in loaded.items():
            if k in cfg:
                cfg[k] = v
            else:
                print("warning: ignoring unknown config key", k)
    return cfg


def write_config_sidecar(out_dir, config):
    # record the effective config plus library versions next to the cache so the
    # crops are self-describing and reproducible
    try:
        import mediapipe as _mp
        mp_ver = getattr(_mp, "__version__", "unknown")
    except Exception:
        mp_ver = "unavailable"
    info = dict(config)
    info["numpy"] = np.__version__
    info["cv2"] = cv2.__version__
    info["mediapipe"] = mp_ver
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "preprocess_config.json"), "w") as f:
        json.dump(info, f, indent=2)


def sample_frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, min(n, total)).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append((int(i), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def make_detector(confidence):
    # import mediapipe lazily so the module and its pure functions load without
    # mediapipe installed and without a GPU (tests inject a stub detector instead)
    import mediapipe as mp
    return mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=confidence)


def crop_face(fd, img, margin=DEFAULT_MARGIN, size=DEFAULT_SIZE):
    # fd is any object exposing process(img) -> result with a .detections attr,
    # so a stub detector can stand in for mediapipe in tests
    h, w = img.shape[:2]
    res = fd.process(img)
    if not res.detections:
        return None
    box = res.detections[0].location_data.relative_bounding_box
    x, y = int(box.xmin * w), int(box.ymin * h)
    bw, bh = int(box.width * w), int(box.height * h)
    mx, my = int(bw * margin), int(bh * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
    face = img[y0:y1, x0:x1]
    if face.size == 0:
        return None
    return cv2.resize(face, (size, size))


def load_official_split(splits_dir):
    # optional FF++ train/val/test json files (lists of source id pairs)
    mapping = {}
    if not splits_dir or not os.path.isdir(splits_dir):
        return mapping
    for role in ["train", "val", "test"]:
        f = os.path.join(splits_dir, role + ".json")
        if os.path.exists(f):
            for pair in json.load(open(f)):
                for sid in pair:
                    mapping[str(sid)] = role
    return mapping


def process_clip(detector, frames, method, clip_id, source_id, official_split,
                 out_dir, size, margin, compression):
    # turn a list of (frame_idx, RGB ndarray) frames plus a detector into manifest
    # rows and cached .npy crops. Returns (rows, faces_detected).
    os.makedirs(out_dir, exist_ok=True)
    label = 0 if method == "real" else 1
    rows = []
    faces_detected = 0
    for fidx, frame in frames:
        face = crop_face(detector, frame, margin, size)
        if face is None:
            continue
        path = os.path.join(out_dir, f"f{fidx:04d}.npy")
        np.save(path, face)
        faces_detected += 1
        rows.append({
            "crop_id": f"{method}_{clip_id}_f{fidx:04d}",
            "clip_id": clip_id,
            "source_id": source_id,
            "method": method,
            "label": label,
            "official_split": official_split,
            "frame_idx": fidx,
            "compression": compression,
            "path": path,
        })
    return rows, faces_detected


def run(raw, out, manifest_path, config, official, detector=None, frame_reader=None):
    # orchestration entry; detector and frame_reader are injectable seams so the
    # whole pipeline is exercisable on synthetic inputs with no corpus/mediapipe/GPU
    built = detector is None
    if built:
        detector = make_detector(config["confidence"])
    if frame_reader is None:
        frame_reader = sample_frames

    rows = []
    for method in METHODS:
        vids = glob.glob(os.path.join(raw, method, "*.mp4"))
        for vid in vids:
            clip_id = os.path.splitext(os.path.basename(vid))[0]
            source_id = clip_id.split("_")[0]
            osplit = official.get(source_id, "train")
            out_dir = os.path.join(out, method, clip_id)
            frames = frame_reader(vid, config["frames"])
            clip_rows, _ = process_clip(
                detector, frames, method, clip_id, source_id, osplit,
                out_dir, config["size"], config["margin"], config["compression"])
            rows.extend(clip_rows)
        print(method, "done")

    if built and hasattr(detector, "close"):
        detector.close()

    # write the manifest with the committed column order; fail loudly on drift
    df = pd.DataFrame(rows, columns=CROP_COLUMNS)
    assert list(df.columns) == CROP_COLUMNS, "crops manifest schema drift"
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    df.to_parquet(manifest_path, index=False)
    print("wrote", manifest_path, len(df), "crops")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Cache FF++ face crops into data/processed and write the "
                    "crops manifest. Every knob is config-driven: defaults, then "
                    "--config YAML, then explicit flags.")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--manifest", default="data/manifests/crops.parquet")
    ap.add_argument("--official-splits", default="")
    ap.add_argument("--config", default="",
                    help="optional YAML overriding the documented defaults")
    ap.add_argument("--frames", type=int, default=None,
                    help="frames sampled per clip (default: %d)" % DEFAULT_FRAMES)
    ap.add_argument("--size", type=int, default=None,
                    help="output crop square size (default: %d)" % DEFAULT_SIZE)
    ap.add_argument("--margin", type=float, default=None,
                    help="crop margin fraction (default: %.2f)" % DEFAULT_MARGIN)
    ap.add_argument("--confidence", type=float, default=None,
                    help="detector min confidence (default: %.2f)" % DEFAULT_CONFIDENCE)
    ap.add_argument("--seed", type=int, default=None,
                    help="sampling seed (default: %d)" % DEFAULT_SEED)
    args = ap.parse_args()

    # precedence: defaults, then YAML from --config, then any flag passed explicitly
    config = load_config(args.config)
    for k in ["frames", "size", "margin", "confidence", "seed"]:
        v = getattr(args, k)
        if v is not None:
            config[k] = v
    np.random.seed(config["seed"])

    official = load_official_split(args.official_splits)
    write_config_sidecar(args.out, config)
    raise SystemExit(run(args.raw, args.out, args.manifest, config, official))


if __name__ == "__main__":
    main()
