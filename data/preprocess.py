import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# expects a normalized layout: raw/<method>/<clip>.mp4
METHODS = ["real", "DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]

# the committed data/manifests/crops.parquet schema (docs/INTERFACES.md Contract 1).
# data/dataset.py and data/make_splits.py consume these exact columns in this
# order; changing it breaks Phase 4 and Phase 5, so do not reorder or rename.
CROP_COLUMNS = ["crop_id", "clip_id", "source_id", "method", "label",
                "official_split", "frame_idx", "compression", "path"]

# the per-clip face-detection success-rate log schema (data/manifests/detection_log.csv).
# this small CSV evidences ROADMAP criterion 3 (comparable real-vs-fake detection
# failure rates) and doubles as the resume ledger: any clip_id already here was done.
DETECT_LOG_COLUMNS = ["clip_id", "method", "label", "frames_sampled",
                      "faces_detected", "detection_rate"]

# a clip counts as a detection "failure" for the audit summary when fewer than this
# fraction of its sampled frames yielded a face
LOW_RATE = 0.5

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


# mediapipe dropped the legacy mp.solutions API in the 2026 builds, so we use the
# Tasks face detector. It needs a small model file, downloaded once and cached.
BLAZE_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")


def ensure_blaze_model():
    cache = os.path.join(os.path.expanduser("~"), ".cache", "ffpp_preprocess")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "blaze_face_short_range.tflite")
    if not os.path.exists(path):
        import urllib.request
        print("downloading face detector model to", path)
        urllib.request.urlretrieve(BLAZE_URL, path)
    return path


class _Box:
    # the fields crop_face reads off relative_bounding_box
    def __init__(self, xmin, ymin, width, height):
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.height = height


class _Detection:
    def __init__(self, box):
        self.location_data = type("loc", (), {"relative_bounding_box": box})


class _Result:
    def __init__(self, detections):
        self.detections = detections


class TasksDetector:
    # wraps the mediapipe Tasks FaceDetector to expose the same
    # .process(rgb) -> result.detections[i].location_data.relative_bounding_box
    # interface the rest of the pipeline (and the stub detector in tests) uses
    def __init__(self, detector):
        self.detector = detector

    def process(self, rgb):
        import mediapipe as mp
        h, w = rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.detector.detect(image)
        dets = []
        for d in res.detections:
            bb = d.bounding_box  # pixel coords: origin_x, origin_y, width, height
            dets.append(_Detection(_Box(bb.origin_x / w, bb.origin_y / h,
                                        bb.width / w, bb.height / h)))
        return _Result(dets)

    def close(self):
        self.detector.close()


def make_detector(confidence):
    # import mediapipe lazily so the module and its pure functions load without
    # mediapipe installed and without a GPU (tests inject a stub detector instead)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_blaze_model()),
        min_detection_confidence=confidence)
    return TasksDetector(vision.FaceDetector.create_from_options(opts))


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


def detection_rate(faces_detected, frames_sampled):
    # faces per sampled frame, with a zero-frames guard, rounded for the log
    if frames_sampled > 0:
        return round(faces_detected / frames_sampled, 4)
    return 0.0


def summarize_detection(rows, low=LOW_RATE):
    # pure aggregate over detection-log rows (no file I/O) so it is unit-testable.
    # coerces label/detection_rate since carried-over rows read from CSV are strings.
    def group_stats(group):
        n = len(group)
        rates = [float(r["detection_rate"]) for r in group]
        mean = round(sum(rates) / n, 4) if n else 0.0
        below = sum(1 for x in rates if x < low)
        return {"clips": n, "mean_rate": mean, "below_low": below}

    real = [r for r in rows if int(r["label"]) == 0]
    fake = [r for r in rows if int(r["label"]) == 1]

    per_method = {}
    for r in rows:
        per_method.setdefault(r["method"], []).append(float(r["detection_rate"]))
    per_method_mean = {m: round(sum(v) / len(v), 4) for m, v in per_method.items()}

    return {
        "low": low,
        "real": group_stats(real),
        "fake": group_stats(fake),
        "per_method": per_method_mean,
    }


def print_detection_summary(summary):
    real, fake = summary["real"], summary["fake"]
    low = summary["low"]
    print("detection success-rate summary (real vs fake, threshold %.2f):" % low)
    # real and fake means side by side for an at-a-glance comparability check
    print("  real (label 0): clips={:<6} mean_rate={:<8} below_low={}".format(
        real["clips"], real["mean_rate"], real["below_low"]))
    print("  fake (label 1): clips={:<6} mean_rate={:<8} below_low={}".format(
        fake["clips"], fake["mean_rate"], fake["below_low"]))
    print("  per-method mean detection_rate:")
    for m, mean in summary["per_method"].items():
        print("    {:<15} {}".format(m, mean))


def load_done_clips(detect_log_path):
    # the detection log doubles as the resume ledger: read the (method, clip_id)
    # pairs already recorded so a resumed run skips them. Missing file means nothing
    # is done yet. The key is method-scoped because FF++ reuses source-pair
    # filenames across methods (000_003.mp4 exists under both DeepFakes and
    # Face2Face with clip_id 000_003), so a clip_id-only key would wrongly skip the
    # second method's clip and under-cache the fakes.
    done = set()
    if not detect_log_path or not os.path.isfile(detect_log_path):
        return done
    with open(detect_log_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["method"], row["clip_id"]))
    return done


def load_detection_rows(detect_log_path):
    # read existing detection rows to carry over when resuming; empty if absent
    rows = []
    if not detect_log_path or not os.path.isfile(detect_log_path):
        return rows
    with open(detect_log_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def load_manifest_rows(manifest_path):
    # read existing crops manifest rows to carry over when resuming; empty if absent
    if not manifest_path or not os.path.isfile(manifest_path):
        return []
    df = pd.read_parquet(manifest_path)
    return df.to_dict("records")


def dedupe_rows(rows, key):
    # key is a single column name, or a tuple of column names for a composite key.
    # keep the first row seen per key so a resumed clip/crop is never duplicated
    seen = set()
    out = []
    for r in rows:
        if isinstance(key, tuple):
            k = tuple(r[c] for c in key)
        else:
            k = r[key]
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def write_detection_log(detect_log_path, rows):
    parent = os.path.dirname(detect_log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(detect_log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DETECT_LOG_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in DETECT_LOG_COLUMNS})


def run(raw, out, manifest_path, config, official, detector=None, frame_reader=None,
        detect_log="data/manifests/detection_log.csv", resume=False):
    # orchestration entry; detector and frame_reader are injectable seams so the
    # whole pipeline is exercisable on synthetic inputs with no corpus/mediapipe/GPU
    built = detector is None
    if built:
        detector = make_detector(config["confidence"])
    if frame_reader is None:
        frame_reader = sample_frames

    # resume keys on (method, clip_id) already present in detect_log; carry over the
    # prior detection rows and manifest rows so the merged outputs keep earlier work
    done = load_done_clips(detect_log) if resume else set()
    carry_detect = load_detection_rows(detect_log) if resume else []
    carry_manifest = load_manifest_rows(manifest_path) if resume else []

    rows = []
    detect_rows = []
    skipped = 0
    for method in tqdm(METHODS, desc="methods"):
        vids = glob.glob(os.path.join(raw, method, "*.mp4"))
        for vid in tqdm(vids, desc=method, leave=False):
            clip_id = os.path.splitext(os.path.basename(vid))[0]
            if (method, clip_id) in done:
                # already recorded in a prior run: do not re-read/detect/rewrite.
                # keyed on (method, clip_id) so the same clip_id under another
                # method is still processed rather than skipped
                skipped += 1
                continue
            source_id = clip_id.split("_")[0]
            osplit = official.get(source_id, "train")
            out_dir = os.path.join(out, method, clip_id)
            frames = frame_reader(vid, config["frames"])
            clip_rows, faces_detected = process_clip(
                detector, frames, method, clip_id, source_id, osplit,
                out_dir, config["size"], config["margin"], config["compression"])
            rows.extend(clip_rows)
            frames_sampled = len(frames)
            label = 0 if method == "real" else 1
            detect_rows.append({
                "clip_id": clip_id,
                "method": method,
                "label": label,
                "frames_sampled": frames_sampled,
                "faces_detected": faces_detected,
                "detection_rate": detection_rate(faces_detected, frames_sampled),
            })
        print(method, "done")

    if built and hasattr(detector, "close"):
        detector.close()

    if resume and skipped:
        print("resume: skipped", skipped, "clips already in", detect_log)

    # merge carried-over and fresh rows; dedupe so no clip_id/crop_id is duplicated.
    # fresh rows come first so a reprocessed clip overrides its carried-over copy.
    all_manifest = dedupe_rows(rows + carry_manifest, "crop_id")
    all_detect = dedupe_rows(detect_rows + carry_detect, ("method", "clip_id"))

    # write the manifest with the committed column order; fail loudly on drift
    df = pd.DataFrame(all_manifest, columns=CROP_COLUMNS)
    assert list(df.columns) == CROP_COLUMNS, "crops manifest schema drift"
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    df.to_parquet(manifest_path, index=False)
    print("wrote", manifest_path, len(df), "crops")

    write_detection_log(detect_log, all_detect)
    print("wrote", detect_log, len(all_detect), "clips")

    print_detection_summary(summarize_detection(all_detect))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Cache FF++ face crops into data/processed and write the "
                    "crops manifest. Every knob is config-driven: defaults, then "
                    "--config YAML, then explicit flags.")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--manifest", default="data/manifests/crops.parquet")
    ap.add_argument("--detect-log", default="data/manifests/detection_log.csv",
                    help="per-clip face-detection success-rate log "
                         "(default: data/manifests/detection_log.csv)")
    ap.add_argument("--resume", action="store_true",
                    help="skip clips already recorded in the existing detect-log "
                         "and merge rather than overwrite prior work")
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
    raise SystemExit(run(args.raw, args.out, args.manifest, config, official,
                         detect_log=args.detect_log, resume=args.resume))


if __name__ == "__main__":
    main()
