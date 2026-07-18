import argparse
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# expects a normalized layout: raw/<method>/<clip>.mp4
METHODS = ["real", "DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]


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


def crop_face(fd, img, margin=0.3, size=256):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--manifest", default="data/manifests/crops.parquet")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--official-splits", default="")
    args = ap.parse_args()

    official = load_official_split(args.official_splits)
    fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    rows = []
    for method in METHODS:
        vids = glob.glob(os.path.join(args.raw, method, "*.mp4"))
        for vid in vids:
            clip_id = os.path.splitext(os.path.basename(vid))[0]
            source_id = clip_id.split("_")[0]
            label = 0 if method == "real" else 1
            osplit = official.get(source_id, "train")
            out_dir = os.path.join(args.out, method, clip_id)
            os.makedirs(out_dir, exist_ok=True)
            for fidx, frame in sample_frames(vid, args.frames):
                face = crop_face(fd, frame)
                if face is None:
                    continue
                p = os.path.join(out_dir, f"f{fidx:04d}.npy")
                np.save(p, face)
                rows.append(dict(
                    crop_id=f"{method}_{clip_id}_f{fidx:04d}",
                    clip_id=clip_id, source_id=source_id, method=method,
                    label=label, official_split=osplit, frame_idx=fidx,
                    compression="c23", path=p,
                ))
        print(method, "done")

    fd.close()
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(args.manifest, index=False)
    print("wrote", args.manifest, len(df), "crops")


if __name__ == "__main__":
    main()
