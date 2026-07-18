import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.simswap_generator import SimSwapGenerator

 """Samples exactly n evenly spaced frames from the given video."""
def sample_frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Return empty if video is invalid or empty
    if total <= 0:             
        cap.release()
        return []
    
    # Generate index numbers for sampled frames
    idxs = np.linspace(0, total - 1, min(n, total)).astype(int)
    frames = []
    
    # Loop through indices, seek to the frame, and read
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append((int(i), frame))  # keep BGR here; generator wants BGR
    cap.release()
    return frames

"""Extracts the middle frame of a video to use as the source identity."""
def middle_frame(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2 - 1, 0))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

  """
    Detects and crops a face using MediaPipe.
    Maintains the same crop convention as FF++ to allow cached matching.
    """
def crop_face_simple(fd, img_rgb, margin=0.3, size=256):
    h, w = img_rgb.shape[:2]
    res = fd.process(img_rgb)
    if not res.detections:
        return None
    
    # Get bounding box coordinates and rescale to image pixels
    box = res.detections[0].location_data.relative_bounding_box
    x, y = int(box.xmin * w), int(box.ymin * h)
    bw, bh = int(box.width * w), int(box.height * h)
    
    # Add a margin to the bounding box
    mx, my = int(bw * margin), int(bh * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
    face = img_rgb[y0:y1, x0:x1]
    if face.size == 0:
        return None
    return cv2.resize(face, (size, size))   # Resize cropped face to standard size


# Setup argument parser to manage input paths, output configs, and generator settings
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir of real FF++ clips, e.g. data/raw/real")
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--simswap-repo", required=True)
    ap.add_argument("--out", default="data/simswap")
    ap.add_argument("--manifest", default="data/manifests/crops.parquet",
                     help="existing manifest to append SimSwap rows to")
    ap.add_argument("--split-out", default="data/splits/simswap-test.csv")
    ap.add_argument("--pairs", type=int, default=200,
                     help="number of (source identity, target clip) swap pairs to generate")
    ap.add_argument("--frames", type=int, default=20, help="frames sampled per swapped clip")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()


    # Seed random number generators for reproducible clip/frame selection
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # Collect all .mp4 videos in the raw directory
    clips = sorted(glob.glob(os.path.join(args.raw, "*.mp4")))
    if len(clips) < 2:
        sys.exit(f"need at least 2 real clips in {args.raw}, found {len(clips)}")

    # Initialize MediaPipe Face Detection and SimSwap Generator
    import mediapipe as mp
    fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    gen = SimSwapGenerator(args.weights_dir, args.simswap_repo)

    # Pick target clips and generate swap pairs
    n_pairs = min(args.pairs, len(clips))
    targets = random.sample(clips, n_pairs)

    rows = []
    n_ok = n_fail = 0
    for target_path in targets:
        target_id = os.path.splitext(os.path.basename(target_path))[0]

        # Select a different video to act as the source identity
        source_path = random.choice([c for c in clips if c != target_path])
        source_id = os.path.splitext(os.path.basename(source_path))[0]
        
        # Get the source face for the swap
        source_frame = middle_frame(source_path)
        if source_frame is None:
            n_fail += 1
            continue

        clip_id = f"{source_id}_to_{target_id}"
        out_dir = os.path.join(args.out, clip_id)
        os.makedirs(out_dir, exist_ok=True)

         # Iterate over frames in the target video and apply the face swap
        for fidx, frame_bgr in sample_frames(target_path, args.frames):
            swapped_bgr = gen.swap(source_frame, frame_bgr)
            if swapped_bgr is None:
                n_fail += 1
                continue

            # Convert swapped result to RGB for MediaPipe detection    
            swapped_rgb = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2RGB)
            face = crop_face_simple(fd, swapped_rgb)
            if face is None:
                n_fail += 1
                continue

            # Save the cropped face as a NumPy array
            p = os.path.join(out_dir, f"f{fidx:04d}.npy")
            np.save(p, face)
            n_ok += 1

              # Record metadata for the dataset manifest
            rows.append(dict(
                crop_id=f"SimSwap_{clip_id}_f{fidx:04d}",
                clip_id=clip_id,
                source_id=target_id,
                method="SimSwap",
                label=1,
                official_split="test",   # SimSwap is always an unseen/test-only set
                frame_idx=fidx,
                compression="n/a",
                path=p,
            ))

    fd.close()
    print(f"generated {n_ok} crops across {n_pairs} pairs, {n_fail} frame/detection failures")

    if not rows:
        sys.exit("no crops were generated; check SimSwap weights/repo paths and inputs")
   
   # Format the dataset records into a DataFrame
    new_df = pd.DataFrame(rows)
   
   # Append to existing Parquet manifest or create a new one
    if os.path.exists(args.manifest):
        existing = pd.read_parquet(args.manifest)
        existing = existing[existing["method"] != "SimSwap"]  # replace any prior SimSwap rows
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    combined.to_parquet(args.manifest, index=False)
    print("updated manifest:", args.manifest, "total rows:", len(combined))

    split_df = pd.DataFrame({"crop_id": new_df["crop_id"], "role": "test"})
    os.makedirs(os.path.dirname(args.split_out), exist_ok=True)
    split_df.to_csv(args.split_out, index=False)
    print("wrote split:", args.split_out, len(split_df), "rows")


if __name__ == "__main__":
    main()
