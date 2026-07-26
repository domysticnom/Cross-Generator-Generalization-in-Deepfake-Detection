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
# reuse the FF++ detector/crop rather than a second copy: these crops must land in
# the same manifest as the FF++ ones, so the crop convention has to be identical by
# construction, not by restating it. preprocess also already migrated off the
# mediapipe `solutions` API, which was removed in the 2026 builds.
from data.preprocess import make_detector, crop_face


def sample_frames(path, n):
    """Samples exactly n evenly spaced frames from the given video."""
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

def middle_frame(path):
    """Extracts the middle frame of a video to use as the source identity."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2 - 1, 0))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

# Face detection + cropping is imported from data/preprocess.py (make_detector /
# crop_face) so SimSwap crops match the FF++ cache exactly: same detector, same
# margin, same output size. The previous local copy duplicated that logic and used
# the removed `mp.solutions` API.


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

    # Initialize the FF++ face detector (mediapipe Tasks API) and SimSwap generator
    fd = make_detector(0.5)
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

        # embed the source identity ONCE per pair instead of once per frame:
        # the source is constant across the whole clip, and its face detection is
        # the CPU-bound step
        source_latent = gen.prepare_source(source_frame)
        if source_latent is None:
            n_fail += 1
            continue

        clip_id = f"{source_id}_to_{target_id}"
        out_dir = os.path.join(args.out, clip_id)
        os.makedirs(out_dir, exist_ok=True)

         # Iterate over frames in the target video and apply the face swap
        for fidx, frame_bgr in sample_frames(target_path, args.frames):
            swapped_bgr = gen.swap(source_frame, frame_bgr, source_latent=source_latent)
            if swapped_bgr is None:
                n_fail += 1
                continue

            # Convert swapped result to RGB for MediaPipe detection
            swapped_rgb = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2RGB)
            face = crop_face(fd, swapped_rgb)
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

    # The split MUST carry real negatives. evaluate.py scores every method as
    # "real vs that method"; a SimSwap-only split is single-class, so
    # roc_auc_score raises ValueError, evaluate.py catches it, and the SimSwap
    # column lands in the transfer matrix as NaN. Use the real clips from the
    # official FF++ test split, matching how the held-out FF++ methods are scored.
    real_test = combined[(combined["method"] == "real") &
                         (combined["official_split"] == "test")]["crop_id"]
    if real_test.empty:
        print("WARNING: no real crops tagged official_split='test' in the manifest, so "
              "the SimSwap set has no negatives and its AUC will be NaN. Re-run "
              "preprocess with --official-splits, or repair the manifest with "
              "data/tag_official_splits.py, then re-run this script.")

    split_df = pd.DataFrame({
        "crop_id": pd.concat([new_df["crop_id"], real_test], ignore_index=True),
        "role": "test",
    })
    os.makedirs(os.path.dirname(args.split_out), exist_ok=True)
    split_df.to_csv(args.split_out, index=False)
    print(f"wrote split: {args.split_out} {len(split_df)} rows "
          f"({len(new_df)} SimSwap + {len(real_test)} real negatives)")


if __name__ == "__main__":
    main()
