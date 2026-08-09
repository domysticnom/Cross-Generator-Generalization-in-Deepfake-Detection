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

CROP_COLUMNS = ["crop_id", "clip_id", "source_id", "method", "label",
                "official_split", "frame_idx", "compression", "path"]


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


def crop_face_simple(fd, img_rgb, margin=0.3, size=256):
    """
    Detects and crops a face using MediaPipe.
    Maintains the same crop convention as FF++ to allow cached matching.
    """
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


def pick_source(target_path, clips, seed):
    """
    Deterministically picks a source clip for the given target, independent
    of processing order. A plain shared/sequential random.choice() call
    would NOT have this property: skipping an already-done pair on resume
    would shift which random numbers get consumed for every pair after it,
    silently changing which source gets paired with each remaining target
    compared to an uninterrupted run. Seeding a fresh, per-target generator
    from (seed, target_id) instead makes each pair's source choice fully
    independent of what's been skipped or already processed.
    """
    target_id = os.path.splitext(os.path.basename(target_path))[0]
    pair_rng = random.Random(f"{seed}-{target_id}")
    candidates = [c for c in clips if c != target_path]
    return pair_rng.choice(candidates)


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

    os.makedirs(args.out, exist_ok=True)

    # Collect all .mp4 videos in the raw directory
    clips = sorted(glob.glob(os.path.join(args.raw, "*.mp4")))
    if len(clips) < 2:
        sys.exit(f"need at least 2 real clips in {args.raw}, found {len(clips)}")

    # Deterministic target selection: a plain random.Random(seed) instance
    # (not the shared random module) so this doesn't interact with anything
    # else that might also call the global random functions.
    target_rng = random.Random(args.seed)
    n_pairs = min(args.pairs, len(clips))
    targets = target_rng.sample(clips, n_pairs)

    # Resume support: load whatever manifest already exists and treat any
    # clip_id already present under method="SimSwap" as done -- a pair is
    # only ever written to the manifest after ALL of its frames finish, so
    # "present in the manifest" reliably means "fully complete", the same
    # invariant train.py's per-epoch checkpointing relies on.
    if os.path.exists(args.manifest):
        manifest_df = pd.read_parquet(args.manifest, engine="fastparquet")
    else:
        manifest_df = pd.DataFrame(columns=CROP_COLUMNS)
    existing_simswap = manifest_df[manifest_df["method"] == "SimSwap"]
    done_clip_ids = set(existing_simswap["clip_id"].unique()) if len(existing_simswap) else set()

    if os.path.exists(args.split_out):
        split_df = pd.read_csv(args.split_out)
    else:
        split_df = pd.DataFrame(columns=["crop_id", "role"])

    if done_clip_ids:
        print(f"resume: {len(done_clip_ids)} pair(s) already complete, will be skipped")

    # Initialize MediaPipe Face Detection and SimSwap Generator
    # Uses the same Tasks-API detector as preprocess.py (data/preprocess.py's
    # make_detector), not the legacy mp.solutions.face_detection API directly.
    # Confirmed this session: this venv's mediapipe version has dropped
    # mp.solutions entirely (AttributeError: module 'mediapipe' has no
    # attribute 'solutions'), and preprocess.py already proved the Tasks API
    # works correctly in this exact environment. Both files live in data/, so
    # this is a plain sibling import, not a package-relative one.
    from preprocess import make_detector
    fd = make_detector(0.5)
    gen = SimSwapGenerator(args.weights_dir, args.simswap_repo)

    n_ok = n_fail = n_skipped = 0
    try:
        for target_path in targets:
            target_id = os.path.splitext(os.path.basename(target_path))[0]
            source_path = pick_source(target_path, clips, args.seed)
            source_id = os.path.splitext(os.path.basename(source_path))[0]
            clip_id = f"{source_id}_to_{target_id}"

            if clip_id in done_clip_ids:
                n_skipped += 1
                continue

            # Get the source face for the swap
            source_frame = middle_frame(source_path)
            if source_frame is None:
                n_fail += 1
                continue

            out_dir = os.path.join(args.out, clip_id)
            os.makedirs(out_dir, exist_ok=True)

            pair_rows = []
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

                pair_rows.append(dict(
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

            if not pair_rows:
                # every frame in this pair failed detection; nothing to
                # checkpoint, don't mark it done so a future run retries it
                continue

            # Checkpoint: persist this pair's rows immediately, so a killed
            # process / disconnected kernel loses at most the ONE pair that
            # was in progress, not the whole run.
            new_df = pd.DataFrame(pair_rows, columns=CROP_COLUMNS)
            manifest_df = pd.concat([manifest_df, new_df], ignore_index=True)
            os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
            manifest_df.to_parquet(args.manifest, engine="fastparquet", index=False)

            new_split_rows = pd.DataFrame({"crop_id": new_df["crop_id"], "role": "test"})
            split_df = pd.concat([split_df, new_split_rows], ignore_index=True)
            os.makedirs(os.path.dirname(args.split_out) or ".", exist_ok=True)
            split_df.to_csv(args.split_out, index=False)

            done_clip_ids.add(clip_id)
            print(f"  checkpoint: {clip_id} done ({len(pair_rows)} crops) -- "
                  f"{len(done_clip_ids)}/{n_pairs} pairs complete")
    finally:
        fd.close()

    total_simswap = (manifest_df["method"] == "SimSwap").sum()
    print(f"\ngenerated {n_ok} new crops this run, {n_fail} frame/detection failures, "
          f"{n_skipped} pair(s) skipped (already done)")
    print(f"manifest: {args.manifest} -- {len(manifest_df)} total rows, "
          f"{total_simswap} SimSwap rows")
    print(f"split: {args.split_out} -- {len(split_df)} rows")


if __name__ == "__main__":
    main()
