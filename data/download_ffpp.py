import argparse
import glob
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

# must match data/preprocess.py exactly: raw/<method>/<clip>.mp4
METHODS = ["real", "DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]
EXPECTED_PER_METHOD = 1000

# native FF++ source subdir (relative to the dataset root) -> normalized method name.
# FaceShifter and any compression other than c23 are intentionally excluded.
NATIVE_DIRS = {
    "real": "original_sequences/youtube/c23/videos",
    "DeepFakes": "manipulated_sequences/Deepfakes/c23/videos",
    "Face2Face": "manipulated_sequences/Face2Face/c23/videos",
    "FaceSwap": "manipulated_sequences/FaceSwap/c23/videos",
    "NeuralTextures": "manipulated_sequences/NeuralTextures/c23/videos",
}


def resolve_native_dir(root, native_rel):
    # the native path relative to the dataset root
    direct = os.path.join(root, *native_rel.split("/"))
    if os.path.isdir(direct):
        return direct
    # fallback: a Kaggle zip often extracts into one extra nested folder,
    # so look for the native tail anywhere under root
    tail = os.path.join(*native_rel.split("/"))
    for cur, _dirs, _files in os.walk(root):
        cand = os.path.join(cur, tail)
        if os.path.isdir(cand):
            return cand
    return direct  # does not exist; caller reports a count of 0


def list_videos(src_dir):
    if not src_dir or not os.path.isdir(src_dir):
        return []
    return sorted(glob.glob(os.path.join(src_dir, "*.mp4")))


def build_plan(root, raw):
    # one entry per method: (method, source_dir, dest_dir, [source videos])
    plan = []
    for method in METHODS:
        src = resolve_native_dir(root, NATIVE_DIRS[method]) if root else None
        dst = os.path.join(raw, method)
        plan.append((method, src, dst, list_videos(src)))
    return plan


def place_file(src, dst, mode):
    # relocate one video, preserving its basename verbatim (the filename stem is
    # the clip_id / source_id contract consumed by data/preprocess.py). Idempotent:
    # an existing destination is never overwritten.
    if os.path.exists(dst):
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(src, dst)
    elif mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    return "placed"


def kaggle_download(dataset, download_dir):
    if shutil.which("kaggle") is None:
        print("kaggle CLI not found on PATH.")
        print("Install it and set KAGGLE_USERNAME / KAGGLE_KEY, then retry.")
        print("See docs/DATASET_ACCESS.md for the full setup.")
        return False
    os.makedirs(download_dir, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", download_dir, "--unzip"]
    print("running:", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def write_provenance(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if header:
            f.write("# FF++ acquisition provenance\n\n")
            f.write("Append-only log, one JSON record per acquisition run.\n\n")
        f.write("## " + record["timestamp"] + "\n\n")
        f.write("```json\n")
        f.write(json.dumps(record, indent=2))
        f.write("\n```\n\n")


def print_plan(plan, raw, route, source):
    print("route:", route, "  source:", source)
    print("normalized target root:", raw)
    print("per-method source -> destination (found / expected):")
    for method, src, dst, vids in plan:
        shown = src if src else "(source not resolved yet)"
        print("  {:<15} {} -> {}  ({} found / {} expected)".format(
            method, shown, dst, len(vids), EXPECTED_PER_METHOD))


def main():
    ap = argparse.ArgumentParser(
        description="Acquire FaceForensics++ c23 and normalize it into "
                    "data/raw/<method>/*.mp4 (see docs/DATASET_ACCESS.md).")
    ap.add_argument("--raw", default="data/raw",
                    help="normalized output root (default: data/raw)")
    ap.add_argument("--route", choices=["kaggle", "official", "local"], default="kaggle",
                    help="kaggle: pull the c23 mirror via the kaggle CLI; "
                         "official/local: normalize an already-downloaded --source-dir")
    ap.add_argument("--kaggle-dataset", default="xdxd003/ff-c23",
                    help="Kaggle dataset id for route=kaggle (default: xdxd003/ff-c23)")
    ap.add_argument("--source-dir", default="",
                    help="root of an already-downloaded FF++ tree (route=official/local)")
    ap.add_argument("--link-mode", choices=["copy", "symlink", "move"], default="copy",
                    help="how to place videos into data/raw (default: copy)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned mapping and counts, create nothing")
    ap.add_argument("--provenance", default="data/PROVENANCE.md",
                    help="append the run record here (default: data/PROVENANCE.md)")
    args = ap.parse_args()

    # decide where the native FF++ tree lives
    if args.route in ("official", "local"):
        root = args.source_dir
        source = args.source_dir or "(unset)"
    else:  # kaggle
        # for a real run we download into a sibling of --raw; for dry-run we do
        # not touch the network, so the tree may not exist yet
        root = args.source_dir or os.path.join(os.path.dirname(args.raw) or ".", "ffpp_kaggle")
        source = args.kaggle_dataset

    if args.dry_run:
        plan = build_plan(root if os.path.isdir(root) else "", args.raw)
        print_plan(plan, args.raw, args.route, source)
        print("dry-run: nothing downloaded, nothing written under", args.raw)
        return

    # real run: make sure the native tree is present
    if args.route == "kaggle":
        if not kaggle_download(args.kaggle_dataset, root):
            raise SystemExit(1)
    else:
        if not args.source_dir or not os.path.isdir(args.source_dir):
            print("route={} needs --source-dir to point at a downloaded FF++ tree.".format(args.route))
            print("See docs/DATASET_ACCESS.md for the official download script.")
            raise SystemExit(1)

    plan = build_plan(root, args.raw)
    counts = {}
    for method, src, dst, vids in plan:
        placed = 0
        skipped = 0
        for vid in vids:
            # basename preserved verbatim: 033_097.mp4 stays 033_097.mp4
            out = os.path.join(dst, os.path.basename(vid))
            if place_file(vid, out, args.link_mode) == "placed":
                placed += 1
            else:
                skipped += 1
        counts[method] = placed + skipped
        print("{:<15} placed {}, skipped(existing) {}".format(method, placed, skipped))

    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "route": args.route,
        "source": source,
        "link_mode": args.link_mode,
        "raw": args.raw,
        "expected_per_method": EXPECTED_PER_METHOD,
        "counts": counts,
    }
    write_provenance(args.provenance, record)

    total = sum(counts.values())
    summary = ", ".join("{}={}".format(m, counts[m]) for m in METHODS)
    print("done:", summary, " total", total, " provenance:", args.provenance)


if __name__ == "__main__":
    main()
