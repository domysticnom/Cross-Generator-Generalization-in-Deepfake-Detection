import argparse
import csv
import glob
import os
import subprocess

# must match data/preprocess.py exactly: raw/<method>/<clip>.mp4
METHODS = ["real", "DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]
EXPECTED_PER_METHOD = 1000

CSV_COLUMNS = ["method", "expected", "found", "corrupt", "status"]


def list_method_videos(raw, method):
    # the count/walk path, kept separate from any ffprobe call so --counts-only
    # can bypass the decode check and so this is unit-testable in isolation
    return sorted(glob.glob(os.path.join(raw, method, "*.mp4")))


def ffprobe_ok(path):
    # a file is readable if ffprobe returns 0 AND prints a codec name line
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=codec_name", "-of", "csv=p=0", path]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        # ffprobe not on PATH: cannot verify, treat as unreadable
        return False
    if res.returncode != 0:
        return False
    return res.stdout.strip() != ""


def count_corrupt(videos):
    corrupt = 0
    for vid in videos:
        if not ffprobe_ok(vid):
            corrupt += 1
    return corrupt


def method_status(expected, found, corrupt):
    if corrupt > 0:
        return "CORRUPT"
    if found != expected:
        return "MISMATCH"
    return "OK"


def inventory(raw, expected, counts_only):
    rows = []
    for method in METHODS:
        videos = list_method_videos(raw, method)
        found = len(videos)
        corrupt = 0 if counts_only else count_corrupt(videos)
        status = method_status(expected, found, corrupt)
        rows.append({
            "method": method,
            "expected": expected,
            "found": found,
            "corrupt": corrupt,
            "status": status,
        })
    return rows


def total_row(rows):
    found = sum(r["found"] for r in rows)
    corrupt = sum(r["corrupt"] for r in rows)
    expected = sum(r["expected"] for r in rows)
    status = "OK" if all(r["status"] == "OK" for r in rows) else "FAIL"
    return {
        "method": "TOTAL",
        "expected": expected,
        "found": found,
        "corrupt": corrupt,
        "status": status,
    }


def write_csv(out, rows, total):
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow(total)


def print_summary(rows, total):
    print("{:<15} {:>8} {:>6} {:>8} {}".format(
        "method", "expected", "found", "corrupt", "status"))
    for r in rows:
        print("{:<15} {:>8} {:>6} {:>8} {}".format(
            r["method"], r["expected"], r["found"], r["corrupt"], r["status"]))
    print("{:<15} {:>8} {:>6} {:>8} {}".format(
        total["method"], total["expected"], total["found"],
        total["corrupt"], total["status"]))


def run(raw, out, expected, counts_only):
    rows = inventory(raw, expected, counts_only)
    total = total_row(rows)
    write_csv(out, rows, total)
    print_summary(rows, total)
    print("wrote", out)

    bad = [r for r in rows if r["status"] != "OK"]
    if bad:
        reasons = ", ".join("{}={}".format(r["method"], r["status"]) for r in bad)
        print("FAIL:", reasons)
        return 1
    print("OK: all methods complete and readable")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Inventory + integrity check for the normalized FF++ tree "
                    "(data/raw/<method>/*.mp4). Verifies EXPECTED_PER_METHOD "
                    "clips per method and, unless --counts-only, ffprobe-decodes "
                    "every file. Writes data/manifests/ff_inventory.csv and exits "
                    "non-zero on any count mismatch or corruption.")
    ap.add_argument("--raw", default="data/raw",
                    help="normalized input root (default: data/raw)")
    ap.add_argument("--out", default="data/manifests/ff_inventory.csv",
                    help="inventory CSV output (default: data/manifests/ff_inventory.csv)")
    ap.add_argument("--counts-only", action="store_true",
                    help="skip the ffprobe decode check (count validation only)")
    ap.add_argument("--expected", type=int, default=EXPECTED_PER_METHOD,
                    help="expected clips per method (default: %d)" % EXPECTED_PER_METHOD)
    args = ap.parse_args()

    raise SystemExit(run(args.raw, args.out, args.expected, args.counts_only))


if __name__ == "__main__":
    main()
