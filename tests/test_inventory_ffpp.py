import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "data" / "inventory_ffpp.py"

# import the module under test to reuse its method vocabulary
sys.path.insert(0, str(REPO_ROOT / "data"))
import inventory_ffpp  # noqa: E402

METHODS = inventory_ffpp.METHODS


def build_tree(raw, per_method, short_method=None):
    # create `per_method` placeholder .mp4 files under raw/<method>/, dropping one
    # file for `short_method` to simulate an incomplete corpus
    for method in METHODS:
        d = raw / method
        d.mkdir(parents=True, exist_ok=True)
        n = per_method - 1 if method == short_method else per_method
        for i in range(n):
            (d / "{}_{:03d}.mp4".format(method, i)).write_bytes(b"stub")


def run_cli(raw, out, expected):
    # invoke through subprocess so the exit-code contract is what is tested
    cmd = [sys.executable, str(SCRIPT),
           "--raw", str(raw), "--out", str(out),
           "--counts-only", "--expected", str(expected)]
    return subprocess.run(cmd, capture_output=True, text=True)


def read_csv(out):
    with open(out, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_happy_path_all_ok(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "manifests" / "ff_inventory.csv"
    expected = 3
    build_tree(raw, expected)

    res = run_cli(raw, out, expected)
    assert res.returncode == 0, res.stdout + res.stderr

    rows = read_csv(out)
    method_rows = [r for r in rows if r["method"] != "TOTAL"]
    total = [r for r in rows if r["method"] == "TOTAL"][0]

    assert len(method_rows) == len(METHODS)
    assert all(r["status"] == "OK" for r in method_rows)
    assert int(total["found"]) == expected * len(METHODS)
    assert total["status"] == "OK"


def test_short_count_mismatch(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "manifests" / "ff_inventory.csv"
    expected = 3
    build_tree(raw, expected, short_method="FaceSwap")

    res = run_cli(raw, out, expected)
    assert res.returncode != 0, res.stdout + res.stderr

    rows = read_csv(out)
    by_method = {r["method"]: r for r in rows}
    assert by_method["FaceSwap"]["status"] == "MISMATCH"
    assert int(by_method["FaceSwap"]["found"]) == expected - 1
    # the other methods are still fine
    assert by_method["real"]["status"] == "OK"
