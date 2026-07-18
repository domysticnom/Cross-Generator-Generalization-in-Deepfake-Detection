import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# import the module under test the same way tests/test_inventory_ffpp.py does:
# put the data/ dir on sys.path so `import preprocess` finds data/preprocess.py
sys.path.insert(0, str(REPO_ROOT / "data"))
import preprocess  # noqa: E402


# ---------------------------------------------------------------------------
# injectable fakes: no mediapipe, no ffmpeg, no GPU, no real corpus
# ---------------------------------------------------------------------------

def fake_frame_reader(path, n):
    # deterministic synthetic frames, ignores the actual file contents. A clip
    # whose filename contains "miss" yields all-zero frames so the detector can
    # be made to miss every frame of that clip.
    name = os.path.basename(path)
    frames = []
    for i in range(n):
        if "miss" in name:
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
        else:
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        frames.append((i, frame))
    return frames


class _Box:
    def __init__(self, xmin, ymin, width, height):
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.height = height


class _LocData:
    def __init__(self, box):
        self.relative_bounding_box = box


class _Detection:
    def __init__(self, box):
        self.location_data = _LocData(box)


class _Result:
    def __init__(self, detections):
        self.detections = detections


class FakeDetector:
    # stands in for mediapipe FaceDetection: exposes process(img) -> result with a
    # .detections attr, matching the attribute path crop_face reads
    # (detection.location_data.relative_bounding_box). Returns one central box for
    # a normal frame and no detection for an all-zero frame (a forced miss).
    def process(self, img):
        if int(img.max()) == 0:
            return _Result([])
        return _Result([_Detection(_Box(0.25, 0.25, 0.5, 0.5))])

    def close(self):
        pass


def build_raw(raw, clips_by_method):
    # create stub .mp4 files under raw/<method>/; contents ignored by the fake
    # frame reader, so empty stubs are fine
    for method, clips in clips_by_method.items():
        d = raw / method
        d.mkdir(parents=True, exist_ok=True)
        for clip_id in clips:
            (d / (clip_id + ".mp4")).write_bytes(b"stub")


def small_config(frames=4, size=96):
    # start from the real documented defaults, then shrink for a fast test
    cfg = preprocess.load_config(None)
    cfg["frames"] = frames
    cfg["size"] = size
    return cfg


def do_run(raw, out, manifest, detect_log, clips_by_method, resume=False, size=96):
    build_raw(raw, clips_by_method)
    return preprocess.run(
        str(raw), str(out), str(manifest), small_config(size=size), {},
        detector=FakeDetector(), frame_reader=fake_frame_reader,
        detect_log=str(detect_log), resume=resume)


def read_log(detect_log):
    with open(detect_log, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    return header, rows


# ---------------------------------------------------------------------------
# Test 1: schema and crops
# ---------------------------------------------------------------------------

def test_schema_and_crops(tmp_path):
    import pandas as pd

    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    manifest = tmp_path / "manifests" / "crops.parquet"
    detect_log = tmp_path / "manifests" / "detection_log.csv"
    clips = {"real": ["000_003", "001_004"], "DeepFakes": ["000_003"]}

    rc = do_run(raw, out, manifest, detect_log, clips, size=96)
    assert rc == 0

    df = pd.read_parquet(manifest)
    # columns exactly the committed schema, in order
    assert list(df.columns) == preprocess.CROP_COLUMNS

    # real is label 0, every manipulation is label 1
    for _, row in df.iterrows():
        expected = 0 if row["method"] == "real" else 1
        assert int(row["label"]) == expected
        # source_id is the identity prefix of the clip_id
        assert row["source_id"] == row["clip_id"].split("_")[0]

    # at least one referenced crop is on disk and is an (size, size, 3) .npy
    path = df.iloc[0]["path"]
    assert os.path.exists(path)
    arr = np.load(path)
    assert arr.shape == (96, 96, 3)


# ---------------------------------------------------------------------------
# Test 2: detection log math and a forced all-miss clip
# ---------------------------------------------------------------------------

def test_detection_log(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    manifest = tmp_path / "manifests" / "crops.parquet"
    detect_log = tmp_path / "manifests" / "detection_log.csv"
    # 000_miss forces the detector to miss every frame (all-zero synthetic frames)
    clips = {"real": ["000_001", "000_miss"], "DeepFakes": ["000_002"]}

    do_run(raw, out, manifest, detect_log, clips)

    header, rows = read_log(detect_log)
    assert header == preprocess.DETECT_LOG_COLUMNS

    by_clip = {r["clip_id"]: r for r in rows}
    # per-row detection_rate equals faces_detected over frames_sampled
    for r in rows:
        fd = int(r["faces_detected"])
        fs = int(r["frames_sampled"])
        assert float(r["detection_rate"]) == preprocess.detection_rate(fd, fs)

    # the forced-miss clip detects nothing and records rate 0.0
    assert int(by_clip["000_miss"]["faces_detected"]) == 0
    assert float(by_clip["000_miss"]["detection_rate"]) == 0.0

    # summarize_detection returns separate real (label 0) and fake (label 1) means.
    # real clips: 000_001 rate 1.0 and 000_miss rate 0.0 -> mean 0.5; fake 000_002
    # rate 1.0 -> mean 1.0.
    summary = preprocess.summarize_detection(rows)
    assert summary["real"]["mean_rate"] == 0.5
    assert summary["fake"]["mean_rate"] == 1.0


# ---------------------------------------------------------------------------
# Test 3: resume adds no duplicate rows
# ---------------------------------------------------------------------------

def test_resume_no_duplication(tmp_path):
    import pandas as pd

    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    manifest = tmp_path / "manifests" / "crops.parquet"
    detect_log = tmp_path / "manifests" / "detection_log.csv"
    clips = {"real": ["000_001", "001_002"], "DeepFakes": ["002_003"]}

    do_run(raw, out, manifest, detect_log, clips)
    # second run over the same inputs with resume on: skips everything, merges
    build_raw(raw, clips)
    preprocess.run(
        str(raw), str(out), str(manifest), small_config(), {},
        detector=FakeDetector(), frame_reader=fake_frame_reader,
        detect_log=str(detect_log), resume=True)

    df = pd.read_parquet(manifest)
    assert not df["crop_id"].duplicated().any()

    header, rows = read_log(detect_log)
    keys = [(r["method"], r["clip_id"]) for r in rows]
    assert len(keys) == len(set(keys))

    # the detection log doubles as the resume ledger, keyed on (method, clip_id)
    done = preprocess.load_done_clips(str(detect_log))
    assert done == {("real", "000_001"), ("real", "001_002"),
                    ("DeepFakes", "002_003")}


# ---------------------------------------------------------------------------
# Test 4: copy-and-consume portability through CropDataset
# ---------------------------------------------------------------------------

def test_copy_and_consume_portability(tmp_path):
    pytest.importorskip("torch")
    import pandas as pd
    import dataset  # noqa: F401  (data/ already on sys.path)

    raw = tmp_path / "raw"
    out = tmp_path / "processed"          # absolute tmp path, so stored crop paths resolve
    manifest = tmp_path / "manifests" / "crops.parquet"
    detect_log = tmp_path / "manifests" / "detection_log.csv"
    clips = {"real": ["000_001"], "DeepFakes": ["000_002"]}

    do_run(raw, out, manifest, detect_log, clips, size=96)

    # a tiny split CSV: every produced crop_id assigned role train
    df = pd.read_parquet(manifest)
    split_csv = tmp_path / "splits" / "all-train.csv"
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    split = pd.DataFrame({"crop_id": df["crop_id"], "role": "train"})
    split.to_csv(split_csv, index=False)

    ds = dataset.CropDataset(str(manifest), str(split_csv), role="train", input_size=96)
    assert len(ds) > 0

    img, label, method, clip_id = ds[0]
    assert tuple(img.shape) == (3, 96, 96)
    assert isinstance(label, int)


# ---------------------------------------------------------------------------
# Test 5: same clip_id under two methods both cache under --resume
#   FF++ reuses source-pair filenames across methods (000_003.mp4 lives under
#   both DeepFakes and Face2Face with clip_id 000_003). A clip_id-only resume key
#   would skip the second method and under-cache the fakes, so the ledger key must
#   be (method, clip_id).
# ---------------------------------------------------------------------------

def test_resume_same_clip_id_across_methods(tmp_path):
    import pandas as pd

    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    manifest = tmp_path / "manifests" / "crops.parquet"
    detect_log = tmp_path / "manifests" / "detection_log.csv"

    # first run: only DeepFakes/000_003 exists
    do_run(raw, out, manifest, detect_log, {"DeepFakes": ["000_003"]})

    # now Face2Face/000_003 appears (same clip_id, different method); resume must
    # process it rather than skip it as already done
    build_raw(raw, {"Face2Face": ["000_003"]})
    preprocess.run(
        str(raw), str(out), str(manifest), small_config(), {},
        detector=FakeDetector(), frame_reader=fake_frame_reader,
        detect_log=str(detect_log), resume=True)

    df = pd.read_parquet(manifest)
    methods_for_clip = set(df[df["clip_id"] == "000_003"]["method"])
    assert methods_for_clip == {"DeepFakes", "Face2Face"}

    # both methods' crops are on disk
    for crop_id in ["DeepFakes_000_003_f0000", "Face2Face_000_003_f0000"]:
        assert crop_id in set(df["crop_id"])

    # the detection log keeps a row per (method, clip_id), not one collapsed row
    _, rows = read_log(detect_log)
    keys = {(r["method"], r["clip_id"]) for r in rows}
    assert ("DeepFakes", "000_003") in keys
    assert ("Face2Face", "000_003") in keys
