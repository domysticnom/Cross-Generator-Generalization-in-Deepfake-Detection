# Interfaces and Contracts

**Status: proposal for the team to react to at kickoff.** Nothing here is locked.
The point is to agree on the *seams* between our pieces so we can build in
parallel without stepping on each other. Each person still implements the module
they own; this document only fixes the inputs and outputs where our modules meet.

If a field or convention below does not fit your part, flag it before you start
coding, not after, so we only agree once.

## Why this exists

All eight training runs depend on the same chain: raw data, then cached face
crops, then frozen splits, then training, then evaluation. If everyone agrees on
the data manifest, the config keys, and the results format up front, then the
eight runs drop straight into one transfer matrix at the end. If we skip this
step, the last week becomes reconciling five slightly different file formats.

## Critical path (what blocks what)

```
Phase 1 raw data       (shared download, one copy)
   -> Phase 2 EDA              reads raw + manifest
   -> Phase 3 crop cache       produces the crops + manifest
   -> Phase 4 splits           reads manifest, writes the folds
   -> Phase 5 training         reads crops + a split + a config
   -> Phase 6 evaluation       reads results files, builds the matrix
```

The SimSwap synthesis runs alongside phases 3 to 5 and produces one extra test
set. It does not block the eight runs. Who takes which piece is for the team to
decide; this document only fixes where the pieces connect.

## Directory conventions

```
data/
  raw/                      # FF++ c23 videos as downloaded (git-ignored)
  processed/                # cached face crops, .npy or .pt (git-ignored)
  manifests/crops.parquet   # index of every cached crop (committed, small)
  splits/holdout-<method>.csv   # one file per leave-one-out fold (committed)
  simswap/                  # self-generated SimSwap fakes (git-ignored)
models/                     # detector + generator code
experiments/
  train.py  evaluate.py     # shared entry points
  results/<run_name>.json   # one results file per run (committed, small)
configs/<model>_<fold>.yaml # one config per run
```

Only small index and result files are committed. Videos, crops, and checkpoints
are git-ignored.

## Contract 1: the crop manifest

`data/manifests/crops.parquet`, one row per cached crop.

| column          | type | meaning |
|-----------------|------|---------|
| `crop_id`       | str  | unique id, e.g. `FaceSwap_033_097_f012` |
| `clip_id`       | str  | source clip, e.g. `033_097` |
| `source_id`     | str  | identity / source sequence, e.g. `033`. Used to enforce identity-disjoint splits |
| `method`        | str  | one of `real`, `DeepFakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `SimSwap` |
| `label`         | int  | `0` real, `1` fake |
| `official_split`| str  | FF++ split of the source: `train`, `val`, `test` (the official 720 / 140 / 140) |
| `frame_idx`     | int  | frame index sampled from the clip |
| `compression`   | str  | `c23` |
| `path`          | str  | path to the cached crop, relative to repo root |

Rules we rely on:
- `source_id` is the same across a real clip and every manipulation made from it,
  so splitting on `source_id` keeps identities disjoint.
- Crop size is fixed by the model at load time (see config), so the cache can
  store one canonical size and each loader resizes if needed.

## Contract 2: the split manifests

One CSV per leave-one-manipulation-out fold, in `data/splits/`:

```
holdout-deepfakes.csv
holdout-face2face.csv
holdout-faceswap.csv
holdout-neuraltextures.csv
```

Each file: two columns, `crop_id` and `role`, where `role` is `train`, `val`, or
`test`.

Fold rule, using `holdout-faceswap.csv` as the example:
- `train` and `val`: `real` plus `DeepFakes`, `Face2Face`, `NeuralTextures`
- `test`: `real` plus `FaceSwap` only (the held-out method, never seen in training)

Hard invariant, checked by an audit script before any training: no `source_id`
appears in both a `train`/`val` role and the `test` role of the same fold. If the
audit fails, the fold is not valid and training does not start.

## Contract 3: the run config

`configs/<model>_<fold>.yaml`. Every run reads exactly these keys, so `train.py`
never hard-codes anything.

```yaml
run_name: efficientnet_holdout-faceswap   # also the results filename and W&B run
seed: 1337                                 # identical across all eight runs

model:
  backbone: efficientnet_b4                # or: xception
  pretrained: true
  input_size: 224                          # 299 for xception
  normalize: imagenet

data:
  manifest: data/manifests/crops.parquet
  split: data/splits/holdout-faceswap.csv
  held_out_method: FaceSwap                # the unseen method for this fold

train:
  epochs: 15
  batch_size: 64
  lr: 0.0003
  optimizer: adamw
  amp: bf16                                # fp16 on the P100
  checkpoint_dir: checkpoints/efficientnet_holdout-faceswap

eval:
  video_level: true                        # video score = mean of frame scores
  extra_test_sets:                         # never trained on, evaluated as unseen
    - name: SimSwap
      split: data/splits/simswap-test.csv
```

Anything a run needs to differ on lives here. Two runs that share a backbone
differ only by the `split` and `held_out_method` lines.

## Contract 4: the results file

Each run writes `experiments/results/<run_name>.json` and logs the same numbers
to W&B. This is the single format the transfer matrix is built from, so it is the
most important contract to get right.

```json
{
  "run_name": "efficientnet_holdout-faceswap",
  "backbone": "efficientnet_b4",
  "held_out_method": "FaceSwap",
  "seed": 1337,
  "level": "video",
  "results": [
    {"tested_on": "DeepFakes",      "seen": true,  "auc": 0.981, "acc": 0.95, "precision": 0.95, "recall": 0.94, "f1": 0.94},
    {"tested_on": "Face2Face",      "seen": true,  "auc": 0.976, "acc": 0.94, "precision": 0.94, "recall": 0.93, "f1": 0.93},
    {"tested_on": "NeuralTextures", "seen": true,  "auc": 0.969, "acc": 0.93, "precision": 0.93, "recall": 0.92, "f1": 0.92},
    {"tested_on": "FaceSwap",       "seen": false, "auc": 0.612, "acc": 0.60, "precision": 0.61, "recall": 0.58, "f1": 0.59},
    {"tested_on": "SimSwap",        "seen": false, "auc": 0.688, "acc": 0.66, "precision": 0.67, "recall": 0.64, "f1": 0.65}
  ]
}
```

Why each field matters:
- `seen` marks whether the tested method was in this run's training set. The
  seen-vs-unseen table and the ΔAUC column come straight from this flag, so we
  never fold unseen numbers into an average by accident.
- `tested_on` includes the held-out FF++ method and `SimSwap`, so the
  self-generated generator shows up as its own column for every run.
- Report both frame level and video level. Either write two files
  (`<run_name>_frame.json`, `<run_name>_video.json`) or add a top-level `level`
  and emit both. Pick one at kickoff.

## Contract 5: experiment tracking

- W&B project: `cross-generator-deepfake` (one project for the whole team).
- `group`: the fold, e.g. `holdout-faceswap`, so a model's four folds sit together.
- `name`: the config `run_name`.
- `tags`: `[backbone, held_out_method]`.
- If a teammate has no internet on their training box, log to TensorBoard locally
  and still write the JSON results file. The JSON is the source of truth; W&B is
  the shared dashboard.

## Contract 6: branches and reviews

- `main` stays working and is not committed to directly.
- Module work: `feat/<area>`, e.g. `feat/preprocess`, `feat/eda`, `feat/train`,
  `feat/simswap`.
- Training runs: `run/<model>-<fold>`, e.g. `run/efficientnet-holdout-faceswap`.
- Open a pull request into `main`, one teammate reviews, then merge. Commit
  messages say what changed and why in one line.

## Environment contract

- Python 3.11.
- Install PyTorch from the cu128 wheel index, not plain PyPI. The default wheel
  silently fails on RTX 50-series GPUs. See the root `README.md` for the command.
- `ffmpeg` 6.x or 7.x on the system PATH.
- A `check_env.py` smoke script prints the torch version and whether CUDA is
  visible. Run it once on your machine before committing GPU time.

## The eight runs

Two backbones times four held-out methods gives eight cross-generator training
runs. In-distribution baselines (train on all four methods) are optional
reference runs and are not part of the eight. How the runs get divided up is a
team decision, not something this document prescribes.

| Run | Backbone     | Held-out (unseen test) | Trained on (3 methods + real) |
|-----|--------------|------------------------|-------------------------------|
| 1   | EfficientNet | DeepFakes              | F2F, FS, NT, real             |
| 2   | EfficientNet | Face2Face              | DF, FS, NT, real              |
| 3   | EfficientNet | FaceSwap               | DF, F2F, NT, real             |
| 4   | EfficientNet | NeuralTextures         | DF, F2F, FS, real             |
| 5   | XceptionNet  | DeepFakes              | F2F, FS, NT, real             |
| 6   | XceptionNet  | Face2Face              | DF, FS, NT, real              |
| 7   | XceptionNet  | FaceSwap               | DF, F2F, NT, real             |
| 8   | XceptionNet  | NeuralTextures         | DF, F2F, FS, real             |

Every run additionally scores the SimSwap set as a fifth, unseen column.

## Keep identical across all eight runs

So the eight results are comparable in one matrix, only the held-out method and
backbone change. Everything else is fixed:

| Setting              | Value |
|----------------------|-------|
| Dataset, compression | FaceForensics++ at c23 |
| Crops                | the shared cached crops, same pipeline for every run |
| Split                | official 720 / 140 / 140, identity-disjoint |
| Seed                 | the same fixed seed, recorded in the config |
| Primary metric       | AUC, at frame and video level, plus a confusion matrix |
| Test discipline      | the held-out method is never opened during training or tuning |

## A note on who does what

This document deliberately does not assign people to modules or runs. It fixes
only the seams: the file formats, the config keys, and the branch model. Who
owns preprocessing, who owns the detectors, who takes which of the eight runs,
and what counts as "done" are for the team to decide together. Once those
decisions are made, they can be recorded here or wherever the team prefers.
