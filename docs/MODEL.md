# Detector model

Architecture, training procedure, and evaluation for the CNN detectors. Applies to all
four to eight runs. Why these methods were chosen is in [`RESEARCH.md`](RESEARCH.md); the module
contracts are in [`INTERFACES.md`](INTERFACES.md).

## Task

Binary classification of a cached face crop: real (0) vs fake (1). Each run is one cell of
the cross-generator transfer matrix, not a standalone detector.

## Architecture

`models/detector.py` wraps `timm`:

```python
timm.create_model(backbone, pretrained=pretrained, num_classes=2)
```

This keeps the ImageNet pretrained convolutional backbone and replaces its 1000-way head
with a fresh 2-way head transfer learning, not training from scratch. **The whole network
is then fine tuned at a low learning rate**.

Training uses `CrossEntropyLoss` over the two logits; evaluation takes
`softmax(logits)[:, 1]` as the probability that a crop is fake.

Because the backbone is a config field, EfficientNet and Xception run through identical
code, so any difference between them is attributable to the architecture.

## Inputs

Crops are pre-extracted by `data/preprocess.py`, so training does no video decoding.

- **Crop size:** `input_size` from the config 224 for EfficientNet, 299 for Xception.
- **Normalization:** ImageNet mean/std, matching the pretrained weights.
- **Cache format:** `.npy` face crops indexed by `data/manifests/crops.parquet`.

## Training

Config-driven via `configs/<run_name>.yaml`; two runs differ only by `backbone` and
`held_out_method`. `resolve_device()` selects CUDA, then Apple MPS, then CPU. Mixed
precision is enabled on CUDA only.

| Choice | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Decoupled weight decay, robust for fine tuning |
| Learning rate | 3e-4 | Low, so pretrained features are refined rather than destroyed |
| Batch size | 64 | Fits a mid range GPU at 224² |
| Epochs | 15 | Transfer learning converges quickly |
| Seed | 1337 | Identical across all eight runs so they stay comparable |

The seed and every hyperparameter live in the committed config, so a result traces back to
a versioned file.

## Evaluation

Seen methods are scored on the `val` role and the held out method on `test`. Each method
becomes one row flagged `seen`, written to `experiments/results/<run_name>.json`.

Metrics: ROC-AUC (primary), accuracy, precision, recall, F1. With `video_level: true` a
clip's score is the mean of its frame scores, so the unit of analysis is the video.

Seen and unseen are never averaged together the gap between them is the result.
`experiments/transfer_matrix.py` combines every results JSON into the matrix, so only
real runs belong in `experiments/results/`.

## Data splits

`data/make_splits.py` builds one "leave one manipulation out" fold per method: train and val
draw from real plus the three trained methods, test is the held out method.

Roles come from the manifest's `official_split`, so preprocessing must run with
`--official-splits`. If it does not, every clip is labelled `train`, leaving no validation
set `make_splits.py` fails loudly on this, and `data/tag_official_splits.py` repairs a
manifest already on disk without recropping.

## Known gaps

Two places where the implementation does not yet match the proposal. Both are deliberate
and unresolved, not oversights:

**Augmentation is not implemented.** The proposal specifies horizontal flip plus mild
brightness/contrast jitter, and deliberately avoids blur or compression because those erase
the manipulation artifacts the detector needs. `data/dataset.py` currently only resizes and
normalizes. Adding it would change training for every run, so it needs a team decision on
timing.

**Folds are not identity disjoint.** Every held out clip goes to test whatever its official
split, so one identity can appear in train (as real or another manipulation) and in test at
once. `INTERFACES.md` Contract 2 calls for the disjoint form. Note the empty role guard does
not catch this: a leaky fold still has all three roles populated.

Measured on a 40 clip per method subset, this moved the unseen DeepFakes video AUC by
**+0.0015** (0.6815 vs 0.6800) noise at that scale. So it is a protocol correctness issue
rather than wrong numbers, but the write up should not claim identity disjoint splits
unless it is fixed first.
