# Portable Preprocessing Cache Format

**Status: proposal for the team, consistent with docs/INTERFACES.md.**

This document describes the cache that `data/preprocess.py` produces so a teammate
can copy it to another machine and consume it without re-running preprocessing.
It covers the on-disk layout, the self-describing config sidecar, the schemas
(pointed at INTERFACES.md rather than duplicated), the repo-relative path rule,
and a numbered copy-and-consume procedure. This is ROADMAP Phase 3 criterion 4
(the cache is portable and documented).

Re-running preprocessing on the real FF++ corpus is expensive (face detection over
thousands of clips), so the whole point of the cache is that it travels: crop it
once, copy the crops plus the two manifests, and every teammate's loader reads the
same tensors.

## On-disk layout

```
data/
  processed/                              # cached face crops (git-ignored, copied out of band)
    preprocess_config.json                # effective config plus library versions (the sidecar)
    <method>/<clip_id>/f%04d.npy          # one .npy per sampled frame, e.g. DeepFakes/000_003/f0012.npy
  manifests/
    crops.parquet                         # index of every cached crop (committed, small)
    detection_log.csv                     # per-clip face-detection success-rate log (committed, small)
```

What each piece is:

| Artifact | What it is | Committed? |
|----------|-----------|------------|
| `data/processed/<method>/<clip_id>/f%04d.npy` | one cached face crop per sampled frame, an `(size, size, 3)` uint8 RGB array saved with `numpy.save` | No, git-ignored (`data/processed/` and `*.npy` in `.gitignore`); copied out of band |
| `data/manifests/crops.parquet` | one row per cached crop, the index the loaders read | Yes, small |
| `data/manifests/detection_log.csv` | one row per clip: frames sampled, faces detected, detection_rate; the real-vs-fake detection-failure evidence and also the resume ledger | Yes, small |
| `data/processed/preprocess_config.json` | the effective run config (frames, size, margin, confidence, seed, compression) plus numpy / opencv / mediapipe versions, so the crops are self-describing | No, lives with the git-ignored crops and travels with them |

The crops and the sidecar are git-ignored on purpose (they are large and machine
generated). The two manifest files are small and committed, so the schema is
version controlled even though the pixel data is not.

## Schemas

The `crops.parquet` schema is Contract 1 in `docs/INTERFACES.md`. It is not
duplicated here so the two documents can never drift; read Contract 1 for the
exact columns and their meaning. In code the same columns live in
`data/preprocess.py` as `CROP_COLUMNS`, written in that fixed order with a
schema-drift assertion, and `data/dataset.py` and `data/make_splits.py` consume
them.

Manifest-name reconciliation: the ROADMAP refers to the crop index generically as
`manifest.parquet`, and `docs/INTERFACES.md` plus the code commit to the concrete
name `data/manifests/crops.parquet`. They are the same artifact under two names;
`crops.parquet` is the real filename.

The `detection_log.csv` schema is `DETECT_LOG_COLUMNS` in `data/preprocess.py`:

| column | meaning |
|--------|---------|
| `clip_id` | source clip, e.g. `000_003` |
| `method` | one of `real`, `DeepFakes`, `Face2Face`, `FaceSwap`, `NeuralTextures` |
| `label` | `0` real, `1` fake |
| `frames_sampled` | how many frames were read from the clip |
| `faces_detected` | how many of those frames yielded a face crop |
| `detection_rate` | `faces_detected / frames_sampled`, rounded |

Note that the log is keyed on `(method, clip_id)`, not `clip_id` alone: FF++
reuses the same source-pair filenames across methods (`000_003.mp4` exists under
both DeepFakes and Face2Face), so each method gets its own row for a shared
`clip_id`.

## The repo-relative path rule

The `path` column in `crops.parquet` points at the cached `.npy` for that crop.
The rule the team relies on: crop paths follow the same relative layout on every
machine (`data/processed/<method>/<clip_id>/f%04d.npy`). No teammate's absolute
home directory is baked into the committed manifest. A run writes crops under the
`--out` root it was given, so as long as everyone lays the cache out the same way
under their own repo root, the loader resolves the crops. When you copy the cache,
put `data/processed/` and the two `data/manifests/` files at the same relative
paths inside your checkout and the loader finds them with no edits.

## Copy-and-consume procedure

To move the cache to another machine and use it without re-running preprocessing:

1. Copy `data/processed/` (the crops plus `preprocess_config.json`) and both
   `data/manifests/crops.parquet` and `data/manifests/detection_log.csv` to the
   same relative paths inside your checkout on the target machine. The `.npy`
   crops are git-ignored, so move them out of band (rsync, a shared drive, a zip),
   not through git.
2. Read `data/processed/preprocess_config.json` to confirm the parameters (frames,
   size, margin, confidence, seed, compression) and the numpy / opencv / mediapipe
   versions the crops were made with, so you know exactly what you are consuming.
3. Build a split manifest as in `docs/INTERFACES.md` Contract 2 (`crop_id`, `role`
   columns), then load crops through `data/dataset.py` `CropDataset(manifest_path,
   split_path, role, input_size)`. It reads `crops.parquet`, filters to your split,
   and `numpy.load`s each `.npy`. No preprocessing rerun is needed.

The hermetic test `tests/test_preprocess.py` proves step 3 end to end on synthetic
crops: it runs `preprocess.run()`, then loads the produced manifest through
`CropDataset` from a fresh path and pulls a `(3, size, size)` tensor, so the
copy-and-consume seam is exercised without the real corpus.

## Notes

- End-to-end verification on the real FF++ corpus is a later human-run step, done
  once the Phase 1 download completes. This document and the hermetic tests prove
  the format and the seams on synthetic inputs; the real-corpus crop-and-consume
  pass is out of scope here.
- The parameter defaults in `configs/preprocess.yaml` are Phase 2 EDA placeholders.
  A later EDA pass may revise them by editing that YAML rather than the code; the
  effective values always land in `preprocess_config.json` next to the crops, so a
  copied cache records the values it was actually built with.
