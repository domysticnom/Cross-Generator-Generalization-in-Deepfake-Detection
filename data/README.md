# data/

Dataset information, acquisition notes, and preprocessing/loading scripts.

Raw and processed data are **git-ignored** (see `.gitignore`); only code and
small manifest/index files live here.

## Planned contents

- `download_ffpp.py`: acquire FaceForensics++ (c23) and verify per-method counts
- `preprocess.py`: frame extraction, face detection/crop, normalize, then cache
- `make_splits.py`: build leave-one-manipulation-out, identity-disjoint split manifests
- `synthesize_simswap.py`: generate the self-produced SimSwap "unseen generator" set
- `manifests/`: small CSV/parquet indexes linking cached crops to clip / method / identity

## Layout (created at runtime, ignored by git)

```
data/
├── raw/         # FF++ videos as downloaded
├── processed/   # cached face crops (.npy/.pt)
├── splits/      # LOMO split manifests
└── simswap/     # self-generated SimSwap fakes
```
