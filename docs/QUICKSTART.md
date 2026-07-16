# Quickstart

From a fresh clone to a running training job. Steps 1 to 4 are one-time setup.
Steps 5 to 7 you repeat per run.

> Note: the training and evaluation scripts referenced in steps 6 and 7 are being
> implemented now. This quickstart is the workflow they follow, so the commands
> are the target once those scripts land.

## 1. Clone

    git clone https://github.com/domysticnom/Cross-Generator-Generalization-in-Deepfake-Detection.git
    cd Cross-Generator-Generalization-in-Deepfake-Detection

## 2. Create an environment

Python 3.11 is recommended.

    python -m venv .venv
    # Windows:        .venv\Scripts\activate
    # macOS / Linux:  source .venv/bin/activate

## 3. Install PyTorch from the cu128 wheel index (read this)

Install torch FIRST, and NOT with a plain `pip install torch`. The default PyPI
wheel ships without the kernels for RTX 50-series GPUs and will fall back to CPU
without telling you. The cu128 wheel also runs on older cards, so the whole team
uses the same command:

    pip install torch==2.9.* torchvision==0.24.* torchaudio==2.9.* \
        --index-url https://download.pytorch.org/whl/cu128

Then the rest:

    pip install -r requirements.txt

You also need `ffmpeg` 6.x or 7.x on your PATH for video decoding.

## 4. Confirm your setup

    python check_env.py

You want CUDA visible and the CUDA tensor op passing. If CUDA is not available on
a machine that has an NVIDIA GPU, you installed the wrong torch wheel. Redo step 3.

## 5. Pull the shared data

Do not re-download and re-crop FaceForensics++ yourself. Pull the shared cached
crops and manifest that the preprocessing owner published, into:

    data/processed/               # cached crops
    data/manifests/crops.parquet  # the crop index
    data/splits/                  # the frozen leave-one-out folds

The location of the shared copy is posted in the team channel. Everyone trains
from the same crops so the eight runs stay comparable.

## 6. Run a fold

Each run is one config. Branch, then run:

    git checkout -b run/efficientnet-holdout-faceswap
    python experiments/train.py    --config configs/efficientnet_holdout-faceswap.yaml
    python experiments/evaluate.py --config configs/efficientnet_holdout-faceswap.yaml

Each config corresponds to one run in the run table in `docs/INTERFACES.md`.

## 7. Publish your results

- Your run writes `experiments/results/<run_name>.json` in the shared format
  (see Contract 4 in `docs/INTERFACES.md`).
- It logs to the shared W&B project `cross-generator-deepfake`.
- Commit the small results JSON, open a pull request into `main`, get one review.

Checkpoints and crops are git-ignored, so committing results never drags large
files into the repo.

## Where to look

- `docs/INTERFACES.md` : the contracts every module and run agrees on
- `README.md` : project overview and method summary
- `data/README.md` : what the preprocessing pipeline produces
