# Cross-Generator Generalization in Deepfake Detection

**IE7374: Generative AI, Northeastern University**

Train conventional CNN deepfake detectors (**EfficientNet**, **XceptionNet**) on
FaceForensics++ and measure **how well they generalize to manipulation methods
they were never trained on**, reported as a cross-generator transfer matrix
rather than a single in-distribution accuracy number. A pretrained **SimSwap**
face-swap generator is used to synthesize a fifth, self-produced "unseen
generator" test set, adding a generative component to the pipeline.

> **Status:** The pipeline runs end to end — dataset acquisition, face-crop preprocessing,
> leave-one-out splits, training, evaluation, and transfer-matrix assembly. The eight
> training runs are in progress; see [`RUNS.md`](RUNS.md) for the board.

## Team

| Member | Role |
|---|---|
| Dominic Rivas | Principal researcher |
| Jonathan Jude Regalado | Principal researcher |
| Lyxelis Rodriguez Navarro | Principal researcher |
| Obinna Okonkwo | Principal researcher |
| Sagar Ayare | Principal researcher |

Every member independently trains and evaluates an assigned run so the transfer matrix is
produced in parallel. Run ownership is tracked in [`RUNS.md`](RUNS.md).

## Documentation

| Doc | Covers |
|---|---|
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | Objectives, literature review, backbone benchmarking, preliminary experiments |
| [`docs/MODEL.md`](docs/MODEL.md) | Detector architecture, training, evaluation, known gaps |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Setup through a finished run |
| [`docs/INTERFACES.md`](docs/INTERFACES.md) | Contracts every module and run agrees on |
| [`docs/DATASET_ACCESS.md`](docs/DATASET_ACCESS.md) | Obtaining FaceForensics++ |
| [`RUNS.md`](RUNS.md) | Who owns which of the eight runs |

## Usage

Three notebooks, run in order:

```
experiments/00_setup_and_preprocess.ipynb   # install, download, face-crop cache, splits
experiments/01_train.ipynb                  # claim a run, train, evaluate
experiments/02_transfer_matrix.ipynb        # assemble the cross-generator matrix
```

`00` installs its own pinned stack (torch 2.4.1+cu121, then a numpy-1.x island for
mediapipe), so run it before anything else. The CLI equivalent is in
[`docs/QUICKSTART.md`](docs/QUICKSTART.md).

## Repository structure

```
.
├── data/          # Acquisition, face-crop preprocessing, splits, crop loading
├── models/        # detector.py (timm backbones) + simswap_generator.py
├── experiments/   # 00/01/02 notebooks, train.py, evaluate.py, transfer_matrix.py, results
├── configs/       # One YAML per run (8 runs) + preprocess.yaml
├── docs/          # Research, model, quickstart, interfaces, dataset access
├── tests/         # Unit tests for preprocessing and inventory
├── requirements.txt
└── README.md
```

See the `README.md` inside each directory for its specific contents.

## Generative component (SimSwap)

The milestone calls for a generative model in the pipeline, not just detection
of generated media. We use a **pretrained SimSwap** face-swap generator to
synthesize our own fake videos from FaceForensics++ real source footage. Those
self-generated forgeries become a fifth "unseen generator" category, evaluated
alongside FF++'s four manipulation methods.

SimSwap is a modern, subject-agnostic learned face swap, so it sits in a
different generator family from FF++'s DeepFakes (autoencoder swap), FaceSwap
(computer-graphics swap), Face2Face, and NeuralTextures (reenactment). That
makes it a genuine out-of-distribution test rather than a re-run of a method the
detectors already know. We *use* the pretrained generator off the shelf; we do
not train or fine-tune a generator from scratch.

## Setup

Python **3.11** is recommended.

**Easiest path:** run `experiments/00_setup_and_preprocess.ipynb`, which installs the
pinned stack in the right order and then builds the crop cache.

To install manually, order matters — torch first, then requirements, then the numpy-1.x
pins that mediapipe needs:

```bash
# 1. Create and activate an environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

# 2. Install PyTorch FIRST from a CUDA wheel index, NOT plain PyPI.
#    cu121 matches the school VM's driver; pick the wheel matching your own.
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121
# macOS (no CUDA, runs on CPU/MPS):  pip install torch torchvision

# 3. Install the rest
pip install -r requirements.txt

# 4. mediapipe/TensorFlow need numpy 1.x, so re-pin after requirements
pip install "numpy<2" "opencv-python-headless<5" "protobuf>=3.20.3,<5"
```

`ffmpeg` (6.x/7.x) must be available on the system PATH for video decoding.

Verify with `python check_env.py`. Per-run workflow: `docs/QUICKSTART.md`. Module contracts:
`docs/INTERFACES.md`.

## Method overview

- **Task:** binary deepfake detection (real vs. fake) framed as a
  cross-generator *generalization* study.
- **Data:** FaceForensics++ (c23), with 1,000 real and 4,000 fake videos across
  four manipulation methods (DeepFakes, Face2Face, FaceSwap, NeuralTextures).
- **Protocol:** leave-one-manipulation-out. Train on 3 of 4 methods, test on the
  held-out 4th, rotated to build a transfer matrix. A pretrained SimSwap set and
  (stretch) a DFDC subset act as additional never-seen generators.
- **Models:** ImageNet-pretrained EfficientNet and XceptionNet, fine-tuned.
- **Metrics:** ROC-AUC (primary), plus accuracy, precision, recall, F1, and
  confusion matrices, reported **seen-vs-unseen separately**.

## Reproducibility

Runs are driven by version-controlled YAML configs with fixed seeds and
checkpoint/resume, so results are portable across the team's heterogeneous
hardware. Datasets and checkpoints are git-ignored (see `.gitignore`).
