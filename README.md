# Cross-Generator Generalization in Deepfake Detection

**IE7374: Generative AI, Northeastern University**

Train conventional CNN deepfake detectors (**EfficientNet**, **XceptionNet**) on
FaceForensics++ and measure **how well they generalize to manipulation methods
they were never trained on**, reported as a cross-generator transfer matrix
rather than a single in-distribution accuracy number. A pretrained **SimSwap**
face-swap generator is used to synthesize a fifth, self-produced "unseen
generator" test set, adding a generative component to the pipeline.

> **Status:** Skeleton / scaffolding. This is the initial repository structure
> for the data-pipeline and infrastructure milestone. Implementation lands in
> subsequent commits.

## Team

Dominic Rivas, Jonathan Jude Regalado, Lyxelis Rodriguez Navarro,
Obinna Okonkwo, Sagar Ayare

## Repository structure

```
.
├── data/          # Dataset notes + preprocessing/loading scripts (FF++, splits, SimSwap set)
├── models/        # Detector implementations (EfficientNet, XceptionNet) + generator wrapper
├── experiments/   # Notebooks, training runs, results (EDA, transfer matrix)
├── configs/       # YAML hyperparameter / path configs (reproducible, per-machine overrides)
├── docs/          # Proposal, reports, figures
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

```bash
# 1. Create and activate an environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

# 2. Install PyTorch FIRST from the cu128 wheel index (NOT plain PyPI).
#    The default PyPI torch wheel lacks sm_120 kernels required by RTX 50-series GPUs.
pip install torch==2.9.* torchvision==0.24.* torchaudio==2.9.* \
    --index-url https://download.pytorch.org/whl/cu128

# 3. Install the rest
pip install -r requirements.txt
```

`ffmpeg` (6.x/7.x) must be available on the system PATH for video decoding.

Verify the install with `python check_env.py`. For the full per-run workflow,
see `docs/QUICKSTART.md`, and for the module/run contracts see `docs/INTERFACES.md`.

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
