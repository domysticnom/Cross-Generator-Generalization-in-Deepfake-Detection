# models/

Model definitions. Backbones come from `timm`; hyperparameters live in `configs/`.
Checkpoints are git-ignored.

## Contents

- `detector.py` — `build_model(backbone, pretrained, num_classes=2)`. One factory for both
  EfficientNet and Xception, so the two architectures share an identical pipeline.
- `simswap_generator.py` — wrapper around the pretrained SimSwap face-swap generator used
  to synthesize the self-produced unseen-generator set. We use a pretrained generator; we
  do not train one.

Architecture and training rationale: [`../docs/MODEL.md`](../docs/MODEL.md).
