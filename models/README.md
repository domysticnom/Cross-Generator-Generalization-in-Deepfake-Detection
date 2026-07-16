# models/

Model implementations and architecture definitions.

## Planned contents

- `efficientnet.py`: EfficientNet detector (ImageNet-pretrained, fine-tuned head)
- `xception.py`: XceptionNet detector (the canonical FF++ baseline)
- `simswap_generator.py`: thin wrapper around the **pretrained** SimSwap face-swap
  generator used to synthesize the self-produced unseen-generator set. We *use* a
  pretrained generator; we do not train one from scratch.
- `build.py`: factory that instantiates a detector from a config

Detector backbones come from `timm`. Hyperparameters and paths are kept out of
the code and live in `configs/` so runs differ only by a config file.

Trained weights / checkpoints are **git-ignored**.
