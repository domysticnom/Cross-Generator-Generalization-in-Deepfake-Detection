# configs/

Version-controlled YAML configuration files. Keeping hyperparameters and paths
out of the code makes runs reproducible and portable: a laptop run and a remote
GPU run differ only by their config, and every result traces back to a committed
file.

## Planned contents

- `base.yaml`: shared defaults (seed, metrics, preprocessing params)
- `efficientnet.yaml`, `xception.yaml`: per-model overrides
- `splits/`: one config per leave-one-manipulation-out fold

Each config records the fixed random seed so training is deterministic.
