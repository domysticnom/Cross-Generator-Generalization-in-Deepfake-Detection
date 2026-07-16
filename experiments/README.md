# experiments/

Notebooks, training/evaluation scripts, and results.

## Planned contents

- `01_eda.ipynb`: FF++ class balance, per-method counts, identity coverage, quality
- `train.py`: training loop (config-driven, checkpoint/resume)
- `evaluate.py`: metrics per split; assembles the cross-generator transfer matrix
- `results/`: figures, transfer-matrix tables, seen-vs-unseen reports

Large outputs (checkpoints, cached predictions) are **git-ignored**; commit only
small result tables and figures needed for the report.
