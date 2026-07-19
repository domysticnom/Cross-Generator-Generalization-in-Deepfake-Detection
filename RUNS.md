# Training runs board

Eight runs = 2 backbones (EfficientNet, XceptionNet) times 4 held-out FF++ methods.
Five of us, so we split them. Claim a run here before you train it so no two people
do the same one.

## How to use this

1. Pick an open run below and put your name in **Owner**, set **Status** to `in progress`,
   then commit + push this file (that is your claim).
2. Open `experiments/01_train.ipynb`, set `RUN` in step 0 to your run, and run it top to
   bottom (needs the crops cache from `00_setup_and_preprocess.ipynb` on your VM first).
3. When step 5 writes `experiments/results/<run>.json`, push that JSON and set your
   **Status** here to `done`. That JSON is what feeds the transfer matrix; the notebook's
   run board also reads it, so the whole team sees your run flip to DONE.

A run is only really DONE when its results JSON is on `main`. Until then it stays open.

## Board

| Run | Owner | Status |
|-----|-------|--------|
| efficientnet_holdout-deepfakes | Jonathan | in progress (on `jonathan/work-on-model` side branch; needs reconciling with main before the result counts) |
| efficientnet_holdout-face2face | Obinna | in progress |
| efficientnet_holdout-faceswap | | todo |
| efficientnet_holdout-neuraltextures | | todo |
| xception_holdout-deepfakes | Lyxelis | claimed (wrote the config) |
| xception_holdout-face2face | Lyxelis | claimed (wrote the config) |
| xception_holdout-faceswap | | todo |
| xception_holdout-neuraltextures | | todo |

## Notes

- Prereq for every run: the crops cache on your VM (`data/processed/` + `data/manifests/crops.parquet`),
  produced by `00_setup_and_preprocess.ipynb`. Crops are gitignored and regenerated per VM, not downloaded.
- GPU-specific tweaks (amp, batch size) go through the optional overrides in step 0 of the
  train notebook; they never change the committed config, only how the run executes.
- Heads-up on `jonathan/work-on-model`: it branched from an older commit and rebuilt the
  model/training layer separately, so it is missing most of main's pipeline (preprocess,
  notebooks, splits, transfer matrix). It is a divergent branch, not a fast-forward; merging
  it into main is a team decision, not a clean merge.
