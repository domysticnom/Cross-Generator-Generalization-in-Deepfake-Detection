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

All eight are now done and on `main`. The board below reflects who actually
produced the results JSON that is committed, which is not the same as who
originally claimed the run -- see Notes.

| Run | Owner | Status |
|-----|-------|--------|
| efficientnet_holdout-deepfakes | Dominic | done (Jonathan had claimed it; see Notes) |
| efficientnet_holdout-face2face | Dominic | done (Obinna had claimed it; see Notes) |
| efficientnet_holdout-faceswap | Dominic | done |
| efficientnet_holdout-neuraltextures | Dominic | done |
| xception_holdout-deepfakes | Lyxelis | done |
| xception_holdout-face2face | Lyxelis | done |
| xception_holdout-faceswap | Dominic | done |
| xception_holdout-neuraltextures | Dominic | done |

## Notes

- Prereq for every run: the crops cache on your VM (`data/processed/` + `data/manifests/crops.parquet`),
  produced by `00_setup_and_preprocess.ipynb`. Crops are gitignored and regenerated per VM, not downloaded.
- GPU-specific tweaks (amp, batch size) go through the optional overrides in step 0 of the
  train notebook; they never change the committed config, only how the run executes.
- `jonathan/work-on-model` **no longer exists on the remote** -- it was deleted. Jonathan's
  Dockerfile and models README work did land on `main`; the efficientnet_holdout-deepfakes
  run that branch was carrying did not, and the committed result for that run is Dominic's.
- `obinna/efficientnet-face2face` is still open and unmerged. It claims
  efficientnet_holdout-face2face as done, but its results JSON has `"auc": NaN` and only 1
  of the 5 test rows, so it would overwrite a complete result with a broken one. Left alone
  until that run is redone.
- Two runs (efficientnet deepfakes, efficientnet face2face) were done twice because the
  board was never updated -- the claims lived on side branches that `main` could not see.
  Claim runs by pushing THIS file to `main`, not by pushing a branch.

## What `experiments/results_cleantest/` is

It is the **identity-disjoint evaluation** of the same eight runs, produced by evaluating
against `data/splits_clean/` instead of `data/splits/`. Neither was documented, so both
sat in the repo looking like leftovers.

`data/splits_clean/` holds the same four folds with the test role restricted to the 140
source identities that never appear in training (~280 clips, ~5,480 crops per fold, and
near class-balanced). It is the honest split set, and the audit confirms it:

    python data/audit_splits.py --splits data/splits_clean   # PASS
    python data/audit_splits.py                              # FAIL (the leaky default)

The runs themselves were trained on `data/splits/`, so `splits_clean/` affects evaluation
only. Rebuilding the folds identity-disjoint at TRAINING time still requires retraining.

Confirmed by reproducing it: `experiments/leakage_impact.py --subset clean` restricts
each held-out test set to the 140 identities that never appear in training (~280 clips,
~5,480 crops, near class-balanced). Its output matches every `results_cleantest` number
to within 0.0003 AUC, with accuracies identical to 4 decimal places, on all eight runs.
See `experiments/results/identity_leakage.csv`.

So: `results/` is the leaky evaluation, `results_cleantest/` is the clean one. Cite the
clean numbers, and say which you are citing.

**The leakage does NOT inflate the headline.** Removing it *raises* held-out AUC in 6 of
8 folds (mean +0.030, range -0.009 to +0.070). ROC-AUC is rank-based and so is not biased
by the class-ratio difference between the two subsets, which makes the AUC comparison a
fair one. Accuracy and recall are not comparable that way -- the clean subset is balanced
and the full one is ~7:1 fake -- so only AUC should be compared across the two.

The below-chance FaceSwap result survives the correction: 0.362 -> 0.405 (Xception) and
0.465 -> 0.476 (EfficientNet). Still below chance on both backbones.

Splits are still not identity-disjoint at TRAINING time -- `data/audit_splits.py` fails on
all four folds and should keep failing until the splits are rebuilt. What the clean
evaluation shows is that the effect on the reported numbers is small and, where it matters,
conservative.

## Analysis scripts

Run these after a run's results JSON exists; none of them retrain anything.

| Script | What it does |
|---|---|
| `experiments/check_results.py` | Sanity-checks every result JSON before the numbers go in the report. Exit 1 on any FAIL. |
| `data/audit_splits.py` | DATA-05 identity-disjointness audit. Exit 1 if any fold leaks (all four currently do). |
| `experiments/leakage_impact.py` | Measures what the leak is worth by re-scoring on never-seen identities. Writes `experiments/results/identity_leakage.csv`. |
| `experiments/eval_simswap.py` | Re-scores only the SimSwap row of an existing run, in place. Use this instead of `evaluate.py` when you must not overwrite FF++ numbers a teammate produced. |

## SimSwap column caveat

The SimSwap column is not comparable across every row. Two disjoint SimSwap sets exist
(a 994-pair set and a 241-pair set, zero crop_id overlap, generated independently). The six
Dominic runs are scored on the 994-pair set and carry a `simswap_split` field recording
that. Lyxelis's two xception runs are scored on the 241-pair set and carry no stamp.
`experiments/check_results.py` flags this automatically -- do not average down the SimSwap
column, and do not compare a stamped row's SimSwap AUC against an unstamped one's.
