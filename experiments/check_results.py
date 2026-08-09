"""Sanity checks for the cross-generator transfer results.

Run this before any number goes into the report:

    python experiments/check_results.py

It reads every experiments/results*/ *.json and flags the failure modes that
produce a plausible-looking but wrong generalization claim. Nothing here
recomputes a metric -- it only checks the metrics we already wrote out are
internally consistent and mean what the report says they mean.

Exit code is 0 if only INFO/WARN fired, 1 if any FAIL fired.
"""

import glob
import json
import math
import os
import sys
from collections import defaultdict

# The four FF++ manipulation methods the models train on. SimSwap is a separate
# unseen-generator set synthesized locally, never trained on by anyone.
FFPP_METHODS = {"DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"}
EXTERNAL_METHODS = {"SimSwap"}

RESULT_DIRS = sorted(glob.glob(os.path.join("experiments", "results*")))

findings = []


def record(level, run, msg):
    findings.append((level, run, msg))


def load_runs():
    """Returns {dir: {run_name: parsed_json}}."""
    out = {}
    for d in RESULT_DIRS:
        runs = {}
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                with open(path, encoding="utf-8") as fh:
                    # NaN is not legal JSON but json.load accepts it by default,
                    # which is exactly how a NaN AUC slips through unnoticed.
                    runs[os.path.basename(path)] = json.load(fh)
            except Exception as e:
                record("FAIL", path, f"unparseable: {e}")
        if runs:
            out[d] = runs
    return out


def check_nan(d, fname, run):
    """A NaN/None AUC means the eval had one class only, or all-constant scores."""
    for row in run.get("results", []):
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                record("FAIL", f"{d}/{fname}",
                       f"{k} is NaN for tested_on={row.get('tested_on')} "
                       "(single-class test split, or degenerate scores)")


def check_seen_flags(d, fname, run):
    """`seen` must be False for exactly the held-out method and for externals.

    If this is wrong the seen-vs-unseen gap -- the entire contribution -- is
    computed over the wrong partition.
    """
    held = run.get("held_out_method")
    for row in run.get("results", []):
        tested = row.get("tested_on")
        seen = row.get("seen")
        if tested in EXTERNAL_METHODS:
            expected = False
        elif tested == held:
            expected = False
        elif tested in FFPP_METHODS:
            expected = True
        else:
            record("WARN", f"{d}/{fname}", f"unrecognized tested_on={tested!r}")
            continue
        if seen != expected:
            record("FAIL", f"{d}/{fname}",
                   f"tested_on={tested} held_out={held} has seen={seen}, expected {expected}")


def check_coverage(d, fname, run):
    """Every run should test on all four FF++ methods."""
    tested = {r.get("tested_on") for r in run.get("results", [])}
    missing = FFPP_METHODS - tested
    if missing:
        record("FAIL", f"{d}/{fname}", f"missing test methods: {sorted(missing)}")
    if not (tested & EXTERNAL_METHODS):
        record("WARN", f"{d}/{fname}", "no external (SimSwap) unseen set evaluated")


def check_below_chance(d, fname, run):
    """AUC < 0.5 is anti-correlation, not weak generalization.

    A detector that scores 0.36 is reliably ranking fakes as MORE real than
    reals. That is a sign flip / label-polarity problem somewhere in the eval,
    and reporting it as 'poor transfer' would be wrong.
    """
    for row in run.get("results", []):
        auc = row.get("auc")
        if isinstance(auc, (int, float)) and not math.isnan(auc) and auc < 0.5:
            record("FAIL", f"{d}/{fname}",
                   f"AUC={auc} < 0.5 on tested_on={row.get('tested_on')} "
                   "(worse than chance -> label polarity / threshold bug, not transfer failure)")


def check_auc_acc_divergence(d, fname, run, tol=0.35):
    """High AUC with floor-level accuracy means the threshold is miscalibrated.

    AUC is threshold-free; accuracy is not. AUC 0.99 with acc 0.13 does not mean
    the model is bad -- it means the 0.5 cut point is in the wrong place for this
    test distribution. Worth separating in the report.
    """
    for row in run.get("results", []):
        auc, acc = row.get("auc"), row.get("acc")
        if not all(isinstance(v, (int, float)) for v in (auc, acc)):
            continue
        if math.isnan(auc) or math.isnan(acc):
            continue
        if auc - acc > tol:
            record("WARN", f"{d}/{fname}",
                   f"tested_on={row.get('tested_id') or row.get('tested_on')}: "
                   f"AUC={auc} but acc={acc} (gap {auc - acc:.2f}) -> "
                   "threshold miscalibrated on this set; prefer AUC in the writeup")


def check_unseen_aggregation(d, fname, run, spread=0.25):
    """The reported unseen_auc averages two very different things.

    seen_vs_unseen_gap.csv collapses {held-out FF++ method, SimSwap} into one
    'unseen' mean. Those are not interchangeable: the held-out method is a
    same-dataset, same-pipeline manipulation, while SimSwap is an externally
    synthesized generator. When they disagree sharply the mean is not a
    meaningful quantity, and it can hide a below-chance component.
    """
    unseen = [(r.get("tested_on"), r.get("auc")) for r in run.get("results", [])
              if r.get("seen") is False
              and isinstance(r.get("auc"), (int, float))
              and not math.isnan(r.get("auc"))]
    if len(unseen) < 2:
        return
    vals = [a for _, a in unseen]
    lo, hi = min(vals), max(vals)
    if hi - lo > spread:
        mean = sum(vals) / len(vals)
        parts = ", ".join(f"{t}={a}" for t, a in unseen)
        record("FAIL", f"{d}/{fname}",
               f"unseen components disagree by {hi - lo:.2f} ({parts}); "
               f"reporting their mean {mean:.4f} as 'unseen AUC' conflates a held-out "
               "FF++ method with an external generator -- report them separately")


def check_seed_consistency(runs_by_dir):
    """Same run_name + same seed must not produce different numbers."""
    by_key = defaultdict(list)
    for d, runs in runs_by_dir.items():
        for fname, run in runs.items():
            key = (run.get("run_name"), run.get("backbone"),
                   run.get("held_out_method"), run.get("seed"))
            by_key[key].append((d, fname, run))
    for key, entries in by_key.items():
        if len(entries) < 2:
            continue
        base_d, base_f, base = entries[0]
        for d, fname, run in entries[1:]:
            for rb, rr in zip(base.get("results", []), run.get("results", [])):
                if rb.get("tested_on") != rr.get("tested_on"):
                    continue
                a, b = rb.get("auc"), rr.get("auc")
                if not all(isinstance(v, (int, float)) for v in (a, b)):
                    continue
                if math.isnan(a) or math.isnan(b):
                    continue
                if abs(a - b) > 1e-9:
                    record("WARN", f"{key[0]} seed={key[3]}",
                           f"tested_on={rb.get('tested_on')}: "
                           f"{base_d}={a} vs {d}={b} (delta {abs(a - b):.4f}) "
                           "-- same run_name and seed, different result")


def check_gap_direction(d, fname, run):
    """Unseen should underperform seen. If not, the split may be leaking."""
    seen = [r["auc"] for r in run.get("results", [])
            if r.get("seen") and isinstance(r.get("auc"), (int, float))
            and not math.isnan(r["auc"])]
    unseen = [r["auc"] for r in run.get("results", [])
              if r.get("seen") is False and r.get("tested_on") == run.get("held_out_method")
              and isinstance(r.get("auc"), (int, float)) and not math.isnan(r["auc"])]
    if seen and unseen:
        mean_seen = sum(seen) / len(seen)
        if unseen[0] >= mean_seen:
            record("WARN", f"{d}/{fname}",
                   f"held-out AUC {unseen[0]} >= mean seen AUC {mean_seen:.4f} "
                   "-- no generalization gap; check the split actually held this method out")


def main():
    runs_by_dir = load_runs()
    if not runs_by_dir:
        print("no result files found under experiments/results*/")
        return 1

    for d, runs in runs_by_dir.items():
        for fname, run in runs.items():
            check_nan(d, fname, run)
            check_seen_flags(d, fname, run)
            check_coverage(d, fname, run)
            check_below_chance(d, fname, run)
            check_auc_acc_divergence(d, fname, run)
            check_gap_direction(d, fname, run)
            check_unseen_aggregation(d, fname, run)
    check_seed_consistency(runs_by_dir)

    order = {"FAIL": 0, "WARN": 1, "INFO": 2}
    findings.sort(key=lambda f: (order.get(f[0], 3), f[1]))

    n_fail = sum(1 for lvl, _, _ in findings if lvl == "FAIL")
    n_warn = sum(1 for lvl, _, _ in findings if lvl == "WARN")

    total_runs = sum(len(r) for r in runs_by_dir.values())
    print(f"checked {total_runs} run files across {len(runs_by_dir)} result dirs\n")
    for lvl, where, msg in findings:
        print(f"[{lvl}] {where}\n       {msg}")
    print(f"\n{n_fail} FAIL, {n_warn} WARN")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
