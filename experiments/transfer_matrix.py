import argparse
import glob
import json
import os

import pandas as pd

# Amalgamate every experiments/results/<run>.json into the cross-generator
# transfer matrix and the seen-vs-unseen gap. Each run wrote its rows in the
# shared schema (docs/INTERFACES.md Contract 4): tested_on + seen flag per method.


def load_results(results_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(f).startswith("transfer"):
            continue
        r = json.load(open(f))
        for res in r.get("results", []):
            rows.append({
                "backbone": r["backbone"],
                "held_out": r["held_out_method"],
                "run": r["run_name"],
                "level": r.get("level"),
                "tested_on": res["tested_on"],
                "seen": bool(res["seen"]),
                "auc": res.get("auc"),
                "acc": res.get("acc"),
                "f1": res.get("f1"),
            })
    return pd.DataFrame(rows)


def matrix_for(df, backbone):
    # rows = held-out method (the run), cols = tested-on method, value = AUC.
    # The diagonal (held_out == tested_on) is the UNSEEN cross-generator number;
    # off-diagonal cells are seen (in-distribution). The SimSwap column is unseen
    # for every run (the self-generated generator).
    sub = df[df.backbone == backbone]
    return sub.pivot_table(index="held_out", columns="tested_on", values="auc")


def gap_summary(df):
    seen = df[df.seen].groupby(["backbone", "held_out"])["auc"].mean().rename("seen_auc")
    unseen = df[~df.seen].groupby(["backbone", "held_out"])["auc"].mean().rename("unseen_auc")
    g = pd.concat([seen, unseen], axis=1).reset_index()
    g["gap"] = g["seen_auc"] - g["unseen_auc"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()

    df = load_results(args.results)
    if df.empty:
        print("no run results found in", args.results,
              "- run experiments/evaluate.py for at least one run first")
        return

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "all_results_long.csv"), index=False)
    print("runs found:", sorted(df.run.unique()))

    for bb in sorted(df.backbone.unique()):
        m = matrix_for(df, bb)
        path = os.path.join(args.out, f"transfer_matrix_{bb}.csv")
        m.to_csv(path)
        print(f"\n=== transfer matrix: {bb} (rows=held-out run, cols=tested-on, value=AUC) ===")
        print(m.round(3).to_string())
        print("wrote", path)

    g = gap_summary(df)
    g.to_csv(os.path.join(args.out, "seen_vs_unseen_gap.csv"), index=False)
    print("\n=== seen vs unseen gap (per run) ===")
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
