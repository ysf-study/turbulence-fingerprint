#!/usr/bin/env python3
"""Feature-set ablation (Table 4 of the paper).

For each feature subset the model is trained and scored under the released
five-fold assignment, and the reported value is the mean and standard deviation
of the five out-of-fold AUROCs -- the same protocol as Table 4.  The pooled
out-of-fold AUROC is written as well, for comparison with the headline 0.918.

Usage
-----
    python src/ablation.py --out results
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from train_eval import FEATURES, PARAMS, load_tables

SUBSETS = [
    ("all4", FEATURES),
    ("F_low", ["F_low"]),
    ("omega_max", ["omega_max"]),
    ("max_temporal", ["max_temporal"]),
    ("p999_temporal", ["p999_temporal"]),
    ("without_F_low", ["omega_max", "max_temporal", "p999_temporal"]),
    ("without_omega_max", ["F_low", "max_temporal", "p999_temporal"]),
    ("without_max_temporal", ["F_low", "omega_max", "p999_temporal"]),
    ("without_p999_temporal", ["F_low", "omega_max", "max_temporal"]),
    ("spatial", ["F_low", "omega_max"]),
    ("temporal", ["max_temporal", "p999_temporal"]),
]


def evaluate(df: pd.DataFrame, cols: list[str], n_jobs: int) -> dict:
    X = df[cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    fold = df["fold"].to_numpy(dtype=int)
    oof = np.zeros(len(y), dtype=float)
    for k in range(int(fold.max()) + 1):
        tr, te = fold != k, fold == k
        clf = XGBClassifier(**PARAMS, n_jobs=n_jobs)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    per_fold = [float(roc_auc_score(y[fold == k], oof[fold == k])) for k in range(int(fold.max()) + 1)]
    return {
        "mean_auroc": float(np.mean(per_fold)),
        "sd_auroc": float(np.std(per_fold, ddof=1)),
        "pooled_auroc": float(roc_auc_score(y, oof)),
        "per_fold": per_fold,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/features_4d.csv")
    ap.add_argument("--folds", default="data/fold_assignment.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    df = load_tables(args.features, args.folds)
    if "fold" not in df.columns:
        raise SystemExit("ablation needs a fold assignment; pass --folds")
    os.makedirs(args.out, exist_ok=True)

    rows, full = [], {}
    base = None
    for name, cols in SUBSETS:
        res = evaluate(df, cols, args.n_jobs)
        if base is None:
            base = res["mean_auroc"]
        rows.append(
            {
                "subset": name,
                "n_features": len(cols),
                "features": "+".join(cols),
                "auroc_mean": res["mean_auroc"],
                "auroc_sd": res["sd_auroc"],
                "auroc_pooled": res["pooled_auroc"],
                "delta_vs_all4": res["mean_auroc"] - base,
            }
        )
        full[name] = res
        print("%-22s %.4f ± %.4f  (pooled %.4f)" % (name, res["mean_auroc"], res["sd_auroc"], res["pooled_auroc"]))

    pd.DataFrame(rows).to_csv(os.path.join(args.out, "ablation.csv"), index=False)
    with open(os.path.join(args.out, "ablation.json"), "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2)
    print("written -> %s/ablation.csv" % args.out)


if __name__ == "__main__":
    main()
