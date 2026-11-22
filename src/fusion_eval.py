#!/usr/bin/env python3
"""Fusion with D3 (Table 9 of the paper).

D3 ("Detection by Difference of Differences", arXiv:2508.00701) is a
training-free detector whose decision variable is a single scalar: the standard
deviation of the second difference of a frozen encoder's frame embeddings.  This
script concatenates that scalar with the four turbulence statistics and retrains
the same XGBoost configuration used everywhere else in this package, so that the
four rows of Table 9 can be regenerated from released per-clip scores.

    B  D3 only, no training          1-D, threshold on the raw score
    A  TF only                       4-D, XGBoost
    C  D3 only (capacity control)    1-D, XGBoost
    D  TF + D3                       5-D, XGBoost

Two aggregations are reported, because the paper uses both in different places
and the distinction matters for reading Table 9:

* **pooled** -- AUROC over all 1,635 out-of-fold scores at once.
* **macro**  -- mean over the ten generators of the AUROC between that
  generator's AI clips and all 485 real clips.  This is the "mean per-generator"
  convention of Table A.11 and of the paired tests in A.12.

Table 9 prints B, C and D as macro values and row A as the pooled value of the
TF-only reference (0.9178, the same number as Supplementary A.0); the two
aggregations differ by less than 0.001 on this data set.  ``fusion.csv`` writes
both columns for every row so the choice is never implicit.

Split-finding method
--------------------
Tables 3-5 pin ``tree_method="exact"``.  The fusion run behind Table 9 was made
before that pin was added and therefore used the XGBoost 2.x default,
``hist``: with ``hist`` the D3-only control C comes out at 0.9771/0.9777 against
0.9770/0.9776 printed, and the fused row D at 0.9838/0.9843 against
0.9835/0.9840.  With ``exact`` the same two rows move by up to 0.0015.  The
default here is ``exact``, to match the rest of the package; pass
``--tree-method hist`` to sit next to the printed Table 9 values.  The
difference never changes a ranking or a conclusion, and ``docs/TABLES.md``
tabulates both columns side by side.

Outputs (under --out)
---------------------
fusion.csv   feature_set, dimensions, training, pooled_auroc, macro_auroc
fusion.json  the same, plus the per-generator AUROCs behind each macro value

Usage
-----
    python src/fusion_eval.py --out results
    python src/fusion_eval.py --out results --tree-method hist
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

ROWS = [
    # (label, columns, training description, needs a model)
    ("B: D3 only", ["d3_score"], "None (threshold)", False),
    ("A: TF only", FEATURES, "XGBoost", True),
    ("C: D3 only (control)", ["d3_score"], "XGBoost", True),
    ("D: TF + D3 (this work)", FEATURES + ["d3_score"], "XGBoost", True),
]


def fit_predict(X_tr, y_tr, X_te, n_jobs: int, tree_method: str) -> np.ndarray:
    params = dict(PARAMS)
    params["tree_method"] = tree_method
    clf = XGBClassifier(**params, n_jobs=n_jobs)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def oof_scores(X, y, fold, n_jobs: int, tree_method: str) -> np.ndarray:
    """Out-of-fold scores under the released five-fold assignment."""
    oof = np.zeros(len(y), dtype=float)
    for k in range(int(fold.max()) + 1):
        tr, te = fold != k, fold == k
        oof[te] = fit_predict(X[tr], y[tr], X[te], n_jobs, tree_method)
    return oof


def load_fusion_table(features_csv: str, folds_csv: str, d3_csv: str) -> pd.DataFrame:
    df = load_tables(features_csv, folds_csv)
    if "fold" not in df.columns:
        raise ValueError("fusion_eval.py needs the released fold assignment")
    d3 = pd.read_csv(d3_csv)
    for c in ("file", "d3_score"):
        if c not in d3.columns:
            raise ValueError(f"{d3_csv}: missing column '{c}'")
    if d3["file"].duplicated().any():
        raise ValueError("duplicate 'file' values in the D3 score file")
    df = df.merge(d3[["file", "d3_score"]], on="file", how="left", validate="one_to_one")
    if df["d3_score"].isna().any():
        raise ValueError("some clips have no D3 score")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--features", default="data/features_4d.csv")
    ap.add_argument("--folds", default="data/fold_assignment.csv")
    ap.add_argument("--d3", default="data/d3_scores.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument(
        "--tree-method",
        choices=["exact", "hist"],
        default="exact",
        help="split-finding method; 'hist' is what the printed Table 9 used",
    )
    args = ap.parse_args()

    df = load_fusion_table(args.features, args.folds, args.d3)
    y = df["label"].to_numpy(dtype=int)
    gen = df["generator"].astype(str).to_numpy()
    fold = df["fold"].to_numpy(dtype=int)
    generators = sorted(set(gen) - {"Real"})
    os.makedirs(args.out, exist_ok=True)

    rows, per_generator = [], {}
    for label, cols, training, needs_model in ROWS:
        X = df[cols].to_numpy(dtype=float)
        scores = (
            oof_scores(X, y, fold, args.n_jobs, args.tree_method)
            if needs_model
            else X[:, 0]
        )
        pooled = float(roc_auc_score(y, scores))
        pg = {
            g: float(roc_auc_score(y[(gen == g) | (y == 1)], scores[(gen == g) | (y == 1)]))
            for g in generators
        }
        macro = float(np.mean(list(pg.values())))
        rows.append(
            {
                "feature_set": label,
                "dimensions": len(cols),
                "training": training,
                "pooled_auroc": round(pooled, 4),
                "macro_auroc": round(macro, 4),
            }
        )
        per_generator[label] = {g: round(v, 4) for g, v in pg.items()}
        print(f"{label:26s} {len(cols)}-D  pooled {pooled:.4f}  macro {macro:.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(args.out, "fusion.csv"), index=False)
    payload = {
        "rows": rows,
        "per_generator": per_generator,
        "n_videos": int(len(y)),
        "split_source": args.folds,
        "tree_method": args.tree_method,
        "xgboost_params": PARAMS,
        "note": (
            "Table 9 prints B/C/D as macro (mean per-generator) values and A as the "
            "pooled value of the TF-only reference; both columns are written here."
        ),
    }
    with open(os.path.join(args.out, "fusion.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"written -> {args.out}/fusion.csv, {args.out}/fusion.json")


if __name__ == "__main__":
    main()
