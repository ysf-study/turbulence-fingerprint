#!/usr/bin/env python3
"""Five-fold and leave-one-generator-out (LOO) evaluation of the TF features.

Both protocols read the *released* fold assignment (``data/fold_assignment.csv``)
instead of re-deriving it, so the folds used here are the folds used in the
paper.  If that file is absent the folds are regenerated with
``StratifiedKFold(5, shuffle=True, random_state=42)``, which reproduces the
released assignment exactly (checked by ``scripts/verify_release.py``).

Protocols
---------
1. Five-fold, pooled out-of-fold (OOF).  Each clip is scored once, by a model
   that did not see it; AUROC is computed over all 1,635 scores.
2. Per-generator five-fold.  For generator g, AUROC is computed between the AI
   clips of g and all 485 real clips, both scored out of fold.
3. Leave-one-generator-out.  For held-out generator g and real-video fold k,
   train on (all AI generators except g) + (real clips not in fold k), test on
   (AI clips of g) + (real clips in fold k).  The reported value is the mean
   over the 10 generators of the mean over the 5 real folds.

Outputs (under --out)
---------------------
metrics.json        headline numbers, per-generator values, environment
per_generator.csv   generator, n_ai, auroc_5fold, auroc_loo
scores_oof.csv      file, generator, label, fold, score      (one row per clip)
scores_loo.csv      heldout, fold, file, generator, label, score
"""
from __future__ import annotations

import argparse
import json
import os
import platform

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

FEATURES = ["F_low", "omega_max", "max_temporal", "p999_temporal"]
PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    eval_metric="logloss",
    # 'exact' is the split-finding method XGBoost used by default for a data set
    # this small before 2.0; it is stated explicitly so that the numbers do not
    # depend on the library's default.  'hist' changes the third decimal.
    tree_method="exact",
)


def load_tables(features_csv: str, folds_csv: str | None) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    missing = [c for c in FEATURES + ["file", "label", "generator"] if c not in df.columns]
    if missing:
        raise ValueError(f"{features_csv}: missing columns {missing}")
    if df["file"].duplicated().any():
        raise ValueError("duplicate 'file' values in the feature table")

    if folds_csv and os.path.isfile(folds_csv):
        fa = pd.read_csv(folds_csv)
        for c in ("file", "fold"):
            if c not in fa.columns:
                raise ValueError(f"{folds_csv}: missing column '{c}'")
        if len(fa) != len(df):
            raise ValueError(
                f"fold assignment has {len(fa)} rows, the feature table has {len(df)}"
            )
        df = df.merge(fa[["file", "fold"]], on="file", how="left", validate="one_to_one")
        if df["fold"].isna().any():
            raise ValueError("some clips have no fold in the assignment file")
        df["fold"] = df["fold"].astype(int)
        if sorted(df["fold"].unique()) != list(range(df["fold"].nunique())):
            raise ValueError("folds must be labelled 0..K-1")

    y = df["label"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("labels must be 0/1")
    return df


def fit_predict(X_tr, y_tr, X_te, n_jobs: int) -> np.ndarray:
    clf = XGBClassifier(**PARAMS, n_jobs=n_jobs)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--features", default="data/features_4d.csv")
    ap.add_argument(
        "--folds",
        default="data/fold_assignment.csv",
        help="released fold assignment; pass '' to regenerate the split",
    )
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--skip-loo", action="store_true", help="skip leave-one-generator-out")
    args = ap.parse_args()

    df = load_tables(args.features, args.folds)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    gen = df["generator"].astype(str).to_numpy()
    os.makedirs(args.out, exist_ok=True)

    if "fold" in df.columns:
        fold = df["fold"].to_numpy(dtype=int)
        split_source = args.folds
    else:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        fold = np.full(len(y), -1, dtype=int)
        for k, (_, te) in enumerate(skf.split(X, y)):
            fold[te] = k
        split_source = f"StratifiedKFold(n_splits={args.n_splits}, seed={args.seed})"

    n_folds = int(fold.max()) + 1
    for k in range(n_folds):
        if len(np.unique(y[fold == k])) < 2:
            raise ValueError(f"fold {k} contains a single class")

    # ---------- five-fold, pooled out-of-fold ----------
    oof = np.zeros(len(y), dtype=float)
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        oof[te] = fit_predict(X[tr], y[tr], X[te], args.n_jobs)

    pooled = float(roc_auc_score(y, oof))
    per_fold = [float(roc_auc_score(y[fold == k], oof[fold == k])) for k in range(n_folds)]
    mean_fold = float(np.mean(per_fold))
    sd_fold = float(np.std(per_fold, ddof=1))

    generators = sorted(set(gen) - {"Real"})
    per_gen_5 = {}
    for g in generators:
        m = (gen == g) | (y == 1)
        per_gen_5[g] = float(roc_auc_score(y[m], oof[m]))
    mean_gen_5 = float(np.mean(list(per_gen_5.values())))

    pd.DataFrame(
        {"file": df["file"], "generator": gen, "label": y, "fold": fold, "score": oof}
    ).to_csv(os.path.join(args.out, "scores_oof.csv"), index=False)

    # ---------- leave-one-generator-out ----------
    per_gen_loo, loo_rows = {}, []
    if not args.skip_loo:
        for g in generators:
            aucs = []
            for k in range(n_folds):
                tr = ((gen != g) & (y == 0)) | ((y == 1) & (fold != k))
                te = (gen == g) | ((y == 1) & (fold == k))
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    continue
                idx = np.where(te)[0]
                pred = fit_predict(X[tr], y[tr], X[te], args.n_jobs)
                aucs.append(float(roc_auc_score(y[te], pred)))
                for j, i in enumerate(idx):
                    loo_rows.append(
                        {
                            "heldout": g,
                            "fold": k,
                            "file": df["file"].iloc[i],
                            "generator": gen[i],
                            "label": int(y[i]),
                            "score": float(pred[j]),
                        }
                    )
            if not aucs:
                raise RuntimeError(f"no valid LOO folds for generator {g}")
            per_gen_loo[g] = float(np.mean(aucs))

        loo_mean = float(np.mean(list(per_gen_loo.values())))
        loo_sd = float(np.std(list(per_gen_loo.values()), ddof=1))
        pd.DataFrame(loo_rows).to_csv(os.path.join(args.out, "scores_loo.csv"), index=False)
    else:
        loo_mean = loo_sd = float("nan")

    metrics = {
        "n_videos": int(len(y)),
        "n_real": int((y == 1).sum()),
        "n_ai": int((y == 0).sum()),
        "n_folds": n_folds,
        "split_source": split_source,
        "pooled_5fold_auroc": pooled,
        "mean_per_fold_auroc": mean_fold,
        "sd_per_fold_auroc": sd_fold,
        "per_fold_auroc": per_fold,
        "mean_per_generator_5fold_auroc": mean_gen_5,
        "loo_mean_auroc": loo_mean,
        "loo_sd_over_generators": loo_sd,
        "per_generator_5fold": per_gen_5,
        "per_generator_loo": per_gen_loo,
        "xgboost_params": PARAMS,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(
        [
            {
                "generator": g,
                "n_ai": int((gen == g).sum()),
                "auroc_5fold": per_gen_5[g],
                "auroc_loo": per_gen_loo.get(g, float("nan")),
            }
            for g in generators
        ]
    ).to_csv(os.path.join(args.out, "per_generator.csv"), index=False)

    print(f"clips: {len(y)} ({int((y == 1).sum())} real / {int((y == 0).sum())} AI)")
    print(f"folds: {split_source}")
    print(f"5-fold pooled AUROC       : {pooled:.4f}")
    print(f"5-fold mean over folds    : {mean_fold:.4f} (SD {sd_fold:.4f})")
    print(f"mean per-generator 5-fold : {mean_gen_5:.4f}")
    if not args.skip_loo:
        print(f"LOO mean over generators  : {loo_mean:.4f} (SD {loo_sd:.4f})")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
