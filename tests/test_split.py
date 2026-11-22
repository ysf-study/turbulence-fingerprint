"""The released split must be the one sklearn produces from the seed."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _frames(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(os.path.join(data_dir, "features_4d.csv")),
        pd.read_csv(os.path.join(data_dir, "fold_assignment.csv")),
    )


def test_fold_assignment_is_stratified_kfold_seed_42(data_dir: str) -> None:
    feats, folds = _frames(data_dir)
    merged = feats.merge(folds, on="file", suffixes=("", "_f"), validate="one_to_one")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rec = np.full(len(merged), -1, dtype=int)
    y = merged["label"].to_numpy(dtype=int)
    for k, (_, test) in enumerate(skf.split(np.zeros(len(merged)), y)):
        rec[test] = k
    assert (rec == merged["fold"].to_numpy(dtype=int)).all()


def test_folds_are_label_stratified_not_generator_stratified(data_dir: str) -> None:
    """97 real + 230 AI per fold; the generator mix is free to vary."""
    feats, folds = _frames(data_dir)
    merged = feats.merge(folds, on="file", suffixes=("", "_f"), validate="one_to_one")
    for k in range(5):
        sub = merged[merged["fold"] == k]
        assert int((sub["label"] == 1).sum()) == 97
        assert int((sub["label"] == 0).sum()) == 230
    # a generator must not be confined to a single fold, or LOO would be untestable
    assert (merged.groupby("generator")["fold"].nunique() >= 3).all()


def test_generator_and_label_agree_between_tables(data_dir: str) -> None:
    feats, folds = _frames(data_dir)
    merged = feats.merge(folds, on="file", suffixes=("", "_f"), validate="one_to_one")
    assert (merged["generator"] == merged["generator_f"]).all()
    assert (merged["label"] == merged["label_f"]).all()
