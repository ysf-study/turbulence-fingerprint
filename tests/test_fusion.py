"""Table 9: the D3 scalar, the capacity control, and the fused model.

The slow test refits the models (about 20 s on four cores).  The fast test only
recomputes AUROCs from the released scores, which is what a reviewer needs.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score


def _macro_auroc(df: pd.DataFrame, score_col: str) -> float:
    """Mean over generators of AUROC(that generator's AI clips vs all real clips)."""
    y = df["label"].to_numpy(dtype=int)
    s = df[score_col].to_numpy(dtype=float)
    gen = df["generator"].astype(str).to_numpy()
    gens = sorted(set(gen) - {"Real"})
    return float(
        np.mean([roc_auc_score(y[(gen == g) | (y == 1)], s[(gen == g) | (y == 1)]) for g in gens])
    )


def test_d3_only_row_is_exactly_reproducible(data_dir: str) -> None:
    """Row B needs no training: the released scores alone give 0.9730 macro."""
    df = pd.read_csv(os.path.join(data_dir, "d3_scores.csv"))
    assert _macro_auroc(df, "d3_score") == pytest.approx(0.9730, abs=5e-4)
    assert float(roc_auc_score(df["label"], df["d3_score"])) == pytest.approx(0.9724, abs=5e-4)


def test_fusion_reference_rows_match_the_paper(reference_dir: str) -> None:
    fusion = pd.read_csv(os.path.join(reference_dir, "fusion.csv")).set_index("feature_set")
    assert len(fusion) == 4
    assert fusion.loc["B: D3 only", "macro_auroc"] == pytest.approx(0.9730, abs=5e-4)
    assert fusion.loc["C: D3 only (control)", "macro_auroc"] == pytest.approx(0.9776, abs=2e-3)
    assert fusion.loc["D: TF + D3 (this work)", "macro_auroc"] == pytest.approx(0.9840, abs=2e-3)


@pytest.mark.slow
def test_fusion_script_reruns_to_the_reference(root: str, tmp_path) -> None:
    proc = subprocess.run(
        [sys.executable, os.path.join(root, "src", "fusion_eval.py"), "--out", str(tmp_path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    fresh = pd.read_csv(tmp_path / "fusion.csv")
    ref = pd.read_csv(os.path.join(root, "results", "reference", "fusion.csv"))
    assert list(fresh["feature_set"]) == list(ref["feature_set"])
    for col in ("pooled_auroc", "macro_auroc"):
        assert np.allclose(fresh[col], ref[col], atol=1e-4), col
