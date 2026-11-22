"""The release must be internally consistent before anything is trained."""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pandas as pd


def test_verify_release_passes(root: str) -> None:
    """scripts/verify_release.py is the single source of truth for integrity."""
    proc = subprocess.run(
        [sys.executable, os.path.join(root, "scripts", "verify_release.py")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAIL" not in proc.stdout
    match = re.search(r"(\d+)/(\d+) checks passed", proc.stdout)
    assert match is not None, proc.stdout
    passed, total = int(match.group(1)), int(match.group(2))
    assert passed == total
    assert total >= 26


def test_tables_share_the_same_clip_identifiers(data_dir: str) -> None:
    names = ["features_4d.csv", "fold_assignment.csv", "demamba_scores.csv", "d3_scores.csv"]
    sets = []
    for name in names:
        df = pd.read_csv(os.path.join(data_dir, name))
        assert not df["file"].duplicated().any(), f"{name} has duplicate clip ids"
        sets.append(set(df["file"]))
    assert all(s == sets[0] for s in sets), "the four tables disagree on the clip set"
    assert len(sets[0]) == 1635


def test_checksums_cover_every_table(data_dir: str) -> None:
    listed = {
        line.split(maxsplit=1)[1].lstrip("*").strip()
        for line in open(os.path.join(data_dir, "SHA256SUMS"), encoding="utf-8")
        if line.strip()
    }
    assert listed == {
        "features_4d.csv",
        "fold_assignment.csv",
        "demamba_scores.csv",
        "d3_scores.csv",
        "nsgvd_per_generator.csv",
    }


def test_class_and_generator_counts(data_dir: str) -> None:
    feats = pd.read_csv(os.path.join(data_dir, "features_4d.csv"))
    assert int((feats["label"] == 1).sum()) == 485
    assert int((feats["label"] == 0).sum()) == 1150
    counts = feats["generator"].value_counts().to_dict()
    assert counts.pop("Real") == 485
    assert sum(counts.values()) == 1150
    assert len(counts) == 10
