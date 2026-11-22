# -*- coding: utf-8 -*-
"""Compare a fresh reproduction against the released reference outputs.

The CI environment can differ in the Python patch version and in the last
floating-point digits while reproducing every printed number.  Byte-for-byte
`diff` is therefore too strict for `metrics.json`; this script compares the
scientific values with a tolerance and ignores the `environment` metadata block.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ATOL = 1e-6
RTOL = 1e-6


def compare_numbers(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise AssertionError(f"{path}: key mismatch {set(a) ^ set(b)}")
        for k in a:
            compare_numbers(a[k], b[k], f"{path}.{k}" if path else k)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise AssertionError(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare_numbers(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not np.isclose(float(a), float(b), atol=ATOL, rtol=RTOL):
            raise AssertionError(f"{path}: {a} != {b}")
    else:
        if a != b:
            raise AssertionError(f"{path}: {a!r} != {b!r}")


def compare_metrics(fresh: Path, ref: Path) -> None:
    a = json.loads((fresh / "metrics.json").read_text(encoding="utf-8"))
    b = json.loads((ref / "metrics.json").read_text(encoding="utf-8"))
    a.pop("environment", None)
    b.pop("environment", None)
    compare_numbers(a, b, "metrics.json")


def compare_table(fresh: Path, ref: Path, name: str) -> None:
    a = pd.read_csv(fresh / name)
    b = pd.read_csv(ref / name)
    if list(a.columns) != list(b.columns):
        raise AssertionError(f"{name}: columns differ")
    if len(a) != len(b):
        raise AssertionError(f"{name}: rows {len(a)} != {len(b)}")
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            if not np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), atol=ATOL, rtol=RTOL):
                raise AssertionError(f"{name}.{col}: numeric values differ")
        else:
            if not (a[col].astype(str).to_numpy() == b[col].astype(str).to_numpy()).all():
                raise AssertionError(f"{name}.{col}: string values differ")


def compare_analysis_json(fresh: Path, ref: Path, name: str) -> None:
    a = json.loads((fresh / name).read_text(encoding="utf-8"))
    b = json.loads((ref / name).read_text(encoding="utf-8"))
    compare_numbers(a, b, name)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare_reference.py FRESH_DIR REFERENCE_DIR", file=sys.stderr)
        return 2
    fresh, ref = Path(sys.argv[1]), Path(sys.argv[2])
    compare_metrics(fresh, ref)
    for name in ["per_generator.csv", "ablation.csv", "fusion.csv"]:
        compare_table(fresh, ref, name)
    for name in ["ablation.json", "fusion.json"]:
        compare_analysis_json(fresh, ref, name)
    print("Reference outputs match within atol/rtol = 1e-6")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"reference mismatch: {exc}", file=sys.stderr)
        raise SystemExit(1)
