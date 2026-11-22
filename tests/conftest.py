"""Shared fixtures.

The scripts in ``src/`` are written to be run from the repository root
(``python src/train_eval.py``), so they import each other as top-level modules.
The tests do the same by putting ``src/`` on ``sys.path``.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture(scope="session")
def root() -> str:
    return ROOT


@pytest.fixture(scope="session")
def data_dir(root: str) -> str:
    return os.path.join(root, "data")


@pytest.fixture(scope="session")
def reference_dir(root: str) -> str:
    return os.path.join(root, "results", "reference")
