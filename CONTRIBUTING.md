# Contributing

This repository exists to support a specific manuscript, so the bar for a change
is "does it keep the published numbers reproducible", not "is it a nice feature".

## Reporting a number that does not reproduce

Open an issue with:

1. the output of `python scripts/verify_release.py`;
2. the output of
   ```bash
   python -c "import platform, numpy, pandas, sklearn, xgboost; print(platform.python_version(), numpy.__version__, pandas.__version__, sklearn.__version__, xgboost.__version__)"
   ```
3. the value you expected and the value you got.

Most drift traces to one of three causes, all documented in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): an XGBoost version whose
default `tree_method` is `hist` rather than `exact`, a different fold split, or
the LOO aggregation order (mean of fold AUROCs versus pooling first).

## Before opening a pull request

```bash
python -m pytest -q          # 15 tests, about 25 s
python scripts/verify_release.py
ruff check src scripts tests
```

All three must pass. A pull request that changes a published number is only
accepted together with a note in `CHANGELOG.md` and an update to
`results/reference/`, because those files are the contract with the paper.

## Rules that are not negotiable

- **Do not add raw videos, checkpoints or flow caches.** `*.mp4`, `*.npz`,
  `*.pth` are ignored on purpose; redistributing them would break the licences of
  GenVideo, MSR-VTT and the third-party detectors (see [`NOTICE.md`](NOTICE.md)).
- **Do not change `data/features_4d.csv` or `data/fold_assignment.csv`.** They
  are the inputs of record. If a feature definition changes, ship a new column
  and a new table, not an edited one.
- **Do not reorder the rows of `features_4d.csv`.** The fold assignment is
  derived from that row order; reordering silently changes every split.
- **Pin versions you rely on.** A dependency bump that moves an AUROC by more
  than 0.005 is a behavioural change, not a maintenance update.

## Style

`ruff` with the configuration in `pyproject.toml` (line length 100). Scripts in
`src/` stay runnable as `python src/<name>.py` from the repository root — that is
why they import each other as top-level modules rather than as a package.
