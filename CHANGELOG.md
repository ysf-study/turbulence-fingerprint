# Changelog

All notable changes to this reproducibility package. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version numbers
track the DOI-bearing releases, not the manuscript revisions: `v1.0.0` is the
version reviewed with the submission, and any later tag is additive.

## [1.1.0] — 2026-11-18

Adds the Table 9 fusion artifacts and the engineering scaffolding a public
repository is expected to carry. No number that existed in 1.0.0 changes.

### Added

- `data/d3_scores.csv` — 1,635 per-clip D3 scores, so Table 9 is no longer
  "protocol only".
- `src/fusion_eval.py` — regenerates rows B (D3 only), A (TF only), C (capacity
  control) and D (TF + D3) of Table 9, reporting both the pooled and the
  per-generator-mean aggregation.
- `results/reference/fusion.csv`, `results/reference/fusion.json` — the
  reference fusion run.
- `data/nsgvd_per_generator.csv` — the per-generator NSG-VD values plotted in
  Figure 5(b), transcribed so the figure can be redrawn and the 0.442–0.753
  range checked.
- `tests/` — 15 pytest tests over release integrity, the split, the feature
  extractor and the fusion rows.
- `.github/workflows/ci.yml`, `pyproject.toml`, `requirements-dev.txt`,
  `requirements-features.txt`, `environment.yml`, `Dockerfile` — CI, tooling and
  container entry points. OpenCV lives in the separate
  `requirements-features.txt` so that a missing wheel cannot fail the whole test
  job; the five feature tests skip cleanly instead.
- `.zenodo.json` — deposit metadata for the DOI release.
- `AUTHORS.md`, `CONTRIBUTING.md`, `NOTICE.md`, `CHANGELOG.md`,
  `docs/REVIEWER_GUIDE.md`, `docs/PIPELINE.md`.

### Changed

- `scripts/verify_release.py` now runs 26 checks instead of 21: the D3 score
  file, its clip set, the D3-only AUROC and the three Table 9 rows are checked
  too.
- `README.md` documents the test suite, the container route, and the Table 9 row.
- `Makefile` gains `test`, `lint`, `fusion` and `docker` targets.

### Documented

- Table 9's fusion models were fitted before `tree_method="exact"` was pinned in
  the main pipeline, so they used the XGBoost 2.x default, `hist`. The D3-only
  control comes out at 0.9771/0.9777 under `hist` against 0.9770/0.9776 printed,
  and the fused row at 0.9838/0.9843 against 0.9835/0.9840; under `exact` the two
  rows move by at most 0.0015 in the other direction. Row B is unaffected because
  it involves no training. The difference changes no ranking and no conclusion,
  and both columns are now tabulated in `docs/TABLES.md`.

## [1.0.0] — 2026-11-18

First release, archived for the submission.

### Added

- `data/features_4d.csv` — four optical-flow statistics for 1,635 clips
  (485 real + 1,150 AI across ten generators).
- `data/fold_assignment.csv` — the five-fold split used in every experiment.
- `data/demamba_scores.csv` — per-clip foundation-model scores (AUROC 0.9989).
- `src/train_eval.py` — pooled out-of-fold AUROC (0.9182), per-generator values,
  leave-one-generator-out (0.9166).
- `src/ablation.py` — the eleven feature sets of Table 4.
- `src/extract_features.py`, `src/feature_utils.py` — the extraction pipeline,
  for extending the table with new videos.
- `scripts/verify_release.py` (21 checks), `scripts/reproduce_all.sh`.
- `docs/REPRODUCIBILITY.md`, `docs/TABLES.md`, `docs/PUBLISHING.md`.
