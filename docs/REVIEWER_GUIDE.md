# For reviewers: five minutes, no GPU

The manuscript's statistical claims can be checked without downloading a video,
a checkpoint or a GPU. Everything below runs from the repository root on a laptop.

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py     # ~5 s, 26 checks, no training
python -m pytest -q                  # ~25 s, 15 tests
```

`verify_release.py` prints one line per check and exits non-zero if anything has
drifted. A healthy run ends with `26/26 checks passed`.

## Claim → artifact

| Claim in the manuscript | Where to look | Value | Tolerance |
|---|---|---|---|
| Pooled 5-fold out-of-fold AUROC 0.918 | `results/reference/metrics.json`, recomputed by `verify_release.py` | 0.9182 | 0.005 |
| Per-generator AUROC, Table 3 | `results/reference/per_generator.csv` vs `docs/TABLES.md` | 5 of 10 exact to three decimals, rest ≤ 0.003 | — |
| Leave-one-generator-out mean 0.915 | `results/reference/metrics.json` | 0.9166 | 0.005 |
| Feature ablation ordering, Table 4 | `results/reference/ablation.csv` | identical ordering of 11 rows | max Δ 0.0048 |
| Foundation-model baseline 0.9989 | `data/demamba_scores.csv`, recomputed in `verify_release.py` | 0.9989 | 0.001 |
| D3-only row of Table 9, 0.9730 | `data/d3_scores.csv`, recomputed in `verify_release.py` | 0.9730 | 0.0005 |
| Fused TF + D3 row of Table 9, 0.9840 | `results/reference/fusion.csv` | 0.9838 (exact) / 0.9843 (hist) | 0.002 |
| The split is the one stated | `verify_release.py` regenerates `StratifiedKFold(5, seed=42)` | 100 % match | exact |
| 485 real + 1,150 AI, 97/230 per fold | `verify_release.py`, `data/README.md` | as stated | exact |

The fusion rows carry a documented caveat: the models behind Table 9 were fitted
with XGBoost's default `hist` splitter, before `tree_method="exact"` was pinned
elsewhere. `src/fusion_eval.py --tree-method hist` reproduces the printed
Table 9 values to within 0.0003; the default `exact` path reproduces the package's
reference. No ranking and no conclusion depends on the difference. See
[`TABLES.md`](TABLES.md).

## What is deliberately not here

- **Raw videos.** GenVideo and MSR-VTT are third-party benchmarks; only clip
  identifiers are shipped, so the tables can be joined against your own copies.
- **Third-party checkpoints.** DeMamba, D3 and NSG-VD are represented by their
  per-clip scores, not by weights. [`../NOTICE.md`](../NOTICE.md) records what is
  redistributed and under which terms.
- **Runnable attack code (Table 7).** The attacks are described by protocol in
  Supplementary A.7–A.8. Reimplementing them would require the videos.
- **Re-extraction of features from videos (Table 6).** The extractor is included
  (`src/extract_features.py`), the re-encoding grid is in Supplementary A.6, and
  the videos are not redistributable.

[`TABLES.md`](TABLES.md) marks every table and figure as **runs** or
**protocol only**; nothing is claimed to run that does not.

## If a number does not match

Send the failing line, the value you expected, and the output of

```bash
python -c "import platform, numpy, pandas, sklearn, xgboost; print(platform.python_version(), numpy.__version__, pandas.__version__, sklearn.__version__, xgboost.__version__)"
```

The contact address is in [`../AUTHORS.md`](../AUTHORS.md).
