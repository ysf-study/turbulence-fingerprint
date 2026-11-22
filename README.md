# Turbulence Fingerprint — code and derived data

[![ci](https://github.com/ysf-study/turbulence-fingerprint/actions/workflows/ci.yml/badge.svg)](https://github.com/ysf-study/turbulence-fingerprint/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](LICENSE-data)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Reproducibility package for

> **Higher-Order Intermittency in Learned Neural Motion Representations: A Testable
> Statistical Prediction**
> Shufei Yan, Dapeng Gao — submitted to *Neurocomputing*

The package contains the four-dimensional feature table for the 1,635-clip
cross-generator evaluation set, the fold assignment used in every five-fold
experiment of the paper, the per-clip scores behind the headline numbers — the
foundation-model baseline and the D3 detector included — and the code that
regenerates them.

Everything here runs on a laptop in about two minutes; no GPU and no raw videos
are required. Reviewers who want the short version: [`docs/REVIEWER_GUIDE.md`](docs/REVIEWER_GUIDE.md).

---

## Quick start

```bash
python -m pip install -r requirements.txt

# 1. integrity and consistency checks (seconds, no training)
python scripts/verify_release.py

# 2. five-fold + leave-one-generator-out evaluation (~1 min)
python src/train_eval.py --out results

# 3. feature-set ablation, Table 4 (~15 s)
python src/ablation.py --out results

# 4. fusion with D3, Table 9 (~20 s)
python src/fusion_eval.py --out results
```

Or all four in order: `bash scripts/reproduce_all.sh`, `make all`, or

```bash
docker build -t tf-repro . && docker run --rm tf-repro
```

`verify_release.py` prints 26 PASS/FAIL lines and exits non-zero if anything
drifts. The other three scripts write `metrics.json`, `per_generator.csv`,
`ablation.csv`, `fusion.csv` and the per-clip score files under `results/`.

Tests: `python -m pytest -q` (15 tests, about 25 s; five of them exercise the
feature extractor and need OpenCV, so they are skipped unless you first run
`pip install -r requirements-features.txt`).

## What the scripts reproduce

| Paper | Quantity | Script | Released reference |
|---|---|---|---|
| Table 3 | per-generator five-fold AUROC | `src/train_eval.py` | `results/reference/per_generator.csv` |
| §5.2 | pooled out-of-fold AUROC 0.918 | `src/train_eval.py` | `results/reference/metrics.json` |
| Table 4 | 11 feature sets, mean ± SD over folds | `src/ablation.py` | `results/reference/ablation.csv` |
| Table 5 | LOO AUROC per generator, mean 0.915 | `src/train_eval.py` | `results/reference/per_generator.csv` |
| §6.1 | foundation-model baseline 0.9989 | `scripts/verify_release.py` | `data/demamba_scores.csv` |
| Table 9 | D3 only / D3-only control / TF + D3 fusion | `src/fusion_eval.py` | `results/reference/fusion.csv` |
| Figure 5(b) | per-generator NSG-VD AUROC, mean 0.605 | any plotting tool | `data/nsgvd_per_generator.csv` |

Not reproduced here, and why, in [`docs/TABLES.md`](docs/TABLES.md): Table 1–2
(theory), Table 6 (compression / resolution, needs re-extraction from videos),
Table 7 (attacks and defenses), Table 8 (other published detectors), Table 10
(conceptual).

## Expected output

Produced with Python 3.13, XGBoost 2.1.4, scikit-learn 1.9.1, on 2026-11-18:

```
clips: 1635 (485 real / 1150 AI)
5-fold pooled AUROC       : 0.9182      paper: 0.918
5-fold mean over folds    : 0.9186 (SD 0.0084)   paper: 0.9188 ± 0.0046
mean per-generator 5-fold : 0.9190      paper: 0.919
LOO mean over generators  : 0.9166 (SD 0.0372)   paper: 0.915

Table 9                   pooled    macro   paper (macro)
B: D3 only                0.9724    0.9730  0.9730
A: TF only                0.9182    0.9190  0.9178 (pooled)
C: D3 only + XGBoost      0.9755    0.9761  0.9776
D: TF + D3                0.9834    0.9838  0.9840
```

The pooled AUROC and the per-generator values agree with the paper to the third
decimal for 5 of 10 generators and to within 0.003 for the rest. The LOO mean
sits 0.0016 above the published value. Both offsets are discussed under
[Numerical agreement](#numerical-agreement); the fusion row is discussed under
[the Table 9 caveat](#the-table-9-caveat).

Re-running the scripts reproduces `results/reference/` **byte for byte** on the
pinned versions — `metrics.json`, `per_generator.csv`, `ablation.csv`,
`fusion.csv` and the score files all come out identical.

## Repository layout

```
data/
  features_4d.csv       1,635 clips × 4 statistics + label + generator
  fold_assignment.csv   file, generator, label, fold      (the split used in the paper)
  demamba_scores.csv    per-clip foundation-model scores (the 0.9989 baseline)
  d3_scores.csv         per-clip D3 scalars (Table 9)
  nsgvd_per_generator.csv  NSG-VD values plotted in Figure 5(b)
  SHA256SUMS            checksums of the five tables
src/
  train_eval.py         five-fold pooled OOF and leave-one-generator-out
  ablation.py           Table 4 feature-set ablation
  fusion_eval.py        Table 9: D3, capacity control, TF + D3
  extract_features.py   optional: recompute the four statistics from your own videos
  feature_utils.py      optical-flow preprocessing and the four statistics
scripts/
  verify_release.py     integrity + consistency checks (26, no training)
  reproduce_all.sh      the four commands above, in order
tests/                  pytest: release integrity, split, extractor, fusion
results/reference/      outputs of the reference run, for diffing
docs/
  REVIEWER_GUIDE.md     five minutes, no GPU
  REPRODUCIBILITY.md    protocols, counts, tolerances
  TABLES.md             paper table / figure → script → expected value
  PIPELINE.md           data flow and module map
  PUBLISHING.md         GitHub settings, Zenodo DOI, release notes
```

## Data

| File | Rows | Contents |
|---|---:|---|
| `features_4d.csv` | 1,635 | `F_low`, `omega_max`, `max_temporal`, `p999_temporal` |
| `fold_assignment.csv` | 1,635 | fold index 0–4 per clip |
| `demamba_scores.csv` | 1,635 | real-class score from the foundation-model baseline |
| `d3_scores.csv` | 1,635 | D3 scalar (std of the second difference of frame embeddings) |
| `nsgvd_per_generator.csv` | 10 | one AUROC per generator, no per-clip data released |

Clip identifiers are the clip stem with the `.npz` suffix of the cached flow file
(`Crafter_1003.npz` corresponds to `Crafter_1003.mp4` in GenVideo). Real clips are
from MSR-VTT; AI clips are the ten generators of the GenVideo cross-generator
split. Raw videos are **not** redistributed — see [`data/README.md`](data/README.md)
and [`NOTICE.md`](NOTICE.md).

Counts: 485 real + 1,150 AI = 1,635, with 56–140 clips per generator; each fold
holds 97 real and 230 AI clips.

## Numerical agreement

The pipeline is deterministic given the fold assignment and the library versions,
but three things move the last digit:

1. **XGBoost version.** 200 trees, depth 4, learning rate 0.05, subsample 0.8,
   `random_state=42`, `tree_method="exact"`. `exact` is stated explicitly because
   XGBoost switched its default to `hist` in 2.0; with `hist` the pooled AUROC
   moves from 0.9182 to 0.9189 and several per-generator values by 0.002.
2. **The split.** The released `fold_assignment.csv` is exactly what
   `StratifiedKFold(5, shuffle=True, random_state=42)` produces on the released
   row order — `verify_release.py` checks this. A different random split of the
   same data gives pooled AUROC within ±0.005 and a fold-to-fold SD anywhere in
   0.008–0.020, which is why the SD quoted in Table 4 is not reproduced exactly.
3. **LOO aggregation.** The paper reports the mean over generators of the mean
   over the five real folds; that is what `train_eval.py` computes (0.9166). Pooling
   the five folds of a generator before computing its AUROC gives 0.9161. Neither
   reaches 0.9150 to the last digit, and no conclusion depends on the third decimal.

### The Table 9 caveat

Table 9 also mixes two aggregations: rows B, C and D are means over the ten
generators (the Table A.11 convention, and the one the paired tests use), while
row A is the pooled value of the TF-only reference (0.9178; its generator mean is
0.9187). `fusion.csv` writes both columns for every row.

The fusion models were also fitted before `tree_method="exact"` was pinned, so
they used the XGBoost default `hist`. Under `hist` the D3-only control comes out
at 0.9771/0.9777 against 0.9770/0.9776 printed and the fused row at
0.9838/0.9843 against 0.9835/0.9840; under `exact` the same rows move by up to
0.0015 the other way. Row B is unaffected — it involves no training. Reproduce
either column with `src/fusion_eval.py --tree-method hist|exact`; the default is
`exact`, to match the rest of the package. No ranking and no conclusion depends
on the difference.

## Scope

This package supports the *statistical* claims: that four optical-flow statistics
separate real from generated video (0.918), that the separation survives
leave-one-generator-out (0.917), that each of the four dimensions carries
independent signal (Table 4), and that the four add to a third-party detector
(Table 9). It does not ship third-party checkpoints, attack code, or the raw
videos, so the robustness and comparison tables are described by protocol rather
than by runnable code.

## Contributing

Bug reports and corrections are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md).
The short version: `pytest`, `verify_release.py` and `ruff` must pass, and a pull
request that changes a published number must say so in
[`CHANGELOG.md`](CHANGELOG.md).

## License

Code: MIT ([`LICENSE`](LICENSE)). Derived tables (`features_4d.csv`,
`fold_assignment.csv`, `demamba_scores.csv`, `d3_scores.csv`,
`nsgvd_per_generator.csv`): CC BY 4.0 ([`LICENSE-data`](LICENSE-data)). Raw
videos remain under the GenVideo and MSR-VTT licences, and the score files remain
subject to the terms of the detectors they came from — see [`NOTICE.md`](NOTICE.md).

## Citation

See [`CITATION.cff`](CITATION.cff). If you use the feature table, please cite the
paper above; if you use the code, cite the archived release as well.
