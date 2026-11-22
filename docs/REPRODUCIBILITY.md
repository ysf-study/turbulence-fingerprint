# Reproducibility notes

## Environment

| Component | Version used for the reference run |
|---|---|
| Python | 3.13.12 |
| numpy | 2.5.5 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| xgboost | 2.1.4 |

`requirements.txt` pins these. Python 3.10+ is expected to work; the numbers below
were produced on the versions above.

## Evaluation set

The GenVideo cross-generator split has 10 generators (Crafter, Gen2, HotShot,
Lavie, ModelScope, MoonValley, MorphStudio, Show-1, Sora, WildScrape); real clips
come from MSR-VTT.

- All videos standardized to 256 × 256, 8 fps, CRF 23.
- Near-static clips removed by a motion-magnitude filter (mean optical flow > 0.1).
- With a fixed seed (42), 1,906 candidate clips were drawn by stratified sampling;
  1,635 survive the filter — 485 real + 1,150 AI, a retention of 85.8 %.
- 56–140 clips per generator (`data/README.md` lists the counts).

## Split

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the class label,
applied to the row order of `features_4d.csv`. Stratification is on the label only,
not on the generator, so the real/AI ratio is constant across folds (97 / 230) while
the generator mix varies. `scripts/verify_release.py` regenerates the split and
asserts a 100 % match with the released `fold_assignment.csv`.

Every five-fold value in the paper uses this one assignment.

## Model

`xgboost.XGBClassifier` with

```python
n_estimators=200, max_depth=4, learning_rate=0.05,
subsample=0.8, random_state=42, eval_metric="logloss",
tree_method="exact"
```

Four inputs, no scaling, no feature engineering. `tree_method` is stated explicitly:
XGBoost changed its default from `exact` to `hist` in 2.0, and on this data set the
choice moves the pooled AUROC from 0.9182 to 0.9189.

The one exception is the fusion run behind Table 9, which was fitted before that
pin existed and so used `hist`. `src/fusion_eval.py` defaults to `exact` for
consistency and takes `--tree-method hist` to sit next to the printed values;
both columns are tabulated in [`TABLES.md`](TABLES.md).

## Metrics

| Name | Definition |
|---|---|
| **pooled OOF AUROC** | each clip scored once by the model of the fold it is not in; AUROC over all 1,635 scores. This is the 0.918 quoted throughout the paper. |
| **mean over folds** | AUROC computed inside each fold's test set, then averaged over the 5 folds; the SD is over those 5 values. This is the protocol of Table 4. |
| **per-generator five-fold** | AUROC between the AI clips of one generator and all 485 real clips, all out of fold. |
| **LOO** | for held-out generator *g* and real fold *k*: train on (all AI generators except *g*) + (real clips outside fold *k*), test on (*g*'s AI clips) + (real clips in fold *k*); average over the 5 values of *k*, then over the 10 generators. |

## Determinism and expected drift

Given the fold file and the pinned versions, `train_eval.py` is deterministic.
Three sources of drift remain:

1. **XGBoost version** — `hist` versus `exact`, and library-internal changes between
   2.x and 3.x, move the third decimal (±0.003).
2. **Split choice** — a different random split of the same data gives a pooled
   AUROC within ±0.005 and a fold-to-fold SD between 0.008 and 0.020; the SD of
   0.0046 quoted in Table 4 is specific to the split used there.
3. **LOO aggregation** — mean of fold AUROCs (0.9166) versus pooling the folds
   before computing each generator's AUROC (0.9161). Both are defensible; the paper
   used the former.

`scripts/verify_release.py` uses a tolerance of 0.005 on both headline AUROCs and
fails if either drifts further.

## Runtime

| Command | Time | Output |
|---|---|---|
| `scripts/verify_release.py` | ~5 s | 26 pass/fail lines |
| `src/train_eval.py` | ~60 s (5 + 50 model fits) | `metrics.json`, `per_generator.csv`, `scores_oof.csv`, `scores_loo.csv` |
| `src/ablation.py` | ~15 s (55 fits) | `ablation.csv`, `ablation.json` |
| `src/fusion_eval.py` | ~20 s (15 fits) | `fusion.csv`, `fusion.json` |
| `pytest -q` | ~25 s | 15 tests, 5 of them skipped without OpenCV |

Measured on 4 CPU cores; no GPU.

The five skipped tests cover `src/feature_utils.py` (Helmholtz projection, flatness,
temporal maxima) and need `cv2`; `pip install -r requirements-features.txt` enables
them. They are kept out of `requirements-dev.txt` on purpose, so that a missing
OpenCV wheel on some interpreter cannot stop the other ten tests from reporting.
No number in the paper depends on them.

## Regenerating the features

Only needed if you want to extend the table or verify the extraction. Requires the
videos (not distributed) and, for RAFT, the RAFT repository plus `raft-things.pth`:

```bash
python src/extract_features.py --videos /data/crafter --label 0 \
    --generator Crafter --backend raft \
    --raft-root /path/to/RAFT --weights /path/to/raft-things.pth --out crafter.csv
```

`--backend farneback` runs without a GPU and without weights, and is a cheaper
substitute if you only need the flow pipeline end to end.

Frame sampling is the detail most likely to bite: 20 frames are drawn uniformly at
random with `RandomState(42)` and then sorted. Taking the first 20 frames instead
gives visibly different statistics.
