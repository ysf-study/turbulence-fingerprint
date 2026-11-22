# Paper tables and figures → what this package can regenerate

Status legend: **runs** = script here regenerates it; **protocol only** = the
procedure is described in the paper and Supplementary Material, but the inputs are
third-party and not redistributed.

| # | Table / figure | Status | What it needs | Where |
|---|---|---|---|---|
| T1 | Flatness evolution under recursive averaging | protocol only | Gaussianized simulation, ~10⁴ realizations | §4.4; simulation code not included |
| T2 | Levels of evidence | — | conceptual table | — |
| T3 | Per-generator five-fold AUROC | **runs** | `features_4d.csv` + `fold_assignment.csv` | `src/train_eval.py` → `per_generator.csv` |
| T4 | Feature ablation, 11 sets | **runs** | same | `src/ablation.py` → `ablation.csv` |
| T5 | Leave-one-generator-out AUROC | **runs** | same | `src/train_eval.py` → `per_generator.csv` |
| T6 | Compression / resolution sensitivity | protocol only | re-extraction from re-encoded videos | `src/extract_features.py` provides the extractor; the re-encoding grid is in Supplementary A.6 |
| T7 | Attacks and defenses | protocol only | attack implementations (Square Attack and the black-box adaptive attack) | Supplementary A.7–A.8 |
| T8 | Comparison with published methods | protocol only | third-party checkpoints (D3, NSG-VD, DeMamba) | Supplementary A.1–A.4 |
| T9 | Fusion with D3 | **runs** | `features_4d.csv` + `fold_assignment.csv` + `d3_scores.csv` | `src/fusion_eval.py` → `fusion.csv` |
| T10 | Three physical paradigms | — | conceptual table | — |
| F1 | Pipeline overview | — | schematic | `docs/PIPELINE.md` redraws it |
| F2 | Gaussianization simulation | protocol only | as T1 | — |
| F3 | Main results | **runs** | `per_generator.csv` (T3 values) | any plotting tool |
| F4 | Compression / resolution | protocol only | as T6 | — |
| F5 | Fusion with NSG-VD | **runs** (from reported values) | `nsgvd_per_generator.csv` | any plotting tool; panel (a) is protocol only |

## Verifying the headline numbers without training

`scripts/verify_release.py` recomputes, from the released per-clip scores:

- pooled out-of-fold AUROC = 0.9182 (paper 0.918), from
  `results/reference/scores_oof.csv`;
- the foundation-model baseline AUROC = 0.9989 (paper 0.9989), from
  `data/demamba_scores.csv`;
- the D3-only AUROC = 0.9730 (paper 0.9730), from `data/d3_scores.csv`;
- the three Table 9 rows that involve a learner, from
  `results/reference/fusion.csv`;
- the 485 / 1,150 split, the per-generator counts, the 97 / 230 composition of
  each fold, and that all four tables cover the same clip identifiers;
- that `fold_assignment.csv` is exactly `StratifiedKFold(5, seed=42)`.

All of these are independent of XGBoost and complete in about five seconds; the
fusion rows are read from the reference file rather than refitted.

## Table 9, three ways

Table 9 prints B, C and D as means over the ten generators and A as a pooled
value. The fusion models were fitted before `tree_method="exact"` was pinned in
the main pipeline, so they used the XGBoost 2.x default `hist`. Both paths are
reproducible:

| Row | Paper | `--tree-method exact` (package default) | `--tree-method hist` (paper's setting) |
|---|---|---|---|
| B: D3 only (no training) | 0.9730 | 0.9730 | 0.9730 |
| A: TF only (pooled; macro in brackets) | 0.9178 | 0.9182 (0.9190) | 0.9189 (0.9197) |
| C: D3 only + XGBoost | 0.9776 | 0.9761 | 0.9777 |
| D: TF + D3 | 0.9840 | 0.9838 | 0.9843 |

`hist` reproduces the printed C row to 0.0001 and the printed D row to 0.0003;
`exact` is the setting the rest of the paper uses. The gap never exceeds 0.0015
and changes no ordering, but a reviewer comparing the repository with Table 9
will notice it, so it is stated here rather than hidden. If the table is revised,
the cleanest fix is to re-fit the fusion models with `exact` and print those
values; nothing else in the manuscript depends on the third decimal of Table 9.

## Per-generator values, side by side

| Generator | Paper (Table 3) | This package |
|---|---:|---:|
| Crafter | 0.923 | 0.9232 |
| Gen2 | 0.958 | 0.9557 |
| HotShot | 0.946 | 0.9481 |
| Lavie | 0.864 | 0.8625 |
| ModelScope | 0.889 | 0.8870 |
| MoonValley | 0.978 | 0.9785 |
| MorphStudio | 0.882 | 0.8849 |
| Show-1 | 0.941 | 0.9414 |
| Sora | 0.911 | 0.9119 |
| WildScrape | 0.897 | 0.8970 |
| **mean** | **0.919** | **0.9190** |

## Ablation, side by side

| Feature set | Paper (Table 4) | This package |
|---|---:|---:|
| All 4 dimensions | 0.9188 ± 0.0046 | 0.9186 ± 0.0084 |
| F_low only | 0.8039 ± 0.0146 | 0.7991 ± 0.0155 |
| omega_max only | 0.8628 ± 0.0233 | 0.8631 ± 0.0258 |
| max temporal only | 0.8689 ± 0.0225 | 0.8676 ± 0.0242 |
| p99.9 temporal only | 0.8776 ± 0.0168 | 0.8781 ± 0.0192 |
| Without F_low | 0.8837 ± 0.0163 | 0.8818 ± 0.0173 |
| Without omega_max | 0.9159 ± 0.0052 | 0.9171 ± 0.0074 |
| Without max temporal | 0.9168 ± 0.0061 | 0.9175 ± 0.0089 |
| Without p99.9 temporal | 0.9029 ± 0.0081 | 0.9034 ± 0.0119 |
| Spatial subset | 0.8974 ± 0.0130 | 0.8971 ± 0.0145 |
| Temporal subset | 0.8816 ± 0.0096 | 0.8789 ± 0.0138 |

Mean absolute difference: 0.0018; maximum 0.0048 (F_low only). The ordering of the
eleven rows is identical, which is what the argument in §5.2.2 relies on. The SD
column is systematically wider here because it is computed on the released split;
see `docs/REPRODUCIBILITY.md`.
