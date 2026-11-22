# Reference outputs

Committed so that a re-run can be diffed without trusting anyone's screenshot.
Produced on 2026-11-18 with the versions pinned in `requirements.txt`.

| File | Produced by | Contents |
|---|---|---|
| `metrics.json` | `src/train_eval.py` | headline numbers, per-generator values, environment |
| `per_generator.csv` | `src/train_eval.py` | generator, n_ai, AUROC five-fold and LOO |
| `scores_oof.csv` | `src/train_eval.py` | one row per clip: out-of-fold score |
| `scores_loo.csv` | `src/train_eval.py` | one row per (held-out generator, fold, clip) |
| `ablation.csv` | `src/ablation.py` | the eleven feature sets of Table 4 |
| `ablation.json` | `src/ablation.py` | same, with the per-fold values kept |
| `fusion.csv` | `src/fusion_eval.py` | Table 9: rows B, A, C, D with pooled and macro AUROC |
| `fusion.json` | `src/fusion_eval.py` | same, plus the per-generator values behind each macro |

Headline values:

```
pooled_5fold_auroc              0.9182
mean_per_fold_auroc             0.9186   (SD 0.0084)
mean_per_generator_5fold_auroc  0.9190
loo_mean_auroc                  0.9166   (SD 0.0372 over generators)

Table 9 (fusion.csv)            pooled   macro
B: D3 only                      0.9724   0.9730
A: TF only                      0.9182   0.9190
C: D3 only + XGBoost            0.9755   0.9761
D: TF + D3                      0.9834   0.9838
```

To regenerate into `results/` and compare:

```bash
python src/train_eval.py  --out results
python src/ablation.py    --out results
python src/fusion_eval.py --out results
diff results/per_generator.csv results/reference/per_generator.csv
```

A diff confined to the fourth decimal is expected; see
`docs/REPRODUCIBILITY.md` for the three sources of drift, and `docs/TABLES.md`
for why the printed Table 9 sits closest to the `--tree-method hist` column.
