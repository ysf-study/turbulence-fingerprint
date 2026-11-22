# Data files

Five tables, all keyed on `file` and all covering the same 1,635 clips — except
`nsgvd_per_generator.csv`, which holds ten summary values. Checksums are in
[`SHA256SUMS`](SHA256SUMS); `scripts/verify_release.py` checks them.

## `features_4d.csv`

One row per clip.

| Column | Meaning |
|---|---|
| `file` | clip identifier — the clip stem plus the `.npz` suffix of the cached flow file (`Crafter_1003.npz` ⇄ `Crafter_1003.mp4` in GenVideo) |
| `F_low` | mean flatness over scales ℓ = 1, 2, 3, computed on the solenoidal flow component |
| `omega_max` | maximum vorticity magnitude over the sampled frames |
| `max_temporal` | maximum temporal flow increment between consecutive sampled frames |
| `p999_temporal` | 99.9th percentile of the temporal flow increment |
| `label` | 1 = real, 0 = AI-generated |
| `generator` | `Real`, `Crafter`, `Gen2`, `HotShot`, `Lavie`, `ModelScope`, `MoonValley`, `MorphStudio`, `Show-1`, `Sora`, `WildScrape` |

Counts:

| Class | Clips |
|---|---:|
| Real (MSR-VTT) | 485 |
| Crafter | 108 |
| Gen2 | 126 |
| HotShot | 131 |
| Lavie | 115 |
| ModelScope | 140 |
| MoonValley | 112 |
| MorphStudio | 126 |
| Show-1 | 98 |
| Sora | 56 |
| WildScrape | 138 |
| **AI total** | **1,150** |
| **Total** | **1,635** |

How the values were produced (`src/feature_utils.py`, `src/extract_features.py`):

1. Video resized to 256 × 256, 8 fps, CRF 23.
2. **Twenty frames drawn uniformly at random** with `numpy.random.RandomState(42)`
   and then sorted, not the first twenty frames. This matters: the two choices
   give different statistics for the same clip.
3. Dense optical flow between consecutive frames with RAFT (`raft-things.pth`).
4. Global camera motion removed with a RANSAC homography fitted on a 500-point
   grid (about 22 × 22 points).
5. Helmholtz decomposition; only the solenoidal component is kept.
6. The four statistics computed as defined in Section 3 of the paper.

## `fold_assignment.csv`

| Column | Meaning |
|---|---|
| `file` | clip identifier |
| `generator` | generator label (duplicated from the feature table for convenience) |
| `label` | 1 = real, 0 = AI |
| `fold` | 0–4 |

Derived with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the
label, applied to the row order of `features_4d.csv`. `scripts/verify_release.py`
recomputes it and asserts a 100 % match, so this file can be regenerated from the
feature table and the seed alone. Each fold holds 97 of the 485 real clips and 230
of the 1,150 AI clips; the split is stratified on the class label only, not on the
generator.

## `demamba_scores.csv`

Per-clip output of the DeMamba pipeline, run under the same five-fold split with
`microsoft/xclip-base-patch16` weights as described in Supplementary A.14.
Higher score = more real.

| Column | Meaning |
|---|---|
| `file` | clip identifier |
| `generator`, `label` | as above |
| `fold` | the fold whose model produced the score (out-of-fold, so each clip is scored once) |
| `demamba_score` | real-class score |

`roc_auc_score(label, demamba_score)` over all 1,635 clips is **0.9989**, the
value quoted in Section 6.1 of the paper. These scores are provided so that the
baseline statement can be checked without redistributing a third-party checkpoint.

## `d3_scores.csv`

Per-clip output of the D3 detector (Zheng et al., ICCV 2025, arXiv:2508.00701),
reimplemented as described in Section 5.6: 16 frames per clip, 10 % centre crop,
X-CLIP frame encoder, score = standard deviation of the second difference of the
frame embeddings. Higher score = more real.

| Column | Meaning |
|---|---|
| `file` | clip identifier |
| `generator`, `label` | as above |
| `fold` | the released fold (kept for convenience; D3 itself needs no training) |
| `d3_score` | the D3 scalar |

This is the input of Table 9 row B. Its AUROC is **0.9730** as the mean over the
ten generators and 0.9724 pooled over the 1,635 clips; `verify_release.py` checks
the first of these. `src/fusion_eval.py` uses the same column for rows C and D.

## `nsgvd_per_generator.csv`

The ten values plotted in Figure 5(b) of the paper: AUROC of the NSG-VD detector
under the reduced-scale sub-protocol (300 real + 50 AI per generator, about 800
clips, SEINE checkpoint applied out of domain).

| Column | Meaning |
|---|---|
| `generator` | generator name |
| `auroc_nsgvd` | AUROC of that generator's AI clips against the real clips of the sub-protocol |

Mean 0.605, minimum 0.442 (Sora, below chance), maximum 0.753 (WildScrape). These
are transcribed from the run reported in Section 5.6.2, not recomputed here: the
detector is third-party and neither its code nor its checkpoint is redistributed.
They are included so that Figure 5(b) can be redrawn and the range quoted in the
text can be checked against a file.

## What is not here

The raw videos. GenVideo and MSR-VTT are third-party benchmarks with their own
licences; neither is redistributed, and the clip identifiers above let you match
rows to your own copies. `src/extract_features.py` recomputes the four statistics
from a directory of videos if you need to extend the table.

Also absent: the DeMamba, D3 and NSG-VD checkpoints and their code. Only the
per-clip scores those detectors produced are shipped, and those files remain
subject to the upstream terms — see [`../NOTICE.md`](../NOTICE.md).
