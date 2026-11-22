# Third-party components and provenance

This file records what in the package comes from someone else, so that the MIT
licence on the code and the CC BY 4.0 licence on the tables are not read as a
claim over third-party work.

## Scores derived from third-party detectors

| File | Source | What is redistributed |
|---|---|---|
| `data/demamba_scores.csv` | DeMamba (Chen et al., arXiv:2405.19707) | 1,635 real-class scores produced by running the DeMamba pipeline with `microsoft/xclip-base-patch16` weights, as described in Supplementary A.14. The DeMamba checkpoint and its code are **not** redistributed. |
| `data/d3_scores.csv` | D3 (Zheng et al., *D3: Training-free AI-generated video detection using second-order features*, ICCV 2025, arXiv:2508.00701) | 1,635 D3 scalars from a reimplementation following the paper (16 frames, 10 % centre crop, X-CLIP frame encoder). The D3 code is **not** redistributed. |
| `data/nsgvd_per_generator.csv` | NSG-VD (Zhang et al., *Physics-driven spatiotemporal modeling for AI-generated video detection*, NeurIPS 2025) | The ten per-generator AUROC values plotted in Figure 5(b), transcribed from the run described in Section 5.6.2. No NSG-VD code or weights are included. |

These score files are redistributed so that the comparison rows of the paper can
be checked without downloading checkpoints. If the upstream authors' terms are
stricter than CC BY 4.0, their terms govern those three files.

## Data sets

| Data set | Use here | Redistributed? |
|---|---|---|
| GenVideo (Chen et al., arXiv:2405.19707) | the 1,150 AI-generated clips, ten generators | No — clip identifiers only |
| MSR-VTT | the 485 real clips | No — clip identifiers only |

Neither data set is redistributed and neither licence is subsumed by this
package. `data/features_4d.csv` contains derived statistics, not pixels.

## Software

| Component | Version used | Licence |
|---|---|---|
| numpy, scipy | 2.5.5 / 1.18.1 | BSD-3-Clause |
| pandas | 3.1.1 | BSD-3-Clause |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| XGBoost | 2.1.4 | Apache-2.0 |
| OpenCV (optional, extraction only) | 4.12.0 | Apache-2.0 |
| RAFT (optional, extraction only) | ECCV 2020 release | BSD-3-Clause |

RAFT is used to compute optical flow when features are re-extracted from videos.
Neither the RAFT code nor `raft-things.pth` is redistributed; the Farneback
backend in `src/feature_utils.py` is the dependency-free substitute.
