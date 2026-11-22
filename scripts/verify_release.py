#!/usr/bin/env python3
"""Integrity and consistency checks for this release.

Runs in a few seconds and needs no model training: it re-derives the fold
assignment, recomputes AUROCs from the released per-clip scores, and compares
both against the values quoted in the paper.

    python scripts/verify_release.py

Exit code 0 means every check passed.  Nothing here touches the network.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
REF = os.path.join(ROOT, "results", "reference")

# Values quoted in the manuscript.  The tolerances are set to twice the spread
# observed between XGBoost 2.x and 3.x on this data set.
EXPECTED = {
    "n_clips": 1635,
    "n_real": 485,
    "n_ai": 1150,
    "real_per_fold": 97,
    "ai_per_fold": 230,
    "pooled_auroc": 0.918,
    "loo_auroc": 0.915,
    "demamba_auroc": 0.9989,
    # Table 9, macro (mean per-generator) convention for B/C/D; A is pooled.
    "d3_auroc_macro": 0.9730,
    "fusion_c_macro": 0.9776,
    "fusion_d_macro": 0.9840,
    "fusion_a_pooled": 0.9178,
}
TOL_AUROC = 0.005

GEN_COUNTS = {
    "Real": 485, "Crafter": 108, "Gen2": 126, "HotShot": 131, "Lavie": 115,
    "ModelScope": 140, "MoonValley": 112, "MorphStudio": 126, "Show-1": 98,
    "Sora": 56, "WildScrape": 138,
}

results: list[tuple[bool, str]] = []


def check(cond: bool, msg: str) -> None:
    results.append((bool(cond), msg))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    feats = pd.read_csv(os.path.join(DATA, "features_4d.csv"))
    folds = pd.read_csv(os.path.join(DATA, "fold_assignment.csv"))
    dem = pd.read_csv(os.path.join(DATA, "demamba_scores.csv"))
    d3 = pd.read_csv(os.path.join(DATA, "d3_scores.csv"))

    # ---- 1. checksums ----
    sums = os.path.join(DATA, "SHA256SUMS")
    if os.path.isfile(sums):
        bad = []
        for line in open(sums, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            digest, name = line.split(maxsplit=1)
            p = os.path.join(DATA, name.lstrip("*"))
            if not os.path.isfile(p) or sha256(p) != digest:
                bad.append(name)
        check(not bad, "SHA256SUMS match the files on disk" + (f" (bad: {bad})" if bad else ""))
    else:
        check(False, "SHA256SUMS missing")

    # ---- 2. table shape and counts ----
    check(len(feats) == EXPECTED["n_clips"], f"feature table has {len(feats)} rows")
    check(len(folds) == EXPECTED["n_clips"], f"fold assignment has {len(folds)} rows")
    check(len(dem) == EXPECTED["n_clips"], f"DeMamba score file has {len(dem)} rows")
    check(len(d3) == EXPECTED["n_clips"], f"D3 score file has {len(d3)} rows")
    check(set(feats["file"]) == set(folds["file"]) == set(dem["file"]) == set(d3["file"]),
          "the four tables cover the same 1,635 clip identifiers")

    y = feats["label"].to_numpy(int)
    check(int((y == 1).sum()) == EXPECTED["n_real"], f"real clips = {(y == 1).sum()}")
    check(int((y == 0).sum()) == EXPECTED["n_ai"], f"AI clips = {(y == 0).sum()}")

    got = feats["generator"].value_counts().to_dict()
    check(got == GEN_COUNTS, "per-generator counts match the paper (Table A.2)")

    merged = feats.merge(folds, on="file", suffixes=("", "_f"), validate="one_to_one")
    check((merged["generator"] == merged["generator_f"]).all()
          and (merged["label"] == merged["label_f"]).all(),
          "generator and label agree between the feature table and the fold file")

    for k in range(5):
        sub = merged[merged["fold"] == k]
        ok = (int((sub["label"] == 1).sum()) == EXPECTED["real_per_fold"]
              and int((sub["label"] == 0).sum()) == EXPECTED["ai_per_fold"])
        check(ok, f"fold {k}: {int((sub['label']==1).sum())} real + "
                  f"{int((sub['label']==0).sum())} AI")

    # ---- 3. the released split is the one sklearn produces from the seed ----
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rec = np.full(len(merged), -1, dtype=int)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(merged)), merged["label"].to_numpy(int))):
        rec[te] = k
    check(float((rec == merged["fold"].to_numpy(int)).mean()) == 1.0,
          "released folds == StratifiedKFold(5, shuffle=True, random_state=42)")

    # ---- 4. AUROCs recomputed from the released scores ----
    metrics_path = os.path.join(REF, "metrics.json")
    if os.path.isfile(metrics_path):
        m = json.load(open(metrics_path, encoding="utf-8"))
        pooled = float(m["pooled_5fold_auroc"])
        loo = float(m["loo_mean_auroc"])
        check(abs(pooled - EXPECTED["pooled_auroc"]) <= TOL_AUROC,
              f"5-fold pooled AUROC = {pooled:.4f} (paper {EXPECTED['pooled_auroc']:.3f})")
        check(abs(loo - EXPECTED["loo_auroc"]) <= TOL_AUROC,
              f"LOO AUROC = {loo:.4f} (paper {EXPECTED['loo_auroc']:.3f})")
        check(m["n_videos"] == EXPECTED["n_clips"], "reference metrics were produced on 1,635 clips")
    else:
        check(False, "results/reference/metrics.json missing")

    oof_path = os.path.join(REF, "scores_oof.csv")
    if os.path.isfile(oof_path):
        oof = pd.read_csv(oof_path)
        check(len(oof) == EXPECTED["n_clips"], "OOF score file has one row per clip")
        recomputed = float(roc_auc_score(oof["label"], oof["score"]))
        check(abs(recomputed - float(json.load(open(metrics_path, encoding="utf-8"))["pooled_5fold_auroc"])) < 1e-6,
              f"pooled AUROC recomputed from scores_oof.csv = {recomputed:.4f}")
    else:
        check(False, "results/reference/scores_oof.csv missing")

    # ---- 5. the foundation-model baseline quoted in the paper ----
    auc_dem = float(roc_auc_score(dem["label"], dem["demamba_score"]))
    check(abs(auc_dem - EXPECTED["demamba_auroc"]) <= 1e-3,
          f"DeMamba baseline AUROC from released scores = {auc_dem:.4f} "
          f"(paper {EXPECTED['demamba_auroc']:.4f})")

    # ---- 6. D3 scalar and the Table 9 fusion rows ----
    d3m = feats.merge(d3[["file", "d3_score"]], on="file", validate="one_to_one")
    yb = d3m["label"].to_numpy(dtype=int)
    sb = d3m["d3_score"].to_numpy(dtype=float)
    gb = d3m["generator"].astype(str).to_numpy()
    gens = sorted(set(gb) - {"Real"})
    macro_b = float(
        np.mean(
            [
                roc_auc_score(yb[(gb == g) | (yb == 1)], sb[(gb == g) | (yb == 1)])
                for g in gens
            ]
        )
    )
    check(abs(macro_b - EXPECTED["d3_auroc_macro"]) <= 5e-4,
          f"D3-only AUROC from released scores = {macro_b:.4f} "
          f"(paper {EXPECTED['d3_auroc_macro']:.4f})")

    fusion_path = os.path.join(REF, "fusion.csv")
    if os.path.isfile(fusion_path):
        fu = pd.read_csv(fusion_path).set_index("feature_set")
        a_pooled = float(fu.loc["A: TF only", "pooled_auroc"])
        c_macro = float(fu.loc["C: D3 only (control)", "macro_auroc"])
        d_macro = float(fu.loc["D: TF + D3 (this work)", "macro_auroc"])
        check(abs(a_pooled - EXPECTED["fusion_a_pooled"]) <= TOL_AUROC,
              f"Table 9 A (TF only, pooled) = {a_pooled:.4f} "
              f"(paper {EXPECTED['fusion_a_pooled']:.4f})")
        check(abs(c_macro - EXPECTED["fusion_c_macro"]) <= 0.002,
              f"Table 9 C (D3 + XGBoost) = {c_macro:.4f} "
              f"(paper {EXPECTED['fusion_c_macro']:.4f})")
        check(abs(d_macro - EXPECTED["fusion_d_macro"]) <= 0.002,
              f"Table 9 D (TF + D3) = {d_macro:.4f} "
              f"(paper {EXPECTED['fusion_d_macro']:.4f})")
    else:
        check(False, "results/reference/fusion.csv missing")

    # ---- report ----
    for ok, msg in results:
        print(("  PASS  " if ok else "  FAIL  ") + msg)
    n_fail = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
