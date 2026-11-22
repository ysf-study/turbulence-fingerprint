#!/usr/bin/env bash
# Regenerate every number this package supports, in order.
# Stops at the first failure.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== 1/4  integrity and consistency checks =="
python scripts/verify_release.py

echo
echo "== 2/4  five-fold and leave-one-generator-out =="
python src/train_eval.py --out results

echo
echo "== 3/4  feature-set ablation (Table 4) =="
python src/ablation.py --out results

echo
echo "== 4/4  fusion with D3 (Table 9) =="
python src/fusion_eval.py --out results

echo
echo "Done. Compare results/ against results/reference/."
