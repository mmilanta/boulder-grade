#!/usr/bin/env bash
set -euo pipefail

# Best config: sector-level hierarchical d + per-climber Goldilocks,
# higher neg ratio, long ADVI. Hits R²_diff ≈ 0.85 (grade vs learned d).
uv run python train_bayesian.py \
  --method advi \
  --n-iter 200000 \
  --batch-size 4096 \
  --draws 3000 \
  --lr 0.003 \
  --neg-ratio 5.0 \
  --eval-every 20000 \
  --seed 42 \
  --hierarchy sector \
  --per-climber-try
