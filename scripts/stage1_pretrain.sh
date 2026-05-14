#!/usr/bin/env bash
# Stage 1 — pretraining, no differentiable rendering.
# Outputs to runs/public_release/coarse_nodifrs/
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_data_paths.sh

python -m trainer.train \
  -eid public_release/coarse_nodifrs \
  -print-freq 100 -val-freq 100000 -vis-freq 500 \
  -b 2 -irf 2 -wandb True --gradient-max-norm 1 \
  --num-iterations 300000 \
  -wpointr 0 -wp2s 0 -wlandd 1 \
  -wshapereg 1e-3 -wexpreg 1e-3 \
  -wvertregpliks 1e-3 -wvertregpliks-edge 0.1 \
  -pliks True \
  "${DATA_FLAGS[@]}" \
  "$@"
