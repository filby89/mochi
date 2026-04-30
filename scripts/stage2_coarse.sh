#!/usr/bin/env bash
# Stage 2 — coarse training with differentiable rendering.
# Requires a stage 1 checkpoint at PRETRAINED_CKPT.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_data_paths.sh

PRETRAINED_CKPT="${PRETRAINED_CKPT:-runs/coarse/restart/coarse_nodifrs/checkpoints/model_00309000.pth}"

python -m trainer.train_global \
  -eid restart/coarse_difrs \
  -print-freq 100 -val-freq 20000 -vis-freq 1000 \
  -b 2 -irf 2 -wandb True --gradient-max-norm 1 \
  --input-image-type color_images --num-iterations 300000 \
  -wpointr 0 -wp2s 0 -wlandd 0.5 \
  -wshapereg 1e-3 -wexpreg 1e-3 \
  -wvertregpliks 1e-3 -wvertregpliks-edge 0.1 \
  -pliks True \
  -difr True -wpointmaps 10 -wnorm 10 \
  --pretrained-path "$PRETRAINED_CKPT" \
  "${DATA_FLAGS[@]}" \
  "$@"
