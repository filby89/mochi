#!/usr/bin/env bash
# Stage 3 — local refinement training.
# Requires a stage 2 checkpoint at PRETRAINED_CKPT.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_data_paths.sh

PRETRAINED_CKPT="${PRETRAINED_CKPT:-pretrained_models/global.pth}"

python -m trainer.train \
  -eid public_release_v2/local_wland_stricter \
  -print-freq 100 -val-freq 100000 -vis-freq 100 \
  -b 2 -irf 1 -wandb True --gradient-max-norm 1 \
  --num-iterations 300000 \
  -wpointr 0.01 -wp2s 0 -wlandd 0.0 --weight-landmarks 0.2 \
  -wshapereg 1e-6 -wexpreg 1e-5 \
  -wvertregpliks 1e-5 -wvertregpliks-edge 0.1 \
  -difr True -wpointmaps 10 -wnorm 4 -wedge 0.01 \
  --enable-local True \
  --point-mask-weights '{"w_point_face":0.0,"w_point_ears":0.0,"w_point_eyeballs":1.0,"w_point_eye_region":0.0,"w_point_lips":0.0,"w_point_neck":0.0,"w_point_nostrils":0.0,"w_point_scalp":0.0,"w_point_boundary":0.0}' \
  --pretrained-path "$PRETRAINED_CKPT" \
  "${DATA_FLAGS[@]}" \
  "$@"
