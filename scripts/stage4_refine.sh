#!/usr/bin/env bash
# Stage 4 — optional test-time optimization (TTO).
# Set PRETRAINED_CKPT / PRETRAINED_LOCAL_CKPT
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_data_paths.sh

PRETRAINED_CKPT="${PRETRAINED_CKPT:-pretrained_models/global.pth}"
PRETRAINED_LOCAL_CKPT="${PRETRAINED_LOCAL_CKPT:-pretrained_models/local.pth}"
# Blender binary used for the high-quality TTO visualizations. Override with your own path,
# e.g. BLENDER_BIN=/path/to/blender bash scripts/stage4_refine.sh
BLENDER_BIN="${BLENDER_BIN:-/path/to/blender}"


python -m trainer.train_refine \
  -eid "public_release/refine_t" \
  --gradient-max-norm 1 \
  -b 1 -irf 1 -difr True -wpointmaps 10 -wnorm 4 \
  -wp2s 0 -wlandd 0 -wpointr 0.01 -wedge 0.01 \
  --weight-landmarks 0.5 \
  --enable-local True \
  --point-mask-weights '{"w_point_face":0.0,"w_point_ears":0.0,"w_point_eyeballs":1.0,"w_point_eye_region":0.0,"w_point_lips":0.0,"w_point_neck":0.0,"w_point_nostrils":0.0,"w_point_scalp":0.0,"w_point_boundary":0.0}' \
  -refine-lr 1e-3 -refine-steps 51 \
  --refine-vis True -refine-vis-freq 1 -vis-freq 1 \
  --pretrained-path "$PRETRAINED_CKPT" \
  --pretrained-local-path "$PRETRAINED_LOCAL_CKPT" \
  --visualization-renderer blender --blender-bin "$BLENDER_BIN" \
  --render-half-sides-overlay true \
  "${DATA_FLAGS[@]}" \
  "$@"
