#!/usr/bin/env bash
# Stage 4 — optional test-time optimization (TTO).
# Set PRETRAINED_CKPT / PRETRAINED_LOCAL_CKPT
# (and optionally TRAIN_LIST / VAL_LIST).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_data_paths.sh

PRETRAINED_CKPT="${PRETRAINED_CKPT:-runs/public_release/coarse_difrs/checkpoints/model_00309000.pth}"
PRETRAINED_LOCAL_CKPT="${PRETRAINED_LOCAL_CKPT:-runs/public_release/local_wland02/checkpoints/local_00300000.pth}"


python -m trainer.train_refine \
  -eid "public_release/refine" \
  --gradient-max-norm 1 \
  -b 1 -irf 1 -difr True -wpointmaps 10 -wnorm 4 \
  -wp2s 0 -wlandd 0 -wpointr 0.01 -wedge 0.01 \
  --weight-landmarks 0.5 \
  --enable-local True \
  --point-mask-weights '{"w_point_face":0.0,"w_point_ears":0.0,"w_point_eyeballs":1.0,"w_point_eye_region":0.0,"w_point_lips":0.0,"w_point_neck":0.0,"w_point_nostrils":0.0,"w_point_scalp":0.0,"w_point_boundary":0.0}' \
  -tdl "$TRAIN_LIST" \
  -vdl "$VAL_LIST" \
  -wshapereg 1e-6 -wvertregpliks 1e-5 -wvertregpliks-edge 0.01 -wexpreg 1e-5 \
  -refine-lr 1e-3 -refine-steps 101 \
  --refine-vis True -refine-vis-freq 1 -vis-freq 1 \
  --pretrained-path "$PRETRAINED_CKPT" \
  --pretrained-local-path "$PRETRAINED_LOCAL_CKPT" \
  "${DATA_FLAGS[@]}" \
  "$@"
