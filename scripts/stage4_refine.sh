#!/usr/bin/env bash
# Stage 4 — optional test-time optimization (TTO).
# Set REF_START, REF_END, PRETRAINED_CKPT (and optionally TRAIN_LIST / VAL_LIST).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_data_paths.sh

REF_START="${REF_START:-0}"
REF_END="${REF_END:-1}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-runs/coarse/restart/local_wland005/checkpoints/model_00300000.pth}"
TRAIN_LIST="${TRAIN_LIST:-/fast/pfilntisis/TEMPEH_data/data/aws_data/seventy_subj__all_seq_frames_per_seq_40_head_rot_120_train.json}"
VAL_LIST="${VAL_LIST:-assets/meshes_list_test.json}"

# Stage 4 overrides the train / val data lists with TTO-specific JSONs;
# we strip the -tdl / -vdl flags from DATA_FLAGS and supply our own.
TTO_DATA_FLAGS=()
skip=0
for arg in "${DATA_FLAGS[@]}"; do
  if [[ "$skip" == 1 ]]; then skip=0; continue; fi
  if [[ "$arg" == "-tdl" || "$arg" == "-vdl" ]]; then skip=1; continue; fi
  TTO_DATA_FLAGS+=("$arg")
done

python -m trainer.train_refine \
  -eid "tto/${REF_START}_${REF_END}" \
  --gradient-max-norm 1 --input-image-type color_images \
  -b 1 -irf 1 -difr True -wpointmaps 10 -wnorm 4 \
  -wp2s 0 -wlandd 0 -wpointr 0.01 -wedge 0.01 \
  --weight-landmarks 0.5 \
  --enable-local True \
  -tdl "$TRAIN_LIST" \
  -vdl "$VAL_LIST" \
  -refine-lr 1e-3 -refine-steps 105 \
  --refine-vis True -refine-vis-freq 1 -vis-freq 1 \
  --refine-start-index "$REF_START" --refine-end-index "$REF_END" \
  --pretrained-path "$PRETRAINED_CKPT" \
  "${TTO_DATA_FLAGS[@]}" \
  "$@"
