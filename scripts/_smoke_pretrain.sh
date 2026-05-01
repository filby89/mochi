#!/usr/bin/env bash
# Short smoke-run of stage 1 to verify the pipeline on a real GPU.
# Trains for only 200 iterations into a throwaway experiment id.
set -euo pipefail
cd "$(dirname "$0")/.."

source /fast/pfilntisis/.virtualenvs/TEMPEH/bin/activate
source scripts/_data_paths.sh

python -m trainer.train_global \
  -eid smoke/stage1_gpu \
  -print-freq 10 -val-freq 100000 -vis-freq 100000 -save-freq 100000 \
  -b 2 -irf 2 -wandb False --gradient-max-norm 1 \
  --input-image-type color_images --num-iterations 200 \
  -wpointr 0 -wp2s 0 -wlandd 1 \
  -wshapereg 1e-3 -wexpreg 1e-3 \
  -wvertregpliks 1e-3 -wvertregpliks-edge 0.1 \
  -pliks True \
  "${DATA_FLAGS[@]}"
