#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd $V2E
for split in train valtest; do
  if [ $split = valtest ]; then X=(--mode_ext val --val_include_test); else X=(--mode_ext train); fi
  $PY -u precompute_teacher_latents.py "${DATA_FLAGS[@]}" "${MOGE_FLAGS[@]}" --init_material_weight $MINC "${X[@]}" --cache_multiscale \
    --teacher_cache_path $WORK/latents/teacher_$split.h5 --batchSize 32 --nThreads 6 --seed 1 --no_audio_augment \
    --exp_name precompute_$split 2>&1 | tee $WORK/logs/precompute_$split.log
done
