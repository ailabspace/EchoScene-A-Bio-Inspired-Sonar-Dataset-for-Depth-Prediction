#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
SEED=${1:-1}; shift || true
EXP=v2e_ccl_s$SEED
cd $V2E
$PY -u train_ccl.py "${TRAIN_FLAGS[@]}" "${MOGE_FLAGS[@]}" --seed $SEED \
  --teacher_cache_path $WORK/latents/teacher_train.h5 --validation_cache_path $WORK/latents/teacher_valtest.h5 \
  --moge_cache_enc_dim_out 384 --cache_freeze_teacher_proj \
  --lambda_teacher_depth 0.5 --teacher_depth_align --lambda_ccl_depth 0.5 --lambda_ldc 0.2 --lambda_cc 0.1 \
  "$@" --exp_name $EXP 2>&1 | tee $WORK/logs/$EXP.log
mkdir -p $WORK/runs && ln -sfn $V2E/checkpoint/$EXP/biosonar $WORK/runs/$EXP
