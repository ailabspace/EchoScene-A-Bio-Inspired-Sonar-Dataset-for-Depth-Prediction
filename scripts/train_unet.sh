#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
SEED=${1:-1}; shift || true
EXP=unet_s$SEED
cd $V2E
$PY -u train_ccl.py "${TRAIN_FLAGS[@]}" --seed $SEED \
  --lambda_ccl_depth 0 --lambda_teacher_depth 0 --lambda_ldc 0 --lambda_cc 0 \
  "$@" --exp_name $EXP 2>&1 | tee $WORK/logs/$EXP.log
mkdir -p $WORK/runs && ln -sfn $V2E/checkpoint/$EXP/biosonar $WORK/runs/$EXP
