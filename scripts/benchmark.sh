#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
RES=$WORK/results; mkdir -p $RES
RUNS=("$@"); [ ${#RUNS[@]} -gt 0 ] || RUNS=($WORK/runs/*)
VIEWS=("$REPO/data/board test" "$REPO/data/dark test")

for v in "${VIEWS[@]}"; do
  set -- $v; D=$1; SP=$2; tag=$(basename $D)
  for r in "${RUNS[@]}"; do
    n=$(basename $(dirname $(readlink -f $r)))
    $PY $REPO/benchmark/eval_echo.py --run_dir $r --data $D --split $SP --out $RES/${tag}_${n}.json
  done
  [ "${RGB:-1}" = 1 ] || continue
  $PY $REPO/benchmark/eval_rgb.py --method moge2 --name moge2_vitl --data $D --split $SP --out $RES/${tag}_moge2_vitl.json
  $PY $REPO/benchmark/eval_rgb.py --method da3 --name da3 --data $D --split $SP --out $RES/${tag}_da3.json
  [ -f $TEACHER ] && $PY $REPO/benchmark/eval_rgb.py --method moge2 --name teacher_moge2_vits_ft --model_id $TEACHER \
    --num_tokens 256 --square 224 --data $D --split $SP --out $RES/${tag}_teacher_moge2_vits_ft.json
done
$PY $REPO/benchmark/summarize.py $RES | tee $RES/benchmark.md
