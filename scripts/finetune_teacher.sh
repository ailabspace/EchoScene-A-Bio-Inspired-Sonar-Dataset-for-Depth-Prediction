#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
$PY -u $REPO/teacher/ft_moge_teacher.py --data $DATA --out $TEACHER "$@" 2>&1 | tee $WORK/logs/ft_teacher.log
