#!/bin/bash
# ============================================================
#  单地图单场景冒烟测试
#  用法: bash run_smoke.sh [地图] [场景]
#
#  示例:
#    bash run_smoke.sh                      # 默认 Town10HD + S00
#    bash run_smoke.sh Town01 S10           # 指定地图和场景
#    bash run_smoke.sh Town04 S30           # 夜间场景
# ============================================================

set -e
cd "$(dirname "$0")"

source /home/aisecurity/miniconda3/etc/profile.d/conda.sh
conda activate stk

MAP=${1:-Town10HD}
SCENARIO=${2:-S00}

echo "============================================"
echo " Smoke Test: ${MAP} + ${SCENARIO}"
echo "============================================"

python scripts/pipeline/smoke_test.py --map "$MAP" --scenario "$SCENARIO"
