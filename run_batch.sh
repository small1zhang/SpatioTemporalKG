#!/bin/bash
# ============================================================
#  SpatioTemporalKG 批量采集一键脚本
#  用法: bash run_batch.sh [选项]
#
#  示例:
#    bash run_batch.sh                          # 全部 70 任务 (3 GPU)
#    bash run_batch.sh --maps Town10HD,Town01   # 只跑指定地图
#    bash run_batch.sh --scenarios S00,S10,S30  # 只跑指定场景
#    bash run_batch.sh --gpus 1,2 --parallel 2  # 自定义 GPU
#    bash run_batch.sh --resume                 # 从断点继续 (默认)
# ============================================================

set -e
cd "$(dirname "$0")"

# 激活 conda 环境
source /home/aisecurity/miniconda3/etc/profile.d/conda.sh
conda activate stk

echo "============================================"
echo " SpatioTemporalKG Batch Collection"
echo " Maps:      5 (Town10HD, Town01, Town02, Town04, Town05)"
echo " Scenarios: 14 (S00-S33)"
echo " Total:     70 tasks"
echo " GPUs:      RTX 5090 x4"
echo "============================================"
echo ""

python scripts/pipeline/batch_collect.py "$@"
