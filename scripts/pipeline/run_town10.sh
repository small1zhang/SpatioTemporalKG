#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# run_town10.sh — 快速运行 Town10 知识图谱提取
# 
# 用法: 在 VNC 已启动 CARLA (localhost:2000, Town10HD_Opt) 后,
#       cd /home/aisecurity/01_ZHB/SpatioTemporalKG
#       bash scripts/pipeline/run_town10.sh
# 
# 或者手动:
#       conda activate stk
#       cd /home/aisecurity/01_ZHB/SpatioTemporalKG
#       CUDA_VISIBLE_DEVICES=0 python3 scripts/pipeline/run_phases_1_5.py \\
#           --host localhost --port 2000 --carla-port 2000 \\
#           --frames 60 --vehicles 30 --walkers 15 \\
#           --out data/runs/town10_vnc

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
echo "REPO=$REPO_DIR"
echo "CARLA on localhost:2000 (VNC, Town10HD_Opt)"
echo ""
echo "[*] Starting 5-phase KG extraction..."
echo "    60 frames, 30 vehicles, 15 walkers"
echo ""

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/home/aisecurity/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate stk

cd "$REPO_DIR" || exit 1

CUDA_VISIBLE_DEVICES=0 python3 scripts/pipeline/run_phases_1_5.py \
    --host localhost \
    --port 2000 \
    --carla-port 2000 \
    --frames 60 \
    --vehicles 30 \
    --walkers 15 \
    --out data/runs/town10_vnc

echo ""
echo "[OK] Done — run_phases_1_5.py finished"
echo "Output in: $REPO_DIR/data/runs/town10_vnc/"
