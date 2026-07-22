#!/bin/bash
# =====================================================================
# build_anomaly_dataset_manual.sh
#   异常检测数据集采集 + 构建全流程操作手册
#
# 目标产出:
#   data/dataset/stk_anomaly_v1/
#     frame_labels.csv    帧级标签 (22 列)
#     frame_actors.csv    帧×actor 运动学 (35 列)
#     event_labels.json   事件级标签
#     dataset_index.json  元数据 / 划分 / 类别分布
#
# 采集规模: 3 地图 × (1 长 20min 跑 + 14 短场景) ≈ 65min 数据
# 划分   : 长跑帧号 70/15/15 = train/val/test
#          短场景 S00-S02→train / S10-S22→val / S30-S33→test
# =====================================================================
# 重要前置:
#   1. 先启动 CARLA 服务器: ./CarlaUE4.sh -RenderOffScreen (限时模式可省略)
#   2.conda activate stk                       # 含 carla python 包
#   3. 注意每跑一次长跑会改 spawn_offset 获得不同起位置
# =====================================================================

set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

# 用 stk conda env (含 carla)
PYTHON_BIN="${PYTHON_BIN:-/home/aisecurity/miniconda3/envs/stk/bin/python3.10}"
[ ! -x "$PYTHON_BIN" ] && PYTHON_BIN=python3
echo "[+] python = $PYTHON_BIN"

# ====================================================================
# Step 1: 短场景批量采集 (3 地图 × 14 场景 = 42 任务, 每任务 ≤ 7.5s)
# ====================================================================
echo "============================================================"
echo "[Step 1] Batch short scenario collection"
echo "  3 maps × 14 scenarios = 42 tasks (each ≤ 150 frames @ 20fps)"
echo "============================================================"

$PYTHON_BIN scripts/pipeline/batch_collect.py \
    --maps "Town10HD,Town01,Town05" \
    --scenarios "S00,S01,S02,S10,S11,S12,S13,S20,S21,S22,S30,S31,S32,S33" \
    --gpus 1,2,3 \
    --parallel 3 \
    --out data/runs/batch

# 输出: data/runs/batch/<map>/<scenario>/phases_<ts>_<frames>f/

# ====================================================================
# Step 2: 长跑采集 (3 地图 × 各 1 次 20min)
#   每个 20min 长跑通过 --weather-cycle (clear->cloud->rain->night) +
#   --density-ramp (15v/6w -> 25v/10w -> 35v/15w) +
#   --spawn-offset (参考起点偏移) 增强多样性
# ====================================================================
echo ""
echo "============================================================"
echo "[Step 2] Long-run collection (3 maps × 1 × 20min)"
echo "============================================================"

# ---------- 2.1: Town10HD ----------
echo "--- [2.1] Town10HD run ---"
TOWN=Town10HD SPAWN_OFFSET=1 \
bash scripts/long_run/run_long_run.sh full
# 重命名以便后续索引清晰
mv data/long_run/run_*_24000f data/long_run/Town10HD_run1 || true

# ---------- 2.2: Town01 ----------
echo "--- [2.2] Town01 run ---"
TOWN=Town01 SPAWN_OFFSET=5 \
bash scripts/long_run/run_long_run.sh custom 24000 2000 2.0 30 15 Town01
mv data/long_run/run_*_24000f data/long_run/Town01_run1 || true

# ---------- 2.3: Town05 ----------
echo "--- [2.3] Town05 run ---"
TOWN=Town05 SPAWN_OFFSET=10 \
bash scripts/long_run/run_long_run.sh custom 24000 2000 2.0 30 15 Town05
mv data/long_run/run_*_24000f data/long_run/Town05_run1 || true

# ====================================================================
# Step 3: 跑 pipeline.py (chunk -> phase5 KG, 含 SafetyViolation 节点)
#   若 run_long_run.sh 已自动跑过 pipeline, 此步可跳过
# ====================================================================
echo ""
echo "============================================================"
echo "[Step 3] Pipeline (chunk -> phase5 KG)"
echo "============================================================"
for RUN in data/long_run/Town10HD_run1 data/long_run/Town01_run1 data/long_run/Town05_run1; do
    if [ ! -d "$RUN/phase5" ]; then
        MAP_NAME=$(basename "$RUN" | cut -d_ -f1)
        echo "--- pipeline: $RUN (map=$MAP_NAME) ---"
        $PYTHON_BIN scripts/long_run/pipeline.py \
            --run-dir "$RUN" \
            --map-name "$MAP_NAME" \
            --tick-s 0.05 \
            --seed 42
    else
        echo "--- $RUN/phase5 already exists, skip ---"
    fi
done

# ====================================================================
# Step 4: 构建异常检测数据集 (chunk + anomaly_log + phase5 -> 标签 CSV)
# ====================================================================
echo ""
echo "============================================================"
echo "[Step 4] Build anomaly detection dataset"
echo "============================================================"

$PYTHON_BIN scripts/long_run/build_anomaly_dataset.py \
    --run-dir data/long_run/Town10HD_run1 \
    --run-dir data/long_run/Town01_run1 \
    --run-dir data/long_run/Town05_run1 \
    --batch-dir data/runs/batch \
    --out      data/dataset/stk_anomaly_v1

# 或用 --all 自动扫描 data/long_run/ + data/runs/batch/:
#   $PYTHON_BIN scripts/long_run/build_anomaly_dataset.py --all \
#       --out data/dataset/stk_anomaly_v1_all

# ====================================================================
# 完成
# ====================================================================
echo ""
echo "============================================================"
echo "✅ Dataset built at data/dataset/stk_anomaly_v1/"
echo "============================================================"
ls -lh data/dataset/stk_anomaly_v1/
