#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# 端到端 5min 数据采集 + 知识图谱提取 (新参数全面启用)
#
# 流程:
#   Phase1: collect.py        采集 5min @ 20fps = 6000帧, ego-centric spawn
#   Phase2-5: pipeline.py     处理 + 序列化, shard + importance + edge prune
#
# 输出:
#   data/long_run/test_5min/run_<timestamp>_6000f/
#     chunk_*.json                (采集数据)
#     metadata.json               (含 spawn_mode=ego_centric)
#     phase5/
#       graph_0001_0_1999.json
#       graph_0002_2000_3999.json
#       graph_0003_4000_5999.json
#       phase5_kg_summary.json
# -----------------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── 用 stk conda env (含 carla) ─────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-/home/aisecurity/miniconda3/envs/stk/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "[FATAL] python not found at $PYTHON_BIN"
    exit 1
fi
echo "[+] using python: $PYTHON_BIN"

# ── 配置 ─────────────────────────────────────────────────────────────────
TOTAL_FRAMES=6000    # 5 min @ 20 fps
CHUNK_FRAMES=2000    # 每块 2000 帧 ~= 1.67 min
SHARD_FRAMES=2000    # 与 chunk 对齐
TOWN="Town10HD"
VEHICLES=20
WALKERS=8
FPS=20
DENSITY=2.0
SEED=42
OUT_BASE="data/long_run/test_5min"

# ego-centric spawn 参数 (FE-14)
EGO_CENTRIC=1                # 启用时 = 1, 禁用时 = 0
NPC_RADIUS_FRONT=70.0
NPC_RADIUS_REAR=30.0
NPC_RADIUS_SIDE=50.0

# 重要性/边裁剪/背景外移参数 (阶段3, FE-13)
IMPORTANCE_THRESHOLD=0.30    # -1 表示不启用
PRUNE_EDGES=1                # 0 / 1
EXCLUDE_LANES=1              # 0 / 1

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUT_BASE}/run_${TS}_${TOTAL_FRAMES}f"
mkdir -p "$(dirname "$RUN_DIR")"

echo "================================================================"
echo "  End-to-End 5min KG Extraction (ego-centric)"
echo "  RUN_DIR     = $RUN_DIR"
echo "  TOTAL_FRAMES= $TOTAL_FRAMES"
echo "  EGO_CENTRIC = $EGO_CENTRIC"
echo "  IMPORTANCE  = $IMPORTANCE_THRESHOLD"
echo "  PRUNE_EDGES = $PRUNE_EDGES"
echo "  EXCLUDE_LANES=$EXCLUDE_LANES"
echo "================================================================"

# ── Phase1: 采集 ──────────────────────────────────────────────────────────
echo ""
echo "[Phase 1] CARLA data collection"
echo "------------------------------------------------------------"
COLLECT_CMD=("$PYTHON_BIN" scripts/long_run/collect.py
    --town "$TOWN"
    --total-frames "$TOTAL_FRAMES"
    --chunk-frames "$CHUNK_FRAMES"
    --vehicles "$VEHICLES"
    --walkers "$WALKERS"
    --fps "$FPS"
    --density "$DENSITY"
    --seed "$SEED"
    --out "$OUT_BASE"
)
if [ "$EGO_CENTRIC" = "1" ]; then
    COLLECT_CMD+=(--ego-centric)
    COLLECT_CMD+=(--npc-radius-front "$NPC_RADIUS_FRONT")
    COLLECT_CMD+=(--npc-radius-rear "$NPC_RADIUS_REAR")
    COLLECT_CMD+=(--npc-radius-side "$NPC_RADIUS_SIDE")
fi

"${COLLECT_CMD[@]}"

# 找最近生成的 run_dir (collect.py 会自动加时间戳)
RUN_DIR=$(ls -td ${OUT_BASE}/run_*_${TOTAL_FRAMES}f | head -1)
echo ""
echo "[+] Phase1 done. RUN_DIR=$RUN_DIR"

# ── Phase2-5: 处理 + 分片知识图谱 ────────────────────────────────────────
echo ""
echo "[Phase 2-5] Pipeline + sharded KG output"
echo "------------------------------------------------------------"
PIPELINE_CMD=("$PYTHON_BIN" scripts/long_run/pipeline.py
    --run-dir "$RUN_DIR"
    --shard-frames "$SHARD_FRAMES"
    --no-resume
)
if [ "$IMPORTANCE_THRESHOLD" != "-1" ]; then
    PIPELINE_CMD+=(--importance-threshold "$IMPORTANCE_THRESHOLD")
fi
if [ "$PRUNE_EDGES" = "1" ]; then
    PIPELINE_CMD+=(--prune-edges)
fi
if [ "$EXCLUDE_LANES" = "1" ]; then
    PIPELINE_CMD+=(--exclude-lanes)
fi

"${PIPELINE_CMD[@]}"

echo ""
echo "================================================================"
echo "  ✅ End-to-End done"
echo "  RUN_DIR = $RUN_DIR"
echo "  KG      = $RUN_DIR/phase5/"
echo "================================================================"
ls -lh "$RUN_DIR/phase5/"
