#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# 端到端 5min 数据采集 + 知识图谱提取 (验证 Phase5 优化输出)
#
# 流程:
#   Phase1: collect.py        采集 5min @ 20fps = 6000帧, chunk=2000 → 3 chunks
#   Phase2-5: pipeline.py     处理 + 序列化, shard=2000 → 3 个分片
#
# 输出:
#   data/long_run/test_5min/run_<timestamp>_6000f/
#     chunk_0001.json ... chunk_0003.json  (采集数据)
#     phase5/
#       graph_0001_0_1999.json     (第 1 分片知识图谱)
#       graph_0002_2000_3999.json (第 2 分片知识图谱)
#       graph_0003_4000_5999.json (第 3 分片知识图谱)
#       phase5_kg_summary.json    (汇总索引)
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

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUT_BASE}/run_${TS}_${TOTAL_FRAMES}f"
mkdir -p "$(dirname "$RUN_DIR")"

echo "================================================================"
echo "  End-to-End 5min KG Extraction"
echo "  RUN_DIR     = $RUN_DIR"
echo "  TOTAL_FRAMES= $TOTAL_FRAMES"
echo "  CHUNK_FRAMES= $CHUNK_FRAMES"
echo "  SHARD_FRAMES= $SHARD_FRAMES"
echo "================================================================"

# ── Phase1: 采集 ──────────────────────────────────────────────────────────
echo ""
echo "[Phase 1] CARLA data collection"
echo "------------------------------------------------------------"
"$PYTHON_BIN" scripts/long_run/collect.py \
    --town "$TOWN" \
    --total-frames "$TOTAL_FRAMES" \
    --chunk-frames "$CHUNK_FRAMES" \
    --vehicles "$VEHICLES" \
    --walkers "$WALKERS" \
    --fps "$FPS" \
    --density "$DENSITY" \
    --seed "$SEED" \
    --out "$OUT_BASE"

# 找最近生成的 run_dir (collect.py 会自动加时间戳)
RUN_DIR=$(ls -td ${OUT_BASE}/run_*_${TOTAL_FRAMES}f | head -1)
echo ""
echo "[+] Phase1 done. RUN_DIR=$RUN_DIR"

# ── Phase2-5: 处理 + 分片知识图谱 ────────────────────────────────────────
echo ""
echo "[Phase 2-5] Pipeline + sharded KG output"
echo "------------------------------------------------------------"
"$PYTHON_BIN" scripts/long_run/pipeline.py \
    --run-dir "$RUN_DIR" \
    --shard-frames "$SHARD_FRAMES" \
    --no-resume

echo ""
echo "================================================================"
echo "  ✅ End-to-End done"
echo "  RUN_DIR = $RUN_DIR"
echo "  KG      = $RUN_DIR/phase5/"
echo "================================================================"
ls -lh "$RUN_DIR/phase5/"
