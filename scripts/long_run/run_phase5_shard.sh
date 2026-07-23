#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Phase5 输出优化版运行脚本 (long-run ≥20min 数据用)
#
# 改动点:
#   1. --shard-frames 2000  按时间窗口分片输出 (每 2000 帧一个 graph_XXXX.json)
#   2. (默认开启) coalesce_containment  把 containsXXX / in_lane 等冗余边
#      合并为区间边, attrs.frames 列表记录覆盖帧
#   3. 阶段3 (FE-13): --importance-threshold / --prune-edges / --exclude-lanes
#   4. 阶段4 启发: 从 metadata.json 读取 spawn_mode 自动传 --ego-id
#
# 预期收益 (20min 24000 帧 数据推算):
#   老模式:  phase5_graph.json ~1.3 GB, ~5.4M 边
#   新模式:  12 个分片, 每片 ~70 MB, 总边数 ~240K (压缩 95%+)
#
# 输出:
#   <RUN_DIR>/phase5/
#     graph_0001_<f_start>_<f_end>.json   # 第 1 分片 (frame 0..1999)
#     graph_0002_<f_start>_<f_end>.json   # 第 2 分片 (frame 2000..3999)
#     ...
#     phase5_kg_summary.json               # 各分片索引 + 全帧统计汇总
# -----------------------------------------------------------------------------

set -euo pipefail

# 切换到仓库根目录
cd "$(dirname "$0")/../.."

# ── 配置 ────────────────────────────────────────────────────────────────────
# RUN_DIR: collect.py 输出的采集目录 (含 chunk_*.json)
# 默认选 20 min 24000 帧的 run, 可由环境变量 RUN_DIR 覆盖
RUN_DIR="${RUN_DIR:-data/long_run/run_20260721_150239_24000f}"

# 每分片包含的帧数
SHARD_FRAMES=2000

# 阶段3 配置 (FE-13)
IMPORTANCE_THRESHOLD="${IMPORTANCE_THRESHOLD:-0.30}"   # -1 = 关闭
PRUNE_EDGES="${PRUNE_EDGES:-1}"                          # 0 / 1
EXCLUDE_LANES="${EXCLUDE_LANES:-1}"                      # 0 / 1

# ── 从 metadata.json 读取 ego_id / spawn_mode (阶段4 启发) ───────────────
META_PATH="$RUN_DIR/metadata.json"
EGO_ID=""
SPAWN_MODE="default"
if [ -f "$META_PATH" ]; then
    EGO_ID=$(python -c "import json; m=json.load(open('$META_PATH')); print(m.get('ego_id') or '')" 2>/dev/null || echo "")
    SPAWN_MODE=$(python -c "import json; m=json.load(open('$META_PATH')); print(m.get('spawn_mode', 'default'))" 2>/dev/null || echo "default")
fi

# ── 执行 ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Phase5 优化输出: 分片 + coalesce + importance + prune"
echo "  RUN_DIR     = $RUN_DIR"
echo "  SHARD_FRAMES= $SHARD_FRAMES"
echo "  SPAWN_MODE  = $SPAWN_MODE"
echo "  EGO_ID      = ${EGO_ID:-<none>}"
echo "  IMPORTANCE  = $IMPORTANCE_THRESHOLD"
echo "  PRUNE_EDGES = $PRUNE_EDGES"
echo "  EXCLUDE_LANES=$EXCLUDE_LANES"
echo "================================================================"

PIPELINE_CMD=(python scripts/long_run/pipeline.py
    --run-dir "$RUN_DIR"
    --shard-frames "$SHARD_FRAMES"
    --no-resume
)

if [ -n "$EGO_ID" ]; then
    PIPELINE_CMD+=(--ego-id "$EGO_ID")
fi
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
