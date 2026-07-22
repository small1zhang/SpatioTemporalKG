#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Phase5 输出优化版运行脚本 (long-run ≥20min 数据用)
#
# 改动点:
#   1. --shard-frames 2000  按时间窗口分片输出 (每 2000 帧一个 graph_XXXX.json)
#   2. (默认开启) coalesce_containment  把 containsXXX / in_lane 等冗余边
#      合并为区间边, attrs.frames 列表记录覆盖帧
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
# 选择 run_20260721_150239_24000f: 完整 20 min 24000 帧, 12 chunks, 190 MB
RUN_DIR="data/long_run/run_20260721_150239_24000f"

# 每分片包含的帧数
SHARD_FRAMES=2000

# ── 执行 ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Phase5 优化输出: 分片 + coalesce"
echo "  RUN_DIR     = $RUN_DIR"
echo "  SHARD_FRAMES= $SHARD_FRAMES"
echo "================================================================"

python scripts/long_run/pipeline.py \
    --run-dir "$RUN_DIR" \
    --shard-frames "$SHARD_FRAMES" \
    --no-resume
