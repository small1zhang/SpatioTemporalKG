#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# =====================================================================
# run_long_run.sh — 一键启动长期连续采集 (Phase1) + Phase2→5 知识图谱构建
#
# 设计目标:
#   1. 自动选 1 张空闲 GPU 给 CARLA 进程 (CARLA 0.9.16 单卡即可)
#   2. 启动 collect.py (含 ego 视角跟随 + 异常调度 + checkpoint 写入)
#      → 在 Ubuntu 可视化窗口上能看到 ego 后方 -8m/+4m 的实时跟随画面
#   3. 采集完成后用 pipeline.py 跑 Phase2-5, 输出 phase5_graph.json
#   4. 出错 / Ctrl+C 也能保留 chunk 和 checkpoint, 下次 --resume 接续
#
# 用法:
#   # 1) 先跑 2 分钟 smoke 测试 (2400 帧 @ 20fps), 验证可视化窗口能看到车
#   bash scripts/long_run/run_long_run.sh smoke
#
#   # 2) 跑 20 分钟正式采集 (24000 帧 @ 20fps)
#   bash scripts/long_run/run_long_run.sh full
#
#   # 3) 跑自定义时长 (例如 10 分钟 = 12000 帧)
#   bash scripts/long_run/run_long_run.sh custom 12000
#
#   # 4) 仅 resume 已有 run (不重新开 collect)
#   bash scripts/long_run/run_long_run.sh resume <run_dir>
#
#   # 5) 仅跑 Phase2-5 (已采集完 chunk)
#   bash scripts/long_run/run_long_run.sh pipeline <run_dir>
#
# 备注:
#   - CARLA 必须已启动且监听 localhost:2000 (窗口模式 / -RenderOffScreen 均可,
#     但只有窗口模式才能在可视化界面看到画面)
#   - 默认使用 conda env: stk 的 python3.10 (有 carla 包); 若你用别的环境,
#     修改下方的 CARLA_PYTHON 变量即可
# =====================================================================

set -euo pipefail

# ---------- 0. 路径与环境 ----------
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

# CARLA Python (有 carla 包的解释器)
CARLA_PYTHON="${CARLA_PYTHON:-/home/aisecurity/miniconda3/envs/stk/bin/python3.10}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$REPO"

# CARLA 服务器连接参数
CARLA_HOST="${CARLA_HOST:-localhost}"
CARLA_PORT="${CARLA_PORT:-2000}"

# 输出根目录
OUT_ROOT="${OUT_ROOT:-data/long_run}"

# ---------- 1. 选空闲 GPU ----------
FREE_GPU_SCRIPT="$REPO/scripts/remote/server_scripts/free_gpu.py"
if [ -x "$FREE_GPU_SCRIPT" ] || [ -f "$FREE_GPU_SCRIPT" ]; then
    GPU_IDX=$(python3 "$FREE_GPU_SCRIPT" --pick-one 2>/dev/null | tail -1 | tr -d ' \n')
    if [ -z "$GPU_IDX" ]; then GPU_IDX=0; fi
else
    GPU_IDX=0
fi
echo "[run_long_run] using GPU=$GPU_IDX"
export CUDA_VISIBLE_DEVICES="$GPU_IDX"

# ---------- 2. 参数派生 ----------
MODE="${1:-smoke}"

# 数据集多样性选项 (默认 full 启用, smoke / custom 不启用; 可被环境变量覆盖)
WEATHER_CYCLE="${WEATHER_CYCLE:-}"
DENSITY_RAMP="${DENSITY_RAMP:-}"
SPAWN_OFFSET="${SPAWN_OFFSET:-0}"

case "$MODE" in
    smoke)
        TOTAL_FRAMES=2400      # 2 min @ 20 fps
        CHUNK_FRAMES=600        # 4 个 chunk, 每块 30 s
        DENSITY=2.0
        VEHICLES=20
        WALKERS=10
        TOWN=Town10HD
        ;;
    full)
        TOTAL_FRAMES=24000     # 20 min @ 20 fps
        CHUNK_FRAMES=2000      # 12 个 chunk, 每块 100 s
        DENSITY=2.0
        VEHICLES=30
        WALKERS=15
        TOWN=Town10HD
        # full 模式默认开启多样性 (可被 WEATHER_CYCLE=0 / DENSITY_RAMP=0 关闭)
        [ -z "$WEATHER_CYCLE" ] && WEATHER_CYCLE=1
        [ -z "$DENSITY_RAMP"  ] && DENSITY_RAMP=1
        ;;
    custom)
        TOTAL_FRAMES="${2:-6000}"
        CHUNK_FRAMES="${3:-1000}"
        DENSITY="${4:-2.0}"
        VEHICLES="${5:-25}"
        WALKERS="${6:-12}"
        TOWN="${7:-Town10HD}"
        ;;
	    resume)
	        RUN_DIR="${2:?usage: $0 resume <run_dir>}"
	        # resume 模式: 不再创建新 run, 直接接续
	        echo "[run_long_run] resume $RUN_DIR"
	        "$CARLA_PYTHON" "$REPO/scripts/long_run/collect.py" \
	            --host "$CARLA_HOST" --port "$CARLA_PORT" \
	            --town "${TOWN:-Town10HD}" \
	            --total-frames "${TOTAL_FRAMES:-24000}" \
	            --chunk-frames "${CHUNK_FRAMES:-2000}" \
	            --density "${DENSITY:-2.0}" \
	            --seed 42 \
	            --vehicles "${VEHICLES:-30}" \
	            --walkers "${WALKERS:-15}" \
	            --fps 20.0 \
	            --out "$OUT_ROOT" \
	            --resume "$RUN_DIR" \
	            $([ "$WEATHER_CYCLE" = "1" ] && echo --weather-cycle) \
	            $([ "$DENSITY_RAMP" = "1" ] && echo --density-ramp)
	        exit $?
	        ;;
    pipeline)
        RUN_DIR="${2:?usage: $0 pipeline <run_dir>}"
        echo "[run_long_run] pipeline-only on $RUN_DIR"
        "$CARLA_PYTHON" "$REPO/scripts/long_run/pipeline.py" \
            --run-dir "$RUN_DIR" \
            --map-name "${TOWN:-Town10HD}" \
            --tick-s 0.05 \
            --seed 42
        exit $?
        ;;
    *)
        echo "Usage: $0 {smoke|full|custom [frames chunk_frames density vehicles walkers town]|resume <run_dir>|pipeline <run_dir>}"
        exit 1
        ;;
esac

echo "[run_long_run] mode=$MODE  total_frames=$TOTAL_FRAMES  chunk_frames=$CHUNK_FRAMES"
echo "[run_long_run] density=$DENSITY vehicles=$VEHICLES walkers=$WALKERS town=$TOWN"
echo "[run_long_run] ego 视角跟随: ON (CARLA 窗口应切到 spectator)"
echo "[run_long_run] python = $CARLA_PYTHON"
echo

# ---------- 3. 健康检查 CARLA ----------
echo "[run_long_run] health check CARLA at ${CARLA_HOST}:${CARLA_PORT} ..."
"$CARLA_PYTHON" -c "
import sys
try:
    import carla
    c = carla.Client('${CARLA_HOST}', ${CARLA_PORT}); c.set_timeout(10.0)
    w = c.get_world()
    print('[ok] CARLA server:', c.get_server_version(), 'map:', w.get_map().name)
except Exception as e:
    print('[fatal] CARLA not reachable:', e)
    sys.exit(1)
" || { echo "[run_long_run] CARLA 不可用, 退出"; exit 1; }
echo

# ---------- 4. Phase1: 采集 (collect.py) ----------
COLLECT_LOG="$REPO/data/long_run/.logs"
mkdir -p "$COLLECT_LOG"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$COLLECT_LOG/collect_${TIMESTAMP}_${MODE}.log"

echo "[run_long_run] 启动 collect.py, 日志: $LOG_FILE"
echo

"$CARLA_PYTHON" "$REPO/scripts/long_run/collect.py" \
    --host "$CARLA_HOST" --port "$CARLA_PORT" \
    --town "$TOWN" \
    --total-frames "$TOTAL_FRAMES" \
    --chunk-frames "$CHUNK_FRAMES" \
    --density "$DENSITY" \
    --seed 42 \
    --vehicles "$VEHICLES" \
    --walkers "$WALKERS" \
    --fps 20.0 \
    --out "$OUT_ROOT" \
    --checkpoint-interval 200 \
    $([ "$WEATHER_CYCLE" = "1" ] && echo --weather-cycle) \
    $([ "$DENSITY_RAMP" = "1" ] && echo --density-ramp) \
    $( [ "${SPAWN_OFFSET:-0}" != "0" ] && echo --spawn-offset "$SPAWN_OFFSET" ) \
    2>&1 | tee "$LOG_FILE"

# 抓最新生成的 run_dir
RUN_DIR=$(ls -td "${OUT_ROOT}"/run_*_${TOTAL_FRAMES}f 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    echo "[run_long_run] [WARN] 没找到新 run 目录, phase2-5 跳过"
    echo "    请手动检查 $OUT_ROOT 下是否有数据"
    exit 0
fi
echo
echo "[run_long_run] 采集完成. run_dir=$RUN_DIR"

# ---------- 5. Phase2-5: 知识图谱构建 (pipeline.py) ----------
PIPELINE_LOG="$COLLECT_LOG/pipeline_${TIMESTAMP}_${MODE}.log"
echo "[run_long_run] 启动 pipeline.py, 日志: $PIPELINE_LOG"
echo

"$CARLA_PYTHON" "$REPO/scripts/long_run/pipeline.py" \
    --run-dir "$RUN_DIR" \
    --map-name "$TOWN" \
    --tick-s 0.05 \
    --seed 42 \
    2>&1 | tee "$PIPELINE_LOG"

echo
echo "[run_long_run] ✓ 全部完成."
echo "    run_dir   = $RUN_DIR"
echo "    phase5    = $RUN_DIR/phase5/phase5_graph.json"
echo "    log files = $LOG_FILE / $PIPELINE_LOG"
echo
echo "可视化页面 (可选):"
echo "    浏览器打开 file://$REPO/viz_output/dashboard_lite.html"
