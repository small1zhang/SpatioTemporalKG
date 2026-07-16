#!/usr/bin/env bash
# ============================================================================
# server_tools.sh  ——  放在服务器 /home/aisecurity/01_ZHB/SpatioTemporalKG/scripts/remote/server_scripts/ 下
# 用法:  在 SSH 登录服务器后 source 这个脚本快捷使用
# ============================================================================
set -e

STK_HOME="/home/aisecurity/01_ZHB/SpatioTemporalKG"
STK_CARLA="/home/aisecurity/Carla"
STK_CONDA="/home/aisecurity/miniconda3/envs/stk"
STK_PYTHON="$STK_CONDA/bin/python"

# ── 别名 ──────────────────────────────────────────────────────────────────────
alias stk-cd="cd $STK_HOME"
alias stk-py="$STK_PYTHON"
alias stk-activate="source $STK_CONDA/bin/activate 2>/dev/null || echo 'conda activate not available, use: conda activate stk'"

# ── 场景回放 ──────────────────────────────────────────────────────────────────
stk-replay() {
    $STK_PYTHON $STK_HOME/scripts/build_replay_from_scenario.py "$@"
}

# ── 测试 ───────────────────────────────────────────────────────────────────────
stk-test() {
    (cd $STK_HOME && $STK_PYTHON -m pytest tests/ -q --tb=line "$@")
}

stk-test-all() {
    (cd $STK_HOME && $STK_PYTHON -m pytest tests/ -v --tb=short "$@")
}

# ── CARLA 操作 ────────────────────────────────────────────────────────────────
stk-carla-ps() {
    ps aux | grep CarlaUE4 | grep -v grep || echo "No CarlaUE4 process"
}
stk-carla-log() {
    tail -50 /home/aisecurity/carla_server.log 2>/dev/null || echo "No carla log found"
}

stk-carla-start() {
    local mode="${1:--RenderOffScreen}"
    cd $STK_CARLA
    nohup ./CarlaUE4.sh $mode -nosound -carla-rpc-port=2000 -quality-level=Low \
        > /home/aisecurity/carla_server.log 2>&1 &
    echo "CARLA PID: $!"
    sleep 20
    echo "--- process ---"
    stk-carla-ps
    echo "--- recent log ---"
    stk-carla-log
}

stk-carla-stop() {
    pkill -f CarlaUE4 && sleep 2 && echo "stopped" || echo "none running"
}

stk-carla-test() {
    $STK_PYTHON -c "
import carla
c = carla.Client('localhost', 2000)
c.set_timeout(20.0)
print('CARLA server:', c.get_server_version())
w = c.get_world()
print('Map:', w.get_map().name)
a = w.get_actors()
print(f'Vehicles: {len(a.filter(\"vehicle.*\"))}, Walkers: {len(a.filter(\"walker.*\"))}, Tls: {len(a.filter(\"traffic.traffic_light*\"))}')
"
}

# ── Git 快捷 ──────────────────────────────────────────────────────────────────
stk-git-status() {
    (cd $STK_HOME && git status -s)
}
stk-git-log() {
    (cd $STK_HOME && git log --oneline -10)
}

# ── Neo4j ─────────────────────────────────────────────────────────────────────
stk-neo4j-status() {
    systemctl status neo4j 2>&1 | head -10
}

# ── GPU / 资源 ────────────────────────────────────────────────────────────────
stk-gpu() {
    nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv
}

stk-disk() {
    df -h /home/aisecurity
}

# ── 帮助 ──────────────────────────────────────────────────────────────────────
stk-help() {
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  SpatioTemporalKG Server Tools                          ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  stk-activate      激活 conda stk 环境                  ║"
    echo "║  stk-cd            跳转项目目录                         ║"
    echo "║  stk-py            使用 stk 环境的 Python               ║"
    echo "║                                                         ║"
    echo "║  stk-replay        跑 14 场景回放 (场景图 JSON)          ║"
    echo "║  stk-test          跑 pytest (快速)                     ║"
    echo "║  stk-test-all      跑全部测试 (详细)                     ║"
    echo "║                                                         ║"
    echo "║  stk-carla-ps      CARLA 进程状态                       ║"
    echo "║  stk-carla-start  启动 CARLA 服务器                     ║"
    echo "║  stk-carla-stop   停止 CARLA 服务器                     ║"
    echo "║  stk-carla-test   测试 CARLA 连通性                     ║"
    echo "║  stk-carla-log    查看 CARLA 日志                       ║"
    echo "║                                                         ║"
    echo "║  stk-gpu           GPU 状态                             ║"
    echo "║  stk-disk          磁盘空间                             ║"
    echo "║  stk-neo4j-status  Neo4j 状态                           ║"
    echo "║  stk-git-status    Git 状态                             ║"
    echo "║                                                         ║"
    echo "║  stk-help          显示本帮助                           ║"
    echo "╚══════════════════════════════════════════════════════════╝"
}

echo ">>> SpatioTemporalKG server tools loaded. Run 'stk-help' for help."
