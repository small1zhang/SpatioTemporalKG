#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cross_validate.py -- 一键运行多 town 进程级隔离 CARLA 交叉验证

功能:
  - 自动检测能 import carla 的 python 解释器 (stk conda env, py3.10)
  - 对每个 town: kill 端口 -> cold-boot 全新 CARLA -> 加载该 town 的地图
    (通过编辑 DefaultEngine.ini + backup/restore) -> 跑 5 阶段管线
    -> kill 该端口的 CARLA -> 进入下一张 town
  - 整个过程不调用 client.load_world(), 根除 UE4 SIGSEGV

默认 town 列表: Town10HD, Town01, Town05
  (Town03 已剔除 -- CARLA 0.9.16 该 town 资产损坏, 即使 cold-boot 直接
   启动也会 SIGSEGV, 不是脚本层能修的问题)

用法 (默认 15 帧, 8 车 3 行人, GPU3):
    python3 scripts/pipeline/run_cross_validate.py

完整参数:
    python3 scripts/pipeline/run_cross_validate.py \\
        --frames 30 \\
        --vehicles 15 \\
        --walkers 5 \\
        --towns Town10HD,Town01,Town05 \\
        --port-base 2300 \\
        --graphics-adapter 3

输出:
    data/runs/cross_validation/
        phases_<timestamp>_<N>f/         -- 每个 town 一份
            metadata.json                 -- 跑参数
            phase5_graph.json              -- KG
            phase5_kg_summary.json        -- KG 摘要
        compare_pm_<timestamp>.json       -- 三镇对比
        cross_validate_report_<ts>.md     -- 对比表 (mk-down)
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
CV_SCRIPT = _REPO / "scripts" / "pipeline" / "cross_validate.py"
REPORT_SCRIPT = _REPO / "scripts" / "pipeline" / "cross_validate_report.py"


def main():
    p = argparse.ArgumentParser(
        description="Run CARLA cross-validation with process-level isolation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames", type=int, default=15,
                   help="采样帧数")
    p.add_argument("--vehicles", type=int, default=8,
                   help="每帧车辆数")
    p.add_argument("--walkers", type=int, default=3,
                   help="每帧行人数")
    p.add_argument("--towns", type=str,
                   default="Town10HD,Town01,Town05",
                   help="逗号分隔的 town 列表 (避开 Town03; CARLA 0.9.16 资产崩)")
    p.add_argument("--port-base", type=int, default=2300,
                   help="CARLA RPC 起始端口, 第 i 个 town 用 port_base + i")
    p.add_argument("--graphics-adapter", type=int, default=3,
                   help="使用的 GPU index (CUDA_VISIBLE_DEVICES)")
    p.add_argument("--report", action="store_true", default=True,
                   help="跑完后自动生成对比 report (cross_validate_report.py)")
    p.add_argument("--no-report", dest="report", action="store_false",
                   help="跳过 report 生成")
    p.add_argument("--out", type=str,
                   default="data/runs/cross_validation",
                   help="输出目录 (相对 repo root)")
    args = p.parse_args()

    if not CV_SCRIPT.exists():
        print(f"[fatal] cross_validate.py not found at {CV_SCRIPT}")
        sys.exit(1)

    cmd = [
        sys.executable, str(CV_SCRIPT),
        "--frames", str(args.frames),
        "--vehicles", str(args.vehicles),
        "--walkers", str(args.walkers),
        "--towns", args.towns,
        "--port-base", str(args.port_base),
        "--graphics-adapter", str(args.graphics_adapter),
        "--out", args.out,
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.graphics_adapter)}

    print("=" * 76)
    print(f"  cross-validation run  --  {datetime.now().isoformat()}")
    print("=" * 76)
    print(f"  towns       : {args.towns}")
    print(f"  frames      : {args.frames}")
    print(f"  vehicles    : {args.vehicles}")
    print(f"  walkers     : {args.walkers}")
    print(f"  port_base   : {args.port_base}")
    print(f"  GPU adapter : {args.graphics_adapter}")
    print(f"  output      : {_REPO / args.out}")
    print(f"  python      : {sys.executable}")
    print("=" * 76)

    # 直接前台跑 (子进程会继承 stdio, 实时进度可见)
    rc = subprocess.call(cmd, cwd=str(_REPO), env=env)
    if rc != 0:
        print(f"\n[error] cross_validate.py exited with rc={rc}")
        sys.exit(rc)

    if args.report:
        print("\n" + "=" * 76)
        print("  Generating cross-validation report ...")
        print("=" * 76)
        rc2 = subprocess.call([sys.executable, str(REPORT_SCRIPT)],
                              cwd=str(_REPO))
        if rc2 != 0:
            print(f"[warn] cross_validate_report.py exited with rc={rc2}")


if __name__ == "__main__":
    main()
