#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
free_gpu.py  --  查询 GPU 占用情况，挑选空闲卡
================================================

用法
----
    # 默认：列出每张卡状态 + 给出可用卡推荐
    python free_gpu.py

    # 只用简表
    python free_gpu.py --brief

    # 直接输出最适合训练的一张卡号（供脚本用）
    export CUDA_VISIBLE_DEVICES=$(python free_gpu.py --pick-one)

    # 输出所有空闲卡号,逗号分隔
    export CUDA_VISIBLE_DEVICES=$(python free_gpu.py --pick-all)

    # 排除某几张卡
    python free_gpu.py --exclude 0,2

    # 输出 JSON 格式
    python free_gpu.py --json

判断规则
--------
一张卡被判为"空闲"，需同时满足：
  - 显存占用 < mem_threshold (默认 2 GiB = 2048 MiB)
  - GPU 利用率 < util_threshold (默认 10%)
  - 计算进程数 == 0（不算 Xorg 这种图形进程）

依赖：nvidia-smi（NVIDIA 驱动自带）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class GPUInfo:
    index: int
    name: str
    temperature: float
    utilization: float
    memory_used: float
    memory_total: float
    processes: List[dict] = field(default_factory=list)
    is_free: bool = True
    free_reason: str = ""

    @property
    def memory_free_mib(self) -> float:
        return self.memory_total - self.memory_used

    @property
    def memory_free_gib(self) -> float:
        return self.memory_free_mib / 1024.0


def _run_nvidia_smi(args):
    try:
        return subprocess.check_output(["nvidia-smi", *args], text=True, timeout=15)
    except FileNotFoundError:
        print("[FATAL] nvidia-smi 不在 PATH", file=sys.stderr)
        sys.exit(2)
    except subprocess.CalledProcessError as e:
        print(f"[FATAL] nvidia-smi 失败: {e}", file=sys.stderr)
        sys.exit(e.returncode)
    except subprocess.TimeoutExpired:
        print("[FATAL] nvidia-smi 超时", file=sys.stderr)
        sys.exit(3)


def parse_gpus():
    out = _run_nvidia_smi([
        "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            continue
        gpus.append(GPUInfo(
            index=int(parts[0]),
            name=parts[1],
            temperature=float(parts[2]),
            utilization=float(parts[3]),
            memory_used=float(parts[4]),
            memory_total=float(parts[5]),
        ))
    return gpus


def parse_processes(gpus):
    out_uuid = _run_nvidia_smi(["--query-gpu=index,uuid", "--format=csv,noheader"])
    uuid_to_idx = {}
    for line in out_uuid.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            uuid_to_idx[parts[1]] = int(parts[0])

    out_proc = _run_nvidia_smi([
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    for line in out_proc.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_uuid, pid, proc_name, mem_used = parts[0], parts[1], parts[2], parts[3]
        idx = uuid_to_idx.get(gpu_uuid)
        if idx is None:
            continue
        try:
            mem = float(mem_used)
        except ValueError:
            mem = 0.0
        for g in gpus:
            if g.index == idx:
                g.processes.append({"pid": pid, "name": proc_name, "memory_mib": mem})
                break


def classify(gpus, mem_threshold, util_threshold):
    for g in gpus:
        if g.memory_used >= mem_threshold:
            g.is_free = False
            g.free_reason = f"显存 {g.memory_used:.0f}MiB >= 阈值{mem_threshold:.0f}MiB"
        elif g.utilization >= util_threshold:
            g.is_free = False
            g.free_reason = f"利用率 {g.utilization:.0f}% >= 阈值{util_threshold:.0f}%"
        elif g.processes:
            g.is_free = False
            g.free_reason = f"有{len(g.processes)}个进程(PIDs={[p['pid'] for p in g.processes]})"
        else:
            g.is_free = True
            g.free_reason = ""


def pick_free(gpus, exclude):
    return [g for g in gpus if g.is_free and g.index not in exclude]


BAR_LEN = 32

def mem_bar(used, total):
    if total <= 0:
        return "[...]"
    pct = used / total * 100
    filled = int(pct / 100 * BAR_LEN)
    return "|" + "#" * filled + "." * (BAR_LEN - filled) + f"| {pct:5.1f}%"


def print_table(gpus, brief=False):
    print()
    print("=" * 84)
    print(f"{'Idx':<4} {'Name':<22} {'Temp':<7} {'Util':<7} {'Memory Used / Total':<30} {'Status'}")
    print("-" * 84)
    for g in gpus:
        status = "[FREE] " if g.is_free else "[BUSY] "
        mem_str = f"{g.memory_used:>7.0f} / {g.memory_total:<7.0f} MiB  {mem_bar(g.memory_used, g.memory_total)}"
        print(f"{g.index:<4} {g.name:<22} {g.temperature:>4.0f}C {g.utilization:>5.0f}%  {mem_str}  {status}")
        if not brief and not g.is_free:
            print(f"     -> {g.free_reason}")
            for p in g.processes:
                name = p["name"][:40] + "..." if len(p["name"]) > 40 else p["name"]
                print(f"     -> PID {p['pid']:<8} {name:<42} {p['memory_mib']:.0f}MiB")
    print("=" * 84)


def print_recommendation(free_gpus):
    print()
    if not free_gpus:
        print("[!] 当前没有空闲 GPU！")
        print("    建议等一会儿再跑, 或用 --mem-threshold / --util-threshold 调整")
        return
    ranked = sorted(free_gpus, key=lambda g: (-g.memory_free_mib, g.utilization, g.temperature))
    best = ranked[0]
    print(f"[OK] 可用 GPU: {[g.index for g in free_gpus]} ({len(free_gpus)} 张)")
    print(f"[*] 推荐使用 GPU {best.index} (空闲 {best.memory_free_gib:.1f}GiB, {best.temperature:.0f}C)")


def main():
    p = argparse.ArgumentParser(description="查询 GPU 占用情况，挑选空闲卡用于训练")
    p.add_argument("--mem-threshold", type=float, default=2048.0,
                   help="显存阈值(MiB), 超则不算空闲. 默认 2048")
    p.add_argument("--util-threshold", type=float, default=10.0,
                   help="利用率阈值(%%), 超则不算空闲. 默认 10")
    p.add_argument("--exclude", type=str, default="",
                   help="排除卡号,逗号分隔. 例: 0,2")
    p.add_argument("--brief", action="store_true")
    p.add_argument("--pick-one", action="store_true")
    p.add_argument("--pick-all", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    exclude = {int(x.strip()) for x in args.exclude.split(",") if x.strip()}
    gpus = parse_gpus()
    if not gpus:
        print("[FATAL] 没检测到 GPU", file=sys.stderr)
        sys.exit(1)

    parse_processes(gpus)
    classify(gpus, args.mem_threshold, args.util_threshold)
    free_gpus = pick_free(gpus, exclude)

    if args.pick_one:
        if not free_gpus:
            sys.exit(1)
        best = sorted(free_gpus, key=lambda g: (-g.memory_free_mib, g.utilization, g.temperature))[0]
        print(best.index)
        return

    if args.pick_all:
        if not free_gpus:
            sys.exit(1)
        print(",".join(str(g.index) for g in free_gpus))
        return

    if args.json:
        data = {"gpus": [asdict(g) for g in gpus],
                "free_indices": [g.index for g in free_gpus],
                "excluded": sorted(exclude),
                "thresholds": {"mem_mib": args.mem_threshold, "util_pct": args.util_threshold}}
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    print_table(gpus, brief=args.brief)
    print_recommendation(free_gpus)

    if free_gpus:
        best = sorted(free_gpus, key=lambda g: (-g.memory_free_mib, g.utilization, g.temperature))[0]
        all_free = ",".join(str(g.index) for g in free_gpus)
        print()
        print("Usage tips:")
        print(f"  CUDA_VISIBLE_DEVICES={best.index} python your_train.py")
        print(f"  CUDA_VISIBLE_DEVICES={all_free} python your_train.py")
        print(f'  export CUDA_VISIBLE_DEVICES=$(python free_gpu.py --pick-one)')
        print(f'  export CUDA_VISIBLE_DEVICES=$(python free_gpu.py --pick-all)')


if __name__ == "__main__":
    main()
