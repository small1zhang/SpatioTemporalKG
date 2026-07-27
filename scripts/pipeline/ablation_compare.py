#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_compare.py -- 退化式 + 配置开关图谱对比实验的批跑与聚合工具

设计目标（对应 PLAN 4.3.1 RQ1.5 + 4.4.1 RQ2.6）:
  A. 退化式对比 (DG-A ~ DG-F): 把完整 STKG 退化为缺少某创新组件的"简化图谱"
  B. 配置开关对比 (CFG-1 ~ CFG-6): 对已暴露的配置开关做配置空间扫描
  C. Tier 增量贡献 (A→D): 按风险梯度逐层级评测图谱增量与检出增量

工作原理:
  1. 从 scenario_library.SCENARIO_REGISTRY 读取 14 场景的 expected_rules/behaviors
  2. 为每个对比体生成对应的 run_phases_1_5.py 命令行参数
  3. (在线模式) 实际调用 run_phases_1_5.py 子进程跑 CARLA
  4. (离线模式) 扫描已有 data/runs/ 目录进行后处理分析
  5. 聚合所有结果 → data/runs/ablation/<config_name>/ → 输出对照表 + Markdown 报告

依赖数据:
  - data/dataset/frame_actors.csv  (1.34M 行 × 38 列，离线复跑 detector 用)
  - data/long_run/<map>_20min/     (chunk_*.json 原始帧，在线跑用)
  - data/runs/ablation/            (本脚本产出目录)

产出:
  - ablation_summary.md           (主对照表，Table 5-A/B/C + Fig.2-A/B)
  - config_comparison.json        (结构化 JSON，供论文图表生成)
  - <config_name>/results.json    (每个对比体的详细指标)

使用示例:
  # 离线模式：在已有 data 上跑退化体分析（需要 CARLA 环境才能在线跑）
  python scripts/pipeline/ablation_compare.py --mode offline --runs-dir data/runs

  # 在线模式：自动调用 run_phases_1_5.py（需要 CARLA server 在运行）
  python scripts/pipeline/ablation_compare.py --mode online --town Town10HD

  # 只跑配置扫描（不需要 CARLA，纯离线分析）
  python scripts/pipeline/ablation_compare.py --mode configs-only --runs data/runs/ablation/baseline

注意:
  退化体 (DG-A ~ DG-F) 的"关闭某功能"需要 stk/ 层级的配置开关配合，
  脚本只负责生成正确的命令行参数。实际开关已暴露在 stk/config.py 中。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# 路径与常量
# ──────────────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

OUT_DIR = _REPO / "data" / "runs" / "ablation"
RUN_PHASES = _REPO / "scripts" / "pipeline" / "run_phases_1_5.py"
FRAME_ACTORS = _REPO / "data" / "dataset" / "frame_actors.csv"
LONG_RUN_DIR = _REPO / "data" / "long_run"
SCENARIO_LIB = _REPO / "stk" / "scenario" / "scenario_library.py"


# ──────────────────────────────────────────────────────────────────────────────
# 退化体定义（DG-A ~ DG-F）
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class DegradedConfig:
    """单个退化体的配置映射。每个字段对应可切换的开关/行为。"""
    name: str
    label: str                       # 论文友好名称
    description: str                 # 一句话说明
    # 对应 stk/config.py 中的可配置字段
    legacy_full_pairing: bool = False        # DG-C 用 True(=全量O(N²))
    filter_behavior_detectors: bool = False  # DG-F 改为 True(=全量扫描)
    filter_scene_spatial: bool = False       # DG-F 改为 True(=全量扫描)
    exclude_lanes: bool = False              # DG-E / DG-F 改为 False(=保留) / DG-C 也可
    exclude_road_elements: bool = True       # 默认不必改
    importance_threshold: float = 0.30       # DG-C / DG-A 用 -1(=disable)
    time_downsample: int = 1                 # DG-D 改为 20(=20帧合并为1)
    # 以下为 PipelineOrchestrator 级行为（非简单 flag）
    dynamic_phase: str = "incremental"       # "incremental" / "full_batch" / "skip"
    rule_phase: str = "enabled"              # "enabled" / "skip"
    cross_layer_bridge: str = "enabled"      # "enabled" / "skip"
    version_manager: str = "enabled"         # "enabled" / "skip"

    def to_thresholds_json(self) -> str:
        """序列化部分能在 --thresholds-json 中覆盖的字段。"""
        d = {"importance_threshold": self.importance_threshold}
        return json.dumps(d)

    def to_cli_args(self) -> List[str]:
        """生成 run_phases_1_5.py 可识别的命令行参数。"""
        args: List[str] = []
        if self.legacy_full_pairing:
            # 需要在 ego_centric.yaml 中 legacy_full_pairing: true
            # 同时传递 --ego-centric 才能启用该路径
            args += ["--ego-centric", "--legacy-full-pairing"]
        if self.exclude_lanes is False:
            args += ["--include-lanes"]  # 需在 pipeline 侧支持
        # 注意：多数开关需要改 YAML 配置或用 --thresholds-json 覆盖
        # 此方法列出"可表达为 CLI 的参量"，非完整覆盖面
        if self.time_downsample > 1:
            args += [f"--time-downsample", str(self.time_downsample)]
        return args


# ──────────────────────────────────────────────────────────────────────────────
# 退化体清单（对应 PLAN 4.3.1 RQ1.5 节表格）
# ──────────────────────────────────────────────────────────────────────────────
DEGRADED_CONFIGS: Dict[str, DegradedConfig] = {
    "DG-A": DegradedConfig(
        name="DG-A", label="STKG-tiny",
        description="去掉属性版本化 → 静态属性图谱",
        importance_threshold=-1.0,  # 关闭重要性打分
        version_manager="skip",
    ),
    "DG-B": DegradedConfig(
        name="DG-B", label="STKG-static",
        description="去掉差分图 + 生命周期 → 静态交通本体",
        dynamic_phase="full_batch",
        version_manager="skip",
    ),
    "DG-C": DegradedConfig(
        name="DG-C", label="STKG-noRule",
        description="去掉规则层（RSS + 14 交规）→ nuScenes 类无规则嵌入",
        rule_phase="skip",
        exclude_lanes=False,
        legacy_full_pairing=True,  # 全量扫描作为退化对照
    ),
    "DG-D": DegradedConfig(
        name="DG-D", label="STKG-flatTime",
        description="50ms 帧级 → 秒级时间合并 → 传统 TKG 对照",
        time_downsample=20,
        legacy_full_pairing=True,
    ),
    "DG-E": DegradedConfig(
        name="DG-E", label="STKG-noCrossLayer",
        description="去掉跨层桥接边 → 场景层与行为层不联动",
        cross_layer_bridge="skip",
    ),
    "DG-F": DegradedConfig(
        name="DG-F", label="STKG-noSceneFilter",
        description="关闭 ROI 与背景过滤 → 全量扫描图谱",
        filter_behavior_detectors=True,
        filter_scene_spatial=True,
        exclude_lanes=False,
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# 配置开关扫描定义（对应 PLAN 4.4.1 RQ2.6 节表格）
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ConfigScan:
    """单个配置扫描项：一个因子 × 一组取值。"""
    name: str
    factor: str                   # CLI / YAML 中对应的配置字段
    values: List[Any]             # 扫描取值
    unit: str = ""                # 单位说明（如 "m" / "s"）
    metrics: List[str] = field(  # 该扫描关注的指标
        default_factory=lambda: [
            "node_count", "edge_count",
            "node_degree_avg", "behavior_f1", "rule_cd", "fps",
        ]
    )


CONFIG_SCANS: Dict[str, ConfigScan] = {
    "CFG-1": ConfigScan(
        name="CFG-1", factor="legacy_full_pairing",
        values=[True, False],
        unit="bool",
        metrics=["sv_count", "rss_runtime_s", "fps", "total_nodes"],
    ),
    "CFG-2": ConfigScan(
        name="CFG-2", factor="importance_threshold",
        values=[-1.0, 0.10, 0.20, 0.30, 0.40, 0.50],
        unit="",
        metrics=["node_cull_rate", "relation_f1", "behavior_f1", "rule_dr", "fps"],
    ),
    "CFG-3": ConfigScan(
        name="CFG-3", factor="exclude_lanes",
        values=[True, False],
        unit="bool",
        metrics=["node_count", "edge_count", "avg_degree", "rule_dr", "peak_mb"],
    ),
    "CFG-4": ConfigScan(
        name="CFG-4", factor="filter_behavior_detectors",
        values=[True, False],
        unit="bool",
        metrics=["maneuver_count", "interaction_count", "behavior_coverage", "fps"],
    ),
    "CFG-5": ConfigScan(
        name="CFG-5", factor="filter_scene_spatial",
        values=[True, False],
        unit="bool",
        metrics=["scene_rel_count", "relation_coverage", "relation_f1", "fps"],
    ),
    "CFG-6": ConfigScan(
        name="CFG-6", factor="threshold_sensitivity",
        values=[  # 组合：(ttc_critical, pedestrian_distance)
            (2.0, 3.0), (2.0, 5.0), (2.0, 7.0), (3.0, 5.0),
            (4.0, 5.0), (5.0, 7.0), (6.0, 10.0),
        ],
        unit="(s, m)",
        metrics=["behavior_f1", "rule_dr", "rule_far"],
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# Tier 增量对比（对应 PLAN 4.3.1 RQ1.5-C 节表格）
# ──────────────────────────────────────────────────────────────────────────────
TIER_SCENARIOS: Dict[str, List[str]] = {
    "A": ["S00", "S01", "S02"],
    "B": ["S10", "S11", "S12", "S13"],
    "C": ["S20", "S21", "S22"],
    "D": ["S30", "S31", "S32", "S33"],
}


# ──────────────────────────────────────────────────────────────────────────────
# 核心逻辑
# ──────────────────────────────────────────────────────────────────────────────

def _load_registry() -> Dict[str, Dict[str, Any]]:
    """动态加载 SCENARIO_REGISTRY（避免硬编码）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "scenario_library", str(SCENARIO_LIB)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SCENARIO_REGISTRY


def _build_run_dir(config_name: str, scenario_id: Optional[str] = None) -> Path:
    """构造产出目录: data/runs/ablation/<config_name>/ 或加 /<scenario_id>/"""
    d = OUT_DIR / config_name
    if scenario_id:
        d = d / scenario_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _gen_run_command(
    town: str,
    frames: int,
    config: DegradedConfig,
    output_dir: Path,
) -> List[str]:
    """为单个退化体生成 run_phases_1_5.py 调用命令。

    ⚠️  注意: 以下 CLI flag 在 run_phases_1_5.py 中可能尚未实现，
    需要先匹配/新增 flag 对应关系。此处作为"脚手架草案"——写完需对照
    run_phases_1_5.py 的 argparse 修正。
    """
    cmd: List[str] = [
        sys.executable, str(RUN_PHASES),
        "--town", town,
        "--frames", str(frames),
        "--out", str(output_dir.parent),  # 父目录，run_phases 内部加时间戳
    ]

    # ── 策略 1: ego-centric 模式 (触发 legacy_full_pairing 路径) ──
    if config.legacy_full_pairing:
        cmd += ["--ego-centric"]
        # legacy_full_pairing 是在 EgoCentricConfig 中，
        # 需要改 ego_centric.yaml 才能生效；这里只记录需求
        # 实际方案见 _write_ego_centric_override()

    # ── 策略 2: 阈值覆盖（importance_threshold、不用时传覆盖） ──
    if config.importance_threshold != 0.30:
        cmd += [
            "--thresholds-json",
            json.dumps({"importance_threshold": config.importance_threshold}),
        ]

    # ── 策略 3: exclude_lanes (需 pipeline 侧支持 --include-lanes) ──
    if not config.exclude_lanes:
        cmd.append("--include-lanes")

    # ── 策略 4: time_downsample (需 pipeline 侧新增 flag) ──
    if config.time_downsample > 1:
        cmd += ["--time-downsample", str(config.time_downsample)]

    # ── 策略 5: 写标识 JSON 让后续识别本跑次是哪个退化体 ──
    meta_path = output_dir / "ablation_meta.json"
    meta_path.write_text(json.dumps({
        "config_name": config.name,
        "label": config.label,
        "description": config.description,
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    return cmd


def _write_ego_centric_override(
    config: DegradedConfig,
    target_path: Path,
) -> Path:
    """为特定跑次生成一份 ego_centric.yaml 覆盖文件。

    当退化体需要 legacy_full_pairing=True / exclude_lanes=False /
    特定 filter_* 值时，写一份 YAML 供 run_phases_1_5.py --config 传入。
    """
    import yaml
    cfg = {
        "ego_id_opt": None,
        "radius_front": 70.0,
        "radius_rear": 30.0,
        "radius_side": 50.0,
        "legacy_full_pairing": config.legacy_full_pairing,
        "filter_behavior_detectors": config.filter_behavior_detectors,
        "filter_scene_spatial": config.filter_scene_spatial,
        "exclude_lanes": config.exclude_lanes,
        "exclude_road_elements": True,
        "importance_threshold": config.importance_threshold,
    }
    target_path.write_text(
        yaml.dump(cfg, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target_path


def _run_online(
    config: DegradedConfig,
    town: str,
    frames: int,
    run_id: str,
) -> Dict[str, Any]:
    """在线模式：调用 run_phases_1_5.py 子进程跑 CARLA。

    需要 CARLA server 已在目标 town 上运行。
    """
    run_dir = _build_run_dir(run_id)
    ego_yaml = run_dir / "ego_centric_override.yaml"
    _write_ego_centric_override(config, ego_yaml)

    cmd = _gen_run_command(town, frames, config, run_dir)
    cmd += ["--ego-centric-config", str(ego_yaml)]

    print(f"\n[ablation] Running {config.label} on {town} ({frames}f)...")
    print(f"  cmd: {' '.join(cmd[:6])} ...")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        wall_s = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_s": 600, "error": "process timeout"}

    result: Dict[str, Any] = {
        "status": "ok" if proc.returncode == 0 else f"rc={proc.returncode}",
        "wall_s": round(wall_s, 2),
        "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
        "run_dir": str(run_dir),
    }

    # 保存元数据
    meta_path = run_dir / "ablation_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"config": asdict(config)}
    meta.update(result)
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return result


def _load_run_results(run_dir: Path) -> Dict[str, Any]:
    """从 run_dir 中加载已有实验结果（phase5_graph.json + metadata.json）。

    兼容 cross_validation 目录格式和 ablation 目录格式。
    """
    phase5 = run_dir / "phase5_graph.json"
    meta_file = run_dir / "metadata.json"
    ablation_meta = run_dir / "ablation_meta.json"

    if not phase5.exists():
        return {}
    graph = json.loads(phase5.read_text(encoding="utf-8"))
    
    meta: Dict[str, Any] = {}
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    if ablation_meta.exists():
        try:
            meta.update(json.loads(ablation_meta.read_text(encoding="utf-8")))
        except Exception:
            pass

    # 从 graph 中提取可比较的指标
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    metrics: Dict[str, Any] = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "avg_degree": round(len(edges) / max(len(nodes), 1), 2),
    }

    # 按类型统计
    node_type_counts: Counter = Counter(
        n.get("entity_type", n.get("type", "unknown")) for n in nodes
    )
    edge_type_counts: Counter = Counter(
        e.get("relation_type", e.get("type", "unknown")) for e in edges
    )
    metrics["node_type_counts"] = dict(node_type_counts)
    metrics["edge_type_counts"] = dict(edge_type_counts)

    # 统计 SafetyViolation / ManeuverNode / InteractionEvent
    metrics["violation_count"] = node_type_counts.get("SafetyViolation", 0)
    metrics["maneuver_count"] = node_type_counts.get("ManeuverNode", 0)
    metrics["interaction_count"] = node_type_counts.get("InteractionEvent", 0)

    # 如果有 n_violations / n_deltas 等帧级字段
    frame_results = graph.get("frame_results", [])
    if frame_results:
        metrics["avg_violations_per_frame"] = round(
            sum(f.get("n_violations", 0) for f in frame_results) / max(len(frame_results), 1),
            2,
        )
        metrics["avg_deltas_per_frame"] = round(
            sum(f.get("n_deltas", 0) for f in frame_results) / max(len(frame_results), 1),
            2,
        )

    metrics["meta"] = meta
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 配置扫描执行（RQ2.6：不需要 CARLA，纯配置参数组合分析）
# ──────────────────────────────────────────────────────────────────────────────
def _run_config_scan_offline(
    scan: ConfigScan,
    existing_base: Path,
    run_id: str,
) -> Dict[str, Any]:
    """离线配置扫描：基于已有基准跑次（baseline），通过统计推断不同配置的 *趋势*。

    真正跑完整扫描需要在线模式（需 CARLA），本方法提供:
    1. 基于已有 run 的统计报告
    2. 为在线模式准备命令矩阵（--config-scan CFG-2 --values ...）
    """
    run_dir = _build_run_dir(run_id)
    scan_dir = run_dir / "config_scan" / scan.name

    # 如果已有 baseline 数据，复制并标记
    if existing_base.exists():
        baseline_metrics = _load_run_results(existing_base)
    else:
        baseline_metrics = {}

    output = {
        "scan_name": scan.name,
        "factor": scan.factor,
        "values": scan.values,
        "baseline_metrics": baseline_metrics,
        "commands": [],  # 各类值对应的 run 命令（在线模式用）
    }

    for v in scan.values:
        # 生成该取值对应的 run 命令
        if scan.factor == "legacy_full_pairing":
            cfg = DegradedConfig(
                name=f"{scan.name}-v{v}",
                label=f"legacy_full_pairing={v}",
                description=f"RSS 全量 vs ROI, pairing={v}",
                legacy_full_pairing=v,
            )
        elif scan.factor == "importance_threshold":
            cfg = DegradedConfig(
                name=f"{scan.name}-thr{v}",
                label=f"importance_threshold={v}",
                description=f"节点裁剪阈值={v}",
                importance_threshold=v,
            )
        elif scan.factor == "exclude_lanes":
            cfg = DegradedConfig(
                name=f"{scan.name}-lanes{v}",
                label=f"exclude_lanes={v}",
                description=f"排除车道={v}",
                exclude_lanes=not v,
            )
        elif scan.factor == "filter_behavior_detectors":
            cfg = DegradedConfig(
                name=f"{scan.name}-bhfilter{v}",
                label=f"filter_behavior_detectors={v}",
                description=f"行为级ROI={v}",
                filter_behavior_detectors=not v,
            )
        elif scan.factor == "filter_scene_spatial":
            cfg = DegradedConfig(
                name=f"{scan.name}-scfilter{v}",
                label=f"filter_scene_spatial={v}",
                description=f"场景级ROI={v}",
                filter_scene_spatial=not v,
            )
        elif scan.factor == "threshold_sensitivity":
            ttc, ped_dist = v
            cfg = DegradedConfig(
                name=f"{scan.name}-ttc{ttc}_ped{ped_dist}",
                label=f"TTC={ttc}, PedDist={ped_dist}",
                description=f"阈值组合 ({ttc}s, {ped_dist}m)",
                importance_threshold=-1.0,  # 不影响裁剪
            )
            # 阈值用 --thresholds-json 传，见 _gen_run_command
            cmd = _gen_run_command(
                town="", frames=0, config=cfg, output_dir=scan_dir,
            )
            override = json.dumps({
                "ttc_critical": ttc,
                "pedestrian_distance": ped_dist,
                "pedestrian_activation_distance": ped_dist + 10.0,
            })
            cmd += ["--thresholds-json", override]
            output["commands"].append({"value": v, "cmd_extra": override})
            continue
        else:
            continue

        cmd = _gen_run_command(town="", frames=0, config=cfg, output_dir=scan_dir)
        output["commands"].append({"value": v, "config": asdict(cfg), "cmd": cmd})

    # 保存 scan 描述文件
    scan_json = scan_dir / "scan_definition.json"
    scan_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return output


# ──────────────────────────────────────────────────────────────────────────────
# Tier 增量分析（RQ1.5-C：不需要 CARLA，纯现有数据分析）
# ──────────────────────────────────────────────────────────────────────────────
def _analyze_tier_increments(
    registry: Dict[str, Dict[str, Any]],
    frame_labels_path: Path,
    run_dir: Path,
) -> Dict[str, Any]:
    """按 Tier A→D 分析每档风险梯度对应的图谱增量。

    输入:
      - SCENARIO_REGISTRY（14 场景含 tier 和 expected_* 字段）
      - frame_labels.csv（每帧含 scenario_id 标签）
      - 已有的 run 产物（或 baseline）

    输出:
      - per-tier 的节点/边/SV/跨层边增量表
    """
    import csv

    # ── 1. 统计每个 tier 的帧数 ──
    tier_frame_counts: Dict[str, int] = defaultdict(int)
    if frame_labels_path.exists():
        with open(frame_labels_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("scenario_id", "")
                tier = registry.get(sid, {}).get("tier", "?")
                tier_frame_counts[tier] += 1
        # 第一条非 A tier 的帧计入累计
    else:
        # fallback：用场景工厂中的帧数估算
        from stk.scenario.scenario_library import get_scenario
        for tier, sids in TIER_SCENARIOS.items():
            tier_frame_counts[tier] = sum(
                len(get_scenario(sid)) for sid in sids
            )

    # ── 2. 期望规则/行为/SV 计数 ──
    tier_rule_counts: Dict[str, int] = {}
    tier_behavior_sets: Dict[str, set] = {}
    tier_sv_counts: Dict[str, int] = {}
    for tier, sids in TIER_SCENARIOS.items():
        rules = set()
        behaviors = set()
        sv_count = 0
        for sid in sids:
            meta = registry.get(sid, {})
            rules.update(meta.get("expected_rules", []))
            behaviors.update(meta.get("expected_behaviors", []))
            sv = meta.get("expected_sv", "-")
            if sv != "-":
                sv_count += 1
        tier_rule_counts[tier] = len(rules)
        tier_behavior_sets[tier] = behaviors
        tier_sv_counts[tier] = sv_count

    # ── 3. 如果有已有 run 的图产物，提取实际节点/边数 ──
    # 尝试从 cross_validation 或 ablation 目录中找 baseline
    baseline_metrics: Dict[str, Any] = {}
    for p in (
        _REPO / "data" / "runs" / "cross_validation"
    ).iterdir():
        if p.is_dir() and (p / "phase5_graph.json").exists():
            baseline_metrics = _load_run_results(p)
            break

    # ── 4. 构造 tier 表格 ──
    tiers = ["A", "B", "C", "D"]
    cum_frames = 0
    cum_scenarios = 0
    cum_nodes = 0
    cum_edges = 0
    cum_rules = 0
    cum_sv = 0
    tier_data: List[Dict[str, Any]] = []

    for t in tiers:
        sids = TIER_SCENARIOS[t]
        n_frames = tier_frame_counts.get(t, len(sids) * 6)
        n_scenarios = len(sids)

        # 估算图谱增量（基于 baseline 的 14 场景平均值估算 per-scenario）
        baseline_node_count = baseline_metrics.get("node_count", 0)
        per_scenario_nodes = (
            baseline_node_count / max(sum(len(v) for v in TIER_SCENARIOS.values()), 1)
        )
        per_scenario_edges = (
            baseline_metrics.get("edge_count", 0)
            / max(sum(len(v) for v in TIER_SCENARIOS.values()), 1)
        )

        tier_data.append({
            "tier": t,
            "scenarios": sids,
            "n_scenarios": n_scenarios,
            "n_frames": n_frames,
            "expected_rules": tier_rule_counts.get(t, 0),
            "expected_sv": tier_sv_counts.get(t, 0),
            "expected_behaviors": len(tier_behavior_sets.get(t, set())),
        })

    return {
        "tier_data": tier_data,
        "scenario_registry": registry,
        "baseline_metrics": baseline_metrics,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 全局批跑编排
# ──────────────────────────────────────────────────────────────────────────────
def run_degraded_comparison(
    town: str,
    frames: int,
    mode: str = "offline",
    existing_runs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """执行退化式对比（RQ1.5-A）。

    Args:
        town:      CARLA 地图名（在线模式需要）
        frames:    每场景帧数
        mode:      "offline" / "online" / "both"
        existing_runs_dir: 已有 run 目录，离线模式读取用
    """
    registry = _load_registry()
    results: Dict[str, Dict[str, Any]] = {}

    for dg_id, dg_cfg in DEGRADED_CONFIGS.items():
        run_id = f"degraded_{dg_id}_{town}_{frames}f"
        print(f"\n[ablation] {dg_id}: {dg_cfg.label} — {dg_cfg.description}")

        if mode in ("online", "both"):
            res = _run_online(dg_cfg, town, frames, run_id)
            results[dg_id] = res
            print(f"  → status={res.get('status')}, wall={res.get('wall_s')}s")

        if mode in ("offline", "both"):
            # 尝试从已有 run 目录加载
            run_suffix = f"degraded_{dg_id}"
            if existing_runs_dir:
                candidates = [
                    p for p in existing_runs_dir.rglob(run_suffix)
                    if p.is_dir() and (p / "phase5_graph.json").exists()
                ]
            else:
                candidates = [
                    p for p in OUT_DIR.rglob(run_suffix)
                    if p.is_dir() and (p / "phase5_graph.json").exists()
                ]
            if candidates:
                metrics = _load_run_results(candidates[0])
                results[dg_id] = {"mode": "offline_load", **metrics}
            else:
                results[dg_id] = {
                    "mode": "offline_no_data",
                    "note": f"no run data found for {run_suffix}",
                }

    # 汇总
    summary_path = OUT_DIR / "degraded_comparison.json"
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[ablation] Summary saved → {summary_path}")
    return results


def run_config_scans(
    mode: str = "offline",
    baseline_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """执行配置开关扫描（RQ2.6 + RQ1.5-B）。

    Args:
        mode:          "offline" / "online"
        baseline_dir:  基准 run 目录（提供 baseline 指标）
    """
    all_scans: Dict[str, Any] = {}
    for scan_id, scan in CONFIG_SCANS.items():
        print(f"\n[ablation] Config scan {scan_id}: {scan.factor}")
        res = _run_config_scan_offline(scan, baseline_dir or Path("."), f"config_{scan_id}")
        all_scans[scan_id] = res

    # 汇总
    summary_path = OUT_DIR / "config_scans.json"
    summary_path.write_text(
        json.dumps(all_scans, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[ablation] Config scan summary → {summary_path}")
    return all_scans


def run_tier_increment_analysis(
    frame_labels: Optional[Path] = None,
) -> Dict[str, Any]:
    """执行 Tier A→D 增量贡献分析（RQ1.5-C）。纯离线，无需 CARLA。"""
    registry = _load_registry()
    labels_path = frame_labels or FRAME_ACTORS
    result = _analyze_tier_increments(registry, labels_path, OUT_DIR)

    # 保存
    tier_path = OUT_DIR / "tier_increment.json"
    tier_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[ablation] Tier increment analysis → {tier_path}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 报告生成：从所有聚合 JSON 生成 Markdown 对照表
# ──────────────────────────────────────────────────────────────────────────────

def generate_markdown_report(
    degraded_results: Optional[Dict] = None,
    config_results: Optional[Dict] = None,
    tier_results: Optional[Dict] = None,
) -> str:
    """生成 Markdown 格式的完整对照表报告。

    输出: data/runs/ablation/ablation_summary.md
    """
    lines: List[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# Ablation & Comparison Summary")
    lines.append(f"\n> Generated: {ts}")
    lines.append(f"> Source: `scripts/pipeline/ablation_compare.py`")
    lines.append("")

    # ── Table 5-A: 退化式对比 ──
    lines.append("## Table 5-A: 退化式图谱对比（RQ1.5-A）")
    lines.append("")
    lines.append("| 退化体 | 标签 | 去掉什么 | 节点数 | 边数 | 平均度 | Violation | Maneuver | Interaction |")
    lines.append("|--------|------|----------|:------:|:----:|:------:|:---------:|:--------:|:-----------:|")

    if degraded_results:
        for dg_id, dg_cfg in DEGRADED_CONFIGS.items():
            res = degraded_results.get(dg_id, {})
            metrics = res if isinstance(res, dict) and "node_count" in res else {}
            nc = metrics.get("node_count", "—")
            ec = metrics.get("edge_count", "—")
            ad = metrics.get("avg_degree", "—")
            sv = metrics.get("violation_count", metrics.get("avg_violations_per_frame", "—"))
            mn = metrics.get("maneuver_count", "—")
            ie = metrics.get("interaction_count", "—")
            if isinstance(sv, float):
                sv = f"{sv:.2f}"
            lines.append(
                f"| {dg_id} | {dg_cfg.label} | {dg_cfg.description} | "
                f"{nc} | {ec} | {ad} | {sv} | {mn} | {ie} |"
            )
    else:
        lines.append("| — | — | *离线模式：无在线数据* | — | — | — | — | — | — |")
    lines.append("")

    # ── Table 5-B: 配置扫描汇总 ──
    lines.append("## Table 5-B: 配置扫描汇总（RQ2.6）")
    lines.append("")
    for scan_id, scan in CONFIG_SCANS.items():
        lines.append(f"### {scan_id}: {scan.factor}")
        lines.append("")
        lines.append(f"扫描取值 ({len(scan.values)} 个): `{scan.values}`")
        lines.append(f"关注指标: `{scan.metrics}`")
        if config_results and scan_id in config_results:
            scan_data = config_results[scan_id]
            commands = scan_data.get("commands", [])
            if commands:
                lines.append("")
                lines.append("| 取值 | 对应标记 |")
                lines.append("|------|----------|")
                for cmd_info in commands:
                    val = cmd_info.get("value", "?")
                    label = cmd_info.get("config", {}).get("label", val)
                    lines.append(f"| {val} | {label} |")
        lines.append("")

    # ── Table 5-C: Tier 增量 ──
    lines.append("## Table 5-C: Tier A→D 增量贡献（RQ1.5-C）")
    lines.append("")
    lines.append("| Tier | 场景数 | 帧数 | 预期规则数 | 预期 SV 数 | 预期行为数 |")
    lines.append("|------|:------:|:----:|:----------:|:----------:|:----------:|")

    if tier_results and "tier_data" in tier_results:
        cum_scenarios = 0
        cum_frames = 0
        cum_rules = 0
        cum_sv = 0
        cum_beh = 0
        for td in tier_results["tier_data"]:
            lines.append(
                f"| {td['tier']} | {td['n_scenarios']} | {td['n_frames']} | "
                f"{td['expected_rules']} | {td['expected_sv']} | {td['expected_behaviors']} |"
            )
            cum_scenarios += td["n_scenarios"]
            cum_frames += td["n_frames"]
            cum_rules += td["expected_rules"]
            cum_sv += td["expected_sv"]
            cum_beh += td["expected_behaviors"]
        lines.append(
            f"| **累计** | **{cum_scenarios}** | **{cum_frames}** | "
            f"**{cum_rules}** | **{cum_sv}** | **{cum_beh}** |"
        )
    else:
        lines.append("| — | — | — | — | — | — | — |")
    lines.append("")

    # ── 注意事项 ──
    lines.append("## 注意事项")
    lines.append("")
    lines.append("- **离线模式**只能加载已有 run 的指标。退化体和配置扫描的完整实验需在 CARLA 环境下以在线模式运行。")
    lines.append("- 各退化体的'关闭某功能'需要在 `stk/config.py` 或对应模块中暴露开关。本脚本记录需求，实际开关已在 `stk/config.py` 中定义。")
    lines.append("- `importance_threshold=-1.0` 表示禁用过滤（不裁剪任何节点/边）。")
    lines.append("- 配置扫描的完整运行需要为每个取值生成一个独立 CARLA run。")
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="退化式 + 配置开关图谱对比实验批跑聚合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["offline", "online", "configs-only", "tier-only", "all"],
        default="offline",
        help="运行模式 (default: offline)",
    )
    parser.add_argument("--town", default="Town10HD", help="CARLA 地图名")
    parser.add_argument("--frames", type=int, default=60, help="每场景帧数")
    parser.add_argument("--runs-dir", type=str, default=None,
                        help="已有 run 目录路径 (offline 模式读取用)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印命令，不实际运行")

    # 退化体选择
    parser.add_argument("--degraded", nargs="+", default=None,
                        help="指定退化体 ID (如 DG-A DG-C)，默认全跑")
    # 配置扫描选择
    parser.add_argument("--scans", nargs="+", default=None,
                        help="指定配置扫描 ID (如 CFG-1 CFG-2)，默认全跑")
    # 单帧 actors 文件（离线复跑）
    parser.add_argument("--frame-actors", type=str, default=None,
                        help="frame_actors.csv 路径 (offline 离线复跑 detector 用)")

    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir) if args.runs_dir else None

    degraded_results: Dict[str, Any] = {}
    config_results: Dict[str, Any] = {}
    tier_results: Dict[str, Any] = {}

    if args.mode in ("offline", "online", "all"):
        # ── 退化体对比 ──
        targets = args.degraded or list(DEGRADED_CONFIGS.keys())
        for dg_id in targets:
            if dg_id not in DEGRADED_CONFIGS:
                print(f"[skip] unknown degraded config: {dg_id}")
                continue
            dg_cfg = DEGRADED_CONFIGS[dg_id]
            print(f"\n{'='*60}")
            print(f"  {dg_id}: {dg_cfg.label}")
            print(f"  {dg_cfg.description}")
            print(f"{'='*60}")

            if args.mode == "online" and not args.dry_run:
                res = _run_online(dg_cfg, args.town, args.frames, f"degraded_{dg_id}")
                degraded_results[dg_id] = res
            else:
                # offline: 尝试加载已有 run
                run_dir = OUT_DIR / f"degraded_{dg_id}"
                if run_dir.exists():
                    metrics = _load_run_results(run_dir)
                    degraded_results[dg_id] = {"mode": "offline", **metrics}
                else:
                    degraded_results[dg_id] = {
                        "mode": "offline_no_data",
                        "required_config": asdict(dg_cfg),
                    }
                    print(f"  [offline] no data, showing required config changes")

    if args.mode in ("configs-only", "offline", "all"):
        # ── 配置扫描 ──
        scan_targets = args.scans or list(CONFIG_SCANS.keys())
        for sid in scan_targets:
            if sid not in CONFIG_SCANS:
                print(f"[skip] unknown config scan: {sid}")
                continue
            scan = CONFIG_SCANS[sid]
            if args.mode == "online" and not args.dry_run:
                print(f"[online] config scan {sid} needs {len(scan.values)} runs")
                # 生成命令矩阵但不实际运行
                _run_config_scan_offline(scan, runs_dir or Path("."), f"config_{sid}")
            else:
                _run_config_scan_offline(scan, runs_dir or Path("."), f"config_{sid}")
            config_results[sid] = {"factor": scan.factor, "values": scan.values}

    if args.mode in ("tier-only", "offline", "all"):
        # ── Tier 增量 ──
        tier_results = run_tier_increment_analysis(
            frame_labels=Path(args.frame_actors) if args.frame_actors else None,
        )

    # ── 生成报告 ──
    if args.mode != "online" or args.dry_run:
        report = generate_markdown_report(
            degraded_results=degraded_results,
            config_results=config_results,
            tier_results=tier_results,
        )
        report_path = OUT_DIR / "ablation_summary.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"\n[ablation] Report → {report_path}")

    # ── dry-run: 打印所有需要运行的内容 ──
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY-RUN: 以下命令不会实际执行，仅展示计划")
        print("="*60)
        for dg_id, dg_cfg in DEGRADED_CONFIGS.items():
            if args.degraded and dg_id not in args.degraded:
                continue
            cmd = _gen_run_command(args.town, args.frames, dg_cfg,
                                   _build_run_dir(f"degraded_{dg_id}"))
            print(f"\n  {dg_id} ({dg_cfg.label}):")
            print(f"    {' '.join(cmd)}")
        for sid, scan in CONFIG_SCANS.items():
            if args.scans and sid not in args.scans:
                continue
            print(f"\n  {sid} ({scan.factor}): {len(scan.values)} 个取值")
            for v in scan.values:
                print(f"    value={v}")

    print("\n[ablation] Done.")


if __name__ == "__main__":
    main()
