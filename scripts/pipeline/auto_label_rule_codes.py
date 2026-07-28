#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_label_rule_codes.py -- 为 frame_labels.csv 的 rule_codes 列填充 ground truth

现状问题:
  data/dataset/frame_labels.csv 共 41,150 行，rule_codes 列 全部为空。
  本脚本把每个场景/异常注入帧对应的"预期应触发的规则码"写入该列。

填充逻辑:
  ① 场景库帧 (scenario_id = S00~S33):
     直接从 SCENARIO_REGISTRY[scenario_id].expected_rules 取规则码。
  ② 长时运行帧 (有 anomaly_type，无 scenario_id):
     用异常类型 → 规则码语义映射表填充（见 ANOMALY_RULE_MAP）。
  ③ 正常帧 (无 scenario_id 且 anomaly_type 为空):
     rule_codes 保持空字符串——正常帧不预期触发任何规则。

输出:
  覆写 data/dataset/frame_labels.csv 的 rule_codes 列（原地修改）。
  同时写入 data/dataset/rule_label_stats.json 汇总统计。

使用:
  python scripts/pipeline/auto_label_rule_codes.py
  python scripts/pipeline/auto_label_rule_codes.py --dry-run   # 只打印不写入
  python scripts/pipeline/auto_label_rule_codes.py --output data/dataset/frame_labels_labeled.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

# ──────────────────────────────────────────────────────────────────────────────
# 路径常量
# ──────────────────────────────────────────────────────────────────────────────
FRAME_LABELS = _REPO / "data" / "dataset" / "frame_labels.csv"
STATS_OUT = _REPO / "data" / "dataset" / "rule_label_stats.json"


# ──────────────────────────────────────────────────────────────────────────────
# SCENARIO_REGISTRY 动态加载（避免硬编码，与 scenario_library.py 同源）
# ──────────────────────────────────────────────────────────────────────────────
def _load_scenario_registry() -> dict:
    """从 scenario_library.py 加载 SCENARIO_REGISTRY。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "scenario_library",
        str(_REPO / "stk" / "scenario" / "scenario_library.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SCENARIO_REGISTRY


# ──────────────────────────────────────────────────────────────────────────────
# 异常类型 → 预期规则码 语义映射
# ──────────────────────────────────────────────────────────────────────────────
# 基于 chapter3_06.md §3.6.4 + stk/rules/traffic/rules.py + stk/rules/rss/model.py
# 的语义关系建立。每种异常注入在 CARLA 中的行为特征 → 哪些规则应当被触发。
ANOMALY_RULE_MAP: dict[str, list[str]] = {
    # sudd_brk: 突然急刹车 → 后车跟车距离不足 → RSS 纵向安全距离违规
    "sudd_brk": ["RSS_R13a"],
    # jun_ny: 路口未让行 → R7 路口让行违规
    "jun_ny": ["R7"],
    # rev_drive: 逆行/反向行驶 → R4 对向会车 + R18 逆行车道
    "rev_drive": ["R4", "R18"],
    # obs_blk: 障碍物阻挡/弱势参与者 → R8 弱势参与者保护
    "obs_blk": ["R8"],
    # avd_col: 紧急避撞 → RSS 纵向/横向安全距离
    "avd_col": ["RSS_R13a", "RSS_R14a"],
    # sudd_stp: 突然停车/禁止停区内停车 → R13 违法停车 + RSS 纵向
    "sudd_stp": ["R13", "RSS_R13a"],
}


# ──────────────────────────────────────────────────────────────────────────────
# 核心逻辑
# ──────────────────────────────────────────────────────────────────────────────

def label_frame(
    row: dict,
    registry: dict,
) -> str:
    """为单帧返回应填充的 rule_codes 字符串（逗号分隔）。

    优先级:
      1. 有 scenario_id → 从 SCENARIO_REGISTRY 取 expected_rules
      2. 有 anomaly_type → 从 ANOMALY_RULE_MAP 取预期规则
      3. 都没有 → 空字符串（正常帧）
    """
    # ── 优先级 1: scenario_id → expected_rules ──
    sid = row.get("scenario_id", "").strip()
    if sid and sid in registry:
        meta = registry[sid]
        rules = meta.get("expected_rules", [])
        return ",".join(rules) if rules else ""

    # ── 优先级 2: anomaly_type → ANOMALY_RULE_MAP ──
    atype = row.get("anomaly_type", "").strip()
    if atype and atype in ANOMALY_RULE_MAP:
        rules = ANOMALY_RULE_MAP[atype]
        return ",".join(rules)

    # ── 优先级 3: 正常帧 → 空 ──
    return ""


def run(
    input_path: Path,
    output_path: Path | None,
    dry_run: bool = False,
    backup: bool = True,
) -> dict:
    """主流程：读取 → 标注 → 写回。

    Returns:
        统计字典
    """
    registry = _load_scenario_registry()

    # 读取
    print(f"[*] Reading {input_path} ...")
    rows: list[dict] = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)
    print(f"    Total rows: {len(rows)}")

    # 验证 rule_codes 列存在
    if "rule_codes" not in fieldnames:
        print("[FATAL] rule_codes column not found in CSV!")
        sys.exit(1)

    # ── 统计计数器 ──
    stats: dict = {
        "total_rows": len(rows),
        "filled_from_scenario": 0,
        "filled_from_anomaly_type": 0,
        "left_empty": 0,
        "scenario_id_dist": Counter(),
        "anomaly_type_dist": Counter(),
        "rule_code_dist": Counter(),
        "per_scenario_rules": {},
    }

    # 标注
    print("[*] Labeling rule_codes ...")
    for row in rows:
        old_val = row.get("rule_codes", "").strip()
        new_val = label_frame(row, registry)
        row["rule_codes"] = new_val

        sid = row.get("scenario_id", "").strip()
        atype = row.get("anomaly_type", "").strip()

        if sid:
            stats["scenario_id_dist"][sid] += 1
            if new_val:
                stats["filled_from_scenario"] += 1
        elif atype:
            stats["anomaly_type_dist"][atype] += 1
            if new_val:
                stats["filled_from_anomaly_type"] += 1
        else:
            stats["left_empty"] += 1

        # 统计每个规则码出现次数
        if new_val:
            for rc in new_val.split(","):
                stats["rule_code_dist"][rc.strip()] += 1

    # per-scenario 规则码明细
    for sid in sorted(registry.keys()):
        meta = registry[sid]
        stats["per_scenario_rules"][sid] = {
            "expected_rules": meta.get("expected_rules", []),
            "tier": meta.get("tier", "?"),
            "category": meta.get("category", "?"),
            "n_frames_labeled": stats["scenario_id_dist"].get(sid, 0),
        }

    # 打印摘要
    print(f"\n[summary]")
    print(f"  filled from scenario_id:      {stats['filled_from_scenario']:>6}")
    print(f"  filled from anomaly_type:      {stats['filled_from_anomaly_type']:>6}")
    print(f"  left empty (normal frames):    {stats['left_empty']:>6}")
    print(f"  total labeled:                 {stats['filled_from_scenario'] + stats['filled_from_anomaly_type']:>6}")
    print(f"\n  rule_code distribution:")
    for rc, cnt in sorted(stats["rule_code_dist"].items(), key=lambda x: -x[1]):
        print(f"    {rc:<15} {cnt:>6}")

    if dry_run:
        print("\n[DRY-RUN] No files written.")
        return stats

    # ── 备份原文件 ──
    if backup and input_path.exists() and output_path is None:
        bak = input_path.with_suffix(".csv.bak")
        import shutil
        shutil.copy2(input_path, bak)
        print(f"[+] Backup → {bak}")

    # ── 写回 CSV ──
    dest = output_path or input_path
    print(f"\n[*] Writing → {dest} ...")
    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"    Done. {len(rows)} rows written.")

    # ── 写统计 JSON ──
    # Counter → dict for JSON serialization
    stats_serializable = {
        "total_rows": stats["total_rows"],
        "filled_from_scenario": stats["filled_from_scenario"],
        "filled_from_anomaly_type": stats["filled_from_anomaly_type"],
        "left_empty": stats["left_empty"],
        "scenario_id_dist": dict(stats["scenario_id_dist"]),
        "anomaly_type_dist": dict(stats["anomaly_type_dist"]),
        "rule_code_dist": dict(stats["rule_code_dist"]),
        "per_scenario_rules": stats["per_scenario_rules"],
        "anomaly_rule_map": ANOMALY_RULE_MAP,
    }
    STATS_OUT.write_text(
        json.dumps(stats_serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[+] Stats → {STATS_OUT}")

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="为 frame_labels.csv 的 rule_codes 列填充 ground truth",
    )
    p.add_argument(
        "--input", type=str, default=str(FRAME_LABELS),
        help="输入 CSV 路径 (default: data/dataset/frame_labels.csv)",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="输出 CSV 路径 (default: 覆写原文件)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="只打印统计，不写入文件",
    )
    p.add_argument(
        "--no-backup", action="store_true",
        help="覆写时不创建 .bak 备份",
    )
    args = p.parse_args()

    stats = run(
        input_path=Path(args.input),
        output_path=Path(args.output) if args.output else None,
        dry_run=args.dry_run,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
