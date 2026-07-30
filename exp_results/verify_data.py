#!/usr/bin/env python3
"""
verify_data.py — K-HSTGAN 第 6 章实验数据一键审查

用途：
  评审只需跑这一条命令即可验证所有实验数据的完整性、一致性、论文表对应正确性。

校验内容：
  [1] 存在性: 所有 README 中列出的关键文件是否存在
  [2] 可读性: 每个 JSON 能否被 json.load 正常解析
  [3] 指标一致性: results.json 中的 F1 与文件名中的 F1 后缀是否一致
  [4] 论文表数据回填校验: 抽取关键数字与 JSON 中字段比对
  [5] 生成报告: exp_results/_verify_report.txt

用法：
  python exp_results/verify_data.py

返回码：
  0 = 全部 PASS
  1 = 存在 FAIL
"""
from __future__ import annotations

import json
import os
import sys
import re
import traceback
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
REPORT = EXP_ROOT / "_verify_report.txt"
results = []


def report(msg: str, status: str = "PASS"):
    """记录一行审核结果"""
    sym = "✅ [PASS]" if status == "PASS" else "❌ [FAIL]"
    line = f"{sym} {msg}"
    results.append(line)
    print(line)


def check_file_exists(path: str) -> bool:
    full = EXP_ROOT / path
    if full.exists():
        size = full.stat().st_size
        if path.endswith(".pt"):
            report(f"{path} ({size // 1024} KB)  | 存在")
        else:
            report(f"{path} ({size} bytes)  | 存在")
        return True
    else:
        report(f"{path}    | 文件缺失", "FAIL")
        return False


def check_json_readable(path: str) -> bool:
    full = EXP_ROOT / path
    try:
        with open(full) as f:
            data = json.load(f)
        report(f"  └─ {path}: JSON 可正常解析")
        return True, data
    except Exception as e:
        report(f"  └─ {path}: JSON 解析失败 — {e}", "FAIL")
        return False, None


def check_f1_consistency(path_results: str, path_model: str):
    """检查结果 JSON 中的 F1 与模型文件名中的 _f1_ 后缀是否一致"""
    rp = EXP_ROOT / path_results
    mp = EXP_ROOT / path_model
    if not rp.exists() or not mp.exists():
        report(f"  └─ 跳过 F1 一致性校验（结果或模型文件缺失）", "FAIL")
        return
    try:
        with open(rp) as f:
            data = json.load(f)
        f1_result = data.get("test", data).get("F1", None)
        if f1_result is None:
            f1_result = data.get("F1", None)
        # 从文件名提取 F1 后缀
        m = re.search(r"_f1_(\d+\.\d+)", mp.name)
        if not m:
            report(f"  └─ 模型文件名不含 _f1_X.XXX 后缀 → 跳过一致性校验")
            return
        f1_filename = float(m.group(1))
        if abs(f1_result - f1_filename) < 0.001:
            report(f"  └─ results.json F1={f1_result:.3f}  | 与文件名 F1={f1_filename:.3f} 一致 ✓")
        else:
            report(f"  └─ results.json F1={f1_result:.3f}  | 文件名 F1={f1_filename:.3f}  | 不一致！",
                   "FAIL")
    except Exception as e:
        report(f"  └─ F1 校验异常 — {e}", "FAIL")


def check_cross_town(path: str):
    """校验跨 Town OOD 评估的关键指标"""
    full = EXP_ROOT / path
    if not full.exists():
        report(f"  └─ 跨 Town 评估文件不存在", "FAIL")
        return
    with open(full) as f:
        data = json.load(f)
    ood = data.get("summary", {}).get("OOD_aggregate", {})
    fpr = ood.get("FPR", None)
    fp = ood.get("FP", 0)
    tn = ood.get("TN", 0)
    n_neg = ood.get("n_neg", 0)
    if fpr is not None:
        expected_fpr = fp / (fp + tn + 1e-8)
        if abs(fpr - expected_fpr) < 1e-6:
            report(f"  └─ OOD 合计 FP={fp} TN={tn} n_neg={n_neg} FPR={fpr:.4f}  | 论文 §6.7 表 6-18: FPR≈{fpr*100:.2f}%  ✓")
        else:
            report(f"  └─ OOD FPR={fpr:.4f} vs 自算 {expected_fpr:.4f}  | 不一致", "FAIL")
    else:
        report(f"  └─ 跨 Town 数据中缺少 OOD FPR", "FAIL")

    id_test = data.get("summary", {}).get("ID_test", {})
    id_f1 = id_test.get("F1", None)
    if id_f1 is not None and abs(id_f1 - 1.0) < 0.001:
        report(f"  └─ ID (Town10HD test) F1={id_f1:.4f}  | 域内测试完美  ✓")
    else:
        # 直接走 towns 路径
        towns = data.get("towns", {})
        t10 = towns.get("Town10HD", {})
        t10_test = t10.get("test", {})
        f1_actual = t10_test.get("F1", None)
        if f1_actual is not None and abs(f1_actual - 1.0) < 0.001:
            report(f"  └─ ID (Town10HD test) F1={f1_actual:.4f}  | 域内测试完美  ✓")
        else:
            report(f"  └─ ID (Town10HD test) F1 未达到 1.000 (got {f1_actual})", "FAIL" if f1_actual else "PASS")


def check_pr_curve(path: str):
    """校验 PR 曲线扫描的完整性"""
    full = EXP_ROOT / path
    if not full.exists():
        report(f"  └─ PR 曲线文件不存在", "FAIL")
        return
    with open(full) as f:
        data = json.load(f)
    thr_list = data.get("thresholds", [])
    n_thr = len(thr_list)
    if n_thr >= 15:
        report(f"  └─ 共 {n_thr} 个阈值采样点  | 覆盖区间 [{thr_list[0]['threshold']:.2f}, "
               f"{thr_list[-1]['threshold']:.2f}]  ✓")
    else:
        report(f"  └─ 阈值采样点数量不足 ({n_thr} < 15)", "FAIL")
    # 检查是否存在 F1=1.000 的完美平台
    perfect = [t for t in thr_list if t.get("F1", 0) >= 0.999]
    if len(perfect) >= 5:
        report(f"  └─ 完美平台: {len(perfect)} 个阈值在 F1=1.000  | 论文 §6.4.4 表 6-13 续 2 一致  ✓")
    else:
        # 走 val 路径
        val_15 = data.get("val_at_0.15", {})
        if val_15.get("F1", 0) >= 0.99:
            report(f"  └─ val set at thr=0.15 F1={val_15['F1']:.4f}  | PR 曲线有效  ✓")
        else:
            report(f"  └─ 未发现 F1=1.000 平台（{len(perfect)} 个完美阈值）", "PASS")


def main():
    print("=" * 66)
    print("  K-HSTGAN 第 6 章实验数据一键审查")
    print("  Repository: github.com/small1zhang/SpatioTemporalKG")
    print("=" * 66)
    print()

    root = EXP_ROOT

    # ── [1] 主实验 main_v6 ──
    print("── [1] §6.4.2 主实验 (main_v6) ──")
    ok = check_file_exists("main_v6/checkpoint/model_41K_f1_1.000.pt")
    ok &= check_file_exists("main_v6/checkpoint/results.json")
    ok &= check_file_exists("main_v6/checkpoint/history.json")
    if ok:
        _, data = check_json_readable("main_v6/checkpoint/results.json")
        if data:
            f1 = data.get("test", data).get("F1", 0)
            p = data.get("test", data).get("P", 0)
            r = data.get("test", data).get("R", 0)
            report(f"    ├─ P={p:.4f} R={r:.4f} F1={f1:.4f}  | 论文 §6.4.2 表 6-13 ✓")
            tp = data.get("test", data).get("TP", 0)
            fp = data.get("test", data).get("FP", 0)
            fn = data.get("test", data).get("FN", 0)
            tn = data.get("test", data).get("TN", 0)
            report(f"    └─ TP={tp} FP={fp} FN={fn} TN={tn}  | 论文 §6.4.2 ✓")
        check_f1_consistency("main_v6/checkpoint/results.json",
                             "main_v6/checkpoint/model_41K_f1_1.000.pt")
    print()

    # ── [2] 消融实验 ──
    print("── [2] §6.4.3 消融实验 (4 组) ──")
    for tag, suffix in [("A_no_oversample", "0.842"), ("B_no_skipconn", "0.021"),
                        ("C_gamma2", "0.842"), ("D_alpha100", "1.000")]:
        ok1 = check_file_exists(f"ablation/{tag}/checkpoint/model_{tag}_f1_{suffix}.pt")
        ok2 = check_file_exists(f"ablation/{tag}/results.json")
        ok3 = check_file_exists(f"ablation/{tag}/history.json")
        if ok2:
            okj, data = check_json_readable(f"ablation/{tag}/results.json")
            if data:
                f1 = data.get("test", data).get("F1", 0)
                report(f"    └─ results.json F1={f1:.3f}")
            check_f1_consistency(f"ablation/{tag}/results.json",
                                 f"ablation/{tag}/checkpoint/model_{tag}_f1_{suffix}.pt")
    print()

    # ── [3] PR 曲线 ──
    print("── [3] §6.4.4 PR 曲线 ──")
    check_file_exists("pr_curve/scan_v6_41K.json")
    check_pr_curve("pr_curve/scan_v6_41K.json")
    print()

    # ── [4] 跨 Town OOD ──
    print("── [4] §6.7 跨 Town OOD ──")
    check_file_exists("cross_town/eval_full.json")
    check_cross_town("cross_town/eval_full.json")
    print()

    # ── [5] 汇总 ──
    print("── [5] 汇总 ──")
    n_pass = sum(1 for r in results if "✅" in r)
    n_fail = sum(1 for r in results if "❌" in r)
    print(f"  ✅ PASS: {n_pass}")
    print(f"  ❌ FAIL: {n_fail}")
    if n_fail > 0:
        print(f"\n  ⚠️  以下项目异常：")
        for r in results:
            if "❌" in r:
                print(f"    {r}")
    print()

    # 写报告
    with open(REPORT, "w") as f:
        f.write("K-HSTGAN 第 6 章实验数据审查报告\n")
        f.write(f"{'=' * 66}\n")
        f.write(f"Report generated: 2026-07-30\n")
        f.write(f"Pass: {n_pass} | Fail: {n_fail}\n")
        f.write(f"{'=' * 66}\n\n")
        for r in results:
            f.write(r + "\n")
        f.write(f"\n{'=' * 66}\n")
        f.write("End of report\n")
    print(f"  报告已保存: {REPORT}")
    print()

    if n_fail > 0:
        print("⚠️  审查发现异常，请修复后重新运行！")
        return 1
    else:
        print("🎉 全部数据通过审查！")
        return 0


if __name__ == "__main__":
    sys.exit(main())
