#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
draw_kg_diagrams.py — 为 4 张关键帧生成知识图谱关系图（论文用）

基于 stk/ontology/types.py 的节点/边 schema：
  节点：SceneSnapshot、Vehicle、Lane、Maneuver、Rule、SafetyViolation、Responsibility
  边：containsVehicle、in_lane、following、changing_lane、violates、definedBy、causedBy、responsibleFor

异常节点/边用红色加粗，正常节点/边用 Duotone 蓝灰配色。

Usage:
    python draw_kg_diagrams.py --out /path/to/output/keyframes/
"""
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx

# ── 中文字体配置（Noto Sans CJK SC） ───────────────────────────────────────────
_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts-droid-fallback/truetype/DroidSansFallback.ttf",
]


def _setup_font():
    """Register a CJK font so matplotlib can render Chinese labels."""
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            try:
                fm.fontManager.addfont(p)
                fp = fm.FontProperties(fname=p)
                fname = fp.get_name()
                plt.rcParams["font.family"] = fname
                plt.rcParams["axes.unicode_minus"] = False
                print(f"[font] using {fname} from {p}")
                return
            except Exception as e:
                print(f"[font] {p} failed: {e}")
    print("[font] WARNING: no CJK font found, labels may show boxes")


_setup_font()

# ── 配色（Duotone 克制）────────────────────────────────────────────────────
COL_ANOMALY_NODE   = "#C0392B"   # 红色异常节点
COL_ANOMALY_EDGE   = "#C0392B"   # 红色异常边
COL_NORMAL_NODE    = "#2C3E50"   # 深蓝灰
COL_NORMAL_EDGE    = "#7F8C8D"   # 中灰
COL_SCENE_NODE     = "#1A5276"   # 场景节点深蓝
COL_RULE_NODE      = "#1E8449"   # 规则节点深绿
COL_BG             = "#FDFEFE"   # 浅灰白背景
COL_TEXT           = "#1C2833"   # 深色文字
COL_LABEL          = "#566573"   # 标签文字

# ── 节点布局参数 ───────────────────────────────────────────────────────────
NODE_RADIUS  = 0.42
LINE_WIDTH_N = 2.0
LINE_WIDTH_A = 3.2  # 异常边加粗
FONT_NORMAL  = 11
FONT_BOLD    = 12
FONT_TITLE   = 16
FONT_CAPTION = 9

# ── 帮助函数 ───────────────────────────────────────────────────────────────

def _add_node(G, node_id, label, pos, node_type="Vehicle", anomaly=False):
    G.add_node(
        node_id,
        label=label,
        pos=pos,
        node_type=node_type,
        anomaly=anomaly,
    )


def _add_edge(G, u, v, relation, anomaly=False, directed=True):
    attrs = {"label": relation, "anomaly": anomaly}
    if directed:
        G.add_edge(u, v, **attrs)
    else:
        G.add_edge(u, v, **attrs)
        G.add_edge(v, u, **attrs)


def _draw_nodes(G, ax, pos, node_colors):
    """绘制节点圆 + 标签"""
    normals = [(n, d) for n, d in G.nodes(data=True) if not d.get("anomaly")]
    anomalies = [(n, d) for n, d in G.nodes(data=True) if d.get("anomaly")]

    # 正常节点
    for n, d in normals:
        pt = pos[n]
        col = node_colors.get(d.get("node_type", "Vehicle"), COL_NORMAL_NODE)
        ax.scatter(pt[0], pt[1], s=600, c=col, zorder=3,
                   edgecolors="white", linewidths=2.0)
        ax.text(pt[0], pt[1] + 0.55, d["label"], ha="center", va="center",
                fontsize=FONT_BOLD, fontweight="bold", color=COL_TEXT, zorder=4)
        # 节点类型小字
        ax.text(pt[0], pt[1] - 0.55, d.get("node_type", ""), ha="center", va="center",
                fontsize=FONT_CAPTION, color=COL_LABEL, style="italic", zorder=4)

    # 异常节点（红色边框+阴影）
    for n, d in anomalies:
        pt = pos[n]
        ax.scatter(pt[0], pt[1], s=600, c=COL_ANOMALY_NODE, zorder=3,
                   edgecolors="#F1948A", linewidths=3.0)
        ax.text(pt[0], pt[1] + 0.55, d["label"], ha="center", va="center",
                fontsize=FONT_BOLD, fontweight="bold", color=COL_ANOMALY_NODE, zorder=4)
        ax.text(pt[0], pt[1] - 0.55, d.get("node_type", ""), ha="center", va="center",
                fontsize=FONT_CAPTION, color=COL_ANOMALY_NODE, style="italic", zorder=4)


def _draw_edges(G, ax):
    """绘制边 + 标签"""
    drawn = set()
    for u, v, d in G.edges(data=True):
        if d.get("directed", True):
            key = (u, v)
        else:
            key = tuple(sorted([u, v]))
        if key in drawn:
            continue
        drawn.add(key)

        p1 = G.nodes[u]["pos"]
        p2 = G.nodes[v]["pos"]
        anomaly = d.get("anomaly", False)
        color  = COL_ANOMALY_EDGE if anomaly else COL_NORMAL_EDGE
        lw     = LINE_WIDTH_A if anomaly else LINE_WIDTH_N

        # 带箭头的边（使用 annotate）
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length < 0.01:
            continue
        nx_norm = dx / length
        ny_norm = dy / length

        # 缩短箭头使其不碰到节点圆
        offset_start = NODE_RADIUS + 0.05
        offset_end   = NODE_RADIUS + 0.15 + 0.25  # 留标签空间
        x1 = p1[0] + nx_norm * offset_start
        y1 = p1[1] + ny_norm * offset_start
        x2 = p2[0] - nx_norm * offset_end
        y2 = p2[1] - ny_norm * offset_end

        if d.get("directed", True):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color=color, lw=lw),
                        zorder=2)
        else:
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=2)

        # 边标签
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        perp_x, perp_y = -ny_norm, nx_norm
        label_offset = 0.25
        ax.text(mx + perp_x * label_offset, my + perp_y * label_offset,
                d["label"], ha="center", va="center",
                fontsize=FONT_NORMAL, color=COL_ANOMALY_EDGE if anomaly else COL_LABEL,
                fontweight="bold" if anomaly else "normal",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                zorder=5)


def _set_style(ax, title, caption):
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-3.5, 4.0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.set_xlabel(caption, fontsize=9, color=COL_LABEL, labelpad=4)
    ax.axis("off")


# ── 四帧的 KG 定义 ────────────────────────────────────────────────────────
# 坐标 (x, y) 手动设计，每帧居中对称

FIGURES = {
    "t01_normal_5.7s": {
        "title": "(a) 正常跟车阶段  t = 5.7 s",
        "caption": "SceneSnapshot → containsVehicle(Ego, V17)；V17 in_adjacent_lane；Ego following V17；两个 Maneuver(cruising)；RSS Rule(R1) definedBy Scene",
        "anomalous": [],   # 无异常
        "nodes": [
            ("Scene",  "SceneSnapshot", (-4, 2.5)),
            ("Ego",    "Vehicle",       (-2,  0)),
            ("V17",    "Vehicle",       ( 2,  0)),
            ("Lane1",  "Lane",          (-2, -2)),
            ("Lane2",  "Lane",          ( 2, -2)),
            ("M_e",    "Maneuver",      (-4, -1)),
            ("M_v",    "Maneuver",      ( 4, -1)),
            ("Rule1",  "Rule",          ( 0,  2.5)),
        ],
        "edges": [
            ("Scene",  "Ego",    "containsVehicle",  False),
            ("Scene",  "V17",    "containsVehicle",  False),
            ("Scene",  "Rule1",  "definedBy",        False),
            ("Ego",    "Lane1",  "in_lane",          False),
            ("V17",    "Lane2",  "in_lane",          False),
            ("Lane1",  "Lane2",  "adjacent_lane",    False),
            ("Ego",    "V17",    "following",        False),
            ("Ego",    "M_e",    "has_maneuver",     False),
            ("V17",    "M_v",    "has_maneuver",     False),
            ("M_e",    "Rule1",  "supportedBy",      False),
        ],
        "node_colors": None,
    },

    "t02_cut-in_9.1s": {
        "title": "(b) 切入行为阶段  t = 9.1 s",
        "caption": "V17 执行 changing_lane 机动；Ego 与 V17 变为 approaching 交互；RSS 初步规则 R13a 开始评估",
        "anomalous": [],   # 切入本身不违规
        "nodes": [
            ("Scene",  "SceneSnapshot", (-4, 2.5)),
            ("Ego",    "Vehicle",       (-2,  0)),
            ("V17",    "Vehicle",       ( 2,  0)),
            ("Lane",   "Lane",          ( 0, -2)),
            ("M_e",    "Maneuver",      (-4, -1)),
            ("M_v",    "Maneuver",      ( 4, -1)),
            ("Rule13a","Rule",          ( 0,  2.5)),
            ("I_app",  "Interaction",   ( 0,  1.5)),
        ],
        "edges": [
            ("Scene",  "Ego",    "containsVehicle",  False),
            ("Scene",  "V17",    "containsVehicle",  False),
            ("Scene",  "Rule13a","definedBy",        False),
            ("Ego",    "Lane",   "in_lane",          False),
            ("V17",    "Lane",   "in_lane",          False),
            ("Ego",    "V17",    "approaching",      False),
            ("Ego",    "I_app",  "has_interaction",  False),
            ("V17",    "I_app",  "has_interaction",  False),
            ("V17",    "M_v",    "has_maneuver",     False),
            ("M_v",    "Rule13a","supportedBy",      False),
            ("Ego",    "M_e",    "has_maneuver",     False),
        ],
        "node_colors": None,
    },

    "t03_unsafe_following_10.3s": {
        "title": "(c) 不安全跟车 + RSS 违反  t = 10.3 s",
        "caption": "⚠ SafetyViolation(RSS_CUTIN) 触发；violates(Ego,V17) 红色加粗；Rule(R13a) → causedBy(SV)；responsibleFor 红色指向 V17",
        "anomalous": ["SafetyViolation", "violates", "causedBy", "responsibleFor"],
        "nodes": [
            ("Scene",       "SceneSnapshot", (-4,  2.5)),
            ("Ego",         "Vehicle",       (-2,  0)),
            ("V17",         "Vehicle",        ( 2,  0)),
            ("Lane",        "Lane",           ( 0, -2)),
            ("M_v",         "Maneuver",       ( 4, -1)),
            ("Rule13a",     "Rule",           ( 0,  2.5)),
            ("SV",          "SafetyViolation",( 0,  1.2), True),
            ("Resp",        "Responsibility", ( 2,  1.2), True),
        ],
        "edges": [
            ("Scene",       "Ego",    "containsVehicle",  False),
            ("Scene",       "V17",    "containsVehicle",  False),
            ("Scene",       "Rule13a","definedBy",        False),
            ("Ego",         "Lane",   "in_lane",          False),
            ("V17",         "Lane",   "in_lane",          False),
            ("Ego",         "V17",    "following",        False),
            ("V17",         "M_v",    "has_maneuver",     False),
            ("Rule13a",     "SV",     "definedBy",        False),
            ("SV",          "Ego",    "violates",         True),
            ("SV",          "V17",    "violates",         True),
            ("SV",          "Resp",   "causedBy",         True),
            ("Resp",        "V17",    "responsibleFor",   True),
        ],
        "node_colors": None,
    },

    "t04_recovery_11.7s": {
        "title": "(d) 恢复阶段  t = 11.7 s",
        "caption": "SV 仍残留（红色虚线表示风险未完全解除）；Ego 减速；V17 加速；following 关系正在恢复",
        "anomalous": ["SafetyViolation", "violates"],
        "nodes": [
            ("Scene",       "SceneSnapshot", (-4,  2.5)),
            ("Ego",         "Vehicle",       (-2,  0)),
            ("V17",         "Vehicle",        ( 2,  0)),
            ("Lane",        "Lane",           ( 0, -2)),
            ("M_e",         "Maneuver",      (-4, -1)),
            ("M_v",         "Maneuver",       ( 4, -1)),
            ("Rule13a",     "Rule",           ( 0,  2.5)),
            ("SV",          "SafetyViolation",( 0,  1.2), True),
        ],
        "edges": [
            ("Scene",       "Ego",    "containsVehicle",  False),
            ("Scene",       "V17",    "containsVehicle",  False),
            ("Scene",       "Rule13a","definedBy",        False),
            ("Ego",         "Lane",   "in_lane",          False),
            ("V17",         "Lane",   "in_lane",          False),
            ("Ego",         "V17",    "following",        False),
            ("V17",         "M_v",    "has_maneuver",     False),
            ("Ego",         "M_e",    "has_maneuver",     False),
            ("M_e",         "Rule13a","supportedBy",      False),
            ("Rule13a",     "SV",     "definedBy",        False),
            ("SV",          "Ego",    "violates",         True),
            ("SV",          "V17",    "violates",         True),
        ],
        "node_colors": None,
    },
}


def build_figure(fig_id, fig_cfg):
    G = nx.DiGraph()
    node_colors = {
        "Vehicle":      COL_NORMAL_NODE,
        "Lane":         "#5D6D7E",
        "Maneuver":     "#1F618D",
        "Rule":         COL_RULE_NODE,
        "SafetyViolation": COL_ANOMALY_NODE,
        "Responsibility": COL_ANOMALY_NODE,
        "Interaction":  "#7D3C98",
        "SceneSnapshot": COL_SCENE_NODE,
    }

    for nid, ntype, pos, *args in fig_cfg["nodes"]:
        anomaly = args[0] if args else False
        _add_node(G, nid, nid, pos, node_type=ntype, anomaly=anomaly)
        if nid == "Scene":
            node_colors[nid] = COL_SCENE_NODE

    for u, v, rel, anom in fig_cfg["edges"]:
        _add_edge(G, u, v, rel, anomaly=anom)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    fig.set_facecolor(COL_BG)
    ax.set_facecolor(COL_BG)

    pos = {n: d["pos"] for n, d in G.nodes(data=True)}
    _draw_nodes(G, ax, pos, node_colors)
    _draw_edges(G, ax)
    _set_style(ax, fig_cfg["title"], fig_cfg["caption"])

    return fig


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt   = "png"
    dpi   = 150
    for fname, cfg in FIGURES.items():
        fig = build_figure(fname, cfg)
        out = out_dir / f"{fname}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=COL_BG)
        plt.close(fig)
        print(f"✅ {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="输出目录（默认 output/knowledge_graphs/）")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).parent / "output" / "knowledge_graphs"
    main(out)
