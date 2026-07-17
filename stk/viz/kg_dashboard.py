"""
stk/viz/kg_dashboard.py  —  四面板 KG 仪表板渲染器 (Pygame)

数据驱动: scene_graph_*.json 离线文件 (含 nodes, edges, risk).
纯渲染, 无外部依赖 (仅需 pygame, networkx).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pygame

# ── 色彩常量 ────────────────────────────────────────────────────────
RISK_COLOR = {
    "LOW": (40, 210, 80), "MEDIUM": (245, 180, 40), "HIGH": (240, 65, 65),
}
NODE_COLOR: Dict[str, Tuple[int,int,int]] = {
    "ego_vehicle": (255, 80, 80), "environment": (100, 220, 120),
    "road_surface_state": (70, 200, 220), "micro_odd": (255, 220, 80),
    "rss_profile": (80, 220, 255), "occluded_area": (170, 170, 190),
    "hidden_actor_hypothesis": (255, 120, 70), "safety_event": (255, 70, 70),
    "fallback_action": (90, 255, 170), "pedestrian": (255, 190, 40),
    "vehicle": (80, 170, 255), "static": (160, 160, 160),
    "lane": (170, 120, 255), "junction": (210, 120, 255),
    "traffic_light": (255, 80, 220),
}
RELATION_COLOR: Dict[str, Tuple[int,int,int]] = {
    "NEAR_BY": (190, 195, 205), "SAME_LANE": (100, 210, 255),
    "APPROACHING": (255, 90, 90), "POTENTIAL_OCCLUDER": (180, 180, 180),
    "VISIBLE_TO": (120, 220, 255), "CREATES_OCCLUDED_AREA": (170, 170, 190),
    "HAS_HIDDEN_ACTOR_HYPOTHESIS": (255, 140, 80), "TIME_TO_CONFLICT": (255, 80, 80),
    "ACTIVE_MICRO_ODD": (255, 220, 80), "ACTIVATES_PROFILE": (70, 220, 255),
    "HAS_ROAD_SURFACE_STATE": (70, 200, 220), "SUPPORTED_BY_EVIDENCE": (255, 180, 70),
    "EXPLAINS_RISK": (130, 255, 180), "EXPLAINS_ACTION": (80, 255, 150),
    "CONSTRAINS": (255, 80, 120), "CONTROLLED_BY": (255, 100, 230),
    "AFFECTS": (110, 220, 130), "ON_LANE": (170, 130, 255),
    "IN_JUNCTION": (210, 120, 255),
}
RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

# ── 工具函数 ────────────────────────────────────────────────────────

def init_pygame(width: int, height: int, title: str, headless: bool = False) -> pygame.Surface:
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(title)
    return screen

def _font(size: int, bold: bool = False) -> pygame.font.Font:
    f = pygame.font.SysFont("Consolas,DejaVu Sans,Arial", size)
    f.set_bold(bold)
    return f

def draw_text(surface: pygame.Surface, text: str, xy: Tuple[int,int], size: int = 18,
              color: Tuple[int,int,int] = (235, 235, 235), bold: bool = False) -> None:
    surface.blit(_font(size, bold).render(str(text), True, color), xy)

def wrap_lines(text: str, max_chars: int = 54) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def graph_edge_attr(graph: nx.MultiDiGraph, u: str, v: str, attr: str, default=None):
    data = graph.get_edge_data(u, v, default={})
    if isinstance(data, dict):
        if attr in data:
            return data.get(attr, default)
        for record in data.values():
            if isinstance(record, dict) and attr in record:
                return record.get(attr, default)
    return default

# ── JSON → Graph ────────────────────────────────────────────────────

def graph_from_json(path: Path) -> Tuple[nx.MultiDiGraph, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    G = nx.MultiDiGraph()
    for node in data.get("nodes", []):
        n = dict(node)
        nid = n.pop("id")
        G.add_node(nid, **n)
    for edge in data.get("edges", []):
        e = dict(edge)
        src = e.pop("source")
        tgt = e.pop("target")
        G.add_edge(src, tgt, **e)
    return G, data.get("risk", {})

class RiskLike:
    pass

def risk_from_dict(risk: Dict[str, Any]) -> Any:
    r = RiskLike()
    r.level = risk.get("level", "LOW")
    r.reasons = risk.get("reasons", [])
    r.response_time = float(risk.get("response_time", 0.5))
    r.accel_max = float(risk.get("accel_max", 3.5))
    r.brake_min = float(risk.get("brake_min", 4.0))
    r.brake_front_max = float(risk.get("brake_front_max", 8.0))
    r.active_profile = risk.get("active_profile", "Urban-Dry")
    r.semantic_margin_m = float(risk.get("semantic_margin_m", 0.0))
    r.explore_speed_mps = float(risk.get("explore_speed_mps", 5.0))
    r.evidence = risk.get("evidence", [])
    return r

def pretty_reason(reason: str) -> str:
    r = str(reason)
    if "降雨" in r or "湿滑" in r:
        return "Wet road/rain: lower accel and braking assumptions"
    if "TTC" in r or "ttc" in r:
        return r.replace("低 TTC 冲突", "Low TTC conflict")
    if "行人" in r or "弱势" in r or "vulnerable" in r.lower():
        return "Vulnerable road user nearby: increase response time"
    if "路口" in r:
        return "Junction multi-agent uncertainty: defensive response"
    return r

# ── 场景合成动画面板 ──────────────────────────────────────────────

def render_synthetic_scene(screen: pygame.Surface, rect: pygame.Rect,
                           graph: nx.MultiDiGraph, risk: Any, tick: int) -> None:
    """CARLA-style 场景动画 (离线无相机帧时使用)"""
    # 天空/城市背景
    pygame.draw.rect(screen, (120, 150, 180), rect)
    horizon = rect.y + int(rect.h * 0.25)
    pygame.draw.rect(screen, (180, 190, 200), pygame.Rect(rect.x, horizon - 65, rect.w, 65))
    # 建筑
    for i in range(7):
        bx = rect.x + 20 + i * int(rect.w / 7)
        bw = int(rect.w / 9)
        bh = 75 + (i % 3) * 22
        clr = [(185,170,150),(160,170,185),(190,180,165)][i % 3]
        pygame.draw.rect(screen, clr, pygame.Rect(bx, horizon - bh, bw, bh))
        for wx in range(bx+10, bx+bw-10, 22):
            for wy in range(horizon-bh+12, horizon-12, 24):
                pygame.draw.rect(screen, (235,225,160), pygame.Rect(wx, wy, 10, 12))
    # 路面
    road = pygame.Rect(rect.x+10, horizon, rect.w-20, rect.h-(horizon-rect.y)-10)
    pygame.draw.rect(screen, (58, 62, 68), road)
    cx, cy = rect.centerx, rect.y + int(rect.h * 0.62)
    pygame.draw.rect(screen, (70,74,80), pygame.Rect(rect.x+10, cy-95, rect.w-20, 190))
    pygame.draw.rect(screen, (70,74,80), pygame.Rect(cx-130, horizon, 260, road.h))
    for x in range(rect.x+35, rect.right-35, 70):
        pygame.draw.line(screen, (235,225,120), (x, cy-95), (x+35, cy-95), 3)
        pygame.draw.line(screen, (235,225,120), (x, cy+95), (x+35, cy+95), 3)
    for y in range(horizon+20, rect.bottom-30, 45):
        pygame.draw.line(screen, (240,240,240), (cx-130, y), (cx-130, y+22), 2)
        pygame.draw.line(screen, (240,240,240), (cx+130, y), (cx+130, y+22), 2)
    # 人行横道
    for i in range(8):
        pygame.draw.rect(screen, (245,245,235), pygame.Rect(cx-105+i*28, cy-18, 14, 36))
    # 雨
    if any(graph.nodes[n].get("type") == "environment"
           and float(graph.nodes[n].get("precipitation",0) or 0) > 10
           for n in graph.nodes):
        for i in range(55):
            rx = rect.x + (i*37 + tick*13) % rect.w
            ry = rect.y + (i*53 + tick*19) % rect.h
            pygame.draw.line(screen, (170,210,255), (rx, ry), (rx-8, ry+18), 1)
    # 语义数据
    ped_dist = 30.0 - tick * 1.5
    ttc = 5.0 - tick * 0.2
    for n in graph.nodes:
        if graph.nodes[n].get("type") == "pedestrian":
            ped_dist = float(graph_edge_attr(graph, "Ego", n, "distance", ped_dist) or ped_dist)
            ttc = float(graph.nodes[n].get("ttc_s", ttc) or ttc)
    # Ego 车辆
    ego_x, ego_y = cx, rect.bottom - 85
    pygame.draw.rect(screen, (235,55,55), pygame.Rect(ego_x-34, ego_y-55, 68, 110), border_radius=10)
    pygame.draw.rect(screen, (40,45,55), pygame.Rect(ego_x-24, ego_y-35, 48, 32), border_radius=5)
    pygame.draw.circle(screen, (30,30,30), (ego_x-38, ego_y-30), 9)
    pygame.draw.circle(screen, (30,30,30), (ego_x+38, ego_y-30), 9)
    draw_text(screen, "EGO", (ego_x-22, ego_y+18), 16, (255,255,255), True)
    # 遮挡卡车
    truck_x, truck_y = cx + 170, cy - 50
    pygame.draw.rect(screen, (120,130,140), pygame.Rect(truck_x-45, truck_y-35, 95, 70), border_radius=8)
    pygame.draw.rect(screen, (85,95,105), pygame.Rect(truck_x+20, truck_y-28, 35, 56), border_radius=5)
    draw_text(screen, "OCCLUDER", (truck_x-50, truck_y+52), 13, (255,255,255), True)
    # 行人位置
    progress = max(0.0, min(1.0, (28.0 - ped_dist) / 24.0))
    ped_x = int(cx + 210 - progress * 300)
    ped_y = cy
    hidden_active = any(graph.nodes[n].get("type") == "hidden_actor_hypothesis" for n in graph.nodes)
    if hidden_active:
        hx, hy = truck_x + 45, cy + 54
        pygame.draw.circle(screen, (255,120,70), (hx, hy), 18, 2)
        pygame.draw.line(screen, (255,120,70), (hx-12, hy-12), (hx+12, hy+12), 2)
        pygame.draw.line(screen, (255,120,70), (hx+12, hy-12), (hx-12, hy+12), 2)
        pygame.draw.line(screen, (255,120,70), (hx, hy), (ped_x, ped_y), 2)
        draw_text(screen, "HiddenActor", (hx-45, hy+26), 13, (255,170,110), True)
    pygame.draw.circle(screen, (255,205,60), (ped_x, ped_y-22), 13)
    for dx, dy in [(0,16),(0,24),(0,46),(-18,12),(18,12),(-15,46),(14,46)]:
        pygame.draw.line(screen, (255,205,60), (ped_x, ped_y-8), (ped_x+dx, ped_y+dy), 4)
    # 风险连线
    rc = RISK_COLOR.get(risk.level, (40,210,80))
    pygame.draw.line(screen, rc, (ego_x, ego_y-60), (ped_x, ped_y),
                     4 if risk.level == "HIGH" else 2)
    # 信息面板
    info = pygame.Rect(rect.x+24, rect.y+55, min(430, rect.w-48), 132)
    pygame.draw.rect(screen, (20,22,28), info, border_radius=8)
    pygame.draw.rect(screen, rc, info, 2, border_radius=8)
    draw_text(screen, f"Scenario: Pedestrian Crossing", (info.x+14, info.y+30), 17, (255,255,255), True)
    draw_text(screen, f"Distance: {ped_dist:.1f}m   TTC: {ttc:.1f}s", (info.x+14, info.y+62), 16)
    draw_text(screen, f"RSS: {risk.active_profile}", (info.x+14, info.y+90), 16, (190,220,255), True)
    draw_text(screen, f"Risk: {risk.level}  explore {risk.explore_speed_mps:.1f}m/s", (info.x+14, info.y+118), 16, rc, True)
    # Banner
    banner = pygame.Rect(rect.right-392, rect.y+58, 362, 76)
    pygame.draw.rect(screen, (20,22,28), banner, border_radius=8)
    pygame.draw.rect(screen, rc, banner, 2, border_radius=8)
    if risk.level == "HIGH":
        draw_text(screen, "RSS shield ON", (banner.x+16, banner.y+29), 22, (255,110,110), True)
        draw_text(screen, "low-speed fallback + evidence", (banner.x+16, banner.y+58), 14)
    elif risk.level == "MEDIUM":
        draw_text(screen, "KG semantic trigger", (banner.x+16, banner.y+29), 20, (245,210,80), True)
        draw_text(screen, "wet road / occlusion profile", (banner.x+16, banner.y+58), 14)
    else:
        draw_text(screen, "Nominal driving", (banner.x+16, banner.y+29), 20, (100,240,140), True)
        draw_text(screen, "monitoring CARLA truth graph", (banner.x+16, banner.y+58), 14)

def render_camera_panel(screen: pygame.Surface, rect: pygame.Rect,
                        graph: nx.MultiDiGraph, risk: Any, tick: int) -> None:
    """场景动画面板 (左栏头部)"""
    pygame.draw.rect(screen, (20,22,28), rect)
    pygame.draw.rect(screen, (80,90,110), rect, 2)
    draw_text(screen, "Application Scene Animation", (rect.x+14, rect.y+10), 20, bold=True)
    inner = pygame.Rect(rect.x+12, rect.y+42, rect.w-24, rect.h-54)
    if graph is not None and risk is not None:
        render_synthetic_scene(screen, inner, graph, risk, tick)
    else:
        pygame.draw.rect(screen, (8,10,14), inner)
        draw_text(screen, "No scene data", (inner.x+20, inner.y+30), 22, (180,190,210))

# ── 四面板核心 ───────────────────────────────────────────────────

def render_legend(screen: pygame.Surface, rect: pygame.Rect) -> None:
    x, y = rect.x, rect.y
    draw_text(screen, "Legend", (x, y), 13, (180,190,210), True)
    y += 18
    items = [("ego","ego_vehicle"),("ped","pedestrian"),("veh","vehicle"),
             ("MicroODD","micro_odd"),("RSS","rss_profile"),("hidden","hidden_actor_hypothesis")]
    for label, typ in items:
        pygame.draw.circle(screen, NODE_COLOR.get(typ, (220,220,220)), (x+8, y+7), 6)
        draw_text(screen, label, (x+20, y), 12, (220,220,220))
        x += 70
    x, y = rect.x, y + 22
    for rel in ["APPROACHING","ACTIVATES_PROFILE","TIME_TO_CONFLICT"]:
        pygame.draw.line(screen, RELATION_COLOR.get(rel, (180,180,180)), (x, y+7), (x+28, y+7), 3)
        draw_text(screen, rel, (x+34, y), 11, (210,210,210))
        x += 180

def semantic_kg_layout(graph: nx.MultiDiGraph) -> Dict[str, Tuple[float, float]]:
    anchors = {"Ego": (0.0, 0.0), "Environment": (0.86, -0.05), "MicroODD": (-0.05, -0.76)}
    type_anchor = {
        "road_surface_state": (0.64, -0.62), "rss_profile": (0.52, -0.82),
        "lane": (-0.82, -0.48), "junction": (-0.90, 0.0),
        "traffic_light": (0.03, -0.98), "vehicle": (0.72, 0.45),
        "pedestrian": (-0.46, 0.72), "static": (0.70, 0.05),
        "occluded_area": (0.25, 0.36), "hidden_actor_hypothesis": (-0.12, 0.55),
        "safety_event": (-0.58, -0.82), "fallback_action": (-0.78, -0.58),
    }
    buckets, pos = {}, {}
    for idx, (node, data) in enumerate(graph.nodes(data=True)):
        if node in anchors:
            pos[str(node)] = anchors[node]
            continue
        typ = str(data.get("type", ""))
        ax, ay = type_anchor.get(typ, (math.cos(idx), math.sin(idx)))
        k = buckets.get(typ, 0)
        buckets[typ] = k + 1
        off = (k-1)*0.10 if k else 0.0
        pos[str(node)] = (max(-0.98, min(0.98, ax+off)), max(-0.98, min(0.98, ay+0.07*(k%3))))
    return pos

def render_kg_panel(screen: pygame.Surface, rect: pygame.Rect, graph: nx.MultiDiGraph, tick: int = 0) -> None:
    """知识图谱面板 (右栏上部)"""
    pygame.draw.rect(screen, (16,18,24), rect)
    pygame.draw.rect(screen, (80,90,110), rect, 2)
    draw_text(screen, "Dynamic Knowledge Graph", (rect.x+14, rect.y+10), 20, bold=True)
    if graph.number_of_nodes() == 0:
        return
    has_semantic = any(d.get("type") in {"micro_odd","rss_profile","occluded_area",
                                          "hidden_actor_hypothesis","safety_event"}
                       for _, d in graph.nodes(data=True))
    if has_semantic:
        pos = semantic_kg_layout(graph)
    else:
        try:
            pos = nx.spring_layout(graph, seed=42, k=0.85, iterations=30)
        except Exception:
            pos = {n: (math.cos(i), math.sin(i)) for i, n in enumerate(graph.nodes())}
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    mnx, mxx = min(xs), max(xs); mny, mxy = min(ys), max(ys)
    pad, off = 45, 55
    area = pygame.Rect(rect.x+pad, rect.y+off, rect.w-2*pad, rect.h-off-70)
    def scale(p):
        x, y = p
        sx = area.x + (x-mnx)/(mxx-mnx+1e-6)*area.w
        sy = area.y + (y-mny)/(mxy-mny+1e-6)*area.h
        return int(sx), int(sy)
    for u, v, data in graph.edges(data=True):
        if u not in pos or v not in pos:
            continue
        a, b = scale(pos[u]), scale(pos[v])
        rel = str(data.get("relation",""))
        color = RELATION_COLOR.get(rel, (95,105,125))
        w = 3 if rel in ("APPROACHING","SAME_LANE","CONTROLLED_BY","ACTIVATES_PROFILE",
                         "TIME_TO_CONFLICT","EXPLAINS_ACTION") else 1
        if rel in ("APPROACHING","TIME_TO_CONFLICT","ACTIVATES_PROFILE"):
            w += 1 + int((tick % 10) / 5)
        pygame.draw.line(screen, color, a, b, w)
        mx, my = (a[0]+b[0])//2, (a[1]+b[1])//2
        if rel in ("SAME_LANE","APPROACHING","NEAR_BY","CONTROLLED_BY","ACTIVATES_PROFILE",
                   "TIME_TO_CONFLICT","EXPLAINS_ACTION","CREATES_OCCLUDED_AREA"):
            draw_text(screen, rel[:16], (mx, my), 11, color)
    for node, data in graph.nodes(data=True):
        x, y = scale(pos[node])
        typ = data.get("type","")
        color = NODE_COLOR.get(typ, (220,220,220))
        r = 18 if node == "Ego" else 12
        pygame.draw.circle(screen, color, (x, y), r)
        pygame.draw.circle(screen, (20,20,20), (x, y), r, 2)
        draw_text(screen, str(node)[:18], (x+r+3, y-8), 13, (235,235,235), node=="Ego")
    render_legend(screen, pygame.Rect(rect.x+14, rect.bottom-70, rect.w-28, 44))

def render_risk_timeline(screen: pygame.Surface, rect: pygame.Rect, history: List[str]) -> None:
    if not history:
        return
    pygame.draw.rect(screen, (12,14,20), rect)
    pygame.draw.rect(screen, (70,78,95), rect, 1)
    draw_text(screen, "Risk timeline", (rect.x+8, rect.y+5), 13, (180,190,210), True)
    max_items = max(1, rect.w // 9)
    recent = history[-max_items:]
    x, y = rect.x+8, rect.y+26
    bw = max(4, (rect.w-16)//max_items)
    for level in recent:
        c = RISK_COLOR.get(level, (180,180,180))
        h = 10 + RISK_ORDER.get(level, 0)*12
        pygame.draw.rect(screen, c, pygame.Rect(x, y+28-h, bw-1, h))
        x += bw

def render_risk_panel(screen: pygame.Surface, rect: pygame.Rect, risk: Any,
                      tick: int, fps: float, history: Optional[List[str]] = None) -> None:
    """Risk / RSS 面板 (左栏下部)"""
    rc = RISK_COLOR.get(risk.level, (220,220,220))
    pygame.draw.rect(screen, (20,22,28), rect)
    pygame.draw.rect(screen, rc, rect, 3)
    draw_text(screen, f"Risk: {risk.level}", (rect.x+14, rect.y+10), 24, rc, True)
    draw_text(screen, f"tick={tick}", (rect.x+170, rect.y+15), 16, (190,195,205))
    draw_text(screen, f"RSS response_time: {risk.response_time:.2f}s", (rect.x+14, rect.y+48), 18)
    draw_text(screen, f"accel_max={risk.accel_max:.2f}  brake_min={risk.brake_min:.2f}", (rect.x+14, rect.y+74), 17)
    draw_text(screen, f"profile={risk.active_profile}  margin={risk.semantic_margin_m:.1f}m", (rect.x+14, rect.y+100), 16, (190,210,255), True)
    draw_text(screen, "Reasons / evidence:", (rect.x+14, rect.y+126), 18, bold=True)
    y = rect.y + 152
    reasons = risk.reasons or ["No risk rule triggered"]
    for reason in reasons[:5]:
        for line in wrap_lines("- " + pretty_reason(reason), 58):
            if y > rect.bottom - 22:
                break
            draw_text(screen, line, (rect.x+20, y), 15, (220,220,220))
            y += 20
    if history is not None:
        render_risk_timeline(screen, pygame.Rect(rect.x+rect.w-300, rect.y+40, 280, 78), history)

def render_actor_table(screen: pygame.Surface, rect: pygame.Rect, graph: nx.MultiDiGraph) -> None:
    """KG 对象表 (右栏下部)"""
    pygame.draw.rect(screen, (20,22,28), rect)
    pygame.draw.rect(screen, (80,90,110), rect, 2)
    draw_text(screen, "KG Objects", (rect.x+12, rect.y+10), 20, bold=True)
    rows = []
    key_types = {"vehicle","pedestrian","traffic_light","lane","micro_odd","rss_profile",
                 "road_surface_state","occluded_area","hidden_actor_hypothesis","safety_event","fallback_action"}
    for node, data in graph.nodes(data=True):
        t = data.get("type","")
        if node in ("Ego","Environment") or t in key_types:
            rows.append((str(node), t, float(data.get("speed_mps", data.get("speed",0.0)) or 0.0), data.get("ttc_s","")))
    y = rect.y + 42
    draw_text(screen, "node / type / speed / ttc", (rect.x+12, y), 14, (160,170,190))
    y += 22
    for node, typ, speed, ttc in rows[:14]:
        color = NODE_COLOR.get(typ, (220,220,220))
        ttc_s = f"{float(ttc):.1f}" if isinstance(ttc,(int,float)) and float(ttc) >= 0 else "-"
        draw_text(screen, f"{node[:22]:22s} {typ[:13]:13s} {speed:4.1f} {ttc_s}", (rect.x+12, y), 13, color)
        y += 20

def render_dashboard(screen: pygame.Surface, graph: nx.MultiDiGraph, risk: Any,
                     tick: int, fps: float, history: Optional[List[str]] = None) -> None:
    """四面板仪表板主渲染"""
    w, h = screen.get_size()
    screen.fill((10, 12, 18))
    left = pygame.Rect(10, 10, int(w*0.58)-15, int(h*0.68))
    graph_rect = pygame.Rect(left.right+10, 10, w-left.right-20, int(h*0.68))
    risk_rect = pygame.Rect(10, left.bottom+10, int(w*0.58)-15, h-left.bottom-20)
    table_rect = pygame.Rect(risk_rect.right+10, graph_rect.bottom+10, w-risk_rect.right-20, h-graph_rect.bottom-20)
    render_camera_panel(screen, left, graph, risk, tick)
    render_kg_panel(screen, graph_rect, graph, tick)
    render_risk_panel(screen, risk_rect, risk, tick, fps, history)
    render_actor_table(screen, table_rect, graph)

