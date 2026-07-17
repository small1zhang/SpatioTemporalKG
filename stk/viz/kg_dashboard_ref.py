#!/usr/bin/env python3
"""Pygame dashboard for ZHB KG+RSS.

Live mode:
  - connects to CARLA
  - finds or spawns an ego vehicle
  - attaches an RGB camera
  - builds the local KG every tick
  - renders camera, KG graph, RSS/risk panel and actor table in one UI

Offline mode:
  - loads exported scene_graph_*.json files
  - renders a replay/screenshot without CARLA, useful for headless validation
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import yaml

try:
    import pygame
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("pygame is required. Install with: python3 -m pip install --user pygame") from exc

try:
    import carla  # type: ignore
except ModuleNotFoundError:
    carla = None  # offline replay can still work

from kg_rss_experiment import KGRiskInferencer, SceneGraphBuilder, ScenarioManager, Visualizer, load_config, spawn_ego


RISK_COLOR = {
    "LOW": (40, 210, 80),
    "MEDIUM": (245, 180, 40),
    "HIGH": (240, 65, 65),
}
NODE_COLOR = {
    "ego_vehicle": (255, 80, 80),
    "environment": (100, 220, 120),
    "road_surface_state": (70, 200, 220),
    "risk_state": (255, 95, 140),
    "explanation_evidence": (130, 255, 180),
    "micro_odd": (255, 220, 80),
    "rss_profile": (80, 220, 255),
    "profile_transition": (140, 255, 210),
    "risk_hold_window": (255, 210, 90),
    "hysteresis_guard": (255, 150, 220),
    "occluded_area": (170, 170, 190),
    "hidden_actor_hypothesis": (255, 120, 70),
    "safety_event": (255, 70, 70),
    "fallback_action": (90, 255, 170),
    "pedestrian": (255, 190, 40),
    "cyclist": (255, 150, 40),
    "vehicle": (80, 170, 255),
    "static": (160, 160, 160),
    "lane": (170, 120, 255),
    "junction": (210, 120, 255),
    "traffic_light": (255, 80, 220),
}
RELATION_COLOR = {
    "NEAR_BY": (190, 195, 205),
    "SAME_LANE": (100, 210, 255),
    "APPROACHING": (255, 90, 90),
    "IN_FRONT_OF": (255, 170, 80),
    "POTENTIAL_OCCLUDER": (180, 180, 180),
    "VISIBLE_TO": (120, 220, 255),
    "PARTIALLY_VISIBLE_TO": (230, 200, 80),
    "HIDDEN_BY": (210, 160, 120),
    "CREATES_OCCLUDED_AREA": (170, 170, 190),
    "HAS_HIDDEN_ACTOR_HYPOTHESIS": (255, 140, 80),
    "TIME_TO_CONFLICT": (255, 80, 80),
    "ACTIVE_MICRO_ODD": (255, 220, 80),
    "ACTIVATES_PROFILE": (70, 220, 255),
    "REQUESTS_PROFILE_CHANGE": (140, 255, 210),
    "HAS_RAW_PROFILE": (180, 220, 255),
    "HAS_ACTIVE_PROFILE": (80, 255, 210),
    "HELD_BY_HYSTERESIS": (255, 210, 90),
    "PREVENTS_OSCILLATION": (255, 150, 220),
    "HAS_ROAD_SURFACE_STATE": (70, 200, 220),
    "SUPPORTED_BY_EVIDENCE": (255, 180, 70),
    "EXPLAINS_RISK": (130, 255, 180),
    "EVIDENCE_SUPPORTS_REASON": (150, 255, 200),
    "FAITHFUL_TO_DECISION": (120, 220, 120),
    "USES_PARAMETER": (255, 210, 120),
    "EXPLAINS_ACTION": (80, 255, 150),
    "CONSTRAINS": (255, 80, 120),
    "CONTROLLED_BY": (255, 100, 230),
    "AFFECTS": (110, 220, 130),
    "ON_LANE": (170, 130, 255),
    "IN_JUNCTION": (210, 120, 255),
}
RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def init_pygame(width: int, height: int, title: str, headless: bool = False) -> pygame.Surface:
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(title)
    return screen


def font(size: int, bold: bool = False) -> pygame.font.Font:
    f = pygame.font.SysFont("DejaVu Sans,Arial,Noto Sans CJK SC", size)
    f.set_bold(bold)
    return f


def draw_text(surface: pygame.Surface, text: str, xy: Tuple[int, int], size: int = 18,
              color: Tuple[int, int, int] = (235, 235, 235), bold: bool = False) -> None:
    surface.blit(font(size, bold).render(str(text), True, color), xy)


def wrap_lines(text: str, max_chars: int = 54) -> List[str]:
    text = str(text)
    if len(text) <= max_chars:
        return [text]
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def graph_from_json(path: Path) -> Tuple[nx.MultiDiGraph, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    G = nx.MultiDiGraph()
    for node in data.get("nodes", []):
        node = dict(node)
        node_id = node.pop("id")
        G.add_node(node_id, **node)
    for edge in data.get("edges", []):
        edge = dict(edge)
        source = edge.pop("source")
        target = edge.pop("target")
        G.add_edge(source, target, **edge)
    return G, data.get("risk", {})


def risk_from_dict(risk: Dict[str, Any]) -> Any:
    class RiskLike:
        pass
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



def graph_edge_attr(graph: nx.MultiDiGraph, u: str, v: str, attr: str, default=None):
    data = graph.get_edge_data(u, v, default={})
    if isinstance(data, dict):
        if attr in data:
            return data.get(attr, default)
        for record in data.values():
            if isinstance(record, dict) and attr in record:
                return record.get(attr, default)
    return default


def render_synthetic_scene(screen: pygame.Surface, rect: pygame.Rect, graph: nx.MultiDiGraph, risk: Any, tick: int) -> None:
    """Bright CARLA-like scenario animation for offline demos.

    It visualizes the application scene even when no CARLA camera frame exists:
    rainy intersection + occluding truck + pedestrian crossing + ego vehicle.
    The animation is driven by KG distance/TTC attributes.
    """
    # Sky / city background
    pygame.draw.rect(screen, (120, 150, 180), rect)
    horizon = rect.y + int(rect.h * 0.25)
    pygame.draw.rect(screen, (180, 190, 200), pygame.Rect(rect.x, horizon - 65, rect.w, 65))
    # Buildings
    for i in range(7):
        bx = rect.x + 20 + i * int(rect.w / 7)
        bw = int(rect.w / 9)
        bh = 75 + (i % 3) * 22
        color = [(185, 170, 150), (160, 170, 185), (190, 180, 165)][i % 3]
        pygame.draw.rect(screen, color, pygame.Rect(bx, horizon - bh, bw, bh))
        for wx in range(bx + 10, bx + bw - 10, 22):
            for wy in range(horizon - bh + 12, horizon - 12, 24):
                pygame.draw.rect(screen, (235, 225, 160), pygame.Rect(wx, wy, 10, 12))
    # Road
    road = pygame.Rect(rect.x + 10, horizon, rect.w - 20, rect.h - (horizon - rect.y) - 10)
    pygame.draw.rect(screen, (58, 62, 68), road)
    # Intersection box and lanes
    cx = rect.centerx
    cy = rect.y + int(rect.h * 0.62)
    pygame.draw.rect(screen, (70, 74, 80), pygame.Rect(rect.x + 10, cy - 95, rect.w - 20, 190))
    pygame.draw.rect(screen, (70, 74, 80), pygame.Rect(cx - 130, horizon, 260, road.h))
    for x in range(rect.x + 35, rect.right - 35, 70):
        pygame.draw.line(screen, (235, 225, 120), (x, cy - 95), (x + 35, cy - 95), 3)
        pygame.draw.line(screen, (235, 225, 120), (x, cy + 95), (x + 35, cy + 95), 3)
    for y in range(horizon + 20, rect.bottom - 30, 45):
        pygame.draw.line(screen, (240, 240, 240), (cx - 130, y), (cx - 130, y + 22), 2)
        pygame.draw.line(screen, (240, 240, 240), (cx + 130, y), (cx + 130, y + 22), 2)
    # Crosswalk
    for i in range(8):
        pygame.draw.rect(screen, (245, 245, 235), pygame.Rect(cx - 105 + i * 28, cy - 18, 14, 36))
    # Rain overlay
    if any((graph.nodes[n].get('type') == 'environment' and float(graph.nodes[n].get('precipitation', 0) or 0) > 10) for n in graph.nodes):
        for i in range(55):
            rx = rect.x + (i * 37 + tick * 13) % rect.w
            ry = rect.y + (i * 53 + tick * 19) % rect.h
            pygame.draw.line(screen, (170, 210, 255), (rx, ry), (rx - 8, ry + 18), 1)
    # Extract semantic values
    ped_dist = float(graph_edge_attr(graph, 'Ego', 'pedestrian:1', 'distance', max(4, 30 - tick * 1.5)) or 10)
    ttc = float(graph.nodes.get('pedestrian:1', {}).get('ttc_s', max(0.8, 5 - tick * 0.2)) or 1)
    # Ego car bottom center, moving slightly forward
    ego_x = cx
    ego_y = rect.bottom - 85
    pygame.draw.rect(screen, (235, 55, 55), pygame.Rect(ego_x - 34, ego_y - 55, 68, 110), border_radius=10)
    pygame.draw.rect(screen, (40, 45, 55), pygame.Rect(ego_x - 24, ego_y - 35, 48, 32), border_radius=5)
    pygame.draw.circle(screen, (30, 30, 30), (ego_x - 38, ego_y - 30), 9)
    pygame.draw.circle(screen, (30, 30, 30), (ego_x + 38, ego_y - 30), 9)
    draw_text(screen, 'EGO', (ego_x - 22, ego_y + 18), 16, (255,255,255), True)
    # Occluding truck right side of crosswalk
    truck_x, truck_y = cx + 170, cy - 50
    pygame.draw.rect(screen, (120, 130, 140), pygame.Rect(truck_x - 45, truck_y - 35, 95, 70), border_radius=8)
    pygame.draw.rect(screen, (85, 95, 105), pygame.Rect(truck_x + 20, truck_y - 28, 35, 56), border_radius=5)
    draw_text(screen, 'OCCLUDER', (truck_x - 50, truck_y + 52), 13, (255,255,255), True)
    # Pedestrian crossing from right to left; distance controls x/y
    progress = max(0.0, min(1.0, (28.0 - ped_dist) / 24.0))
    ped_x = int(cx + 210 - progress * 300)
    ped_y = cy
    hidden_active = any(graph.nodes[n].get('type') == 'hidden_actor_hypothesis' for n in graph.nodes)
    if hidden_active:
        hidden_x, hidden_y = truck_x + 45, cy + 54
        pygame.draw.circle(screen, (255, 120, 70), (hidden_x, hidden_y), 18, 2)
        pygame.draw.line(screen, (255, 120, 70), (hidden_x - 12, hidden_y - 12), (hidden_x + 12, hidden_y + 12), 2)
        pygame.draw.line(screen, (255, 120, 70), (hidden_x + 12, hidden_y - 12), (hidden_x - 12, hidden_y + 12), 2)
        pygame.draw.line(screen, (255, 120, 70), (hidden_x, hidden_y), (ped_x, ped_y), 2)
        draw_text(screen, 'HiddenActor', (hidden_x - 45, hidden_y + 26), 13, (255, 170, 110), True)
    pygame.draw.circle(screen, (255, 205, 60), (ped_x, ped_y - 22), 13)
    pygame.draw.line(screen, (255, 205, 60), (ped_x, ped_y - 8), (ped_x, ped_y + 24), 5)
    pygame.draw.line(screen, (255, 205, 60), (ped_x, ped_y + 2), (ped_x - 18, ped_y + 14), 4)
    pygame.draw.line(screen, (255, 205, 60), (ped_x, ped_y + 2), (ped_x + 18, ped_y + 12), 4)
    pygame.draw.line(screen, (255, 205, 60), (ped_x, ped_y + 24), (ped_x - 15, ped_y + 48), 4)
    pygame.draw.line(screen, (255, 205, 60), (ped_x, ped_y + 24), (ped_x + 14, ped_y + 48), 4)
    # Danger line / TTC label
    risk_color = RISK_COLOR.get(risk.level, (40,210,80))
    pygame.draw.line(screen, risk_color, (ego_x, ego_y - 60), (ped_x, ped_y), 4 if risk.level == 'HIGH' else 2)
    info_rect = pygame.Rect(rect.x + 24, rect.y + 55, min(430, rect.w - 48), 132)
    pygame.draw.rect(screen, (20, 22, 28), info_rect, border_radius=8)
    pygame.draw.rect(screen, risk_color, info_rect, 2, border_radius=8)
    draw_text(screen, "CARLA Truth Replay: Rainy Occluded Crossing", (info_rect.x + 14, info_rect.y + 30), 17, (255,255,255), True)
    draw_text(screen, f"Pedestrian distance: {ped_dist:.1f} m    TTC: {ttc:.1f}s", (info_rect.x + 14, info_rect.y + 62), 16, (235,235,235))
    draw_text(screen, f"MicroODD/RSS: {getattr(risk, 'active_profile', 'Urban-Dry')}", (info_rect.x + 14, info_rect.y + 90), 16, (190, 220, 255), True)
    draw_text(screen, f"Risk: {risk.level}   explore v={getattr(risk, 'explore_speed_mps', 5.0):.1f}m/s", (info_rect.x + 14, info_rect.y + 118), 16, risk_color, True)

    banner_rect = pygame.Rect(rect.right - 392, rect.y + 58, 362, 76)
    pygame.draw.rect(screen, (20, 22, 28), banner_rect, border_radius=8)
    pygame.draw.rect(screen, risk_color, banner_rect, 2, border_radius=8)
    if risk.level == 'HIGH':
        draw_text(screen, "RSS shield ON", (banner_rect.x + 16, banner_rect.y + 29), 22, (255, 110, 110), True)
        draw_text(screen, "low-speed fallback + evidence chain", (banner_rect.x + 16, banner_rect.y + 58), 14, (235, 235, 235))
    elif risk.level == 'MEDIUM':
        draw_text(screen, "KG semantic trigger", (banner_rect.x + 16, banner_rect.y + 29), 20, (245, 210, 80), True)
        draw_text(screen, "wet road / occlusion profile", (banner_rect.x + 16, banner_rect.y + 58), 14, (235, 235, 235))
    else:
        draw_text(screen, "Nominal driving", (banner_rect.x + 16, banner_rect.y + 29), 20, (100, 240, 140), True)
        draw_text(screen, "monitoring CARLA truth graph", (banner_rect.x + 16, banner_rect.y + 58), 14, (235, 235, 235))

def render_camera_panel(screen: pygame.Surface, rect: pygame.Rect, frame: Optional[np.ndarray], graph: Optional[nx.MultiDiGraph] = None, risk: Any = None, tick: int = 0) -> None:
    pygame.draw.rect(screen, (20, 22, 28), rect)
    pygame.draw.rect(screen, (80, 90, 110), rect, 2)
    title = "CARLA RGB Camera" if frame is not None else "Application Scene Animation (CARLA-style)"
    draw_text(screen, title, (rect.x + 14, rect.y + 10), 20, bold=True)
    inner = pygame.Rect(rect.x + 12, rect.y + 42, rect.w - 24, rect.h - 54)
    if frame is None:
        if graph is not None and risk is not None:
            render_synthetic_scene(screen, inner, graph, risk, tick)
        else:
            pygame.draw.rect(screen, (8, 10, 14), inner)
            draw_text(screen, "No camera frame yet / Offline KG replay", (inner.x + 20, inner.y + 30), 22, (180, 190, 210))
        return
    # frame: H,W,3 RGB
    surf = pygame.surfarray.make_surface(np.rot90(frame))
    surf = pygame.transform.smoothscale(surf, (inner.w, inner.h))
    screen.blit(surf, inner)




def pretty_reason(reason: str) -> str:
    """Render compact English explanations in pygame even when CJK fonts are missing."""
    r = str(reason)
    if "强降雨" in r or "湿滑" in r:
        return "Wet road/rain: lower accel and braking assumptions"
    if "微观运行域" in r or "MicroODD" in r:
        return "MicroODD activates certified RSS profile"
    if "隐藏参与者" in r or "HiddenActor" in r or "hidden:" in r:
        return "Hidden actor hypothesis reaches conflict zone: low-speed exploration"
    if "潜在遮挡" in r or "遮挡" in r:
        return "Occluder nearby: defensive RSS response"
    if "弱势交通参与者" in r or "行人" in r:
        return "Vulnerable road user nearby: increase response time"
    if "TTC" in r or "ttc" in r:
        return r.replace("低 TTC 冲突", "Low TTC conflict").replace("：", ":")
    if "路口" in r:
        return "Junction multi-agent uncertainty: defensive response"
    if "红" in r or "黄" in r:
        return "Red/yellow light constraint risk"
    return r

def render_risk_timeline(screen: pygame.Surface, rect: pygame.Rect, history: List[str]) -> None:
    if not history:
        return
    pygame.draw.rect(screen, (12, 14, 20), rect)
    pygame.draw.rect(screen, (70, 78, 95), rect, 1)
    draw_text(screen, "Risk timeline", (rect.x + 8, rect.y + 5), 13, (180, 190, 210), True)
    max_items = max(1, rect.w // 9)
    recent = history[-max_items:]
    x = rect.x + 8
    y = rect.y + 26
    bar_w = max(4, (rect.w - 16) // max_items)
    for level in recent:
        color = RISK_COLOR.get(level, (180, 180, 180))
        h = 10 + RISK_ORDER.get(level, 0) * 12
        pygame.draw.rect(screen, color, pygame.Rect(x, y + 28 - h, bar_w - 1, h))
        x += bar_w


def render_legend(screen: pygame.Surface, rect: pygame.Rect) -> None:
    x, y = rect.x, rect.y
    draw_text(screen, "Legend", (x, y), 13, (180, 190, 210), True)
    y += 18
    items = [("ego", "ego_vehicle"), ("ped", "pedestrian"), ("veh", "vehicle"), ("MicroODD", "micro_odd"), ("RSS", "rss_profile"), ("hidden", "hidden_actor_hypothesis")]
    for label, typ in items:
        pygame.draw.circle(screen, NODE_COLOR.get(typ, (220, 220, 220)), (x + 8, y + 7), 6)
        draw_text(screen, label, (x + 20, y), 12, (220, 220, 220))
        x += 70
    x = rect.x
    y += 22
    for rel in ["APPROACHING", "ACTIVATES_PROFILE", "TIME_TO_CONFLICT"]:
        pygame.draw.line(screen, RELATION_COLOR.get(rel, (180, 180, 180)), (x, y + 7), (x + 28, y + 7), 3)
        draw_text(screen, rel, (x + 34, y), 11, (210, 210, 210))
        x += 180

def render_risk_panel(screen: pygame.Surface, rect: pygame.Rect, risk: Any, tick: int, fps: float, history: Optional[List[str]] = None) -> None:
    pygame.draw.rect(screen, (20, 22, 28), rect)
    pygame.draw.rect(screen, RISK_COLOR.get(risk.level, (220, 220, 220)), rect, 3)
    draw_text(screen, f"Risk: {risk.level}", (rect.x + 14, rect.y + 10), 24, RISK_COLOR.get(risk.level, (230, 230, 230)), True)
    draw_text(screen, f"tick={tick}  FPS={fps:.1f}", (rect.x + 170, rect.y + 15), 16, (190, 195, 205))
    draw_text(screen, f"RSS response_time: {risk.response_time:.2f}s", (rect.x + 14, rect.y + 48), 18)
    draw_text(screen, f"accel_max: {risk.accel_max:.2f}   brake_min: {risk.brake_min:.2f}   front_brake_max: {getattr(risk, 'brake_front_max', 8.0):.1f}", (rect.x + 14, rect.y + 74), 17)
    draw_text(screen, f"profile: {getattr(risk, 'active_profile', 'Urban-Dry')}  semantic_margin={getattr(risk, 'semantic_margin_m', 0.0):.1f}m  explore_v={getattr(risk, 'explore_speed_mps', 5.0):.1f}m/s", (rect.x + 14, rect.y + 100), 16, (190, 210, 255), True)
    draw_text(screen, "KG rule reasons / evidence:", (rect.x + 14, rect.y + 126), 18, bold=True)
    y = rect.y + 152
    reasons = risk.reasons or ["No risk rule triggered"]
    for reason in reasons[:5]:
        for line in wrap_lines("- " + pretty_reason(reason), 58):
            if y > rect.bottom - 22:
                return
            draw_text(screen, line, (rect.x + 20, y), 15, (220, 220, 220))
            y += 20
    if history is not None:
        render_risk_timeline(screen, pygame.Rect(rect.x + rect.w - 300, rect.y + 40, 280, 78), history)


def render_actor_table(screen: pygame.Surface, rect: pygame.Rect, graph: nx.MultiDiGraph) -> None:
    pygame.draw.rect(screen, (20, 22, 28), rect)
    pygame.draw.rect(screen, (80, 90, 110), rect, 2)
    draw_text(screen, "KG Objects", (rect.x + 12, rect.y + 10), 20, bold=True)
    rows = []
    key_types = {
        "vehicle", "pedestrian", "cyclist", "static", "traffic_light", "lane",
        "micro_odd", "rss_profile", "road_surface_state", "occluded_area",
        "hidden_actor_hypothesis", "safety_event", "fallback_action",
    }
    for node, data in graph.nodes(data=True):
        t = data.get("type", "")
        if node in ("Ego", "Environment") or t in key_types:
            rows.append((str(node), t, float(data.get("speed_mps", data.get("speed", 0.0)) or 0.0), data.get("ttc_s", "")))
    y = rect.y + 42
    draw_text(screen, "node / type / speed / ttc", (rect.x + 12, y), 14, (160, 170, 190)); y += 22
    for node, typ, speed, ttc in rows[:14]:
        color = NODE_COLOR.get(typ, (220, 220, 220))
        ttc_s = f"{float(ttc):.1f}" if isinstance(ttc, (int, float)) and float(ttc) >= 0 else "-"
        draw_text(screen, f"{node[:22]:22s} {typ[:13]:13s} {speed:4.1f} {ttc_s}", (rect.x + 12, y), 13, color)
        y += 20


def semantic_kg_layout(graph: nx.MultiDiGraph) -> Dict[str, Tuple[float, float]]:
    """Stable presentation layout for the Word-protocol KG nodes.

    Spring layout is useful for arbitrary graphs but jitters when semantic nodes
    appear/disappear.  For the demo dashboard we keep ontology roles in fixed
    regions so the viewer can track the dynamic evidence chain over time.
    """
    anchors = {
        "Ego": (0.0, 0.0),
        "Environment": (0.86, -0.05),
        "MicroODD": (-0.05, -0.76),
    }
    type_anchor = {
        "road_surface_state": (0.64, -0.62),
        "rss_profile": (0.52, -0.82),
        "lane": (-0.82, -0.48),
        "junction": (-0.90, 0.0),
        "traffic_light": (0.03, -0.98),
        "vehicle": (0.72, 0.45),
        "pedestrian": (-0.46, 0.72),
        "cyclist": (-0.62, 0.62),
        "static": (0.70, 0.05),
        "occluded_area": (0.25, 0.36),
        "hidden_actor_hypothesis": (-0.12, 0.55),
        "safety_event": (-0.58, -0.82),
        "fallback_action": (-0.78, -0.58),
    }
    buckets: Dict[str, int] = {}
    pos: Dict[str, Tuple[float, float]] = {}
    for idx, (node, data) in enumerate(graph.nodes(data=True)):
        if node in anchors:
            pos[str(node)] = anchors[str(node)]
            continue
        typ = str(data.get("type", ""))
        ax, ay = type_anchor.get(typ, (math.cos(idx), math.sin(idx)))
        k = buckets.get(typ, 0)
        buckets[typ] = k + 1
        # Small deterministic offsets for multiple actors of the same type.
        offset = (k - 1) * 0.10 if k else 0.0
        pos[str(node)] = (max(-0.98, min(0.98, ax + offset)), max(-0.98, min(0.98, ay + 0.07 * (k % 3))))
    return pos


def render_kg_panel(screen: pygame.Surface, rect: pygame.Rect, graph: nx.MultiDiGraph, tick: int = 0) -> None:
    pygame.draw.rect(screen, (16, 18, 24), rect)
    pygame.draw.rect(screen, (80, 90, 110), rect, 2)
    draw_text(screen, "Dynamic Local Scene Knowledge Graph", (rect.x + 14, rect.y + 10), 20, bold=True)
    if graph.number_of_nodes() == 0:
        return
    if any(d.get("type") in {"micro_odd", "rss_profile", "occluded_area", "hidden_actor_hypothesis", "safety_event"} for _, d in graph.nodes(data=True)):
        pos = semantic_kg_layout(graph)
    else:
        try:
            pos = nx.spring_layout(graph, seed=42, k=0.85, iterations=30)
        except Exception:
            pos = {n: (math.cos(i), math.sin(i)) for i, n in enumerate(graph.nodes())}
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs); min_y, max_y = min(ys), max(ys)
    pad = 45
    area = pygame.Rect(rect.x + pad, rect.y + 55, rect.w - 2 * pad, rect.h - 95)

    def scale(p):
        x, y = p
        sx = area.x + (x - min_x) / (max_x - min_x + 1e-6) * area.w
        sy = area.y + (y - min_y) / (max_y - min_y + 1e-6) * area.h
        return int(sx), int(sy)

    for u, v, data in graph.edges(data=True):
        if u not in pos or v not in pos:
            continue
        a, b = scale(pos[u]), scale(pos[v])
        rel_full = str(data.get("relation", ""))
        color = RELATION_COLOR.get(rel_full, (95, 105, 125))
        width = 3 if rel_full in ("APPROACHING", "SAME_LANE", "CONTROLLED_BY", "ACTIVATES_PROFILE", "TIME_TO_CONFLICT", "EXPLAINS_ACTION") else 1
        if rel_full in ("APPROACHING", "TIME_TO_CONFLICT", "ACTIVATES_PROFILE"):
            # pulse critical approach edges to make dynamic risk obvious
            pulse = 1 + int((tick % 10) / 5)
            width += pulse
        pygame.draw.line(screen, color, a, b, width)
        mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
        rel = rel_full[:16]
        if rel_full in ("SAME_LANE", "APPROACHING", "NEAR_BY", "CONTROLLED_BY", "POTENTIAL_OCCLUDER", "ACTIVATES_PROFILE", "TIME_TO_CONFLICT", "EXPLAINS_ACTION", "CREATES_OCCLUDED_AREA"):
            draw_text(screen, rel, (mx, my), 11, color)
    for node, data in graph.nodes(data=True):
        x, y = scale(pos[node])
        typ = data.get("type", "")
        color = NODE_COLOR.get(typ, (220, 220, 220))
        radius = 12 if node != "Ego" else 18
        pygame.draw.circle(screen, color, (x, y), radius)
        pygame.draw.circle(screen, (20, 20, 20), (x, y), radius, 2)
        label = str(node)[:18]
        draw_text(screen, label, (x + radius + 3, y - 8), 13, (235, 235, 235), node == "Ego")
    render_legend(screen, pygame.Rect(rect.x + 14, rect.bottom - 70, rect.w - 28, 44))
    rel_counts: Dict[str, int] = {}
    for _, _, data in graph.edges(data=True):
        rel_counts[str(data.get("relation", ""))] = rel_counts.get(str(data.get("relation", "")), 0) + 1
    top_rel = ", ".join(f"{k}:{v}" for k, v in list(rel_counts.items())[:4])
    draw_text(screen, f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}  {top_rel}", (rect.x + 14, rect.bottom - 22), 14, (180, 190, 210))


class CameraManager:
    def __init__(self, world: Any, ego: Any, width: int = 960, height: int = 540):
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", "90")
        transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-8))
        self.sensor = world.spawn_actor(bp, transform, attach_to=ego)
        self.sensor.listen(self._on_image)

    def _on_image(self, image: Any) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.queue.put(array)

    def latest(self) -> Optional[np.ndarray]:
        frame = None
        while not self.queue.empty():
            frame = self.queue.get_nowait()
        return frame

    def destroy(self) -> None:
        try:
            self.sensor.stop()
            self.sensor.destroy()
        except Exception:
            pass


def find_ego(world: Any) -> Optional[Any]:
    actors = world.get_actors().filter("vehicle.*")
    for actor in actors:
        if actor.attributes.get("role_name") == "hero":
            return actor
    return actors[0] if len(actors) else None


def run_offline(args: argparse.Namespace) -> int:
    paths = sorted(Path(args.replay_dir).glob("scene_graph_*.json"))
    if not paths:
        raise SystemExit(f"No scene_graph_*.json found in {args.replay_dir}")
    screen = init_pygame(args.width, args.height, "ZHB KG+RSS Offline Replay", args.headless)
    clock = pygame.time.Clock()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    history: List[str] = []
    paused = False
    for idx, path in enumerate(paths[: max(1, args.ticks)]):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = not paused
        graph, risk_dict = graph_from_json(path)
        risk = risk_from_dict(risk_dict)
        history.append(risk.level)
        render_dashboard(screen, None, graph, risk, idx, clock.get_fps(), history)
        pygame.display.flip()
        pygame.image.save(screen, out_dir / f"dashboard_offline_{idx:04d}.png")
        clock.tick(args.fps)
    pygame.quit()
    print(f"PYGAME_OFFLINE_OK frames={min(len(paths), max(1, args.ticks))} output={out_dir}")
    return 0


def render_dashboard(screen: pygame.Surface, frame: Optional[np.ndarray], graph: nx.MultiDiGraph, risk: Any, tick: int, fps: float, history: Optional[List[str]] = None) -> None:
    w, h = screen.get_size()
    screen.fill((10, 12, 18))
    left = pygame.Rect(10, 10, int(w * 0.58) - 15, int(h * 0.68))
    graph_rect = pygame.Rect(left.right + 10, 10, w - left.right - 20, int(h * 0.68))
    risk_rect = pygame.Rect(10, left.bottom + 10, int(w * 0.58) - 15, h - left.bottom - 20)
    table_rect = pygame.Rect(risk_rect.right + 10, graph_rect.bottom + 10, w - risk_rect.right - 20, h - graph_rect.bottom - 20)
    render_camera_panel(screen, left, frame, graph, risk, tick)
    render_kg_panel(screen, graph_rect, graph, tick)
    render_risk_panel(screen, risk_rect, risk, tick, fps, history)
    render_actor_table(screen, table_rect, graph)


def run_live(args: argparse.Namespace) -> int:
    if carla is None:
        raise SystemExit("CARLA module is required for live dashboard")
    cfg = load_config(args.config)
    cfg["carla"]["host"] = args.host
    cfg["carla"]["port"] = args.port
    cfg["carla"]["town"] = args.town or ""
    screen = init_pygame(args.width, args.height, "ZHB KG+RSS Live Dashboard", args.headless)
    clock = pygame.time.Clock()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    if args.town and world.get_map().name.split("/")[-1] != args.town:
        world = client.load_world(args.town)
    ego = find_ego(world)
    spawned_ego = False
    if ego is None:
        ego = spawn_ego(world, cfg)
        spawned_ego = True
    scenario_manager = None
    if not args.no_scenario:
        cfg.setdefault("scenario", {})["enabled"] = True
        cfg["scenario"]["deterministic_demo"] = True
        scenario_manager = ScenarioManager(cfg)
        scenario_manager.setup(client, world, ego)
    camera = CameraManager(world, ego, args.camera_width, args.camera_height)
    builder = SceneGraphBuilder(cfg)
    inferencer = KGRiskInferencer(cfg)
    visualizer = Visualizer({**cfg, "visualization": {**cfg["visualization"], "output_dir": args.output_dir}})
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    tick = 0
    running = True
    history: List[str] = []
    try:
        while running and (args.ticks <= 0 or tick < args.ticks):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
            try:
                world.tick() if world.get_settings().synchronous_mode else world.wait_for_tick(timeout=args.timeout)
            except Exception:
                pass
            if scenario_manager:
                scenario_manager.tick(tick, world, ego)
            graph = builder.build(ego, world)
            risk = inferencer.infer(graph)
            frame = camera.latest()
            history.append(risk.level)
            render_dashboard(screen, frame, graph, risk, tick, clock.get_fps(), history)
            pygame.display.flip()
            if args.save_every > 0 and tick % args.save_every == 0:
                pygame.image.save(screen, output_dir / f"dashboard_live_{tick:06d}.png")
                visualizer.export_graph_json(graph, tick, risk)
                visualizer.export_graph_graphml(graph, tick, risk)
                visualizer.export_graph_turtle(graph, tick, risk)
                visualizer.export_graph_png(graph, tick, risk)
            tick += 1
            clock.tick(args.fps)
    finally:
        camera.destroy()
        if scenario_manager:
            scenario_manager.destroy()
        if spawned_ego:
            try:
                ego.destroy()
            except Exception:
                pass
        pygame.quit()
    print(f"PYGAME_LIVE_OK ticks={tick} output={output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2001)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--town", default="")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--camera-width", type=int, default=960)
    parser.add_argument("--camera-height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/dashboard")
    parser.add_argument("--headless", action="store_true", help="Use SDL dummy video driver and save screenshots")
    parser.add_argument("--offline", action="store_true", help="Replay exported KG JSON instead of connecting to CARLA")
    parser.add_argument("--no-scenario", action="store_true", help="Do not spawn the deterministic rainy crossing demo actors")
    parser.add_argument("--replay-dir", default="outputs/real_smoke")
    args = parser.parse_args()
    if args.offline:
        return run_offline(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
