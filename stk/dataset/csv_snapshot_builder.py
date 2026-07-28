# -*- coding: utf-8 -*-
"""
CSV + event_labels.json → snapshot dict 转换器（§6.2 数据适配层）

将 data/dataset/ 中的三份数据（frame_actors.csv / frame_labels.csv / event_labels.json）
转换为 extract_stkg_tensors() 可消费的 snapshot dict 格式：
  snapshot = {
    "extracted": {
      "frame_id", "elapsed_seconds", "delta_seconds",
      "vehicles": [...],  "pedestrians": [...],
      "traffic_lights": [...], "lanes": [...],
      "scene_rels": [...], "weather": {...},
    },
    "rule_out": {
      "violations": [SafetyViolation, ...],
    },
    "delta": {...},  # Δg_t 差分特征
  }

CSV 字段 → snapshot 字段映射（尽量复用 _build_node_features() 的 VEHICLE_FEATURE_KEYS）：
  CSV: actor_id          → entity_id
  CSV: x,y,z             → location_x/y/z
  CSV: yaw,pitch,roll    → rotation
  CSV: vx,vy,vz          → velocity_x/y/z
  CSV: speed             → speed (m/s)
  CSV: heading_rad       → heading_rad
  CSV: throttle,brake,steer → control
  CSV: road_id,lane_id   → lane_info
  CSV: is_emergency      → is_emergency
  CSV: is_anomaly,is_anomaly_target → 标签

异常类型 → 规则码映射（用于 rule_out.violations）：
  sudd_brk  → R13  (RSS sudden brake)
  avd_col   → R4   (avoidance/collision, yield)
  rev_drive → R5   (reverse driving)
  obs_blk   → R1   (obstacle blocking = pedestrian danger)
  jun_ny    → R7   (junction non-yield)
  sudd_stp  → R13  (sudden stop = RSS)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 列名常量
# ============================================================
FRAME_LABEL_COLS: List[str] = [
    "frame_id",
    "elapsed_seconds",
    "delta_seconds",
    "map_name",
    "scenario_id",
    "origin_run",
    "n_actors",
    "n_vehicles",
    "n_pedestrians",
    "weather_cloudiness",
    "weather_precipitation",
    "weather_fog_density",
    "weather_sun_altitude",
    "weather_wetness",
    "is_anomaly",
    "anomaly_type",
    "anomaly_event_ids",
    "target_actor_ids",
    "n_violations",
    "max_severity",
    "rule_codes",
    "split",
]

ACTOR_COLS: List[str] = [
    "frame_id",
    "elapsed_seconds",
    "map_name",
    "scenario_id",
    "actor_id",
    "type",
    "is_ego",
    "type_id",
    "is_alive",
    "x",
    "y",
    "z",
    "yaw",
    "pitch",
    "roll",
    "vx",
    "vy",
    "vz",
    "ax",
    "ay",
    "az",
    "speed",
    "speed_kmh",
    "heading_rad",
    "throttle",
    "brake",
    "steer",
    "bbox_x",
    "bbox_y",
    "bbox_z",
    "road_id",
    "lane_id",
    "is_emergency",
    "is_on_crosswalk",
    "is_on_sidewalk",
    "action",
    "is_anomaly",
    "anomaly_type",
    "is_anomaly_target",
]

EVENT_COLS: List[str] = [
    "event_id",
    "anomaly_type",
    "trigger_frame",
    "duration_frames",
    "end_frame",
    "target_actor_id",
    "target_role",
    "intensity",
    "map_name",
    "origin_run",
    "label_source",
]

# 异常类型 → 规则码映射（用于生成 rule_out.violations）
ANOMALY_TYPE_TO_RULE: Dict[str, str] = {
    "sudd_brk":  "R13",  # RSS 紧急制动
    "avd_col":   "R4",   # yield / avoidance collision
    "rev_drive": "R5",   # 逆向行驶
    "obs_blk":   "R1",   # obstacle blocking ≈ pedestrian danger
    "jun_ny":    "R7",   # junction non-yield
    "sudd_stp":  "R13",  # 突然停车（同 RSS 制动）
    "lane_inv":  "R8",   # 越线侵入
    "spd_exc":   "R3",   # 超速
    "tailg":     "R15",  # 跟车过近
}


def _f(v: Any, default: float = 0.0) -> float:
    """安全转 float。"""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _b(v: Any) -> float:
    return 1.0 if bool(v) else 0.0


# ============================================================
# CSV actor row → vehicle/pedestrian dict 映射
# ============================================================
def _row_to_vehicle(row: pd.Series) -> Dict[str, Any]:
    """单行 actor CSV → vehicle dict（兼容 _build_node_features() 的 VEHICLE_FEATURE_KEYS）"""
    return {
        "entity_id":       str(row["actor_id"]),
        "entity_type":     "Vehicle",
        "type_id":         str(row.get("type_id", "")),
        "is_ego":          _b(row.get("is_ego")),
        "is_alive":        _b(row.get("is_alive")),
        "is_emergency":    _b(row.get("is_emergency")),
        "location_x":      _f(row.get("x")),
        "location_y":      _f(row.get("y")),
        "location_z":      _f(row.get("z")),
        "heading_rad":     _f(row.get("heading_rad"), _f(row.get("yaw"), 0.0)),
        "rotation":        {"pitch": _f(row.get("pitch"), 0.0),
                            "yaw":   _f(row.get("yaw"), 0.0),
                            "roll":  _f(row.get("roll"), 0.0)},
        "speed":           _f(row.get("speed"), 0.0),
        "velocity":        {"x": _f(row.get("vx"), 0.0),
                            "y": _f(row.get("vy"), 0.0),
                            "z": _f(row.get("vz"), 0.0)},
        "acceleration":    {"x": _f(row.get("ax"), 0.0),
                            "y": _f(row.get("ay"), 0.0),
                            "z": _f(row.get("az"), 0.0)},
        "control":         {"throttle": _f(row.get("throttle"), 0.0),
                            "brake":    _f(row.get("brake"), 0.0),
                            "steer":    _f(row.get("steer"), 0.0)},
        "bbox":            {"x": _f(row.get("bbox_x"), 2.5),
                            "y": _f(row.get("bbox_y"), 1.2),
                            "z": _f(row.get("bbox_z"), 1.0)},
        # lane info
        "current_lane":    {"road_id": int(_f(row.get("road_id"), -1)),
                            "lane_id": int(_f(row.get("lane_id"), -1))},
        # CSV 无 prev speed，设速度分量为 vx_prev
        "velocity_x":      _f(row.get("vx"), 0.0),
        "velocity_y":      _f(row.get("vy"), 0.0),
        # 标签字段
        "is_anomaly":      _b(row.get("is_anomaly")),
        "is_anomaly_target": _b(row.get("is_anomaly_target")),
        "anomaly_type":    str(row.get("anomaly_type", "") or ""),
        "road_id":         int(_f(row.get("road_id"), -1)),
        "lane_id":         int(_f(row.get("lane_id"), -1)),
    }


def _row_to_pedestrian(row: pd.Series) -> Dict[str, Any]:
    """单行 actor CSV → pedestrian dict"""
    return {
        "entity_id":      str(row["actor_id"]),
        "entity_type":    "Pedestrian",
        "is_alive":       _b(row.get("is_alive")),
        "location_x":     _f(row.get("x")),
        "location_y":     _f(row.get("y")),
        "location_z":     _f(row.get("z")),
        "speed":          _f(row.get("speed"), 0.0),
        "action":         str(row.get("action", "") or ""),
        "labels":         [str(row.get("action", ""))] if row.get("action") else [],
        "crossing":       _b(row.get("is_on_crosswalk")),
        "sidewalk":       _b(row.get("is_on_sidewalk")),
        "is_anomaly_target": _b(row.get("is_anomaly_target")),
        "anomaly_type":   str(row.get("anomaly_type", "") or ""),
        "bbox":           {"x": _f(row.get("bbox_x"), 0.6),
                            "y": _f(row.get("bbox_y"), 0.6),
                            "z": _f(row.get("bbox_z"), 1.7)},
    }


# ============================================================
# 违规对象（用于 rule_out.violations）
# ============================================================
@dataclass
class CSVViolation:
    """从 event_labels.json 转换的 SafetyViolation 兼容对象。"""
    rule_code: str = ""            # 如 "R13", "R4"
    rule_name: str = ""            # 可读名称
    rule_layer: str = "RSS"        # RSS / D-S / STKG
    src_id: str = ""               # target_actor_id
    dst_id: str = ""               # 同 src_id（自指）
    severity: float = 0.0          # event intensity (0~1)
    predicate_str: str = ""
    evidence_path: List[str] = field(default_factory=list)

    def model_dump(self) -> Dict[str, Any]:
        return {
            "rule_code": self.rule_code,
            "rule_name": self.rule_name,
            "rule_layer": self.rule_layer,
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "severity": self.severity,
            "predicate_str": self.predicate_str,
            "evidence_path": self.evidence_path,
            "attrs": {
                "rule_code": self.rule_code,
                "rule_name": self.rule_name,
                "severity": self.severity,
                "src_id": self.src_id,
                "dst_id": self.dst_id,
            },
        }


# ============================================================
# 核心：build_snapshot_from_csv
# ============================================================
def _build_k_nn_edges(
    vehicles: List[Dict],
    pedestrians: List[Dict],
    all_nodes: List[Dict],
    id2row: Dict[str, int],
    K: int = 5,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    空间 K-NN 图构建（与 exporter _build_edge_index fallback 一致）。
    返回 ((src, dst) pairs, edge_type_list)。
    """
    src_list: List[int] = []
    dst_list: List[int] = []
    type_list: List[int] = []

    node_pos: List[Tuple[str, float, float, float, str]] = []
    for v in vehicles:
        nid = str(v.get("entity_id", ""))
        if nid in id2row:
            node_pos.append((nid,
                              _f(v.get("location_x"), 0.0),
                              _f(v.get("location_y"), 0.0),
                              _f(v.get("heading_rad"), 0.0),
                              "vehicle"))
    for p in pedestrians:
        nid = str(p.get("entity_id", ""))
        if nid in id2row:
            node_pos.append((nid,
                              _f(p.get("location_x"), 0.0),
                              _f(p.get("location_y"), 0.0),
                              0.0,
                              "pedestrian"))

    K_nn = min(K, max(len(node_pos) - 1, 0))
    for i, (ni, xi, yi, hi, ti) in enumerate(node_pos):
        dists = []
        for j, (nj, xj, yj, hj, tj) in enumerate(node_pos):
            if i == j:
                continue
            d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            dists.append((d, j, tj))
        dists.sort(key=lambda t: t[0])
        for d2, j, tj in dists[:K_nn]:
            _, xj2, yj2, hj2, tj2_node = node_pos[j]
            nj2 = node_pos[j][0]
            dx = xj2 - xi
            dy = yj2 - yi
            angle_diff = abs(math.atan2(math.sin(hi - hj2), math.cos(hi - hj2)))
            if ti == "pedestrian" or tj2_node == "pedestrian":
                rel = "nearby_pedestrian"
            elif angle_diff < math.pi / 4 and dx * math.cos(hi) + dy * math.sin(hi) > 0:
                rel = "ahead_of"
            elif angle_diff < math.pi / 4:
                rel = "in_lane"
            else:
                rel = "beside"
            # 将 rel 名转为 SCENE_REL_TO_IDX 索引
            from stk.ontology.types import SceneRelationType
            rel_to_idx = {r.value: i for i, r in enumerate(SceneRelationType)}
            rel_idx = rel_to_idx.get(rel, rel_to_idx.get("in_lane", 3))
            if ni in id2row and nj2 in id2row:
                src_list.append(id2row[ni])
                dst_list.append(id2row[nj2])
                type_list.append(rel_idx)

    return (src_list, dst_list), type_list


def build_snapshot_from_csv(
    frame_id: int,
    actors_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    events: List[Dict[str, Any]],
    prev_snapshot: Optional[Dict[str, Any]] = None,
    max_actors: int = 30,
) -> Dict[str, Any]:
    """
    单帧 CSV 数据 → snapshot dict（与 PipelineOrchestrator 输出格式兼容）。

    Args:
        frame_id:      当前帧 ID
        actors_df:     frame_actors.csv 的 DataFrame（内存中）
        labels_df:     frame_labels.csv 的 DataFrame（内存中）
        events:        event_labels.json 的列表（全量，按 trigger_frame 过滤）
        prev_snapshot: 前一帧 snapshot（用于计算 delta_feat；首帧可传 None）
        max_actors:    最大节点数（截断到 ego + 最近车辆 + 最近行人）

    Returns:
        snapshot dict（可直接传给 extract_stkg_tensors）
    """
    # --- 1. 选出当前帧的 actor 行 ---
    frame_actors = actors_df[actors_df["frame_id"] == frame_id].copy()
    if len(frame_actors) == 0:
        raise ValueError(f"frame_id={frame_id} not found in actors_df")

    # 分级：ego 优先 → anomaly_target → 其余
    ego_rows = frame_actors[frame_actors["is_ego"] == 1]
    anom_rows = frame_actors[frame_actors["is_anomaly_target"] == 1]
    other_rows = frame_actors[(frame_actors["is_ego"] != 1) & (frame_actors["is_anomaly_target"] != 1)]

    # 按 distance-to-ego 排序取近的
    if len(ego_rows) > 0:
        ego_x = _f(ego_rows.iloc[0]["x"])
        ego_y = _f(ego_rows.iloc[0]["y"])
        def _dist(r):
            return math.sqrt((_f(r["x"]) - ego_x) ** 2 + (_f(r["y"]) - ego_y) ** 2)
        other_rows = other_rows.copy()
        other_rows["_dist"] = other_rows.apply(_dist, axis=1)
        other_rows = other_rows.sort_values("_dist")

    n_take = max_actors - len(ego_rows) - len(anom_rows)
    if n_take > 0 and len(other_rows) > n_take:
        other_rows = other_rows.head(n_take)

    selected = pd.concat([ego_rows, anom_rows, other_rows.drop(columns=[c for c in other_rows.columns if c.startswith("_")])]).drop_duplicates(subset=["actor_id"])

    selected = pd.concat([ego_rows, anom_rows, other_rows]).drop_duplicates(subset=["actor_id"])
    if len(selected) > max_actors:
        selected = selected.head(max_actors)

    # --- 2. 分类 vehicles / pedestrians ---
    vehicles = []
    pedestrians = []
    for _, row in selected.iterrows():
        atype = str(row.get("type", "")).lower()
        if atype == "vehicle" or atype == "car":
            vehicles.append(_row_to_vehicle(row))
        elif atype == "pedestrian" or atype == "walker":
            pedestrians.append(_row_to_pedestrian(row))
        else:
            # 未知类型默认当 vehicle 处理
            vehicles.append(_row_to_vehicle(row))

    all_nodes = vehicles + pedestrians

    # --- 3. 节点 ID 映射 ---
    node_ids = [str(n.get("entity_id", f"n{i}")) for i, n in enumerate(all_nodes)]
    id2row = {nid: i for i, nid in enumerate(node_ids)}

    # --- 4. y_anomaly 标签（从 actor row 的 is_anomaly 字段）---
    y_anomaly = [0] * len(all_nodes)
    for i, node in enumerate(all_nodes):
        if _b(node.get("is_anomaly")):
            y_anomaly[i] = 1

    # --- 5. 从 event_labels 生成 rule_out ---
    violations = _build_violations_from_events(
        frame_id, events, id2row, node_ids,
    )
    # 用 violations 补充 y_anomaly（event 触发但 actor 不在当前帧的情况）
    for vi in violations:
        src = vi.src_id
        if src in id2row:
            y_anomaly[id2row[src]] = 1

    # --- 6. weather ---
    label_row = labels_df[labels_df["frame_id"] == frame_id]
    if len(label_row) == 0:
        weather = {
            "fog_density": 0.0, "cloudiness": 0.0, "precipitation": 0.0,
            "wetness": 0.0, "sun_altitude_angle": 45.0, "wind_intensity": 0.0,
        }
        elapsed = 0.0
        delta = 0.05
        map_name = ""
        is_anomaly = 0
    else:
        lr = label_row.iloc[0]
        weather = {
            "fog_density":          _f(lr.get("weather_fog_density"), 0.0),
            "cloudiness":           _f(lr.get("weather_cloudiness"), 0.0),
            "precipitation":        _f(lr.get("weather_precipitation"), 0.0),
            "wetness":              _f(lr.get("weather_wetness"), 0.0),
            "sun_altitude_angle":   _f(lr.get("weather_sun_altitude"), 45.0),
            "wind_intensity":       _f(lr.get("weather_wetness"), 0.0),
        }
        elapsed = _f(lr.get("elapsed_seconds"), 0.0)
        delta = _f(lr.get("delta_seconds"), 0.05)
        map_name = str(lr.get("map_name", ""))
        is_anomaly = int(_f(lr.get("is_anomaly"), 0))

    # --- 7. scene_rels（K-NN fallback，与 exporter 相同）---
    (src_list, dst_list), type_list = _build_k_nn_edges(
        vehicles, pedestrians, all_nodes, id2row, K=5,
    )
    scene_rels = []
    for s, d, t in zip(src_list, dst_list, type_list):
        scene_rels.append({
            "src_id": node_ids[s],
            "dst_id": node_ids[d],
            "relation_type": f"rel_{t}",  # 占位；由 extract_stkg_tensors 的 SCENE_REL_TO_IDX 解析
            "frame_id": frame_id,
        })

    # --- 8. kappa_rss 指标（简化版：仅从 ego 和邻居距离估算）---
    # 全量 RSS 计算需要精确的 RSS 参数；此处给简化值：
    #   有 ego 的车辆：d_min_long 用纵向距离近似，speed_residual 用 speed - 50
    #   无 ego 或行人的车辆：全零
    kappa_rss = [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in all_nodes]
    ego_node_idx = None
    for i, n in enumerate(all_nodes):
        if n.get("is_ego"):
            ego_node_idx = i
            break
    if ego_node_idx is not None:
        ego = all_nodes[ego_node_idx]
        ex = _f(ego.get("location_x"), 0.0)
        ey = _f(ego.get("location_y"), 0.0)
        eh = _f(ego.get("heading_rad"), 0.0)
        ev = _f(ego.get("speed"), 0.0)
        for i, n in enumerate(all_nodes):
            if i == ego_node_idx:
                continue
            if n.get("entity_type") != "Vehicle":
                continue
            nx = _f(n.get("location_x"), 0.0)
            ny = _f(n.get("location_y"), 0.0)
            dx = nx - ex
            dy = ny - ey
            dist = math.sqrt(dx * dx + dy * dy)
            # 纵向距离（沿 ego heading）
            long_dist = dx * math.cos(eh) + dy * math.sin(eh)
            # d_min_long ≈ dist * 0.5（简化），d_long_residual = d_min_long - actual_long
            d_min_long = max(dist * 0.5, 1.0)
            d_long_res = max(0.0, d_min_long - abs(long_dist))
            # speed_residual = v - v_limit（v_limit=50 km/h ≈ 13.9 m/s）
            speed_res = max(0.0, _f(n.get("speed"), 0.0) - 13.9)
            # brake_residual = brake - brake_min（brake_min=3.0）
            brake_res = max(0.0, _f(n.get("control", {}).get("brake", 0.0), 0.0) - 3.0)
            kappa_rss[i] = [d_long_res, 0.0, 0.0, speed_res, brake_res]

    # --- 9. kappa_rule（从 violations 映射到 14 维规则向量）---
    rule_codes = ["R1", "R2", "R3", "R4", "R5", "R7", "R8", "R9",
                  "R10", "R11", "R13", "R16", "R17", "R18"]
    rule_idx = {c: i for i, c in enumerate(rule_codes)}
    kappa_rule = [[0.0] * 14 for _ in all_nodes]
    for vi in violations:
        src = str(vi.src_id)
        if src in id2row:
            row = id2row[src]
            code = vi.rule_code.split("a")[0] if vi.rule_code.endswith("a") else vi.rule_code
            if code in rule_idx:
                sev = max(kappa_rule[row][rule_idx[code]], _f(vi.severity))
                kappa_rule[row][rule_idx[code]] = sev

    # --- 10. delta_feat（首帧设零，后续帧由 _compute_deltas 填充）---
    if prev_snapshot is not None:
        delta_feat = _compute_deltas(prev_snapshot, {
            "node_ids": node_ids,
            "violations": violations,
        })
    else:
        delta_feat = [0.0, 0.0, 0.0, 0.0]

    # --- 11. 组装 snapshot ---
    snapshot = {
        "extracted": {
            "frame_id": int(frame_id),
            "elapsed_seconds": elapsed,
            "delta_seconds": delta,
            "map_name": map_name,
            "vehicles": vehicles,
            "pedestrians": pedestrians,
            "traffic_lights": [],        # CSV 无此数据
            "lanes": [],                  # CSV 无此数据
            "scene_rels": scene_rels,
            "weather": weather,
            "_node_ids": node_ids,        # 暴露给下游
            "_y_anomaly": y_anomaly,      # 节点级异常标签
            "_is_frame_anomaly": is_anomaly,  # 帧级标签
        },
        "rule_out": {
            "violations": violations,
        },
        "delta": {
            "delta_feat": delta_feat,
        },
    }
    return snapshot


def _build_violations_from_events(
    frame_id: int,
    events: List[Dict[str, Any]],
    id2row: Dict[str, int],
    node_ids: List[str],
) -> List[CSVViolation]:
    """从 event_labels.json 中的活跃事件生成 CSVViolation 列表。"""
    violations = []
    for ev in events:
        tf = _f(ev.get("trigger_frame"), 0)
        ef = _f(ev.get("end_frame"), tf)
        if tf <= frame_id <= ef:
            target_id = str(ev.get("target_actor_id", ""))
            rule_code = ANOMALY_TYPE_TO_RULE.get(ev.get("anomaly_type", ""), "R1")
            rule_name_map = {
                "R1": "PedestrianDanger", "R2": "RedLightRun",
                "R3": "SpeedExceed", "R4": "YieldFailure",
                "R5": "ReverseDriving", "R7": "JunctionNonYield",
                "R8": "LaneInvasion", "R10": "WrongWay",
                "R11": "UnsafeLaneChange", "R13": "RSSBraking",
                "R15": "Tailgating", "R16": "CutIn", "R17": "EmergencyYield",
                "R18": "RollingStop",
            }
            violations.append(CSVViolation(
                rule_code=rule_code,
                rule_name=rule_name_map.get(rule_code, rule_code),
                rule_layer="RSS" if rule_code.startswith("R1") else "STKG",
                src_id=target_id,
                dst_id=target_id,
                severity=_f(ev.get("intensity"), 0.5),
                predicate_str=f"{rule_name_map.get(rule_code, rule_code)}({target_id})",
                evidence_path=[],
            ))
    return violations


def _compute_deltas(
    prev_snapshot: Dict[str, Any],
    curr_snapshot: Dict[str, Any],
) -> List[float]:
    """计算两帧之间的 Δg_t 差分特征（4 维）：|ΔE|, |ΔA|, ||Δv||, |Δrules|"""
    # ΔE = 新增/删除的节点数
    prev_ids = set(prev_snapshot.get("extracted", {}).get("_node_ids", []))
    curr_ids = set(curr_snapshot.get("extracted", {}).get("_node_ids", []))
    delta_e = len(curr_ids - prev_ids) + len(prev_ids - curr_ids)
    delta_a = len(curr_ids & prev_ids)
    # ||Δv||：节点速度变化 L2 范数（简化：用 count 代理）
    delta_v = float(abs(len(curr_ids) - len(prev_ids)))
    # |Δrules|：新增/删除 violation 数
    delta_r = abs(len(curr_snapshot.get("rule_out", {}).get("violations", []))
                  - len(prev_snapshot.get("rule_out", {}).get("violations", [])))
    return [float(delta_e), float(delta_a), delta_v, float(delta_r)]


# ============================================================
# 批量转换
# ============================================================
def build_snapshots_from_csv(
    actors_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    events: List[Dict[str, Any]],
    frame_ids: Optional[List[int]] = None,
    max_actors: int = 30,
) -> List[Dict[str, Any]]:
    """
    批量转换：CSV → snapshot list。

    Args:
        actors_df:  frame_actors.csv DataFrame
        labels_df:  frame_labels.csv DataFrame
        events:     event_labels.json list
        frame_ids:  指定帧 ID 列表（None → 按 split 过滤）
        max_actors: 每帧最大节点数

    Returns:
        [snapshot, ...] 列表
    """
    if frame_ids is None:
        frame_ids = sorted(actors_df["frame_id"].unique().tolist())

    snapshots = []
    prev = None
    for fid in frame_ids:
        snap = build_snapshot_from_csv(
            frame_id=int(fid),
            actors_df=actors_df,
            labels_df=labels_df,
            events=events,
            prev_snapshot=prev,
            max_actors=max_actors,
        )
        snapshots.append(snap)
        prev = snap
    return snapshots
