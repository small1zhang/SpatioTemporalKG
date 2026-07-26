# -*- coding: utf-8 -*-
"""
STKG → PyTorch Geometric 数据导出器（§4.1 / §6.1 输入层）

将 PipelineOrchestrator.snapshot_store.get(frame_id) 的字典结构转换为
torch_geometric.data.Data 对象，作为 K-HSTGAN 的输入。

输入格式：
    snapshot = {
        "extracted": {
            "frame_id": int,
            "vehicles": [...], "pedestrians": [...],
            "traffic_lights": [...], "lanes": [...],
            "scene_rels": [...], "weather": {...}
        },
        "delta":     <DeltaGraph 对象或其 to_dict()>,
        "rule_out":  {"violations": [...], "responsibilities": [...]}
    }

输出格式 (Data)：
    x:                  [N, F]    节点特征张量（F=18 基础车辆/ped/TL 维度）
    edge_index:         [2, E]    场景层关系边索引
    edge_type:          [E]       场景层关系类型（15 类映射 0-14）
    behavior_edge_index:[2, Eb]
    behavior_edge_type: [Eb]      行为层关系类型（13 类映射 0-12）
    kappa_rss:          [N, 5]    RSS 残差向量
    kappa_rule:         [N, 14]   交规触发强度
    env_feat:           [12]      环境快照特征
    delta_feat:         [4]       Δg_t 四元组（|ΔE.added|, |ΔE.removed|, ||ΔA||_F, |ΔR.added|）
    y_anomaly:          [N]       节点级异常标签（0/1）
    y_scene:            [N]       场景层异常 3 类
    y_behavior:         [N]       行为层异常 7 类
    y_rule:             [N, 14]   规则触发 multi-label
    node_ids:           list[str] 节点 entity_id（按索引序）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from stk.ontology.types import SceneRelationType, BehaviorRelationType


# === 关系类型 → 整数索引的双向映射 ===
SCENE_REL_TO_IDX: Dict[str, int] = {r.value: i for i, r in enumerate(SceneRelationType)}
BEHAVIOR_REL_TO_IDX: Dict[str, int] = {r.value: i for i, r in enumerate(BehaviorRelationType)}
IDX_TO_SCENE_REL = {v: k for k, v in SCENE_REL_TO_IDX.items()}
IDX_TO_BEHAVIOR_REL = {v: k for k, v in BEHAVIOR_REL_TO_IDX.items()}

# === 节点特征维度映射（18 维基础特征）===
# 对 Vehicle：[x, y, z, vx, vy, speed, heading, brake, throttle, steer, vx_prev, vy_prev, lane_x, lane_y, lane_yaw, lane_w, lane_speed, is_emergency]
# 对 Pedestrian/TrafficLight：按字段补齐 → 18 维零向量
VEHICLE_FEATURE_KEYS: List[str] = [
    "location_x", "location_y", "location_z",
    "velocity_x", "velocity_y", "speed",
    "heading_rad", "brake", "throttle", "steer",
    "vx_prev", "vy_prev",
    "lane_x", "lane_y", "lane_yaw", "lane_width",
    "speed_limit",
    "is_emergency",
]

# === 环境特征 12 维（weather 快照 + 帧信息）===
ENV_FEATURE_KEYS: List[str] = [
    "fog_density", "cloudiness", "precipitation", "wetness",
    "sun_altitude_angle", "wind_intensity",
    # 衍生特征：
    "is_night",         # sun_altitude_angle < 0
    "is_rainy",         # precipitation > 0
    "is_foggy",         # fog_density > 30
    " Visibility_km",   # 估算能见度
    " Road_friction",   # 估算路面摩擦（wetness 反向）
    "elapsed_seconds",  # 帧时间
]

# === RSS 残差 5 维 ===
RSS_RESIDUAL_KEYS: List[str] = [
    "d_long_residual",   # d_min_long - d_long
    "d_lat_residual",    # d_min_lat - d_lat
    "ttc_residual",      # TTC - tau_safe
    "speed_residual",    # v - v_limit
    "brake_residual",    # brake - brake_min
]


def _to_float(v: Any, default: float = 0.0) -> float:
    """安全转 float，非数值返回 default。"""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bool_to_float(v: Any) -> float:
    return 1.0 if bool(v) else 0.0


def extract_vehicle_features(v: Dict[str, Any]) -> List[float]:
    """从一辆 vehicle dict 抽取 18 维特征向量。"""
    feats = []
    for key in VEHICLE_FEATURE_KEYS:
        if key == "vx_prev":
            # 速度分量近似（CARLA ticks 之间近似）：缺省 0
            feats.append(_to_float(v.get("velocity_x"), 0.0))
        elif key == "vy_prev":
            feats.append(_to_float(v.get("velocity_y"), 0.0))
        elif key == "lane_x":
            # 当前所在 lane 中心 x：缺省取自身 location_x
            lane = v.get("current_lane") or v.get("lane_info")
            if isinstance(lane, dict):
                feats.append(_to_float(lane.get("center_x"), _to_float(v.get("location_x"), 0.0)))
            else:
                feats.append(_to_float(v.get("location_x"), 0.0))
        elif key == "lane_y":
            lane = v.get("current_lane") or v.get("lane_info")
            if isinstance(lane, dict):
                feats.append(_to_float(lane.get("center_y"), _to_float(v.get("location_y"), 0.0)))
            else:
                feats.append(_to_float(v.get("location_y"), 0.0))
        elif key == "lane_yaw":
            feats.append(0.0)  # 占位，§4.6 实验阶段填充
        elif key == "lane_width":
            feats.append(_to_float(v.get("lane_width"), 3.5))
        elif key == "speed_limit":
            lane = v.get("current_lane") or v.get("lane_info")
            if isinstance(lane, dict):
                feats.append(_to_float(lane.get("speed_limit"), 13.89))  # 50 km/h 默认
            else:
                feats.append(13.89)
        elif key == "is_emergency":
            feats.append(_bool_to_float(v.get("is_emergency", False)))
        elif key == "velocity_x":
            # 若 dict 中提供则直接取，否则从 speed 与 heading 推算
            if "velocity_x" in v:
                feats.append(_to_float(v["velocity_x"]))
            else:
                speed = _to_float(v.get("speed"), 0.0)
                heading = _to_float(v.get("heading_rad"), 0.0)
                feats.append(speed * float(np.cos(heading)))
        elif key == "velocity_y":
            if "velocity_y" in v:
                feats.append(_to_float(v["velocity_y"]))
            else:
                speed = _to_float(v.get("speed"), 0.0)
                heading = _to_float(v.get("heading_rad"), 0.0)
                feats.append(speed * float(np.sin(heading)))
        else:
            feats.append(_to_float(v.get(key), 0.0))
    return feats


def extract_pedestrian_features(p: Dict[str, Any]) -> List[float]:
    """行人 18 维：[x, y, z, vx, vy, speed, heading, b, t, s, vx_prev, vy_prev, lane_x, lane_y, lane_yaw, lane_w, speed_limit, is_child]"""
    # 行人字段较稀疏：location/speed/action；其余填 0
    feats = [0.0] * len(VEHICLE_FEATURE_KEYS)
    feats[0] = _to_float(p.get("location_x"), 0.0)
    feats[1] = _to_float(p.get("location_y"), 0.0)
    feats[2] = _to_float(p.get("location_z"), 0.0)
    speed = _to_float(p.get("speed"), 0.0)
    feats[5] = speed
    # vx / vy 默认 0
    # is_child 作为第 18 维（labels 含 Child）
    labels = p.get("labels") or []
    if isinstance(labels, list) and any("Child" in str(l) for l in labels):
        feats[17] = 1.0
    return feats


def extract_traffic_light_features(tl: Dict[str, Any]) -> List[float]:
    """信号灯 18 维：[x, y, z, 0..0, state_one_hot(*3)]"""
    feats = [0.0] * len(VEHICLE_FEATURE_KEYS)
    # 信号灯字段含 position_x/position_y 或 location_x/location_y
    feats[0] = _to_float(tl.get("position_x", tl.get("location_x", 0.0)))
    feats[1] = _to_float(tl.get("position_y", tl.get("location_y", 0.0)))
    feats[2] = _to_float(tl.get("position_z", tl.get("location_z", 0.0)))
    # 状态独热编码塞到第 9-11 维（throttle/steer/lane_yaw 位置，反正都未用）
    state = str(tl.get("state", "Green")).lower()
    feats[7] = 1.0 if state == "red" else 0.0
    feats[8] = 1.0 if state == "yellow" else 0.0
    feats[14] = 1.0 if state == "green" else 0.0
    return feats


def extract_env_features(weather: Dict[str, Any], elapsed_seconds: float = 0.0) -> List[float]:
    """环境特征 12 维。"""
    sun_alt = _to_float(weather.get("sun_altitude_angle"), 0.0)
    precip = _to_float(weather.get("precipitation"), 0.0)
    fog = _to_float(weather.get("fog_density"), 0.0)
    wet = _to_float(weather.get("wetness"), 0.0)
    feats = [
        _to_float(weather.get("fog_density"), 0.0),
        _to_float(weather.get("cloudiness"), 0.0),
        _to_float(weather.get("precipitation"), 0.0),
        _to_float(weather.get("wetness"), 0.0),
        sun_alt,
        _to_float(weather.get("wind_intensity"), 0.0),
        1.0 if sun_alt < 0 else 0.0,                                    # is_night
        1.0 if precip > 0 else 0.0,                                    # is_rainy
        1.0 if fog > 30 else 0.0,                                      # is_foggy
        max(0.0, 1.0 - fog / 100.0),                                   # Visibility_km 估算
        max(0.0, 1.0 - wet / 100.0),                                   # Road_friction 估算
        float(elapsed_seconds),
    ]
    return feats


def compute_kappa_rss(vehicles: List[Dict], rss_params: Optional[Dict] = None,
                      ego_idx: int = 0) -> np.ndarray:
    """
    计算每个节点的 5 维 RSS 残差向量（§4.4 式 4.24）。
    对非 vehicle 节点返回全 0。

    kappa_rss(v) = [
        d_min_long - d_long,
        d_min_lat  - d_lat,
        TTC        - tau_safe,
        v          - v_limit,
        brake      - brake_min
    ]
    """
    n = len(vehicles)
    out = np.zeros((n, 5), dtype=np.float32)
    if n == 0:
        return out
    ego = vehicles[ego_idx] if ego_idx < n else vehicles[0]
    params = rss_params or {
        "rho": 0.3, "a_max_accel": 0.5, "a_min_brake_long": 3.0,
        "a_brake_long": 8.0, "mu": 0.5,
        "a_min_brake_lat": 1.5, "a_brake_lat": 3.0,
    }
    rho, a_max = params["rho"], params["a_max_accel"]
    a_min_long, a_brk_long = params["a_min_brake_long"], params["a_brake_long"]
    a_min_lat, a_brk_lat = params["a_min_brake_lat"], params["a_brake_lat"]

    ego_speed = _to_float(ego.get("speed"), 0.0)
    ego_x = _to_float(ego.get("location_x"), 0.0)
    ego_y = _to_float(ego.get("location_y"), 0.0)
    ego_brake = _to_float(ego.get("brake"), 0.0)
    ego_lane = ego.get("current_lane") or ego.get("lane_info") or {}
    ego_speed_limit = _to_float(ego_lane.get("speed_limit") if isinstance(ego_lane, dict) else None, 13.89)

    d_min_long = max(0.0, ego_speed * rho + 0.5 * a_max * rho ** 2
                     + (ego_speed + a_max * rho) ** 2 / (2 * a_min_long))

    for i, v in enumerate(vehicles):
        if i == ego_idx:
            # 自车的纵向/横向残差为 0（与自身无距离差）
            out[i, 3] = ego_speed - ego_speed_limit
            out[i, 4] = ego_brake - 0.3  # brake_min
            continue
        dx = _to_float(v.get("location_x"), 0.0) - ego_x
        dy = _to_float(v.get("location_y"), 0.0) - ego_y
        d_long = float(np.hypot(dx, dy))
        d_lat = abs(dx) * 0.3  # 横向距离近似
        v_speed = _to_float(v.get("speed"), 0.0)
        v_lat = 0.5  # 假设最大横向速度
        d_min_lat_i = max(0.0, 0.5 + v_lat ** 2 / (2 * a_min_lat) + rho * v_lat)
        # TTC 简化：相对速度差 / 距离
        rel_v = max(0.0, ego_speed - v_speed)
        ttc = d_long / max(rel_v, 0.1) if rel_v > 0.1 else 30.0
        # 速度限制
        lane = v.get("current_lane") or v.get("lane_info") or {}
        v_speed_limit = _to_float(lane.get("speed_limit") if isinstance(lane, dict) else None, 13.89)
        out[i, 0] = d_min_long - d_long
        out[i, 1] = d_min_lat_i - d_lat
        out[i, 2] = ttc - 2.5  # tau_safe = 2.5 s
        out[i, 3] = v_speed - v_speed_limit
        out[i, 4] = _to_float(v.get("brake"), 0.0) - 0.3
    return out


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    """安全获取属性：兼容 dict / pydantic model / dataclass。

    SafetyViolation pydantic 模型将 src_id/dst_id/severity 等字段
    存储在 `attrs` 字典中，因此 model_dump() 后 src_id 在顶层为 None。
    本函数先查顶层，再查 attrs 子字典，确保可访问这些字段。
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        val = obj.get(key)
        if val is not None:
            return val
        # 尝试 attrs 子字典
        attrs = obj.get("attrs", {})
        if isinstance(attrs, dict):
            return attrs.get(key, default)
        return default
    # pydantic v2 model：尝试 attribute / model_dump
    if hasattr(obj, "model_dump"):
        d = obj.model_dump()
        val = d.get(key)
        if val is not None:
            return val
        attrs = d.get("attrs", {})
        if isinstance(attrs, dict):
            return attrs.get(key, default)
        return default
    if hasattr(obj, "attrs"):
        attrs = getattr(obj, "attrs", {})
        if isinstance(attrs, dict) and key in attrs:
            return attrs[key]
    if hasattr(obj, "__dict__"):
        val = obj.__dict__.get(key)
        if val is not None:
            return val
        attrs = obj.__dict__.get("attrs", {})
        if isinstance(attrs, dict):
            return attrs.get(key, default)
    return getattr(obj, key, default)


def compute_kappa_rule(rule_out: Dict[str, Any], node_count: int,
                       node_ids: List[str]) -> torch.Tensor:
    """
    计算每个节点的 14 维交规触发强度向量（§4.4 式 4.26）。
    severity_i(v) ∈ [0, 1]
    """
    rule_codes = ["R1", "R2", "R3", "R4", "R5", "R7", "R8", "R9",
                 "R10", "R11", "R13", "R16", "R17", "R18"]
    rule_idx = {r: i for i, r in enumerate(rule_codes)}

    out = np.zeros((node_count, 14), dtype=np.float32)
    id2row = {nid: i for i, nid in enumerate(node_ids)}
    violations = rule_out.get("violations") or []
    for sv in violations:
        code = str(_attr(sv, "rule_code", "") or "")
        # 处理 R13a / R15a 等带后缀编码：归并到 R13
        base = code.split("a")[0] if code.endswith("a") else code
        if base not in rule_idx:
            continue
        col = rule_idx[base]
        severity = float(_attr(sv, "severity", 0.0) or 0.0)
        # 取 src_id 与 dst_id 的节点行
        for nid in (_attr(sv, "src_id"), _attr(sv, "dst_id")):
            nid = str(nid) if nid else None
            if nid and nid in id2row:
                row = id2row[nid]
                out[row, col] = max(out[row, col], min(1.0, severity))
    return torch.from_numpy(out)


def _build_node_id_list(snapshot: Dict[str, Any]) -> List[str]:
    """根据 extracted 中的实体列表构造稳定的节点 ID 字典序排列。"""
    ext = snapshot["extracted"]
    ids: List[str] = []
    for v in ext.get("vehicles", []) or []:
        if v.get("entity_id"):
            ids.append(str(v["entity_id"]))
    for p in ext.get("pedestrians", []) or []:
        if p.get("entity_id"):
            ids.append(str(p["entity_id"]))
    for tl in ext.get("traffic_lights", []) or []:
        if tl.get("entity_id"):
            ids.append(str(tl["entity_id"]))
    return sorted(ids)


def _build_node_features(snapshot: Dict[str, Any],
                          node_ids: List[str]) -> torch.Tensor:
    """构造 [N, 18] 节点特征矩阵。"""
    ext = snapshot["extracted"]
    by_id: Dict[str, Dict] = {}
    for v in ext.get("vehicles", []) or []:
        if v.get("entity_id"):
            by_id[str(v["entity_id"])] = ("vehicle", v)
    for p in ext.get("pedestrians", []) or []:
        if p.get("entity_id"):
            by_id[str(p["entity_id"])] = ("pedestrian", p)
    for tl in ext.get("traffic_lights", []) or []:
        if tl.get("entity_id"):
            by_id[str(tl["entity_id"])] = ("tl", tl)

    rows: List[List[float]] = []
    for nid in node_ids:
        if nid in by_id:
            kind, d = by_id[nid]
            if kind == "vehicle":
                rows.append(extract_vehicle_features(d))
            elif kind == "pedestrian":
                rows.append(extract_pedestrian_features(d))
            else:
                rows.append(extract_traffic_light_features(d))
        else:
            rows.append([0.0] * len(VEHICLE_FEATURE_KEYS))
    return torch.tensor(rows, dtype=torch.float32)


def _build_edge_index(snapshot: Dict[str, Any],
                       node_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造场景层 edge_index + edge_type + behavior_edge_index + behavior_edge_type。"""
    ext = snapshot["extracted"]
    id2row = {nid: i for i, nid in enumerate(node_ids)}
    src_list, dst_list, type_list = [], [], []
    bsrc_list, bdst_list, btype_list = [], [], []

    for rel in ext.get("scene_rels", []) or []:
        rtype = str(rel.get("relation_type") or rel.get("type") or "")
        if rtype not in SCENE_REL_TO_IDX:
            continue
        s = rel.get("src_id"); d = rel.get("dst_id")
        if s in id2row and d in id2row:
            src_list.append(id2row[s]); dst_list.append(id2row[d])
            type_list.append(SCENE_REL_TO_IDX[rtype])

    # 行为关系（可能在 extracted.behavior_rels 或顶层 behavior_rels）
    brels = ext.get("behavior_rels") or snapshot.get("behavior_rels") or []
    for rel in brels or []:
        rtype = str(rel.get("relation_type") or rel.get("type") or "")
        if rtype not in BEHAVIOR_REL_TO_IDX:
            continue
        s = rel.get("src_id"); d = rel.get("dst_id")
        if s in id2row and d in id2row:
            bsrc_list.append(id2row[s]); bdst_list.append(id2row[d])
            btype_list.append(BEHAVIOR_REL_TO_IDX[rtype])

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(type_list, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    if bsrc_list:
        b_edge_index = torch.tensor([bsrc_list, bdst_list], dtype=torch.long)
        b_edge_type = torch.tensor(btype_list, dtype=torch.long)
    else:
        b_edge_index = torch.empty((2, 0), dtype=torch.long)
        b_edge_type = torch.empty((0,), dtype=torch.long)

    return edge_index, edge_type, b_edge_index, b_edge_type


def _compute_delta_features(delta: Any) -> List[float]:
    """从 DeltaGraph 提取 4 维 Δg_t 特征。"""
    if delta is None:
        return [0.0, 0.0, 0.0, 0.0]
    if hasattr(delta, "delta_entities"):
        # DeltaGraph 对象
        de = delta.delta_entities
        dr = delta.delta_relations
        da = delta.delta_attrs
        added_e = len(de.added) if hasattr(de, "added") else 0
        removed_e = len(de.removed) if hasattr(de, "removed") else 0
        added_r = len(dr.added) if hasattr(dr, "added") else 0
        # ||ΔA||_F：所有 (old, new) 差的 Frobenius 范数
        norm_a = 0.0
        if isinstance(da, dict):
            for eid, attrs in da.items():
                if isinstance(attrs, dict):
                    for k, v in attrs.items():
                        if isinstance(v, (tuple, list)) and len(v) == 2:
                            try:
                                norm_a += (float(v[1]) - float(v[0])) ** 2
                            except (TypeError, ValueError):
                                pass
        return [float(added_e), float(removed_e), float(np.sqrt(norm_a)), float(added_r)]
    if isinstance(delta, dict):
        de = delta.get("delta_entities") or {}
        dr = delta.get("delta_relations") or {}
        da = delta.get("delta_attrs") or {}
        added_e = len(de.get("added", []))
        removed_e = len(de.get("removed", []))
        added_r = len(dr.get("added", []))
        norm_a = 0.0
        for eid, attrs in (da or {}).items():
            if isinstance(attrs, dict):
                for k, v in attrs.items():
                    if isinstance(v, (tuple, list)) and len(v) == 2:
                        try:
                            norm_a += (float(v[1]) - float(v[0])) ** 2
                        except (TypeError, ValueError):
                            pass
        return [float(added_e), float(removed_e), float(np.sqrt(norm_a)), float(added_r)]
    return [0.0, 0.0, 0.0, 0.0]


def extract_stkg_tensors(snapshot: Dict[str, Any],
                          ego_id: Optional[str] = None,
                          rss_params: Optional[Dict] = None) -> Data:
    """
    将单帧 snapshot 转换为 torch_geometric Data 对象。

    Args:
        snapshot: PipelineOrchestrator.snapshot_store.get(frame_id) 返回的字典
        ego_id:   可选 ego 车辆 entity_id（用于 RSS 残差计算时定位）
        rss_params: RSS 物理参数，None 则用默认值

    Returns:
        torch_geometric.data.Data
    """
    extracted = snapshot["extracted"]
    rule_out = snapshot.get("rule_out") or {}
    node_ids = _build_node_id_list(snapshot)
    id2row = {nid: i for i, nid in enumerate(node_ids)}

    # 节点特征
    x = _build_node_features(snapshot, node_ids)
    # 边
    edge_index, edge_type, b_edge_index, b_edge_type = _build_edge_index(snapshot, node_ids)

    # RSS 残差 —— 只为 vehicle 节点计算，ped/TL 为 0
    vehicles = extracted.get("vehicles", []) or []
    ego_idx = 0
    if ego_id is None:
        # 默认取第一辆车（与 EgoCentricFilter._pick_ego 一致）
        for i, v in enumerate(vehicles):
            if v.get("is_ego") or v.get("is_ego_vehicle"):
                ego_idx = i
                break
    else:
        ego_id_str = str(ego_id)
        for i, v in enumerate(vehicles):
            if str(v.get("entity_id")) == ego_id_str:
                ego_idx = i
                break

    # 注意：ego_idx 是 vehicles 列表中的索引，而我们使用全局节点顺序（字典序）
    # 需要把 kappa_rss 写到对应的全局行
    n = len(node_ids)
    kappa_rss_full = np.zeros((n, 5), dtype=np.float32)
    if vehicles:
        # 计算所有 vehicle 之间的 RSS（这里简化：仅 ego 对其他各算一次）
        rss_vehicles = compute_kappa_rss(vehicles, rss_params=rss_params, ego_idx=ego_idx)
        for i, v in enumerate(vehicles):
            nid = str(v.get("entity_id"))
            if nid in id2row:
                kappa_rss_full[id2row[nid], :] = rss_vehicles[i, :]
    kappa_rss = torch.from_numpy(kappa_rss_full)

    # 交规触发强度
    kappa_rule = compute_kappa_rule(rule_out, n, node_ids)

    # 环境特征
    weather = extracted.get("weather", {}) or {}
    env_feat = torch.tensor(extract_env_features(weather,
                                                  _to_float(extracted.get("elapsed_seconds"), 0.0)),
                            dtype=torch.float32)

    # Δg_t 四元组
    delta_feat = torch.tensor(_compute_delta_features(snapshot.get("delta")), dtype=torch.float32)

    # 标签（异常 / 场景 / 行为 / 规则）—— 训练时由 labeling module 填
    y_anomaly = torch.zeros(n, dtype=torch.long)
    y_scene = torch.zeros(n, dtype=torch.long)
    y_behavior = torch.zeros(n, dtype=torch.long)
    y_rule = torch.zeros((n, 14), dtype=torch.float32)
    # 用 rule_out 中的违规节点反推 y_anomaly = 1
    for sv in rule_out.get("violations", []) or []:
        for nid in (_attr(sv, "src_id"), _attr(sv, "dst_id")):
            nid = str(nid) if nid else None
            if nid and nid in id2row:
                y_anomaly[id2row[nid]] = 1

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        behavior_edge_index=b_edge_index,
        behavior_edge_type=b_edge_type,
        kappa_rss=kappa_rss,
        kappa_rule=kappa_rule,
        env_feat=env_feat,
        delta_feat=delta_feat,
        y_anomaly=y_anomaly,
        y_scene=y_scene,
        y_behavior=y_behavior,
        y_rule=y_rule,
    )
    data.node_ids = node_ids
    data.frame_id = int(extracted.get("frame_id", 0))
    data.ego_idx = int(ego_idx)
    return data


class STKGGraphDataset(torch.utils.data.Dataset):
    """
    多帧 snapshot 序列数据集。

    用法：
        orchestrator.run_scenario('S00', max_frames=6)
        snapshots = [orchestrator.snapshot_store.get(i) for i in range(6)]
        dataset = STKGGraphDataset(snapshots)
        data = dataset[0]
    """

    def __init__(self, snapshots: Sequence[Dict[str, Any]],
                 ego_id: Optional[str] = None,
                 rss_params: Optional[Dict] = None):
        super().__init__()
        self.snapshots = list(snapshots)
        self.ego_id = ego_id
        self.rss_params = rss_params
        self._cache: List[Optional[Data]] = [None] * len(self.snapshots)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> Data:
        if self._cache[idx] is None:
            self._cache[idx] = extract_stkg_tensors(
                self.snapshots[idx], ego_id=self.ego_id, rss_params=self.rss_params)
        return self._cache[idx]

    def Collate(self, indices: Optional[List[int]] = None) -> List[Data]:
        """按时间排序返回多帧列表（供 DHLSTM 时序窗口使用）。"""
        if indices is None:
            indices = list(range(len(self)))
        return [self[i] for i in indices]
