# -*- coding: utf-8 -*-
"""Entity/Relation → Cypher 参数 (v3 §6.2.4)."""
from __future__ import annotations
import json
from typing import Any, Dict, List
from stk.ontology.entity import BaseEntity
from stk.ontology.relation import BaseRelation


def entity_to_cypher_params(entity: BaseEntity) -> Dict[str, Any]:
    params = entity.to_neo4j_dict()
    for k, v in list(params.items()):
        if isinstance(v, float):
            params[k] = round(v, 6)
        elif isinstance(v, (list, dict)):
            params[k] = json.dumps(v, ensure_ascii=False)
    params["_entity_type"] = entity.entity_type
    return params


def relation_to_cypher_params(rel: BaseRelation) -> Dict[str, Any]:
    params = rel.to_neo4j_dict()
    for k, v in list(params.items()):
        if isinstance(v, float):
            params[k] = round(v, 6)
        elif isinstance(v, (list, dict)):
            params[k] = json.dumps(v, ensure_ascii=False)
    return params


def entity_merge_cypher(label: str) -> str:
    return f"MERGE (n:{label} {{entity_id: $entity_id}}) SET n += $params"


def relation_merge_cypher(rel_type: str) -> str:
    return (
        f"MATCH (src {{entity_id: $src_id}})\n"
        f"MATCH (dst {{entity_id: $dst_id}})\n"
        f"MERGE (src)-[r:{rel_type} {{frame_id: $frame_id}}]->(dst)\n"
        f"SET r += $params"
    )

# ---------- 全图序列化 (供 phase5 落盘) ----------

def serialize_graph(frame_snapshot, with_relations: bool = True) -> Dict[str, Any]:
    """把单帧 (或帧列表) 的场景快照序列化为 {nodes, edges} JSON.

    输入可以是:
      - 单帧 dict: 含 vehicles / pedestrians / traffic_lights / lanes / weather / scene_rels
      - 帧列表: 上面那种 dict 的 list, 会做跨帧去重

    输出:
      {
        "nodes": [{"id","type","frame","attrs"},
                  ...],
        "edges": [{"src_id","dst_id","type","frame","attrs"}, ...]
      }
    """
    if isinstance(frame_snapshot, list):
        frames = frame_snapshot
    else:
        frames = [frame_snapshot]

    nodes = {}
    edges = {}

    for snap in frames:
        fid = snap.get("frame_id", 0)

        # --- Vehicles ---
        for v in snap.get("vehicles", []):
            eid = v.get("entity_id") or v.get("id") or None
            if eid is None:
                continue
            eid = str(eid)
            if eid not in nodes:
                nodes[eid] = {
                    "id": eid,
                    "type": v.get("entity_type", "Vehicle"),
                    "first_frame": fid,
                    "last_frame": fid,
                    "attrs": _flatten_attrs(v),
                }
            else:
                nodes[eid]["last_frame"] = fid
                # 把动态字段更新一下 (位置/速度)
                _merge_attrs(nodes[eid]["attrs"], v)

        # --- Pedestrians ---
        for p in snap.get("pedestrians", []):
            eid = p.get("entity_id") or p.get("id") or None
            if eid is None:
                continue
            eid = str(eid)
            if eid not in nodes:
                nodes[eid] = {
                    "id": eid,
                    "type": p.get("entity_type", "Pedestrian"),
                    "first_frame": fid,
                    "last_frame": fid,
                    "attrs": _flatten_attrs(p),
                }
            else:
                nodes[eid]["last_frame"] = fid
                _merge_attrs(nodes[eid]["attrs"], p)

        # --- Traffic lights ---
        for tl in snap.get("traffic_lights", []):
            eid = tl.get("entity_id") or tl.get("id") or None
            if eid is None:
                continue
            eid = str(eid)
            tl_type = tl.get("entity_type", "TrafficLight")
            # 红绿灯的 state 是动态属性, 需要带 frame; 这里 entity 只放首个/stateless
            if eid not in nodes:
                nodes[eid] = {
                    "id": eid,
                    "type": tl_type,
                    "first_frame": fid,
                    "last_frame": fid,
                    "attrs": _flatten_attrs({k: v2 for k, v2 in tl.items()
                                            if k not in ("state", "frame_id")}),
                }
            else:
                nodes[eid]["last_frame"] = fid

        # --- Lanes / roads ---
        for ln in snap.get("lanes", []):
            eid = ln.get("entity_id")
            if eid is None:
                continue
            if eid not in nodes:
                nodes[eid] = {
                    "id": eid,
                    "type": ln.get("entity_type", "RoadElement"),
                    "first_frame": fid,
                    "last_frame": fid,
                    "attrs": _flatten_attrs(ln),
                }
            else:
                nodes[eid]["last_frame"] = fid

        # --- Environment node (1 per frame) ---
        w = snap.get("weather")
        if isinstance(w, dict):
            env_id = f"env_frame_{fid}"
            nodes[env_id] = {
                "id": env_id,
                "type": "EnvironmentSnapshot",
                "first_frame": fid,
                "last_frame": fid,
                "attrs": _flatten_attrs(w),
            }

        # --- Scene relations ---
        if with_relations:
            for rel in snap.get("scene_rels", []):
                _add_edge(edges, rel, fid)

            # 隐式关系: 车辆在车道 (vehicle.lane_id -> in_lane)
            for v in snap.get("vehicles", []):
                lane_id = v.get("lane_id") or v.get("current_lane_id")
                veh_id = v.get("entity_id") or v.get("id")
                if lane_id and veh_id:
                    _add_edge(edges, {
                        "src_id": str(veh_id),
                        "dst_id": str(lane_id) if isinstance(lane_id, str) else f"road_0_lane_{lane_id}",
                        "relation_type": "in_lane",
                        "frame_id": fid,
                    }, fid)

            # 隐式关系: 帧根场景 containsVehicle / containsPedestrian / containsTrafficLight
            scene_id = f"scenario_frame_{fid}"
            for v in snap.get("vehicles", []):
                vid = v.get("entity_id") or v.get("id")
                if vid:
                    _add_edge(edges, {
                        "src_id": scene_id, "dst_id": str(vid),
                        "relation_type": "containsVehicle", "frame_id": fid,
                    }, fid)
            for p in snap.get("pedestrians", []):
                pid = p.get("entity_id") or p.get("id")
                if pid:
                    _add_edge(edges, {
                        "src_id": scene_id, "dst_id": str(pid),
                        "relation_type": "containsPedestrian", "frame_id": fid,
                    }, fid)

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def _flatten_attrs(d: Dict[str, Any]) -> Dict[str, Any]:
    """把不可 JSON 化的属性转成 str/浮点."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = _flatten_attrs(v)
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = str(v)
    return out


def _merge_attrs(dest: Dict[str, Any], src: Dict[str, Any]) -> None:
    """把 src 的最新动态字段合并到 dest."""
    for k in ("location_x", "location_y", "location_z",
              "velocity_x", "velocity_y", "velocity_z",
              "speed", "speed_kmh", "heading_rad", "state", "brake", "throttle"):
        if k in src:
            dest[k] = src[k]


def _add_edge(edges: Dict[str, Any], rel: Dict[str, Any], fid: int) -> None:
    """给 edge 做去重 keying,同 src/dst/type 把 frame 范围合并."""
    src = rel.get("src_id")
    dst = rel.get("dst_id")
    rt = rel.get("relation_type") or rel.get("rel_type") or rel.get("type")
    if not (src and dst and rt):
        return
    src = str(src); dst = str(dst)
    key = (src, dst, rt)
    if key not in edges:
        edges[key] = {
            "src_id": src, "dst_id": dst, "type": rt,
            "first_frame": fid, "last_frame": fid,
            "frame_id": fid,
            "attrs": _flatten_attrs({k: v2 for k, v2 in rel.items()
                                     if k not in ("src_id", "dst_id", "relation_type", "rel_type",
                                                  "type", "frame_id")}),
        }
    else:
        edges[key]["last_frame"] = fid
