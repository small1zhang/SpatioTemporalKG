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




def serialize_graph(frame_snapshot, with_relations: bool = True,
                    maneuvers: List = None,
                    interactions: List = None,
                    behavior_rels: List = None,
                    rule_out: Dict = None,
                    merge_violations: bool = True) -> Dict[str, Any]:
    """把单帧 (或帧列表) 的场景快照序列化为 {nodes, edges} JSON.

    输入可以是:
      - 单帧 dict: 含 vehicles / pedestrians / traffic_lights / lanes / weather / scene_rels
      - 帧列表: 上面那种 dict 的 list, 会做跨帧去重

    可选参数:
      maneuvers: 行为层 ManueverNode 列表 (每帧一个)
      interactions: 行为层 InteractionEvent 列表 (每帧一个)
      rule_out: dict, 含 violations / responsibilities (每帧一个)

    输出:
      {"nodes": [{"id","type","frame","attrs"}, ...],
       "edges": [{"src_id","dst_id","type","frame","attrs"}, ...]}

    merge_violations=True 时:
      同 (rule_code, src_id, dst_id) 的多次触发合并为单个 SafetyViolation,
      ID 改为 sv_<rule_code>_<src>_<dst>, attrs 含 fires_n_frames/first_frame/
      last_frame/fired_frames 列表。ResponsibilityAssignment 同步合并。
    """
    if isinstance(frame_snapshot, list):
        frames = frame_snapshot
    else:
        frames = [frame_snapshot]

    # 如果传了分帧行为/规则数据, 按帧 id 索引
    man_map = {}
    int_map = {}
    vio_map = {}
    resp_map = {}
    beh_rel_map = {}
    cross_rel_map = {}
    if maneuvers is not None:
        for m in maneuvers:
            f = m.get("frame_id") if isinstance(m, dict) else getattr(m, "frame_id", 0)
            man_map.setdefault(f, []).append(m)
    if interactions is not None:
        for i in interactions:
            f = i.get("frame_id") if isinstance(i, dict) else getattr(i, "frame_id", 0)
            int_map.setdefault(f, []).append(i)
    if behavior_rels is not None:
        for br in behavior_rels:
            f = br.get("frame_id", 0) if isinstance(br, dict) else getattr(br, "frame_id", 0)
            beh_rel_map.setdefault(f, []).append(br)
    if rule_out is not None:
        for ro in rule_out:
            f = ro.get("frame_id") if isinstance(ro, dict) else getattr(ro, "frame_id", 0)
            vio = ro.get("violations") if isinstance(ro, dict) else getattr(ro, "violations", [])
            resp = ro.get("responsibilities") if isinstance(ro, dict) else getattr(ro, "responsibilities", [])
            vio_map.setdefault(f, []).extend(vio)
            resp_map.setdefault(f, []).extend(resp)

    def _node_key(eid, ntype):
        return (str(eid), ntype)

    nodes = {}
    edges = {}

    sv_buffer: Dict[int, List] = {}
    ra_buffer: Dict[int, List] = {}
    for snap in frames:
        fid = snap.get("frame_id", 0)

        # --- Vehicles ---
        for v in snap.get("vehicles", []):
            eid = str(v.get("entity_id") or v.get("id", ""))
            if not eid:
                continue
            if eid not in nodes:
                nodes[eid] = {"id": eid, "type": v.get("entity_type", "Vehicle"),
                              "first_frame": fid, "last_frame": fid,
                              "attrs": _flatten_attrs(v)}
            else:
                nodes[eid]["last_frame"] = fid
                _merge_attrs(nodes[eid]["attrs"], v)

        # --- Pedestrians ---
        for p in snap.get("pedestrians", []):
            eid = str(p.get("entity_id") or p.get("id", ""))
            if not eid:
                continue
            if eid not in nodes:
                nodes[eid] = {"id": eid, "type": p.get("entity_type", "Pedestrian"),
                              "first_frame": fid, "last_frame": fid,
                              "attrs": _flatten_attrs(p)}
            else:
                nodes[eid]["last_frame"] = fid
                _merge_attrs(nodes[eid]["attrs"], p)

        # --- Traffic lights ---
        for tl in snap.get("traffic_lights", []):
            eid = str(tl.get("entity_id") or tl.get("id", ""))
            if not eid:
                continue
            if eid not in nodes:
                nodes[eid] = {"id": eid,
                              "type": tl.get("entity_type", "TrafficLight"),
                              "first_frame": fid, "last_frame": fid,
                              "attrs": _flatten_attrs({k: v2 for k, v2 in tl.items()
                                                       if k not in ("state")})}
            else:
                nodes[eid]["last_frame"] = fid
                if "state" in tl:
                    nodes[eid]["attrs"]["state"] = tl["state"]

        # --- Lanes ---
        for ln in snap.get("lanes", []):
            eid = ln.get("entity_id")
            if eid is None:
                continue
            if eid not in nodes:
                nodes[eid] = {"id": eid, "type": ln.get("entity_type", "RoadElement"),
                              "first_frame": fid, "last_frame": fid,
                              "attrs": _flatten_attrs(ln)}
            else:
                nodes[eid]["last_frame"] = fid

        # --- Environment node ---
        w = snap.get("weather")
        if isinstance(w, dict):
            env_id = f"env_frame_{fid}"
            nodes[env_id] = {"id": env_id, "type": "EnvironmentSnapshot",
                             "first_frame": fid, "last_frame": fid,
                             "attrs": _flatten_attrs(w)}

        # --- Maneuver nodes ---
        for m in man_map.get(fid, []):
            eid = getattr(m, "entity_id", m.get("entity_id", "")) if isinstance(m, dict) else m.entity_id
            if not eid:
                continue
            eid = str(eid)
            nodes[eid] = {"id": eid, "type": "Maneuver",
                          "first_frame": fid, "last_frame": fid,
                          "attrs": _flatten_attrs(m.to_dict() if hasattr(m, "to_dict") else m)}

        # --- Interaction nodes ---
        for it in int_map.get(fid, []):
            eid = getattr(it, "entity_id", it.get("entity_id", "")) if isinstance(it, dict) else it.entity_id
            if not eid:
                continue
            eid = str(eid)
            nodes[eid] = {"id": eid, "type": "InteractionEvent",
                          "first_frame": fid, "last_frame": fid,
                          "attrs": _flatten_attrs(it.to_dict() if hasattr(it, "to_dict") else it)}
            # 添加隐式 actor/src/dst 边
            src = str(getattr(it, "src_id", it.get("src_id", ""))) if isinstance(it, dict) else str(it.src_id)
            dst = str(getattr(it, "dst_id", it.get("dst_id", ""))) if isinstance(it, dict) else str(it.dst_id)
            if src:
                _add_edge(edges, {"src_id": eid, "dst_id": src, "relation_type": "actor", "frame_id": fid}, fid)
            if dst:
                _add_edge(edges, {"src_id": eid, "dst_id": dst, "relation_type": "dst", "frame_id": fid}, fid)

        # --- SafetyViolation nodes ---
        if merge_violations:
            # 后处理模式: 暂存 list, 在所有帧处理完后做按 (rule,src,dst) 合并
            sv_buffer.setdefault(fid, []).extend(vio_map.get(fid, []))
            ra_buffer.setdefault(fid, []).extend(resp_map.get(fid, []))
        else:
            for sv in vio_map.get(fid, []):
                eid = getattr(sv, "entity_id", sv.get("entity_id", "")) if isinstance(sv, dict) else sv.entity_id
                if not eid:
                    continue
                eid = str(eid)
                nodes[eid] = {"id": eid, "type": "SafetyViolation",
                              "first_frame": fid, "last_frame": fid,
                              "attrs": _flatten_attrs(sv.to_dict() if hasattr(sv, "to_dict") else sv)}
                dst = str(getattr(sv, "dst_id", sv.get("dst_id", ""))) if isinstance(sv, dict) else str(sv.dst_id)
                if dst:
                    _add_edge(edges, {"src_id": eid, "dst_id": dst,
                                      "relation_type": "violates", "frame_id": fid}, fid)

            for ra in resp_map.get(fid, []):
                eid = getattr(ra, "entity_id", ra.get("entity_id", "")) if isinstance(ra, dict) else ra.entity_id
                if not eid:
                    continue
                eid = str(eid)
                nodes[eid] = {"id": eid, "type": "ResponsibilityAssignment",
                              "first_frame": fid, "last_frame": fid,
                              "attrs": _flatten_attrs(ra.to_dict() if hasattr(ra, "to_dict") else ra)}
                actor = str(getattr(ra, "responsible_actor_id", ra.get("responsible_actor_id", ""))) if isinstance(ra, dict) else str(ra.responsible_actor_id)
                if actor:
                    _add_edge(edges, {"src_id": eid, "dst_id": actor,
                                      "relation_type": "responsibleFor", "frame_id": fid}, fid)

        # --- Behavior relations ---
        for br in beh_rel_map.get(fid, []):
            if isinstance(br, dict):
                _add_edge(edges, br, fid)
            else:
                _add_edge(edges, {
                    "src_id": getattr(br, "src_entity_id", getattr(br, "src_id", "")),
                    "dst_id": getattr(br, "dst_entity_id", getattr(br, "dst_id", "")),
                    "relation_type": getattr(br, "relation_type", ""),
                    "frame_id": getattr(br, "frame_id", fid),
                }, fid)

        # --- Scene relations ---
        if with_relations:
            for rel in snap.get("scene_rels", []):
                _add_edge(edges, rel, fid)

            # 隐式关系: 车辆在车道
            for v in snap.get("vehicles", []):
                lane_id = v.get("lane_id") or v.get("current_lane_id")
                veh_id = v.get("entity_id") or v.get("id")
                if lane_id and veh_id:
                    _add_edge(edges, {
                        "src_id": str(veh_id), "dst_id": str(lane_id) if isinstance(lane_id, str) else f"road_0_lane_{lane_id}",
                        "relation_type": "in_lane", "frame_id": fid,
                    }, fid)

            # containsVehicle / containsPedestrian
            scene_id = f"scenario_frame_{fid}"
            for v in snap.get("vehicles", []):
                vid = v.get("entity_id") or v.get("id")
                if vid:
                    _add_edge(edges, {"src_id": scene_id, "dst_id": str(vid), "relation_type": "containsVehicle", "frame_id": fid}, fid)
            for p in snap.get("pedestrians", []):
                pid = p.get("entity_id") or p.get("id")
                if pid:
                    _add_edge(edges, {"src_id": scene_id, "dst_id": str(pid), "relation_type": "containsPedestrian", "frame_id": fid}, fid)

    if merge_violations and sv_buffer:
        _merge_violations_into_nodes(nodes, edges, sv_buffer, ra_buffer)

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def _flatten_attrs(d):
    """把不可 JSON 化的属性转成 str/浮点."""
    out = {}
    # 支持 pydantic BaseModel
    if not isinstance(d, dict):
        try:
            d = d.to_dict() if hasattr(d, "to_dict") else d.__dict__
        except Exception:
            d = str(d)
    if isinstance(d, str):
        return {"_repr": d}
    for k, v in d.items() if isinstance(d, dict) else []:
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = _flatten_attrs(v)
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            try:
                out[k] = str(v)
            except Exception:
                out[k] = repr(v)
    return out


def _merge_attrs(dest, src):
    """把 src 的最新动态字段合并到 dest."""
    for k in ("location_x", "location_y", "location_z",
              "velocity_x", "velocity_y", "velocity_z",
              "speed", "speed_kmh", "heading_rad", "state",
              "brake", "throttle", "steer"):
        if k in src if isinstance(src, dict) else hasattr(src, k):
            val = src[k] if isinstance(src, dict) else getattr(src, k)
            dest[k] = val




def _merge_violations_into_nodes(nodes, edges, sv_buffer, ra_buffer):
    """把按帧切分的 SafetyViolation/ResponsibilityAssignment 合并成跨帧实例.

    合并键: (rule_code, src_id, dst_id)
      -> 新 ID: sv_<rule_code>_<src_id>_<dst_id>
      -> attrs: rule_code, rule_name, rule_layer, src_id, dst_id,
                severity_max, severity_avg, fired_count, fired_frames,
                first_frame, last_frame, predicate_str
      -> 边:   sv -> dst (violates, attrs.fired_frames = [...])

    同步合并 ResponsibilityAssignment: 按 (sv_merged_id, responsible_actor_id)
      -> 新 ID: resp_<sv_merged_id>_<actor>
      -> attrs: reasons_set, first_frame, last_frame, fired_frames
      -> 边:   resp -> actor (responsibleFor)
    """
    import collections

    # 1) 按合并键 group 所有 SV
    sv_groups = collections.defaultdict(list)
    sv_frames_map = collections.defaultdict(list)
    sv_first_frame = {}
    sv_last_frame = {}

    for fid, sv_list in sv_buffer.items():
        for sv in sv_list:
            rule_code = getattr(sv, "rule_code") if not isinstance(sv, dict) else sv.get("rule_code", "")
            src_id = str(getattr(sv, "src_id", "") if not isinstance(sv, dict) else sv.get("src_id", ""))
            dst_id = str(getattr(sv, "dst_id", "") if not isinstance(sv, dict) else sv.get("dst_id", ""))
            key = (rule_code, src_id, dst_id)
            sv_groups[key].append(sv)
            sv_frames_map[key].append(fid)
            sv_first_frame[key] = min(sv_first_frame.get(key, fid), fid)
            sv_last_frame[key] = max(sv_last_frame.get(key, fid), fid)

    # 2) 生成合并的 SV 节点 + 边
    for (rule_code, src_id, dst_id), svs in sv_groups.items():
        if not rule_code:
            # 没有 rule_code 的 fallback 用原 ID,不合并
            for sv in svs:
                eid_eid = str(getattr(sv, "entity_id", sv.get("entity_id", "")) if isinstance(sv, dict) else sv.entity_id)
                nodes[eid_eid] = {"id": eid_eid, "type": "SafetyViolation",
                                   "first_frame": sv_first_frame[(rule_code, src_id, dst_id)],
                                   "last_frame": sv_last_frame[(rule_code, src_id, dst_id)],
                                   "attrs": _flatten_attrs(sv.to_dict() if hasattr(sv, "to_dict") else sv)}
            continue

        merged_id = f"sv_{rule_code}_{src_id}_{dst_id}"
        first_sv = svs[0]
        attrs = {
            "sv_id": merged_id,
            "rule_code": rule_code,
            "rule_name": getattr(first_sv, "rule_name", first_sv.get("rule_name", "") if isinstance(first_sv, dict) else ""),
            "rule_layer": getattr(first_sv, "rule_layer", first_sv.get("rule_layer", "") if isinstance(first_sv, dict) else ""),
            "src_id": src_id,
            "dst_id": dst_id,
            "severity_max": max(getattr(sv, "severity", sv.get("severity", 0) if isinstance(sv, dict) else 0) for sv in svs),
            "severity_avg": sum(getattr(sv, "severity", sv.get("severity", 0) if isinstance(sv, dict) else 0) for sv in svs) / len(svs),
            "fired_count": len(svs),
            "fired_frames": sorted(sv_frames_map[(rule_code, src_id, dst_id)]),
            "first_frame": sv_first_frame[(rule_code, src_id, dst_id)],
            "last_frame": sv_last_frame[(rule_code, src_id, dst_id)],
            "predicate_str": getattr(first_sv, "predicate_str", first_sv.get("predicate_str", "") if isinstance(first_sv, dict) else ""),
        }
        nodes[merged_id] = {"id": merged_id, "type": "SafetyViolation",
                            "first_frame": attrs["first_frame"],
                            "last_frame": attrs["last_frame"],
                            "attrs": attrs}
        if dst_id:
            edges[(merged_id, dst_id, "violates")] = {
                "src_id": merged_id, "dst_id": dst_id, "type": "violates",
                "first_frame": attrs["first_frame"], "last_frame": attrs["last_frame"],
                "frame_id": attrs["first_frame"], "attrs": {"fired_frames": attrs["fired_frames"]},
            }

    # 3) 同步合并 ResponsibilityAssignment: 按 (sv_merged_id, actor)
    # 建立 original_sv_id -> merged_sv_id 映射
    sv_id_map = {}
    for (rule_code, src_id, dst_id), svs in sv_groups.items():
        if not rule_code:
            continue
        merged = f"sv_{rule_code}_{src_id}_{dst_id}"
        for sv in svs:
            orig_id = str(getattr(sv, "entity_id", sv.get("entity_id", "") if isinstance(sv, dict) else ""))
            if orig_id:
                sv_id_map[orig_id] = merged

    ra_groups = collections.defaultdict(list)
    ra_frames_map = collections.defaultdict(list)

    for fid, ra_list in ra_buffer.items():
        for ra in ra_list:
            sv_id = str(getattr(ra, "sv_id", ra.get("sv_id", "")) if isinstance(ra, dict) else ra.sv_id)
            actor = str(getattr(ra, "responsible_actor_id", ra.get("responsible_actor_id", "")) if isinstance(ra, dict) else ra.responsible_actor_id)
            merged_sv_id = sv_id_map.get(sv_id)
            if merged_sv_id is None:
                continue
            key = (merged_sv_id, actor)
            ra_groups[key].append(ra)
            ra_frames_map[key].append(fid)

    for (merged_sv_id, actor), ras in ra_groups.items():
        merged_resp_id = f"resp_{merged_sv_id}_{actor}"
        reasons = set()
        for ra in ras:
            r = getattr(ra, "reason", ra.get("reason", "") if isinstance(ra, dict) else "")
            if r:
                reasons.add(str(r))
        first_frame = min(ra_frames_map[(merged_sv_id, actor)])
        last_frame = max(ra_frames_map[(merged_sv_id, actor)])
        attrs = {
            "resp_id": merged_resp_id,
            "sv_id": merged_sv_id,
            "responsible_actor_id": actor,
            "reasons": sorted(reasons),
            "fired_count": len(ras),
            "fired_frames": sorted(ra_frames_map[(merged_sv_id, actor)]),
            "first_frame": first_frame,
            "last_frame": last_frame,
        }
        nodes[merged_resp_id] = {"id": merged_resp_id, "type": "ResponsibilityAssignment",
                                  "first_frame": first_frame, "last_frame": last_frame,
                                  "attrs": attrs}
        if actor:
            edges[(merged_resp_id, actor, "responsibleFor")] = {
                "src_id": merged_resp_id, "dst_id": actor, "type": "responsibleFor",
                "first_frame": first_frame, "last_frame": last_frame,
                "frame_id": first_frame, "attrs": {"fired_frames": attrs["fired_frames"]},
            }


def _add_edge(edges, rel, fid):
    """给 edge 做去重 keying, 同 src/dst/type 把 frame 范围合并."""
    if isinstance(rel, dict):
        src = str(rel.get("src_id", ""))
        dst = str(rel.get("dst_id", ""))
        rt = rel.get("relation_type") or rel.get("rel_type") or rel.get("type")
        extra = {k: v for k, v in rel.items() if k not in ("src_id", "dst_id", "relation_type", "rel_type", "type", "frame_id")}
    else:
        src = str(getattr(rel, "src_id", getattr(rel, "src_entity_id", "")))
        dst = str(getattr(rel, "dst_id", getattr(rel, "dst_entity_id", "")))
        rt = getattr(rel, "relation_type", getattr(rel, "rel_type", ""))
        extra = {}
    if not (src and dst and rt):
        # 检查 behavior_rels / cross_layer_rels (它们有 src_entity_id,dst_entity_id)
        if isinstance(rel, dict):
            src = str(rel.get("src_entity_id", ""))
            dst = str(rel.get("dst_entity_id", ""))
            rt = rel.get("relation_type") or rel.get("rel_type") or rel.get("type")
        if not (src and dst and rt):
            return
    key = (src, dst, rt)
    if key not in edges:
        edges[key] = {
            "src_id": src, "dst_id": dst, "type": rt,
            "first_frame": fid, "last_frame": fid,
            "frame_id": fid, "attrs": {},
        }
    else:
        edges[key]["last_frame"] = fid
