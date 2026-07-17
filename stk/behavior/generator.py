"""
BehaviorRelationGenerator — 行为层主生成器 (v3 sec 7.3.2 / sec 3.4)

本模块是行为层的核心驱动:
  1. 从一帧的场景层数据 (实体+场景关系) 通过 detectors.run_all_detectors
     生成所有"行为关系候选" (rel_type, src, dst, condition_met, extra_attrs)
  2. 通过 RelationDebouncer 做防抖处理 — 决定何时创建/删除该行为关系
  3. 当某行为关系首次通过防抖时:
       - 创建对应的行为关系边 (relations.<type>(...))
       - 创建对应的节点 (ManeuverNode / InteractionEvent)
       - 创建对应的跨层桥接边 (manifest.py 的 link_*_to_scene)
  4. 当某行为关系被防抖判定终止时:
       - 调用节点的 close() 函数补完 frame_end / duration_frames / state

BehaviorRelationGenerator.generate() 输出符合 v3 sec 7.3.2 流水线契约:
  返回 dict = {
    "maneuvers":  [ManeuverNode, ...]   新增/更新的个体行为节点
    "interactions": [InteractionEvent, ...]  新增/更新的交互事件节点
    "behavior_rels": [BaseRelation, ...]  新增/更新的行为关系边
    "cross_layer_rels": [BaseRelation, ...]  新增的跨层桥接边 (manifestsAs/actor/src/dst)
  }
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from stk.ontology.relation import BaseRelation
from stk.ontology.types import BehaviorRelationType
from stk.behavior.nodes import (
    ManeuverNode, InteractionEvent,
    make_maneuver_id, make_interaction_id,
)
from stk.behavior.relations import (
    standing_still, changing_lane, following, approaching, yielding_to,
    overtaking, wrong_side_meeting, opposite_direction, same_direction,
    blocked_view, approaching_pedestrian,
    approaching_intersection, crossing,
)
from stk.behavior.debouncer import RelationDebouncer
from stk.behavior.detectors import run_all_detectors
from stk.behavior.manifest import link_maneuver_to_scene, link_interaction_to_scene


# 个体行为 -> ManeuverNode
INDIVIDUAL_RELS = {"standing_still", "changing_lane"}
# 交互/演化行为 -> InteractionEvent
INTERACTION_RELS = {
    "following", "approaching", "yielding_to", "overtaking",
    "wrong_side_meeting", "opposite_direction", "same_direction",
    "blocked_view", "approaching_pedestrian",
    "approaching_intersection", "crossing",
}


# 关系类型到工厂函数的映射
_RELATION_FACTORIES = {
    "standing_still": standing_still,
    "changing_lane": changing_lane,
    "following": following,
    "approaching": approaching,
    "yielding_to": yielding_to,
    "overtaking": overtaking,
    "wrong_side_meeting": wrong_side_meeting,
    "opposite_direction": opposite_direction,
    "same_direction": same_direction,
    "blocked_view": blocked_view,
    "approaching_pedestrian": approaching_pedestrian,
    "approaching_intersection": approaching_intersection,
    "crossing": crossing,
}


class BehaviorRelationGenerator:
    """行为层主生成器，逐帧驱动.

    跨帧状态:
      _debouncer     : RelationDebouncer 实例 (防抖计数)
      _active_nodes  : {(src, dst, rel_type): ManeuverNode|InteractionEvent}  活跃节点
      _active_rels   : {(src, dst, rel_type): BaseRelation}  活跃的行为关系边
    """

    def __init__(self,
                 debouncer: Optional[RelationDebouncer] = None,
                 thresholds: Optional[Dict[str, int]] = None):
        self._debouncer = debouncer or RelationDebouncer(thresholds=thresholds)
        self._active_nodes: Dict[Tuple[str, str, str], Any] = {}
        self._active_rels: Dict[Tuple[str, str, str], BaseRelation] = {}

        # 跨帧累积输出
        self._nodes_emitted: List[Any] = []
        self._relations_emitted: List[BaseRelation] = []
        self._cross_layer_emitted: List[BaseRelation] = []

    # ---------------- 公共 API ----------------

    def generate(
        self,
        frame_id: int,
        vehicles: List[Dict[str, Any]],
        pedestrians: List[Dict[str, Any]] = None,
        traffic_lights: List[Dict[str, Any]] = None,
        junctions: List[Dict[str, Any]] = None,
        crosswalks: List[Dict[str, Any]] = None,
        scene_relations: List[Dict[str, Any]] = None,
    ) -> Dict[str, List[Any]]:
        """对单帧运行行为层生成.

        Args: 见 detectors.run_all_detectors
        Returns:
            {
              "maneuvers":         本帧新增/关闭的 ManeuverNode 列表,
              "interactions":      本帧新增/关闭的 InteractionEvent 列表,
              "behavior_rels":     本帧新增/关闭的行为关系边,
              "cross_layer_rels":  本帧新增的 manifestsAs/actor/src/dst 桥接边,
            }
        """
        pedestrians = pedestrians or []
        traffic_lights = traffic_lights or []
        junctions = junctions or []
        crosswalks = crosswalks or []
        scene_relations = scene_relations or []

        # 1. 运行所有 detect_*
        candidates = run_all_detectors(
            vehicles=vehicles, pedestrians=pedestrians,
            traffic_lights=traffic_lights, junctions=junctions,
            crosswalks=crosswalks, scene_relations=scene_relations,
        )

        new_maneuvers: List[ManeuverNode] = []
        new_interactions: List[InteractionEvent] = []
        new_behavior_rels: List[BaseRelation] = []
        new_cross_layer: List[BaseRelation] = []

        # 2. 对每个候选通过防抖处理
        for rel_type, items in candidates.items():
            for src_id, dst_id, condition_met, extra_attrs in items:
                key = (src_id, dst_id, rel_type)
                action, debounce_extra = self._debouncer.update(
                    relation_type=rel_type, key=key,
                    condition_met=condition_met, frame_id=frame_id,
                )

                if action == "create":
                    # 创建行为关系边
                    rel = self._create_relation(rel_type, key, frame_id, extra_attrs)
                    if rel is not None:
                        new_behavior_rels.append(rel)
                        self._active_rels[key] = rel

                        # 创建节点
                        node, edges = self._create_node_and_links(rel_type, key, frame_id, extra_attrs)
                        if node is not None:
                            self._active_nodes[key] = node
                            new_cross_layer.extend(edges)
                            if isinstance(node, ManeuverNode):
                                new_maneuvers.append(node)
                            else:
                                new_interactions.append(node)

                            # 累积输出
                            self._nodes_emitted.append(node)
                            self._relations_emitted.append(rel)
                            self._cross_layer_emitted.extend(edges)

                elif action == "delete":
                    # 关闭节点
                    if key in self._active_nodes:
                        node = self._active_nodes.pop(key)
                        node.close(frame_id)
                        if isinstance(node, ManeuverNode):
                            new_maneuvers.append(node)
                        else:
                            new_interactions.append(node)

                    # 关闭行为关系
                    if key in self._active_rels:
                        rel = self._active_rels.pop(key)
                        rel.valid_to = frame_id
                        new_behavior_rels.append(rel)

                elif action == "keep":
                    # 已存在的节点 - 不创建新的, 但可以通过派生属性更新
                    pass

        return {
            "maneuvers": new_maneuvers,
            "interactions": new_interactions,
            "behavior_rels": new_behavior_rels,
            "cross_layer_rels": new_cross_layer,
        }

    def all_active(self) -> Dict[str, List[Any]]:
        """返回当前所有活跃的节点与关系 (跨帧累积)."""
        return {
            "maneuvers": [n for n in self._active_nodes.values() if isinstance(n, ManeuverNode)],
            "interactions": [n for n in self._active_nodes.values() if isinstance(n, InteractionEvent)],
            "behavior_rels": list(self._active_rels.values()),
        }

    def stats(self) -> Dict[str, int]:
        """生成器统计信息."""
        return {
            "n_active_maneuvers": sum(1 for n in self._active_nodes.values()
                                       if isinstance(n, ManeuverNode)),
            "n_active_interactions": sum(1 for n in self._active_nodes.values()
                                          if isinstance(n, InteractionEvent)),
            "n_active_relations": len(self._active_rels),
            "n_total_emitted_nodes": len(self._nodes_emitted),
            "n_total_emitted_rels": len(self._relations_emitted),
            "n_total_cross_layer": len(self._cross_layer_emitted),
        }

    def reset(self) -> None:
        """清空所有跨帧状态."""
        self._debouncer = RelationDebouncer()
        self._active_nodes.clear()
        self._active_rels.clear()
        self._nodes_emitted.clear()
        self._relations_emitted.clear()
        self._cross_layer_emitted.clear()

    # ---------------- 内部方法 ----------------

    def _create_relation(self, rel_type: str, key: Tuple[str, str, str],
                          frame_id: int, extra_attrs: Dict[str, Any]) -> Optional[BaseRelation]:
        """通过工厂函数创建一条行为关系边."""
        src_id, dst_id, _ = key
        factory = _RELATION_FACTORIES.get(rel_type)
        if factory is None:
            return None

        # 不同关系类型用不同工厂签名, 通过 kwargs 适配
        try:
            if rel_type == "standing_still":
                return factory(src_id, frame_id, **extra_attrs)
            elif rel_type == "changing_lane":
                return factory(src_id, dst_id, frame_id, **extra_attrs)
            elif rel_type == "following":
                return factory(src_id, dst_id, frame_id, **extra_attrs)
            elif rel_type == "approaching":
                return factory(src_id, dst_id, frame_id, **extra_attrs)
            elif rel_type == "yielding_to":
                return factory(src_id, dst_id, frame_id, **extra_attrs)
            elif rel_type == "overtaking":
                return factory(src_id, dst_id, frame_id, **extra_attrs)
            elif rel_type == "wrong_side_meeting":
                return factory(src_id, dst_id, frame_id, **extra_attrs)
            elif rel_type == "opposite_direction":
                return factory(src_id, dst_id, frame_id, **extra_attrs)
            elif rel_type == "same_direction":
                return factory(src_id, dst_id, frame_id, **extra_attrs)
            elif rel_type == "blocked_view":
                return factory(src_id, dst_id, frame_id, **extra_attrs)
            elif rel_type == "approaching_pedestrian":
                return factory(src_id, dst_id, frame_id, **extra_attrs)
            elif rel_type == "approaching_intersection":
                return factory(src_id, dst_id, frame_id, **extra_attrs)
            elif rel_type == "crossing":
                return factory(src_id, dst_id, frame_id, **extra_attrs)
        except TypeError:
            return None
        return None

    def _create_node_and_links(self, rel_type: str, key: Tuple[str, str, str],
                                  frame_id: int, extra_attrs: Dict[str, Any]) -> Tuple[Optional[Any], List[BaseRelation]]:
        """为新增的行为关系创建对应的节点 + 桥接边.

        Returns:
            (node, cross_layer_edges)
        """
        src_id, dst_id, _ = key

        if rel_type in INDIVIDUAL_RELS:
            # 个体行为 -> ManeuverNode
            maneuver_id = make_maneuver_id(src_id, frame_id)
            node = ManeuverNode(
                entity_id=maneuver_id,
                maneuver_type=rel_type,
                actor_id=src_id,
                frame_start=frame_id,
            )
            cross_edges = link_maneuver_to_scene(maneuver=node,
                                                  vehicle_entity_id=src_id)
            return node, cross_edges

        elif rel_type in INTERACTION_RELS:
            # 交互/演化行为 -> InteractionEvent
            interaction_id = make_interaction_id(src_id, dst_id, rel_type, frame_id)
            node = InteractionEvent(
                entity_id=interaction_id,
                interaction_type=rel_type,
                src_id=src_id,
                dst_id=dst_id,
                frame_start=frame_id,
                derived_attrs=extra_attrs,
            )
            cross_edges = link_interaction_to_scene(interaction=node)
            return node, cross_edges

        return None, []
