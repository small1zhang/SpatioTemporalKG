"""
行为层节点+边双轨关联机制 (v3 sec 3.5 + sec 3.6)

实现 4 类跨层桥接关系 (CrossLayerRelationType):
  manifestsAs  - 行为关系边通过 manifestsAs 与对应的 Maneuver/Interaction 节点关联 (v3 sec 3.5.1)
  actor        - ManeuverNode 关联到场景层 VehicleEntity (v3 sec 3.6.1)
  src          - InteractionEvent 关联到场景层源实体 (v3 sec 3.6.1)
  dst          - InteractionEvent 关联到场景层目标实体 (v3 sec 3.6.1)

输出格式: BaseRelation 实例 (供后续存储层导入 Neo4j)
"""
from __future__ import annotations
from typing import List, Optional

from stk.ontology.relation import BaseRelation
from stk.ontology.types import CrossLayerRelationType, BehaviorRelationType
from stk.behavior.nodes import ManeuverNode, InteractionEvent


def manifestsAs_edge(
    behavior_node_id: str,
    behavior_relation_src: str,
    behavior_relation_dst: str,
    behavior_relation_type: str,
    frame_id: int,
    valid_from: int,
    valid_to: Optional[int] = None,
) -> BaseRelation:
    """ManeuverNode/InteractionEvent -[manifestsAs {relation_type}]-> (VehicleEntity -[behavior_rel]-> target)

    v3 sec 3.5.1 节点+边关联机制.
    示例 (Interaction):
        (int_veh123_veh124_following_2048)
        -[manifestsAs {relation_type: "following"}]->
        (VehicleEntity veh_123)-[following]->(VehicleEntity veh_124)
    源: behavior 节点 (ManeuverNode / InteractionEvent)
    目标: 行为关系边本身 (用 src-dst 对表示)
    边属性: relation_type = 行为关系类型
    """
    target_id = behavior_relation_src + "__" + behavior_relation_dst
    return BaseRelation(
        src_id=behavior_node_id,
        dst_id=target_id,
        relation_type=CrossLayerRelationType.MANIFESTS_AS.value,
        frame_id=frame_id,
        valid_from=valid_from,
        valid_to=valid_to,
        attrs={
            "relation_type": behavior_relation_type,
            "behavior_relation_src": behavior_relation_src,
            "behavior_relation_dst": behavior_relation_dst,
            "frame_id": frame_id,
        },
    )


def actor_edge(
    maneuver_node_id: str,
    vehicle_entity_id: str,
    frame_id: int,
    valid_from: int,
    valid_to: Optional[int] = None,
) -> BaseRelation:
    """ManeuverNode -[actor]-> VehicleEntity

    v3 sec 3.6.1.
    """
    return BaseRelation(
        src_id=maneuver_node_id,
        dst_id=vehicle_entity_id,
        relation_type=CrossLayerRelationType.ACTOR.value,
        frame_id=frame_id,
        valid_from=valid_from,
        valid_to=valid_to,
        attrs={"frame_id": frame_id},
    )


def src_edge(
    interaction_node_id: str,
    src_entity_id: str,
    frame_id: int,
    valid_from: int,
    valid_to: Optional[int] = None,
) -> BaseRelation:
    """InteractionEvent -[src]-> VehicleEntity / PedestrianEntity (v3 sec 3.6.1)"""
    return BaseRelation(
        src_id=interaction_node_id,
        dst_id=src_entity_id,
        relation_type=CrossLayerRelationType.SRC.value,
        frame_id=frame_id,
        valid_from=valid_from,
        valid_to=valid_to,
        attrs={"frame_id": frame_id},
    )


def dst_edge(
    interaction_node_id: str,
    dst_entity_id: str,
    frame_id: int,
    valid_from: int,
    valid_to: Optional[int] = None,
) -> BaseRelation:
    """InteractionEvent -[dst]-> VehicleEntity / PedestrianEntity / TrafficLightEntity / RoadElementEntity (v3 sec 3.6.1)"""
    return BaseRelation(
        src_id=interaction_node_id,
        dst_id=dst_entity_id,
        relation_type=CrossLayerRelationType.DST.value,
        frame_id=frame_id,
        valid_from=valid_from,
        valid_to=valid_to,
        attrs={"frame_id": frame_id},
    )


def link_maneuver_to_scene(maneuver: ManeuverNode,
                            vehicle_entity_id: str) -> List[BaseRelation]:
    """为 ManeuverNode 生成 actor 边。

    Args:
        maneuver: ManeuverNode 实例
        vehicle_entity_id: 场景层 VehicleEntity ID (与 maneuver.actor_id 应一致)
    Returns:
        [manifestsAs 边, actor 边]
    """
    edges: List[BaseRelation] = []
    frame_id = maneuver.frame_start
    valid_from = maneuver.frame_start
    valid_to = maneuver.frame_end

    # manifestsAs 边: ManeuverNode -> (VehicleEntity -[maneuver_type]-> VehicleEntity)
    edges.append(manifestsAs_edge(
        behavior_node_id=maneuver.entity_id,
        behavior_relation_src=maneuver.actor_id,
        behavior_relation_dst=maneuver.actor_id,
        behavior_relation_type=maneuver.maneuver_type,
        frame_id=frame_id, valid_from=valid_from, valid_to=valid_to,
    ))
    # actor 边
    edges.append(actor_edge(
        maneuver_node_id=maneuver.entity_id,
        vehicle_entity_id=vehicle_entity_id,
        frame_id=frame_id, valid_from=valid_from, valid_to=valid_to,
    ))
    return edges


def link_interaction_to_scene(interaction: InteractionEvent) -> List[BaseRelation]:
    """为 InteractionEvent 生成 manifestsAs + src + dst 三条桥接边.

    Args:
        interaction: InteractionEvent 实例

    Returns:
        [manifestsAs 边, src 边, dst 边]
    """
    edges: List[BaseRelation] = []
    frame_id = interaction.frame_start
    valid_from = interaction.frame_start
    valid_to = interaction.frame_end

    edges.append(manifestsAs_edge(
        behavior_node_id=interaction.entity_id,
        behavior_relation_src=interaction.src_id,
        behavior_relation_dst=interaction.dst_id,
        behavior_relation_type=interaction.interaction_type,
        frame_id=frame_id, valid_from=valid_from, valid_to=valid_to,
    ))
    edges.append(src_edge(
        interaction_node_id=interaction.entity_id,
        src_entity_id=interaction.src_id,
        frame_id=frame_id, valid_from=valid_from, valid_to=valid_to,
    ))
    edges.append(dst_edge(
        interaction_node_id=interaction.entity_id,
        dst_entity_id=interaction.dst_id,
        frame_id=frame_id, valid_from=valid_from, valid_to=valid_to,
    ))
    return edges
