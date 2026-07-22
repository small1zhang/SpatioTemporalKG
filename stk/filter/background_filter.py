# -*- coding: utf-8 -*-
"""BackgroundFilter: 静态背景外移 (阶段 3 §4.6.3).

策略:
- 排除 lane 节点 (entity_type=="RoadElement" 或 type=="RoadElement")
- 排除 lane 间关系: lane_connects / adjacent_lane / on_road (lanes 间结构边)
- 排除 in_lane (vehicle→lane) 边, 因为 vehicle.attrs.lane_id 已平铺

开关:
- exclude_lanes=true (默认)
- exclude_road_elements=false (默认关, 预留未来用)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from stk.config import EgoCentricConfig


# lane 间结构边与 in_lane 边一律 drop (语义平铺到 Vehicle.attrs.lane_id)
_DROP_EDGE_TYPES = {
    "lane_connects",    # lane-lane 结构边
    "adjacent_lane",    # lane-lane 相邻边
    "in_lane",          # vehicle-lane 隶属边
    "on_road",          # vehicle-road 隶属边
}


class BackgroundFilter:
    """静态背景外移.

    用法::

        bg = BackgroundFilter(ego_cfg)
        if bg.should_drop_entity(node): continue   # 跳过 lane 节点
        if bg.should_drop_edge(edge): continue     # 跳过 lane 相关边
    """

    def __init__(self, ego_cfg: Optional[EgoCentricConfig] = None):
        self._cfg = ego_cfg or EgoCentricConfig.default()

    def should_drop_entity(self, entity: Dict[str, Any]) -> bool:
        """节点是否应 drop.

        Args:
            entity: 节点 dict, 含 type 或 entity_type 字段.
                    (serialize_graph 生成的节点用 'type',
                     scenario_library 直接用 'entity_type')

        Returns:
            True 表示该节点不进 KG.
        """
        # RoadElement 实体类型 (lane)
        etype = (
            entity.get("entity_type")
            or entity.get("type")
            or ""
        )
        if etype == "RoadElement":
            return self._cfg.exclude_lanes
        # exclude_road_elements 预留扩展: 未来区分 building / static mesh
        return False

    def should_drop_edge(self, edge: Dict[str, Any]) -> bool:
        """边是否应 drop.

        Args:
            edge: 边 dict, 含 type 或 relation_type.

        Returns:
            True 表示该边不进 KG.
        """
        rtype = edge.get("type") or edge.get("relation_type") or ""
        if rtype in _DROP_EDGE_TYPES:
            return self._cfg.exclude_lanes
        # ego 参与的 in_lane (vehicle→lane) 边保留? — 不, lane 节点被 drop,
        # 所以 referencing 它的边也必须 drop; 已由上面的 _DROP_EDGE_TYPES 涵盖.
        # 桥接边 (lane 不参与), 不会受影响.
        return False

    # ── 批量过滤: 一次应用所有节点/边 ──

    def filter_nodes(self, nodes: list) -> list:
        return [n for n in nodes if not self.should_drop_entity(n)]

    def filter_edges(self, edges: list) -> list:
        return [e for e in edges if not self.should_drop_edge(e)]

    def stats(self, nodes: list, edges: list) -> dict:
        """返回 drop 统计用于监控."""
        n_dropped_nodes = sum(1 for n in nodes if self.should_drop_entity(n))
        n_dropped_edges = sum(1 for e in edges if self.should_drop_edge(e))
        return {
            "n_nodes_in": len(nodes),
            "n_nodes_dropped": n_dropped_nodes,
            "n_edges_in": len(edges),
            "n_edges_dropped": n_dropped_edges,
            "exclude_lanes": self._cfg.exclude_lanes,
        }
