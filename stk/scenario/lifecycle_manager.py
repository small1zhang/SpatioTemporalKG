"""
生命周期管理器 (v3 §2.5)

管理动态实体（Vehicle/Pedestrian）进入/离开场景，
以及静态实体（TrafficLight/RoadElement）仅更新属性。

使用 ontology.lifecycle.NodeLifecycle 跟踪每个实体的状态。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from stk.ontology.lifecycle import NodeLifecycle, NodeLifecycleStatus


class LifecycleManager:
    """生命周期管理器。

    维护场景中所有实体的 NodeLifecycle 实例，
    每帧 step() 时更新各实体的状态。

    动态实体 (v3 §2.5.1):
      进入场景 → activate(frame) → 属性更新 → 离开场景 → deactivate(frame)
    静态实体 (v3 §2.5.2):
      地图加载时创建 → 仅更新属性 → 仿真完结
    帧根/环境 (v3 §2.5.3):
      每帧重建
    """

    def __init__(self):
        self._lifecycles: Dict[str, NodeLifecycle] = {}

    def get(self, entity_id: str) -> Optional[NodeLifecycle]:
        """获取实体ID对应的生命周期管理器。"""
        return self._lifecycles.get(entity_id)

    def all_active_ids(self) -> List[str]:
        """获取当前 ACTIVE 状态的实体 ID 列表。"""
        return [eid for eid, lc in self._lifecycles.items()
                if lc.status == NodeLifecycleStatus.ACTIVE]

    def step(self, current_ids: List[str], frame_id: int,
             entity_type_map: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """步进一帧：自动处理实体的激活/失活。

        Args:
            current_ids: 本帧存活的实体 ID 列表
            frame_id: 当前帧号
            entity_type_map: 可选的 {entity_id: entity_type} 映射，用于记录

        Returns:
            状态变更字典: {entity_id: "activated"|"deactivated"|"stable"|"created"}
        """
        entity_type_map = entity_type_map or {}
        changes: Dict[str, str] = {}

        prev_ids = set(self._lifecycles.keys())
        curr_ids = set(current_ids)

        # 新实体：创建 + 激活
        for eid in curr_ids - prev_ids:
            etype = entity_type_map.get(eid, "")
            lc = NodeLifecycle(eid, etype)
            lc.activate(frame_id, reason="actor_entered_scene")
            self._lifecycles[eid] = lc
            changes[eid] = "activated"

        # 持续实体：更新
        for eid in curr_ids & prev_ids:
            self._lifecycles[eid].update(frame_id)
            changes[eid] = "stable"

        # 已离开实体：失活
        for eid in prev_ids - curr_ids:
            if eid in self._lifecycles and self._lifecycles[eid].status == NodeLifecycleStatus.ACTIVE:
                self._lifecycles[eid].deactivate(frame_id, reason="actor_left_scene")
                changes[eid] = "deactivated"

        return changes

    def clear(self):
        """清除所有生命周期记录（仿真重启时使用）。"""
        self._lifecycles.clear()

    def to_dict(self) -> Dict[str, dict]:
        """导出所有生命周期的状态摘要。"""
        return {eid: lc.to_dict() for eid, lc in self._lifecycles.items()}
