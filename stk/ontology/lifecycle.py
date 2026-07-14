"""
节点生命周期状态机 (v3 §1.10, §2.5)
状态: CREATED → ACTIVE → STALE → INACTIVE
属性版本链: (attr, value, valid_from, valid_to)
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeLifecycleStatus(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    STALE = "stale"
    INACTIVE = "inactive"


class LifecycleTransition:
    """生命周期状态转换记录。"""
    def __init__(self, from_status: NodeLifecycleStatus, to_status: NodeLifecycleStatus,
                 frame_id: int, reason: str = ""):
        self.from_status = from_status
        self.to_status = to_status
        self.frame_id = frame_id
        self.reason = reason


class NodeLifecycle:
    def __init__(self, entity_id: str, entity_type: str = ""):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self._status: NodeLifecycleStatus = NodeLifecycleStatus.CREATED
        self._transitions: List[LifecycleTransition] = []
        self._frame_start: Optional[int] = None
        self._frame_end: Optional[int] = None
        self._versions: List[Dict[str, Any]] = []

    @property
    def status(self) -> NodeLifecycleStatus:
        return self._status

    @property
    def frame_start(self) -> Optional[int]:
        return self._frame_start

    @property
    def frame_end(self) -> Optional[int]:
        return self._frame_end

    @property
    def transitions(self) -> List[LifecycleTransition]:
        return list(self._transitions)

    @property
    def version_chain(self) -> List[Dict[str, Any]]:
        return list(self._versions)

    def update(self, frame_id: int):
        """每帧更新，保持 ACTIVE 状态。"""
        if self._status == NodeLifecycleStatus.CREATED:
            self.activate(frame_id)

    def activate(self, frame_id: int, reason: str = ""):
        if self._status != NodeLifecycleStatus.CREATED:
            return
        self._transitions.append(LifecycleTransition(self._status, NodeLifecycleStatus.ACTIVE, frame_id, reason))
        self._status = NodeLifecycleStatus.ACTIVE
        self._frame_start = frame_id

    def deactivate(self, frame_id: int, reason: str = ""):
        if self._status != NodeLifecycleStatus.ACTIVE:
            return
        self._transitions.append(LifecycleTransition(self._status, NodeLifecycleStatus.STALE, frame_id, reason))
        self._status = NodeLifecycleStatus.STALE
        self._frame_end = frame_id

    def add_version(self, attr: str, value: Any, valid_from: int, valid_to: Optional[int] = None):
        self._versions.append({"attr": attr, "value": value, "valid_from": valid_from, "valid_to": valid_to})

    def get_version_at(self, attr: str, frame_id: int) -> Optional[Any]:
        for v in reversed(self._versions):
            if v["attr"] != attr:
                continue
            if v["valid_from"] <= frame_id and (v["valid_to"] is None or frame_id <= v["valid_to"]):
                return v["value"]
        return None

    def is_active_at(self, frame_id: int) -> bool:
        if self._status != NodeLifecycleStatus.ACTIVE:
            return False
        if self._frame_start is not None and frame_id < self._frame_start:
            return False
        if self._frame_end is not None and frame_id > self._frame_end:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {"entity_id": self.entity_id, "status": self._status.value,
                "frame_start": self._frame_start, "frame_end": self._frame_end,
                "n_transitions": len(self._transitions), "n_versions": len(self._versions)}

