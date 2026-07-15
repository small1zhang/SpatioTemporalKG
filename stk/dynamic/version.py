# -*- coding: utf-8 -*-
"""属性版本管理 (v3 §5.4.2, §5.5)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional


class AttrVersion:
    """单个属性版本."""
    def __init__(self, attr_name: str, value: Any, valid_from_frame: int,
                 valid_to_frame: Optional[int] = None):
        self.attr_name = attr_name
        self.value = value
        self.valid_from_frame = valid_from_frame
        self.valid_to_frame = valid_to_frame

    def to_dict(self) -> dict:
        return {"attr_name": self.attr_name, "value": self.value,
                "valid_from_frame": self.valid_from_frame,
                "valid_to_frame": self.valid_to_frame}


class VersionManager:
    """属性版本管理."""

    def __init__(self, threshold: float = 0.01):
        self._versions: Dict[str, Dict[str, List[AttrVersion]]] = {}
        self._threshold = threshold

    def record_change(self, eid: str, attr: str, new_val: Any,
                      frame_id: int) -> Optional[AttrVersion]:
        if eid not in self._versions:
            self._versions[eid] = {}
        vmap = self._versions[eid]
        if attr not in vmap:
            av = AttrVersion(attr_name=attr, value=new_val, valid_from_frame=frame_id)
            vmap[attr] = [av]
            return av
        last = vmap[attr][-1]
        if isinstance(last.value, (int, float)) and isinstance(new_val, (int, float)):
            if abs(float(new_val) - float(last.value)) < self._threshold:
                return None
        if last.value == new_val:
            return None
        last.valid_to_frame = frame_id
        new_av = AttrVersion(attr_name=attr, value=new_val, valid_from_frame=frame_id)
        vmap[attr].append(new_av)
        return new_av

    def close_entity(self, eid: str, frame_id: int):
        for attr, vers in self._versions.get(eid, {}).items():
            for v in vers:
                if v.valid_to_frame is None:
                    v.valid_to_frame = frame_id

    def get_current(self, eid: str, attr: str) -> Any:
        vers = self._versions.get(eid, {}).get(attr, [])
        return vers[-1].value if vers else None

    def get_history(self, eid: str, attr: str) -> List[AttrVersion]:
        return list(self._versions.get(eid, {}).get(attr, []))

    def reset(self):
        self._versions.clear()