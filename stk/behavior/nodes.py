"""
行为层节点定义 (v3 sec 3.2)

定义行为层 2 类核心节点：
  - ManeuverNode (Label: Maneuver) — 单实体持续行为状态
  - InteractionEvent (Label: Interaction) — 多实体交互事件

行为节点继承自 ontology.entity.BaseEntity，entity_type 置为 MANEUVER / INTERACTION_EVENT。
所有时间维度属性通过 attrs 字典存储。
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from stk.ontology.entity import BaseEntity
from stk.ontology.types import EntityType


MANEUVER_TYPES = (
    "standing_still", "changing_lane", "accelerating",
    "decelerating", "cruising", "stopping",
)


class ManeuverNode(BaseEntity):
    def __init__(self, entity_id, maneuver_type, actor_id, frame_start,
                 frame_end=None, state="active", severity=0.0,
                 derived_attrs=None, related_rule=None, confidence=1.0,
                 valid_from=None, valid_to=None):
        if maneuver_type not in MANEUVER_TYPES:
            raise ValueError("Unknown maneuver_type: " + maneuver_type)
        if state not in ("active", "ended"):
            raise ValueError("state must be active/ended, got: " + state)
        if frame_end is None:
            duration = -1
        else:
            if frame_end < frame_start:
                raise ValueError("frame_end < frame_start")
            duration = frame_end - frame_start + 1
        attrs = {
            "maneuver_type": maneuver_type, "actor_id": actor_id,
            "frame_start": frame_start, "frame_end": frame_end,
            "duration_frames": duration, "state": state,
            "severity": severity, "derived_attrs": derived_attrs or {},
            "related_rule": related_rule,
        }
        super().__init__(
            entity_id=entity_id, entity_type=EntityType.MANEUVER,
            valid_from=valid_from if valid_from is not None else frame_start,
            valid_to=valid_to if valid_to is not None else frame_end,
            attrs=attrs, confidence=confidence,
        )

    @property
    def maneuver_type(self): return self.attrs["maneuver_type"]
    @property
    def actor_id(self): return self.attrs["actor_id"]
    @property
    def frame_start(self): return self.attrs["frame_start"]
    @property
    def frame_end(self): return self.attrs["frame_end"]
    @property
    def duration_frames(self): return self.attrs["duration_frames"]
    @property
    def state(self): return self.attrs["state"]
    @property
    def severity(self): return self.attrs["severity"]
    @property
    def related_rule(self): return self.attrs.get("related_rule")

    def close(self, frame_end, final_state="ended"):
        if self.attrs["state"] != "active":
            return
        if frame_end < self.frame_start:
            raise ValueError("frame_end < frame_start")
        self.attrs["frame_end"] = frame_end
        self.attrs["duration_frames"] = frame_end - self.frame_start + 1
        self.attrs["state"] = final_state
        self.valid_to = frame_end

    def is_active_at(self, frame_id):
        if self.state != "active":
            return False
        if frame_id < self.frame_start:
            return False
        if self.frame_end is not None and frame_id > self.frame_end:
            return False
        return True


INTERACTION_TYPES = (
    "following", "approaching", "yielding_to", "overtaking",
    "wrong_side_meeting", "opposite_direction", "same_direction",
    "blocked_view", "approaching_pedestrian",
    "approaching_intersection", "crossing",
    "standing_still", "changing_lane",
)


class InteractionEvent(BaseEntity):
    def __init__(self, entity_id, interaction_type, src_id, dst_id, frame_start,
                 frame_end=None, state="active", severity=0.0,
                 related_rule=None, source_relations=None, confidence=1.0,
                 derived_attrs=None, valid_from=None, valid_to=None):
        if interaction_type not in INTERACTION_TYPES:
            raise ValueError("Unknown interaction_type: " + interaction_type)
        if not src_id or not dst_id:
            raise ValueError("src_id, dst_id required")
        if state not in ("active", "ended"):
            raise ValueError("state must be active/ended, got: " + state)
        if frame_end is None:
            duration = -1
        else:
            if frame_end < frame_start:
                raise ValueError("frame_end < frame_start")
            duration = frame_end - frame_start + 1
        attrs = {
            "interaction_id": entity_id, "interaction_type": interaction_type,
            "src_id": src_id, "dst_id": dst_id,
            "frame_start": frame_start, "frame_end": frame_end,
            "duration_frames": duration, "state": state,
            "severity": severity, "related_rule": related_rule,
            "source_relations": source_relations or [],
            "derived_attrs": derived_attrs or {},
        }
        super().__init__(
            entity_id=entity_id, entity_type=EntityType.INTERACTION_EVENT,
            valid_from=valid_from if valid_from is not None else frame_start,
            valid_to=valid_to if valid_to is not None else frame_end,
            attrs=attrs, confidence=confidence,
        )

    @property
    def interaction_type(self): return self.attrs["interaction_type"]
    @property
    def src_id(self): return self.attrs["src_id"]
    @property
    def dst_id(self): return self.attrs["dst_id"]
    @property
    def frame_start(self): return self.attrs["frame_start"]
    @property
    def frame_end(self): return self.attrs["frame_end"]
    @property
    def duration_frames(self): return self.attrs["duration_frames"]
    @property
    def state(self): return self.attrs["state"]
    @property
    def severity(self): return self.attrs["severity"]
    @property
    def related_rule(self): return self.attrs.get("related_rule")
    @property
    def source_relations(self): return list(self.attrs["source_relations"])

    def close(self, frame_end, final_state="ended"):
        if self.attrs["state"] != "active":
            return
        if frame_end < self.frame_start:
            raise ValueError("frame_end < frame_start")
        self.attrs["frame_end"] = frame_end
        self.attrs["duration_frames"] = frame_end - self.frame_start + 1
        self.attrs["state"] = final_state
        self.valid_to = frame_end

    def is_active_at(self, frame_id):
        if self.state != "active":
            return False
        if frame_id < self.frame_start:
            return False
        if self.frame_end is not None and frame_id > self.frame_end:
            return False
        return True


def make_maneuver_id(veh_id, frame_start):
    return "man_" + veh_id + "_" + str(frame_start)


def make_interaction_id(src_id, dst_id, interaction_type, frame_start):
    return "int_" + src_id + "_" + dst_id + "_" + interaction_type + "_" + str(frame_start)
