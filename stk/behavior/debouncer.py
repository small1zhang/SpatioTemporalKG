"""
行为关系防抖机制 (v3 sec 3.4)

行为关系 (following, approaching, yielding_to 等) 在帧间可能因浮点精度、
CARLA 物理引擎抖动而产生闪烁。引入"持续帧数阈值"防抖机制：

  1. 关系条件连续满足 threshold 帧后, 才正式创建该行为关系
  2. 关系条件连续消失 threshold 帧后, 才删除该行为关系

防抖阈值表 (v3 sec 3.4.2):
  following:                3 帧
  approaching:              3 帧
  yielding_to:              3 帧
  overtaking:               5 帧
  changing_lane:            2 帧
  blocked_view:             3 帧
  approaching_pedestrian:   3 帧
  approaching_intersection: 2 帧
  crossing:                 3 帧
  standing_still:           2 帧
  other (wrong_side_meeting, opposite_direction, same_direction): 1 帧
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


# --- 默认防抖阈值 (v3 sec 3.4.2) ---
DEFAULT_DEBOUNCE_THRESHOLDS: Dict[str, int] = {
    "following": 3,
    "approaching": 3,
    "yielding_to": 3,
    "overtaking": 5,
    "changing_lane": 2,
    "blocked_view": 3,
    "approaching_pedestrian": 3,
    "approaching_intersection": 2,
    "crossing": 3,
    "standing_still": 2,
    # 瞬时关系: 如 wrong_side_meeting, opposite_direction, same_direction
    # 可用 1 帧（零防抖，因为这是单一几何判定）
    "wrong_side_meeting": 1,
    "opposite_direction": 1,
    "same_direction": 1,
}


class DebounceItem:
    """单条关系键的防抖状态。"""

    def __init__(self, threshold: int):
        self.threshold = threshold
        self.counter: int = 0
        self.is_active: bool = False
        self.active_since: Optional[int] = None
        self.last_condition_met: Optional[bool] = None

    def update(self, condition_met: bool, frame_id: int) -> Tuple[str, Optional[Dict[str, Any]]]:
        """更新防抖状态。

        Returns:
            (action, extra_attrs)
            action: "create" / "delete" / "keep" / "none"
        """
        if condition_met:
            self.counter += 1
        else:
            self.counter = 0  # v3 sec 3.4.3: 条件消失立即重置计数器

        if condition_met and not self.is_active and self.counter >= self.threshold:
            # 正式创建
            self.is_active = True
            self.active_since = frame_id
            return ("create", {"debounce_activated": frame_id, "counter": self.counter})
        elif condition_met and self.is_active:
            return ("keep", {"debounce_counter": self.counter})
        elif not condition_met and self.is_active and self.counter <= 0:
            # 条件消失已持续 threshold 帧 -> 正式删除
            self.is_active = False
            self.active_since = None
            return ("delete", {"debounce_deactivated": frame_id, "counter": self.counter})
        elif not condition_met:
            return ("none", None)



class RelationDebouncer:
    """行为关系防抖管理器 (v3 sec 3.4.3).

    Example:
        debouncer = RelationDebouncer(following_threshold=3)
        key = ("veh_001", "veh_002", "following")
        action, extra = debouncer.update("following", key, condition_met=True, frame_id=100)
        if action == "create":
            # 创建 following 关系边 + InteractionEvent 节点
            ...
    """

    def __init__(self, thresholds: Optional[Dict[str, int]] = None):
        self._thresholds: Dict[str, int] = dict(DEFAULT_DEBOUNCE_THRESHOLDS)
        if thresholds:
            self._thresholds.update(thresholds)
        self._items: Dict[Tuple, DebounceItem] = {}

    def get_threshold(self, relation_type: str) -> int:
        return self._thresholds.get(relation_type, 3)

    def update(self, relation_type: str, key: Tuple,
               condition_met: bool, frame_id: int) -> Tuple[str, Optional[Dict[str, Any]]]:
        """更新指定关系键的防抖状态。

        与 v3 sec 3.4.3 伪代码完全对应:
          - condition_met= True -> counter += 1
          - condition_met= False -> counter = 0
          - counter >= threshold -> 正式创建
          - counter <= 0 且曾被激活 -> 正式删除

        Args:
            relation_type: 关系类型 key
            key: (src_id, dst_id, relation_type) 元组
            condition_met: 当前帧该关系条件是否满足
            frame_id: 当前帧号

        Returns:
            (action, extra_attrs):
              "create" -> 可创建该行为了
              "delete" -> 可删除该行为了
              "keep"   -> 已有行为继续
              "none"   -> 仍然不满足条件
        """
        if key not in self._items:
            threshold = self.get_threshold(relation_type)
            self._items[key] = DebounceItem(threshold)

        result = self._items[key].update(condition_met, frame_id)
        if result is None:
            return ("none", None)
        return result

    def active_keys(self) -> list[Tuple]:
        """当前所有"已正式激活"的关系键。"""
        return [k for k, item in self._items.items() if item.is_active]

    def reset(self, key: Optional[Tuple] = None) -> None:
        """重置防抖状态（用于关系清理时）。"""
        if key is not None and key in self._items:
            del self._items[key]
        else:
            self._items.clear()

    def status_summary(self) -> Dict[str, Any]:
        """防抖状态摘要（用于调试 / 测试验证）。"""
        active = []
        pending = []
        for key, item in self._items.items():
            entry = {"key": key, "counter": item.counter,
                     "threshold": item.threshold, "is_active": item.is_active}
            if item.is_active:
                active.append(entry)
            else:
                pending.append(entry)
        return {"active": active, "pending": pending, "n_active": len(active), "n_pending": len(pending)}
