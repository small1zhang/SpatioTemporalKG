#!/usr/bin/env python3
"""验证方案 B: R4 is_opposite_lane 用 lane_id 修复后不再全帧误报"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stk.rules.traffic.rules import check_R4_opposite_meeting

# 模拟 4 种车对组合
cases = [
    # 同向 (lane 同号)
    ("同向 1/1", {"lane_id": 1}, {"lane_id": 1}, 5.0),
    ("同向 -2/-2", {"lane_id": -2}, {"lane_id": -2}, 5.0),
    # 反向 (lane 异号)
    ("反向 1/-1", {"lane_id": 1}, {"lane_id": -1}, 5.0),
    ("反向 -3/3", {"lane_id": -3}, {"lane_id": 3}, 5.0),
    # 缺失
    ("缺失 0/1", {"lane_id": 0}, {"lane_id": 1}, 5.0),
    ("缺失 1/2", {"lane_id": "abc"}, {"lane_id": 2}, 5.0),
    # 不对向但距离近 (反向但 > 距离阈限)
    ("反向远 1/-1 d=20", {"lane_id": 1}, {"lane_id": -1}, 20.0),
]
for name, va, vb, dist in cases:
    la, lb = str(va.get("lane_id", "0")), str(vb.get("lane_id", "0"))
    try:
        is_opposite = int(la) * int(lb) < 0
    except (ValueError, TypeError):
        is_opposite = False
    if is_opposite:
        is_v, sev, _ = check_R4_opposite_meeting(va, vb, dist, is_opposite_lane=True)
    else:
        is_v, sev, _ = False, 0.0, {}
    print(f"{name:25s}  opposite={is_opposite}  trigger={is_v}  sev={sev:.2f}")
