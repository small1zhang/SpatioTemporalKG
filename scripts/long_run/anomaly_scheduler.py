#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anomaly_scheduler.py — 长时间 Phase1 采集的异常事件调度表
(v3 §5.4 / 长期连续采集方案 §四.2-§四.3)

设计要点:
  • 事件调度表 (EventScheduler) 在 20 分钟主循环内按时间窗口随机穿插异常,
    其余时间正常背景车流, 异常触发后自然过渡回正常, 不硬性中断采集.
  • 每个异常事件用 apply_control / set_target_velocity / set_light_state 等
    CARLA 原生 API 直接调用, 不依赖 ScenarioRunner.
  • 异常事件以 "ego 为观察主体" 视角注入: ego 周围某辆背景车做异常行为,
    ego 仅作为传感器触发源 (碰撞/车道入侵传感器 attach 在 ego 上).
  • scheduler 自身是纯 Python, 不直接 import carla, 由 collect.py 在 tick
    循环里调用 scheduler.tick(t, world, ego) 应用异常动作.

异常类型 (与 stk/rules/traffic/rules.py 对齐):
  S31 sudd_brk           : 前车急刹 (ego 跟车前车突然 brake=1.0)
  S32 sudd_stp           : 前车急停 (前车 throttle=0, brake=1.0 持续)
  S20 avd_col            : 紧急避让 (ego 邻车突然变道切入)
  S21 jun_ny             : 路口不让行 (路口背景车不减速通过)
  S22 rev_drive          : 逆向行驶 (背景车 set_target_velocity 反向)
  S10 ped_crs            : 行人横穿 (walker 角色变道朝 ego 行走)
  S33 obs_blk            : 视线遮挡 (在前方 throws 障碍物)
"""
from __future__ import annotations
import argparse
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================
# 1. 数据类
# ============================================================

@dataclass
class AnomalyEvent:
    """单次异常事件描述."""
    event_id: str
    anomaly_type: str            # sudd_brk / sudd_stp / avd_col / ...
    trigger_frame: int           # 触发帧号 (绝对帧号, 从采集开始累计)
    duration_frames: int         # 持续帧数
    target_actor_id: Optional[str] = None  # 受控背景 actor 的 CARLA id (str)
    extra: Dict[str, Any] = field(default_factory=dict)  # 类型相关附件参数
    applied: bool = False        # 是否已开始施加异常控制
    completed: bool = False      # 是否已结束 (恢复正常)

    @property
    def end_frame(self) -> int:
        return self.trigger_frame + self.duration_frames


# ============================================================
# 2. 调度器
# ============================================================

class EventScheduler:
    """在长时间主循环中按事件调度表穿插异常.

    用法:
        sched = EventScheduler(total_frames=24000, seed=42,
                                density_per_minute=2.0,
                                ego_is_target=False)
        events = sched.build_schedule()
        # 在 tick 循环里:
        for frame_id in range(total_frames):
            active = sched.tick(frame_id)  # 返回当前帧应触发的 active 事件
            for ev in active:
                apply_anomaly(world, ev, ego)
    """

    # 各类异常的默认持续帧数 (按 20 fps, 5s ~ 10s)
    DEFAULT_DURATION = {
        "sudd_brk":   100,    # 5 s, 一帧 brake=1.0
        "sudd_stp":   200,    # 10 s 急停后保持
        "avd_col":    100,    # 5 s 紧急变道
        "jun_ny":     100,    # 5 s 路口不让行
        "rev_drive":  150,    # 7.5 s 逆向
        "ped_crs":    120,    # 6 s 行人横穿
        "obs_blk":    200,    # 10 s 障碍物
    }

    def __init__(
        self,
        total_frames: int,
        seed: int = 42,
        density_per_minute: float = 2.0,
        ego_is_target: bool = False,
        anomaly_types: Optional[List[str]] = None,
        frames_per_second: float = 20.0,
    ):
        self.total_frames = total_frames
        self.seed = seed
        # 每分钟异常事件数 (期望值); 20 分钟 * 2 = ~40 次
        self.density_per_minute = density_per_minute
        # 异常是否施加到 ego 本身 (默认 False: ego 是观察方, 背景车做异常)
        self.ego_is_target = ego_is_target
        self.frames_per_second = frames_per_second
        # 异常类型及其权重 (猫眼分布, jun_ny 与 sudd_brk 更常见)
        self.anomaly_types = anomaly_types or [
            "sudd_brk", "sudd_brk", "sudd_stp",
            "avd_col", "jun_ny", "jun_ny",
            "rev_drive", "ped_crs", "obs_blk",
        ]
        self._rng = random.Random(seed)
        self._events: List[AnomalyEvent] = []
        # 已 active 但未 completed 的活跃事件 (frame 维度)
        self._active: List[AnomalyEvent] = []

    # ---------------- 调度表构建 ----------------

    def build_schedule(self) -> List[AnomalyEvent]:
        """采样整个 20min 的异常事件表 (Poisson 过程近似)."""
        total_min = self.total_frames / (self.frames_per_second * 60.0)
        n_expect = int(total_min * self.density_per_minute)
        events = []
        eid = 0
        for _ in range(n_expect):
            eid += 1
            atype = self._rng.choice(self.anomaly_types)
            # 随机触发帧号, 留出首末 60s 安全边距
            margin = int(60 * self.frames_per_second)
            ts = self._rng.randint(margin, max(margin + 1,
                                                self.total_frames - margin))
            dur = self.DEFAULT_DURATION.get(atype, 100)
            ft = "ego" if self.ego_is_target else "background"
            events.append(AnomalyEvent(
                event_id=f"E{eid:04d}",
                anomaly_type=atype,
                trigger_frame=ts,
                duration_frames=dur,
                extra={
                    "target_role": ft,
                    "intensity": self._rng.random(),
                },
            ))
        # 按触发时间排序
        events.sort(key=lambda e: e.trigger_frame)
        self._events = events
        return events

    def to_dict(self) -> List[Dict[str, Any]]:
        return [{
            "event_id": e.event_id, "anomaly_type": e.anomaly_type,
            "trigger_frame": e.trigger_frame, "duration_frames": e.duration_frames,
            "end_frame": e.end_frame, "target_actor_id": e.target_actor_id,
            "extra": e.extra,
            "applied": e.applied, "completed": e.completed,
        } for e in self._events]

    def load_from_list(self, data: List[Dict[str, Any]]) -> None:
        """从 to_dict() 输出重建事件表 (用于 crash-resume)."""
        self._events = []
        self._active = []
        for d in data:
            ev = AnomalyEvent(
                event_id=d["event_id"], anomaly_type=d["anomaly_type"],
                trigger_frame=d["trigger_frame"], duration_frames=d["duration_frames"],
                target_actor_id=d.get("target_actor_id"),
                extra=d.get("extra", {}),
            )
            ev.applied = d.get("applied", False)
            ev.completed = d.get("completed", False)
            self._events.append(ev)
        # 已 active 未 completed 的事件也补回 _active
        for e in self._events:
            if e.applied and not e.completed:
                self._active.append(e)

    # ---------------- 主循环 tick ----------------

    def tick(self, frame_id: int) -> List[AnomalyEvent]:
        """每帧采集前调用: 返回当前帧需要 apply 的活跃事件.

        - 把 trigger_frame == frame_id 的事件激活并加入 _active
        - 把 end_frame < frame_id 的事件标记 completed 并移出 _active
        - 返回 _active 副本, 让 collect.py 对每个 active 事件调用 apply_anomaly
        """
        # 激活
        for e in self._events:
            if (not e.applied) and e.trigger_frame <= frame_id < e.end_frame:
                e.applied = True
                self._active.append(e)
        # 完成
        still_active = []
        for e in self._active:
            if frame_id >= e.end_frame:
                e.completed = True
            else:
                still_active.append(e)
        self._active = still_active
        return list(self._active)


# ============================================================
# 3. 异常动作施加 (CARLA API 直接调用)
# ============================================================

def apply_anomaly(world, ev: AnomalyEvent, ego, carla_module) -> Optional[str]:
    """根据 ev.anomaly_type 对 ev.target_actor_id 施加异常动作.

    Args:
        world: CARLA world
        ev: 异常事件 (target_actor_id 已经在 collect.py 里被解析为背景 actor id)
        ego: ego vehicle actor (carla.Actor)
        carla_module: carla 模块本体

    Returns:
        日志字符串, 用于写采集日志.
    """
    if ev.target_actor_id is None:
        return None  # target 还未绑定 (e.g. obs_blk 在 tick 时才 spawn 障碍物)

    try:
        actor = world.get_actor(int(ev.target_actor_id))
    except Exception:
        return f"  [anom-{ev.event_id}] target {ev.target_actor_id} not found"

    log = f"  [anom-{ev.event_id}] {ev.anomaly_type} on {ev.target_actor_id} (f={ev.trigger_frame})"
    carla = carla_module

    if ev.anomaly_type in ("sudd_brk", "sudd_stp"):
        # 急刹 / 急停: 强制 brake=1.0, throttle=0
        try:
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            if ev.anomaly_type == "sudd_stp":
                actor.set_target_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
        except Exception as e:
            log += f" [ERR {e}]"

    elif ev.anomaly_type == "avd_col":
        # 紧急避让: 邻车突然变道 (steer 跳变, 模拟紧急切线)
        intensity = ev.extra.get("intensity", 0.5)
        steer = 0.4 + 0.3 * intensity  # ±0.4~0.7 的硬切方向
        try:
            sign = 1.0 if (int(ev.event_id[-2:]) % 2 == 0) else -1.0
            actor.apply_control(carla.VehicleControl(
                throttle=0.5, steer=sign * steer, brake=0.0,
            ))
        except Exception as e:
            log += f" [ERR {e}]"

    elif ev.anomaly_type == "jun_ny":
        # 路口不让行: 加速通过路口 (throttle 推满)
        try:
            actor.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0))
        except Exception as e:
            log += f" [ERR {e}]"

    elif ev.anomaly_type == "rev_drive":
        # 逆向: 强制反向 speed (set_target_velocity 反向)
        try:
            tf = actor.get_transform()
            # 取 -forward 方向 5 m/s
            yaw_rad = math.radians(tf.rotation.yaw + 180.0)
            vx = 5.0 * math.cos(math.radians(tf.rotation.yaw + 180.0))
            vy = 5.0 * math.sin(math.radians(tf.rotation.yaw + 180.0))
            actor.set_target_velocity(carla.Vector3D(x=vx, y=vy, z=0.0))
        except Exception as e:
            log += f" [ERR {e}]"

    else:
        # ped_crs / obs_blk 在 collect.py 的 spawn 阶段已处理 target
        log += " [skip]"

    return log


# ============================================================
# 4. 实体选取 — 给 event 绑定 target_actor_id
# ============================================================

def bind_targets(events: List[AnomalyEvent], spawned_vehicles: List[Any],
                 ego_id: Any, seed: int = 42) -> None:
    """把每个异常事件的 target_actor_id 绑定到一个背景车 id (优先 ego 前后 30m).

    Args:
        events: 已构建好的调度表
        spawned_vehicles: CARLA actor 列表 (ego 在内的所有 spawn 车)
        ego_id: ego actor 的 id
    """
    rng = random.Random(seed + 17)
    bg_vehicles = [v for v in spawned_vehicles if v.id != ego_id]
    if not bg_vehicles:
        return
    for ev in events:
        # todo: 真正要做到 "ego 前后 30m", 需要在 collect.py tick 时按位置筛选,
        # 这里先做粗绑定 (随机选一辆背景车作为 target), 留待 collect.py 重绑.
        ev.target_actor_id = str(rng.choice(bg_vehicles).id)


# ============================================================
# 5. CLI 测试 (build_schedule 单跑)
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build 20-min anomaly schedule only")
    p.add_argument("--total-frames", type=int, default=24000, help="总帧数 (20fps*60*20=24000)")
    p.add_argument("--density", type=float, default=2.0, help="每分钟异常事件数 (期望)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="输出 json 路径(可选)")
    args = p.parse_args()

    sched = EventScheduler(
        total_frames=args.total_frames,
        seed=args.seed, density_per_minute=args.density,
    )
    events = sched.build_schedule()
    print(f"[+] Built {len(events)} anomaly events over {args.total_frames} frames "
          f"(~{args.total_frames/1200:.1f} min @ 20fps)")

    # 类型分布
    from collections import Counter
    c = Counter(e.anomaly_type for e in events)
    for t, n in c.most_common():
        print(f"  {t:12s}  {n}x")
    print(f"\n  first 5 events:")
    for e in events[:5]:
        print(f"    {e.event_id} {e.anomaly_type:12s} @f{e.trigger_frame} dur={e.duration_frames}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(sched.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n  saved: {args.out}")
