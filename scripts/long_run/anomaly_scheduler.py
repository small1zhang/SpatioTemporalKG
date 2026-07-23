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

    def tick(self, frame_id: int) -> Tuple[List["AnomalyEvent"], List["AnomalyEvent"]]:
        """每帧采集前调用: 返回 (active, completed_this_frame).

        - 把 trigger_frame == frame_id 的事件激活并加入 _active
        - 把 end_frame <= frame_id 的事件标记 completed 并移出 _active
        - 返回 (active 副本, 本帧刚完成的事件列表):
            * active: 让 collect.py 对每个 active 事件调用 apply_anomaly
            * completed_this_frame: 让 collect.py 恢复 NPC autopilot (异常期间
              被 apply_control 覆盖, 结束后需要 set_autopilot(True) 让交通管理
              器重新接管, 否则 NPC 会以 brake=1.0 永久停在路中央)

        兼容性: 旧调用方 `active = sched.tick(i)` 会拿到 (active, completed) 二元组,
        需要 `active, _ = sched.tick(i)` 才能解构; 为不破坏旧调用, 见 collect.py
        已同步修改.
        """
        # 激活
        for e in self._events:
            if (not e.applied) and e.trigger_frame <= frame_id < e.end_frame:
                e.applied = True
                self._active.append(e)
        # 完成
        still_active = []
        completed_this_frame = []
        for e in self._active:
            if frame_id >= e.end_frame:
                e.completed = True
                completed_this_frame.append(e)
            else:
                still_active.append(e)
        self._active = still_active
        return list(self._active), completed_this_frame


# ============================================================
# 3. 异常动作施加 (CARLA API 直接调用)
# ============================================================

def apply_anomaly(world, ev: AnomalyEvent, ego, carla_module,
                  spawned_walkers: Optional[List[Any]] = None,
                  bp_lib: Any = None,
                  map_: Any = None) -> Optional[str]:
    """根据 ev.anomaly_type 对 ev.target_actor_id 施加异常动作.

    Args:
        world: CARLA world
        ev: 异常事件 (target_actor_id 已经在 collect.py 里被解析为背景 actor id)
        ego: ego vehicle actor (carla.Actor)
        carla_module: carla 模块本体
        spawned_walkers: P1-3 用于 ped_crs — 选一个 walker 重定向其 AI 目标
        bp_lib: P1-3 用于 obs_blk — 查 static.prop 蓝图 spawn 障碍物
        map_: P1-3 用于 obs_blk — 拿 ego 当前 waypoint 投影到车道, 在前方 N 米 spawn

    Returns:
        日志字符串, 用于写采集日志.
    """
    carla = carla_module

    # ── P1-3: obs_blk — tick 时 spawn 障碍物, target_actor_id 用懒绑定 ──
    # obs_blk 的 target_actor_id 在 spawn 之前为 None; 第一次 apply 时 spawn 静态 prop
    # 后把它的 actor.id 写回 ev.target_actor_id, 后续帧就维持控制 (虽然静态 prop 不需
    # 每帧再 apply, 但保持一致调用链)
    if ev.anomaly_type == "obs_blk":
        # 已 spawn 过: 直接返回 (静态障碍物无需每帧操作)
        if ev.target_actor_id is not None:
            try:
                a = world.get_actor(int(ev.target_actor_id))
                if a is not None and a.is_alive:
                    return f"  [anom-{ev.event_id}] obs_blk prop {ev.target_actor_id} alive"
            except Exception:
                pass
            # 已 spawn 但实际找不到: 标记 target=None 让外层清理
            ev.target_actor_id = None

        # 没 spawn 过: 找 ego 前方 8-15m 同车道点 spawn static.prop
        if ego is None or bp_lib is None:
            return f"  [anom-{ev.event_id}] obs_blk [skip: ego/bp_lib missing]"
        try:
            prop_bps = bp_lib.filter("static.prop.*")
            if not prop_bps:
                return f"  [anom-{ev.event_id}] obs_blk [skip: no static.prop bps]"
            bp = random.choice(prop_bps)
            ego_tf = ego.get_transform()
            # 取 ego 朝向前方 12m, 横向偏 0 (车道中央)
            yaw_rad = math.radians(ego_tf.rotation.yaw)
            tx = ego_tf.location.x + 12.0 * math.cos(yaw_rad)
            ty = ego_tf.location.y + 12.0 * math.sin(yaw_rad)
            tz = ego_tf.location.z
            tf = carla.Transform(carla.Location(x=tx, y=ty, z=tz),
                                 carla.Rotation(yaw=ego_tf.rotation.yaw))
            prop = world.spawn_actor(bp, tf)
            ev.target_actor_id = str(prop.id)
            # 把 prop.id 也记到 extra 供 cleanup
            ev.extra.setdefault("obstacle_actor_id", str(prop.id))
            return f"  [anom-{ev.event_id}] obs_blk spawned prop {prop.id} @12m ahead"
        except Exception as e:
            return f"  [anom-{ev.event_id}] obs_blk [spawn failed: {e}]"

    # 其他类型要求 target_actor_id 已绑定
    if ev.target_actor_id is None:
        return None

    try:
        actor = world.get_actor(int(ev.target_actor_id))
    except Exception:
        return f"  [anom-{ev.event_id}] target {ev.target_actor_id} not found"

    log = f"  [anom-{ev.event_id}] {ev.anomaly_type} on {ev.target_actor_id} (f={ev.trigger_frame})"

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

    elif ev.anomaly_type == "ped_crs":
        # P1-3: 行人横穿 — target_actor_id 是 walker (由 bind_targets 绑定)
        # 取该 walker 的 AI controller (spawn_walkers 返回 (walker, controller)),
        # 重定向其 go_to_location 到 ego 当前位置, 提速至 2.0 m/s 朝 ego 走
        if spawned_walkers is None:
            log += " [skip: spawned_walkers not provided]"
            return log
        # 找对应的 controller
        ctl = None
        for w, c in spawned_walkers:
            if w.id == actor.id:
                ctl = c
                break
        if ctl is None:
            log += " [skip: walker controller not found]"
            return log
        try:
            if ego is not None:
                ego_loc = ego.get_location()
                ctl.start()  # 确保活跃
                ctl.go_to_location(ego_loc)
                ctl.set_max_speed(2.0)  # 比正常 1.0 提速 2x, 模拟突发横穿
                log += f" -> walker heading to ego@({ego_loc.x:.1f},{ego_loc.y:.1f})"
            else:
                log += " [skip: ego missing]"
        except Exception as e:
            log += f" [ERR {e}]"

    else:
        log += " [skip]"

    return log


# ============================================================
# 4. 实体选取 — 给 event 绑定 target_actor_id
# ============================================================

def bind_targets(events: List[AnomalyEvent], spawned_vehicles: List[Any],
                 ego_id: Any, seed: int = 42,
                 ego_transform: Any = None,
                 vehicle_waypoints: Dict[str, Tuple[int, int]] = None,
                 spawned_walkers: Optional[List[Any]] = None) -> None:
    """把每个异常事件的 target_actor_id 绑定到与 ego 有合适空间/车道关系的车.

    阶段 4 (FE-15) 实现:
      - 利用 ego_transform (ego 当前 pose) 与 vehicle_waypoints (每车 road_id/lane_id)
        做按车道/距离筛选, 替换原占位的纯 rng.choice
      - 针对不同 anomaly_type 差异化匹配:
        * sudd_brk / sudd_stp: ego 正前方 5-30m 同车道 NPC (前车急刹场景)
        * avd_col / avd_col_track: ego 侧向 3-10m 同向 NPC (侧方切入)
        * cut_in: 相邻车道前方 10-20m NPC
        * ped_crs: 选一个 walker (P1-3: 不再绑定到车辆)
        * others: 距 ego 最近的同车道/相邻车道 NPC
      - 找不到合适 NPC 时回退到 rng.choice (原行为, 不破坏向后兼容)

    Args:
        events: 已构建好的调度表
        spawned_vehicles: CARLA actor 列表 (含 ego)
        ego_id: ego actor id
        ego_transform: ego 的 carla.Transform (可选; 没有则退化到 rng.choice)
        vehicle_waypoints: {vehicle.id_str: (road_id, lane_id)} (可选)
            若未提供, 同车道判定退化为只用距离
        spawned_walkers: P1-3: (walker, controller) 元组列表, 用于 ped_crs 绑定
    """
    rng = random.Random(seed + 17)
    bg_vehicles = [v for v in spawned_vehicles if v.id != ego_id]
    if not bg_vehicles:
        return

    import math as _math
    ego_loc = getattr(ego_transform, "location", None) if ego_transform else None
    ego_yaw_rad = 0.0
    if ego_transform and hasattr(ego_transform, "rotation"):
        ego_yaw_rad = _math.radians(ego_transform.rotation.yaw)

    def _to_ego_frame(npc_loc):
        if ego_loc is None:
            return None, None
        dx = npc_loc.x - ego_loc.x
        dy = npc_loc.y - ego_loc.y
        c, s = _math.cos(ego_yaw_rad), _math.sin(ego_yaw_rad)
        lon = dx * c + dy * s
        lat = -dx * s + dy * c
        return lon, lat

    def _v_id_str(v):
        return str(getattr(v, "id", ""))

    def _same_lane(v, ego_wp_key):
        if not vehicle_waypoints or not ego_wp_key:
            # P1-4: 没有 waypoint 信息时, 退化用 |lat|<2.5m 当作"同车道"
            # (城市道路车道宽 ~3m, 2.5m 内基本是同车道)
            return None  # 返回 None 表示"未知", 让调用方自行用 lat 判定
        vkey = vehicle_waypoints.get(_v_id_str(v))
        return vkey is not None and vkey == ego_wp_key

    ego_wp_key = None
    if vehicle_waypoints and ego_id is not None:
        ego_wp_key = vehicle_waypoints.get(str(ego_id))

    # 按 NPC 与 ego 的空间关系做缓存
    npc_scored = []
    for v in bg_vehicles:
        try:
            vloc = v.get_location()
        except Exception:
            continue
        lon, lat = _to_ego_frame(vloc)
        if lon is None:
            npc_scored.append((v, None, None))
            continue
        npc_scored.append((v, lon, lat))

    def _find_match(predicate):
        candidates = []
        for v, lon, lat in npc_scored:
            if lon is None:
                continue
            ok, score = predicate(v, lon, lat)
            if ok:
                candidates.append((score, v))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    # P1-4: 三级阈值 (strict / relaxed / nearest), 减少直接 rng.choice 降级
    # 同车道判定在无 waypoint 时退化用 |lat| 阈值 (None from _same_lane → 用 lat)
    def _lane_ok(v, lat, thresh=2.5):
        sl = _same_lane(v, ego_wp_key)
        if sl is True:
            return True
        if sl is False:
            return False
        # sl is None: 退化用 lat 距离
        return abs(lat) < thresh

    def _p_ahead_same_lane_strict(v, lon, lat):
        return (_lane_ok(v, lat, 2.5) and 5.0 <= lon <= 30.0 and abs(lat) < 3.0), abs(lon)

    def _p_ahead_same_lane_relaxed(v, lon, lat):
        # 放宽到 0-80m, lat < 4.5m (覆盖大椭圆 spawn 范围)
        return (_lane_ok(v, lat, 3.0) and 0.0 <= lon <= 80.0 and abs(lat) < 4.5), abs(lon)

    def _p_lateral_same_dir_strict(v, lon, lat):
        return (3.0 <= abs(lat) <= 10.0 and -15.0 <= lon <= 15.0), abs(lat)

    def _p_lateral_same_dir_relaxed(v, lon, lat):
        # 放宽到 0.5-15m 横向, -25~25m 纵向
        return (0.5 <= abs(lat) <= 15.0 and -25.0 <= lon <= 25.0), abs(lat)

    def _p_adjacent_lane_ahead_strict(v, lon, lat):
        sl = _same_lane(v, ego_wp_key)
        # 严格: 必须 waypoint 知道车道且不同车道
        if sl is None:
            # 无 waypoint: 用 lat 距离作代理 — 4-7m 算相邻车道
            is_adj = 4.0 <= abs(lat) < 7.0
        else:
            is_adj = (not sl)
        return (is_adj and 10.0 <= lon <= 20.0 and abs(lat) < 5.0), abs(lon)

    def _p_adjacent_lane_ahead_relaxed(v, lon, lat):
        sl = _same_lane(v, ego_wp_key)
        if sl is None:
            is_adj = 3.0 <= abs(lat) <= 10.0
        else:
            is_adj = (not sl)
        return (is_adj and 5.0 <= lon <= 40.0 and abs(lat) < 8.0), abs(lon)

    def _p_nearest(v, lon, lat):
        return (lon is not None), (lon * lon + lat * lat)

    for ev in events:
        atype = ev.anomaly_type
        match = None

        # ── P1-3: ped_crs — 选一个 walker ──
        if atype == "ped_crs" and spawned_walkers:
            walkers = [w for w, _ in spawned_walkers if w.is_alive]
            if walkers:
                rng.shuffle(walkers)
                match = walkers[0]
                ev.target_actor_id = str(match.id)
                continue

        # ── P1-3: obs_blk — target 在 tick 时才 spawn, 此处留 None ──
        if atype == "obs_blk":
            ev.target_actor_id = None
            continue

        # ── P1-4: 三级阈值匹配 (strict → relaxed → nearest → rng.choice) ──
        if ego_loc is not None and atype in ("sudd_brk", "sudd_stp"):
            match = _find_match(_p_ahead_same_lane_strict)
            if match is None:
                match = _find_match(_p_ahead_same_lane_relaxed)
        elif ego_loc is not None and atype in ("avd_col", "avd_col_track"):
            match = _find_match(_p_lateral_same_dir_strict)
            if match is None:
                match = _find_match(_p_lateral_same_dir_relaxed)
        elif ego_loc is not None and atype == "cut_in":
            match = _find_match(_p_adjacent_lane_ahead_strict)
            if match is None:
                match = _find_match(_p_adjacent_lane_ahead_relaxed)
        if match is None:
            # 退化: 距 ego 最近 或 rng.choice
            if ego_loc is not None:
                match = _find_match(_p_nearest)
            if match is None and bg_vehicles:
                match = rng.choice(bg_vehicles)
        if match is not None:
            ev.target_actor_id = str(match.id)


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
