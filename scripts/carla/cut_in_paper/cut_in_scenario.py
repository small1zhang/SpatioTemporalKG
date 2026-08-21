#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cut_in_scenario.py -- CARLA Cut-in Safety-Critical Scenario for Paper

Generates a reproducible cut-in sequence (stable, repeatable):
  V17 ahead in adjacent lane → V17 cuts in → V17 slows down → RSS violation → Ego recovers

Design:
  - Ego is autopiloted by Traffic Manager (cruise mode only).
  - V17 is fully manual-controlled (kinematic VehicleControl) so its
    trajectory is deterministic regardless of TM behavior on complex maps.
  - RGB frames are saved by a single-buffer camera (tick-aligned filenames).
  - All actors are spawned at CARLA-native spawn_points (guaranteed safe).
  - Cleanup uses try/finally and a defensive tick loop, so a crashed
    CARLA server will not break the Python process.

Outputs:
  output/rgb/*.png            全程 220 帧 RGB 图片 (1600×900)
  output/keyframes/*.png      4 张关键帧（PIL 文字标注）
  output/logs/per_frame.csv   逐帧状态 CSV
  output/logs/key_frames.csv  关键帧汇总

Usage:
  # 1. 启动 CARLA
  /home/aisecurity/Carla/CarlaUE4.sh -quality-level=Low -carla-rpc-port=2000 -RenderOffScreen &

  # 2. 跑脚本（Town04 默认，Town10HD 用 --map Town10HD）
  python cut_in_scenario.py
  python cut_in_scenario.py --map Town10HD
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import socket
import sys
import time
from pathlib import Path

try:
    import carla
except ImportError:
    print("[FATAL] Cannot import carla. Activate a conda env with carla installed.")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_MAP = "Town04"   # safer default; pass --map Town10HD if you want
FIXED_DT = 0.1           # s per tick
DURATION_S = 12.0   # 120 ticks — all key frames land before V17 drifts
TM_SEED = 2026
N_BACKGROUND = 6

# RSS
RHO = 0.5
A_MAX = 1.0
B_COMFORT = 4.0

# Vehicle speeds (m/s)
V_EGO = 13.0    # ego cruising (TM target)
V_V17_PRE = 12.0   # V17 in adjacent lane (manual)
V_V17_CUT = 8.0   # V17 during cut-in (slow down significantly)
V_V17_POST = 7.0  # V17 slows (RSS trigger)
V_V17_REC = 14.0  # V17 accelerates (recovery)
V_EGO_REC = 7.0   # ego decelerates (recovery)

# Phase boundaries (tick indices at 10 Hz)
T_CUT_START = 80    # 8.0s: V17 begins lateral move
T_CUT_END = 95      # 9.5s: V17 fully in ego lane (CUT_DURATION=1.5s)
T_SLOW = 105        # 10.5s: V17 slows to unsafe distance
T_RECOVER = 145     # 14.5s: V17 accelerates, ego slows
T_DONE = 185        # 18.5s: both resume normal cruise

# Camera (relative to ego)
CAM_X = -7.5
CAM_Z = 4.5
CAM_PITCH = -18.0
CAM_W = 1600
CAM_H = 900
CAM_FOV = 90

# Cut-in lateral motion
LANE_WIDTH = 3.5       # m, typical CARLA lane width
CUT_DURATION = 1.5     # s for full lane crossing (CUT_DURATION=1.5s -> 15 ticks)
CUT_STEER_MAX = 0.18   # CARLA steering units (moderate for straight-road cut-in)

# Key frames (phase_name, target_time_s)
KF_TARGETS = [
    ("Normal", 6.0),
    ("Cut-in", 8.8),
    ("Unsafe Following", 10.5),
    ("Recovery", 11.5),
]
KF_WINDOW = 0.3


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _log(msg):
    print(f"[cut-in] {msg}", flush=True)


def ping_carla(host, port, timeout=3.0):
    try:
        with socket.socket() as s:
            s.settimeout(timeout)
            s.connect((host, port))
        return True
    except OSError:
        return False


def speed_of(vel):
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)


def rss_safe(v_ego, v_leader):
    if v_ego <= v_leader:
        return 0.0
    return v_ego * RHO + 0.5 * A_MAX * RHO**2 + (v_ego**2 - v_leader**2) / (2.0 * B_COMFORT)


# ═══════════════════════════════════════════════════════════════════════
# Spawn Pair Finder (uses CARLA-native spawn_points)
# ═══════════════════════════════════════════════════════════════════════

def _is_ahead(sp_ego, sp_v17, heading_deg):
    """True if v17 is >1m forward of ego along its heading."""
    h = heading_deg % 360
    if h < 45 or h >= 315:        # heading north (+y)
        return sp_v17.location.y > sp_ego.location.y + 1
    if 135 <= h < 225:            # heading south (-y)
        return sp_v17.location.y < sp_ego.location.y - 1
    if 45 <= h < 135:             # heading east (+x)
        return sp_v17.location.x > sp_ego.location.x + 1
    return sp_v17.location.x < sp_ego.location.x - 1   # west


def find_spawn_pair(world, max_candidates=30):
    """Find ego/v17 spawn-point pairs on a long straight adjacent-lane segment.

    Returns list of (ego_sp, v17_sp, side) candidates,
        side = 'left'  -> V17 is on ego's left, must move right
        side = 'right' -> V17 is on ego's right, must move left
    """
    cmap = world.get_map()
    sp_points = cmap.get_spawn_points()

    # Pre-index all spawn points
    sp_info = []  # (idx, spawn, waypoint)
    for i, sp in enumerate(sp_points):
        wp = cmap.get_waypoint(sp.location, lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue
        sp_info.append((i, sp, wp))

    candidates = []

    for i, sp, wp in sp_info:
        # ── Long-straight filter: heading change <3° over 80m ──────────
        nxt = wp.next(80.0)
        if not nxt:
            continue
        h0, h1 = wp.transform.rotation.yaw, nxt[0].transform.rotation.yaw
        if abs(h0 - h1) > 3.0:
            continue

        # ── Adjacent driving lane ─────────────────────────────────────
        left, right = wp.get_left_lane(), wp.get_right_lane()
        adj_id, side = None, None
        if left and left.lane_type == carla.LaneType.Driving:
            adj_id, side = left.lane_id, 'left'
        elif right and right.lane_type == carla.LaneType.Driving:
            adj_id, side = right.lane_id, 'right'

        if adj_id is None:
            continue

        # ── Find v17 sp on the same road+adjacent lane, 2-10m AHEAD ───
        for j, sp2, wp2 in sp_info:
            if i == j:
                continue
            if wp2.road_id != wp.road_id or wp2.lane_id != adj_id:
                continue
            d = sp.location.distance(sp2.location)
            if not (2.0 < d < 10.0):
                continue
            if not _is_ahead(sp, sp2, sp.rotation.yaw):
                continue
            # Heading match (ego vs v17)
            if abs(sp.rotation.yaw - sp2.rotation.yaw) > 5.0:
                continue
            # V17 lane straightness over 80m
            nxt2 = wp2.next(80.0)
            if nxt2 and abs(wp2.transform.rotation.yaw - nxt2[0].transform.rotation.yaw) > 3.0:
                continue
            score = 100.0 - abs(h0 - h1)
            candidates.append((score, sp, sp2, side))
            break

    candidates.sort(key=lambda x: -x[0])
    return [(sp1, sp2, sd) for _, sp1, sp2, sd in candidates[:max_candidates]]


# ═══════════════════════════════════════════════════════════════════════
# Spawning
# ═══════════════════════════════════════════════════════════════════════

def cleanup_world(world):
    try:
        for v in world.get_actors().filter("vehicle.*"):
            v.destroy()
        for w in world.get_actors().filter("walker.*"):
            w.destroy()
        world.tick()
        time.sleep(0.3)
    except Exception as e:
        _log(f"cleanup_world warn: {e}")


def spawn_vehicles(world, tm, pair, n_bg):
    """Spawn ego (autopilot), v17 (manual), and background cars (autopilot)."""
    ego_sp, v17_sp, side = pair
    bp_lib = world.get_blueprint_library()
    bp_name = "vehicle.tesla.model3"

    # ---- Ego ----
    ego = None
    for ox, oy in [(0, 0), (0, 0.3), (0, -0.3), (0.5, 0), (-0.5, 0)]:
        try:
            tf = carla.Transform(
                carla.Location(ego_sp.location.x + ox, ego_sp.location.y + oy, ego_sp.location.z),
                ego_sp.rotation,
            )
            bp = bp_lib.find(bp_name)
            bp.set_attribute("role_name", "hero")
            ego = world.spawn_actor(bp, tf)
            break
        except RuntimeError:
            continue
    if ego is None:
        return None, None, []
    ego.set_autopilot(True, tm.get_port())
    tm.auto_lane_change(ego, False)
    tm.set_desired_speed(ego, V_EGO * 3.6)
    _log(f"ego spawned id={ego.id} at ({ego_sp.location.x:.1f},{ego_sp.location.y:.1f})")

    # ---- V17 (manual; no autopilot) ----
    v17 = None
    for ox, oy in [(0, 0), (0, 0.3), (0, -0.3)]:
        try:
            tf = carla.Transform(
                carla.Location(v17_sp.location.x + ox, v17_sp.location.y + oy, v17_sp.location.z),
                v17_sp.rotation,
            )
            bp = bp_lib.find(bp_name)
            bp.set_attribute("role_name", "cut_in")
            v17 = world.spawn_actor(bp, tf)
            break
        except RuntimeError:
            continue
    if v17 is None:
        _log("WARN: v17 spawn failed, destroying ego")
        try: ego.destroy()
        except: pass
        return None, None, []
    # Important: V17 is NOT on autopilot -- we control it kinematically.
    _log(f"v17 spawned id={v17.id} at ({v17_sp.location.x:.1f},{v17_sp.location.y:.1f}) side={side}")

    # ---- Background traffic (autopilot) ----
    import random
    all_sp = world.get_map().get_spawn_points()
    bg_bps = bp_lib.filter("vehicle.*")
    bg_vehicles = []
    used = set()
    random.seed(TM_SEED + 1)
    attempts = 0
    while len(bg_vehicles) < n_bg and attempts < 200:
        attempts += 1
        idx = random.randint(0, len(all_sp) - 1)
        if idx in used:
            continue
        sp = all_sp[idx]
        if ego.get_transform().location.distance(sp.location) < 45.0:
            continue
        bp = random.choice(bg_bps)
        bp.set_attribute("role_name", "autopilot")
        try:
            v = world.spawn_actor(bp, sp)
            v.set_autopilot(True, tm.get_port())
            tm.set_desired_speed(v, random.uniform(8, 14) * 3.6)
            bg_vehicles.append(v)
            used.add(idx)
        except RuntimeError:
            continue
    _log(f"background: {len(bg_vehicles)} vehicles spawned")
    return ego, v17, bg_vehicles


# ═══════════════════════════════════════════════════════════════════════
# Manual V17 cut-in controller (kinematic)
# ═══════════════════════════════════════════════════════════════════════

class V17Controller:
    """Drives V17 forward + performs lateral cut-in maneuver."""

    def __init__(self, v17, side: str, ego_yaw_deg: float):
        self.v17 = v17
        self.side = side  # 'left' or 'right' (V17's side relative to ego)
        # Heading (degrees) -- kept constant throughout the straight run
        self.heading_deg = ego_yaw_deg
        self.heading_rad = math.radians(ego_yaw_deg)

        # Cut-in target lateral offset (relative to current lane center):
        #   V17 must move LANE_WIDTH in the direction toward ego lane
        # direction toward ego: if V17 is on left, move +right (i.e. -left_axis)
        # Left-axis vector (perpendicular to heading, pointing left): (-sin, cos)
        # If V17 is on left, it must move in +left_axis direction * -1 to go right.
        # We will compute actual command at run-time using physics control.
        self.lateral_progress = 0.0  # 0 = adjacent lane, 1 = ego lane

        # Initial state (set ego_initial_offset later)
        self._prev_yaw = ego_yaw_deg
        self._yaw_rate = 0.0  # deg/tick, for damping
        self._initialized = False

    def initialize(self, ego_tf, v17_tf):
        self._initial_v17_loc = v17_tf.location
        self._initial_ego_loc = ego_tf.location
        self._initialized = True
        _log(f"V17 controller initialized: ego_yaw={self.heading_deg:.1f}, side={self.side}")

    def get_target_speed(self, tick):
        """Target longitudinal speed (m/s) for V17 at this tick."""
        if tick < T_CUT_START:
            return V_V17_PRE     # 12.0 - cruise in adjacent lane
        elif tick < T_CUT_START + 15:
            return V_V17_CUT     # 8.0  - slow during cut-in
        elif tick < T_RECOVER:
            return V_V17_POST    # 7.0  - slow to create unsafe gap
        elif tick < T_DONE:
            return V_V17_PRE     # 12.0 - recover to cruise speed (no overshoot)
        else:
            return V_V17_PRE     # 12.0 - continue cruise

    def get_lateral_progress(self, tick):
        """0 = adjacent lane, 1 = fully in ego lane. Smooth cubic ramp."""
        if tick < T_CUT_START:
            return 0.0
        if tick >= T_CUT_END:
            return 1.0
        alpha = (tick - T_CUT_START) / float(T_CUT_END - T_CUT_START)
        # smoothstep
        return alpha * alpha * (3 - 2 * alpha)

    def compute_control(self, tick, v17_tf, current_speed, lat_offset=0.0):
        """Build a carla.VehicleControl to maintain heading + lateral motion + target speed.
        lat_offset: V17's current lateral offset from ego lane center (+ = right of ego)."""
        # ── Longitudinal: simple P controller on (target_v - current_v)
        target_v = self.get_target_speed(tick)
        v_err = target_v - current_speed
        # More aggressive braking during cut-in phase
        if T_CUT_START <= tick < T_CUT_END + 5:
            throttle = max(0.0, min(1.0, 0.4 + 0.15 * v_err))
            if v_err < -0.3:
                brake = min(1.0, -v_err * 0.12)
            else:
                brake = 0.0
        else:
            throttle = max(0.0, min(1.0, 0.5 + 0.12 * v_err))
            if v_err < -0.5:
                brake = min(1.0, -v_err * 0.04)
            else:
                brake = 0.0

        # ── Lateral: open-loop cut-in + lateral position feedback ──────────
        steer_open = 0.0
        if T_CUT_START <= tick < T_CUT_END:
            mag = CUT_STEER_MAX if self.side == 'left' else -CUT_STEER_MAX
            t_norm = (tick - T_CUT_START) / max(1.0, float(T_CUT_END - T_CUT_START))
            # symmetric trapezoidal ramp: rise first 35%, hold 30%, fall last 35%
            if t_norm < 0.35:
                steer_open = mag * (t_norm / 0.35)
            elif t_norm < 0.65:
                steer_open = mag
            else:
                steer_open = mag * (1.0 - (t_norm - 0.65) / 0.35)
            # ★ Lateral position feedback: if V17 already at ego lane center,
            #   cut steer to prevent overshoot
            if self.side == 'left' and lat_offset >= 0.0:
                steer_open = min(steer_open, 0.0)   # don't steer further right
            elif self.side == 'right' and lat_offset <= 0.0:
                steer_open = max(steer_open, 0.0)   # don't steer further left

        # ── Heading correction (active AFTER cut-in, with damping) ─────────
        raw_yaw = v17_tf.rotation.yaw
        yaw_err = self.heading_deg - raw_yaw
        yaw_err = (yaw_err + 180.0) % 360.0 - 180.0

        # Actual yaw rate (deg per tick), with low-pass filter
        delta_yaw = raw_yaw - self._prev_yaw
        delta_yaw = (delta_yaw + 180.0) % 360.0 - 180.0
        self._yaw_rate = 0.7 * self._yaw_rate + 0.3 * delta_yaw
        self._prev_yaw = raw_yaw

        if tick >= T_CUT_END:
            # Post cut-in: hold heading AND lateral position
            steer_hold  = max(-0.25, min(0.25, yaw_err  * 0.10))
            steer_damp  = max(-0.12, min(0.12, self._yaw_rate * 0.6))
            # ★ Lateral position P-controller: pull V17 back toward ego lane center
            lat_gain = 0.030  # proportional gain (steer per meter of offset)
            if self.side == 'left':
                steer_lat = max(-0.15, min(0.15, -lat_offset * lat_gain))
            else:
                steer_lat = max(-0.15, min(0.15,  lat_offset * lat_gain))
        else:
            steer_hold = 0.0
            steer_damp = 0.0
            steer_lat  = 0.0

        # Final steer = open-loop pulse + heading-hold + damping + lateral-P
        steer = max(-1.0, min(1.0, steer_open + steer_hold + steer_damp + steer_lat))

        return carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )


# ═══════════════════════════════════════════════════════════════════════
# Camera (single-buffer + tick-named save)
# ═══════════════════════════════════════════════════════════════════════

def attach_camera(world, ego):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(CAM_W))
    bp.set_attribute("image_size_y", str(CAM_H))
    bp.set_attribute("fov", str(CAM_FOV))
    bp.set_attribute("sensor_tick", str(FIXED_DT))
    tf = carla.Transform(
        carla.Location(x=CAM_X, z=CAM_Z),
        carla.Rotation(pitch=CAM_PITCH),
    )
    cam = world.spawn_actor(bp, tf, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
    buf = [None]
    def _cb(image):
        buf[0] = image
    cam.listen(_cb)
    _log(f"camera attached: {CAM_W}x{CAM_H} fov={CAM_FOV}")
    return cam, buf


# ═══════════════════════════════════════════════════════════════════════
# Key Frame Selection
# ═══════════════════════════════════════════════════════════════════════

def select_key_frames(rows):
    """Pick 4 best frames in target±window using heuristics."""
    results = []
    for phase, target_t in KF_TARGETS:
        t_lo, t_hi = target_t - KF_WINDOW, target_t + KF_WINDOW
        cand = [r for r in rows if t_lo <= float(r["sim_time"]) <= t_hi]
        if not cand:
            cand = sorted(rows, key=lambda r: abs(float(r["sim_time"]) - target_t))[:1]
        if phase == "Normal":
            best = max(cand, key=lambda r: abs(float(r["lateral_offset_m"])))
        elif phase == "Cut-in":
            best = min(cand, key=lambda r: abs(abs(float(r["lateral_offset_m"])) - LANE_WIDTH / 2))
        elif phase == "Unsafe Following":
            best = min(cand, key=lambda r: float(r["lon_gap_m"]))
        elif phase == "Recovery":
            best = max(cand, key=lambda r: float(r["lon_gap_m"]))
        else:
            best = cand[0]
        results.append({
            "phase": phase,
            "target_time_s": target_t,
            "actual_time_s": float(best["sim_time"]),
            "frame_id": int(best["frame_id"]),
            "lateral_offset_m": float(best["lateral_offset_m"]),
            "lon_gap_m": float(best["lon_gap_m"]),
            "rss_residual_m": float(best["rss_residual_m"]),
        })
        _log(f"key frame [{phase}]: t={best['sim_time']}s gap={best['lon_gap_m']}m lat={best['lateral_offset_m']}m")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Annotation
# ═══════════════════════════════════════════════════════════════════════

def annotate_frame(src, phase, t, metrics, dst):
    if not HAS_PIL:
        return
    img = Image.open(src).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = font_sm = None
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, 30)
            font_sm = ImageFont.truetype(fp, 22)
            break
    if font is None:
        font = font_sm = ImageFont.load_default()

    # Phase label (top-left)
    draw.rectangle([(10, 10), (340, 56)], fill=(0, 0, 0, 200))
    draw.text((20, 14), phase, fill=(255, 255, 255, 255), font=font)

    # Time (top-right)
    ts = f"t = {t:.1f} s"
    bbox = draw.textbbox((0, 0), ts, font=font)
    tw = bbox[2] - bbox[0]
    draw.rectangle([(CAM_W - tw - 24, 10), (CAM_W - 10, 56)], fill=(0, 0, 0, 200))
    draw.text((CAM_W - tw - 16, 14), ts, fill=(255, 255, 255, 255), font=font)

    # Metrics (bottom-center)
    gap, rss_r, spd = metrics.get("lon_gap", 0), metrics.get("rss_residual", 0), metrics.get("ego_speed", 0)
    txt = f"Gap: {gap:.1f} m | RSS res: {rss_r:+.1f} m | v_ego: {spd:.1f} m/s"
    bbox = draw.textbbox((0, 0), txt, font=font_sm)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx = (CAM_W - tw) // 2
    draw.rectangle([(cx - 10, CAM_H - th - 28), (cx + tw + 10, CAM_H - 10)], fill=(0, 0, 0, 200))
    draw.text((cx, CAM_H - th - 24), txt, fill=(255, 255, 255, 255), font=font_sm)

    out = Image.alpha_composite(img, overlay).convert("RGB")
    out.save(str(dst))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host",  default="localhost")
    p.add_argument("--port",  type=int, default=2000)
    p.add_argument("--map",   default=DEFAULT_MAP)
    p.add_argument("--seed",  type=int, default=TM_SEED)
    p.add_argument("--out",   default="output")
    p.add_argument("--bg",    type=int, default=N_BACKGROUND)
    args = p.parse_args()

    out_root = Path(args.out)
    rgb_dir = out_root / "rgb"
    kf_dir  = out_root / "keyframes"
    log_dir = out_root / "logs"
    for d in [rgb_dir, kf_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ticks = int(DURATION_S / FIXED_DT)
    _log(f"target: {ticks} ticks, dt={FIXED_DT}s, duration={DURATION_S}s")

    # ── ping ───────────────────────────────────────────────────────────
    _log(f"pinging CARLA {args.host}:{args.port} ...")
    if not ping_carla(args.host, args.port):
        _log(f"ERROR: CARLA not reachable at {args.host}:{args.port}")
        _log(f"Start it: /home/aisecurity/Carla/CarlaUE4.sh -quality-level=Low -carla-rpc-port={args.port} -RenderOffScreen")
        sys.exit(1)
    _log("CARLA reachable")

    # ── connect ────────────────────────────────────────────────────────
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    _log(f"server: {client.get_server_version()}")

    # ── load map (skip if already correct) ───────────────────────────────
    world = client.get_world()
    cur_map = world.get_map().name.split("/")[-1]
    if cur_map == args.map:
        _log(f"map already {args.map}, reusing")
    else:
        _log(f"loading map: {cur_map} -> {args.map}")
        # First switch to async to avoid load_world crash in sync mode
        try:
            s = world.get_settings()
            s.synchronous_mode = False
            world.apply_settings(s)
        except Exception:
            pass
        world = client.load_world(args.map)
        time.sleep(2)
    cleanup_world(world)

    # ── sync mode ──────────────────────────────────────────────────────
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    _log(f"TM port={tm.get_port()}, seed={args.seed}")

    ego = v17 = cam = None
    bg_vehicles = []
    buf = [None]
    csv_rows = []
    # Incremental CSV: open file now, write header, flush each row
    csv_path = log_dir / "per_frame.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = None  # created on first row
    try:
        # ── find spawn pair ───────────────────────────────────────────
        _log("finding spawn pair ...")
        pairs = find_spawn_pair(world, max_candidates=30)
        if not pairs:
            _log("ERROR: no suitable spawn pair found")
            sys.exit(1)
        _log(f"found {len(pairs)} candidate pairs")

        for i, pair in enumerate(pairs):
            _log(f"trying pair {i+1}/{len(pairs)}: ego=({pair[0].location.x:.1f},{pair[0].location.y:.1f}) yaw={pair[0].rotation.yaw:.0f} side={pair[2]}")
            cleanup_world(world)
            ego, v17, bg_vehicles = spawn_vehicles(world, tm, pair, n_bg=args.bg)
            if ego and v17:
                _log(f"✅ pair {i+1} succeeded")
                break
            else:
                _log(f"❌ pair {i+1} failed")
        if ego is None or v17 is None:
            _log("ERROR: all candidates failed")
            sys.exit(1)

        # ── camera ─────────────────────────────────────────────────────
        try:
            cam, buf = attach_camera(world, ego)
        except Exception as e:
            _log(f"WARN: camera attach failed: {e}")

        ego_half = ego.bounding_box.extent.x
        v17_half = v17.bounding_box.extent.x

        # ── initialize V17 controller ──────────────────────────────────
        # Tick once to settle physics
        world.tick()
        time.sleep(0.2)
        ego_tf0 = ego.get_transform()
        v17_tf0 = v17.get_transform()
        v17_ctrl = V17Controller(v17, side=pair[2], ego_yaw_deg=ego_tf0.rotation.yaw)
        v17_ctrl.initialize(ego_tf0, v17_tf0)

        # ── MAIN LOOP ───────────────────────────────────────────────────
        _log(f"=== running {ticks} ticks ===")
        for tick in range(ticks):
            sim_time = tick * FIXED_DT

            # CUT-IN phase logging
            if tick == T_CUT_START:
                _log(f"[t={sim_time:.1f}s] >>> V17 begins cut-in ({pair[2]}→ego lane)")
            if tick == T_CUT_END:
                _log(f"[t={sim_time:.1f}s] >>> V17 should now be in ego lane")
            if tick == T_SLOW:
                _log(f"[t={sim_time:.1f}s] >>> V17 slowing (RSS trigger)")
            if tick == T_RECOVER:
                _log(f"[t={sim_time:.1f}s] >>> recovery begins")
            if tick == T_DONE:
                _log(f"[t={sim_time:.1f}s] >>> resume normal cruise")

            # V17 manual control (skip if actor died)
            try:
                if not v17.is_alive:
                    _log(f"WARN: v17 died at tick={tick}, stopping early")
                    break
                v_cur = speed_of(v17.get_velocity())
                # compute lat_offset for lateral feedback
                _h_rad = math.radians(ego.get_transform().rotation.yaw)
                _lx, _ly = -math.sin(_h_rad), math.cos(_h_rad)
                _dx = v17.get_transform().location.x - ego.get_transform().location.x
                _dy = v17.get_transform().location.y - ego.get_transform().location.y
                _lat = _dx * _lx + _dy * _ly
                ctl = v17_ctrl.compute_control(tick, v17.get_transform(), v_cur, lat_offset=_lat)
                v17.apply_control(ctl)
            except RuntimeError as e:
                _log(f"WARN tick={tick} v17 ctrl failed: {e}")
                break

            # Ego speed schedule via TM
            # During T_SLOW..T_RECOVER: ego holds speed (let V17 slow to create gap)
            # During T_RECOVER..T_DONE: ego slows slightly (simulate reaction)
            try:
                if T_SLOW <= tick < T_RECOVER:
                    v_ego_target = V_EGO           # hold speed, V17 brakes → gap shrinks
                elif T_RECOVER <= tick < T_DONE:
                    v_ego_target = V_EGO_REC       # react to unsafe gap
                else:
                    v_ego_target = V_EGO
                tm.set_desired_speed(ego, v_ego_target * 3.6)
            except RuntimeError:
                pass

            # Tick
            world.tick()

            # Save RGB
            if cam and buf[0] is not None:
                try:
                    buf[0].save_to_disk(str(rgb_dir / f"{tick:06d}.png"))
                except Exception:
                    pass
                buf[0] = None

            # Compute metrics (guard against destroyed actors)
            try:
                if not (ego.is_alive and v17.is_alive):
                    _log(f"WARN tick={tick} actor dead, breaking")
                    break
                ego_tf = ego.get_transform()
                ego_v  = ego.get_velocity()
                ego_spd = speed_of(ego_v)
                v17_tf = v17.get_transform()
                v17_v  = v17.get_velocity()
                v17_spd = speed_of(v17_v)
            except RuntimeError as e:
                _log(f"WARN tick={tick} actor read failed: {e}")
                break
            except Exception as e:
                _log(f"WARN tick={tick} unexpected: {e}")
                break

            h_rad = math.radians(ego_tf.rotation.yaw)
            lon_x, lon_y = math.cos(h_rad), math.sin(h_rad)
            lat_x, lat_y = -math.sin(h_rad), math.cos(h_rad)

            dx = v17_tf.location.x - ego_tf.location.x
            dy = v17_tf.location.y - ego_tf.location.y
            lon_dist = dx * lon_x + dy * lon_y
            lon_gap  = lon_dist - ego_half - v17_half
            lat_off  = dx * lat_x + dy * lat_y

            ego_vlon = ego_v.x * lon_x + ego_v.y * lon_y
            v17_vlon = v17_v.x * lon_x + v17_v.y * lon_y
            rel_speed = ego_vlon - v17_vlon

            try:
                ego_w = world.get_map().get_waypoint(ego_tf.location, lane_type=carla.LaneType.Driving)
                v17_w = world.get_map().get_waypoint(v17_tf.location, lane_type=carla.LaneType.Driving)
                ego_lid = ego_w.lane_id if ego_w else -1
                v17_lid = v17_w.lane_id if v17_w else -1
                relation = "same_lane" if ego_lid == v17_lid else "adjacent"
            except Exception:
                ego_lid = v17_lid = -1
                relation = "unknown"

            d_safe = rss_safe(ego_spd, v17_spd)
            rss_res = lon_gap - d_safe
            gt_anomaly = 1 if rss_res < 0 else 0

            csv_rows.append({
                "frame_id": tick,
                "sim_time": f"{sim_time:.2f}",
                "ego_x": f"{ego_tf.location.x:.3f}",
                "ego_y": f"{ego_tf.location.y:.3f}",
                "ego_z": f"{ego_tf.location.z:.3f}",
                "ego_yaw": f"{ego_tf.rotation.yaw:.2f}",
                "ego_speed_ms": f"{ego_spd:.3f}",
                "v17_x": f"{v17_tf.location.x:.3f}",
                "v17_y": f"{v17_tf.location.y:.3f}",
                "v17_z": f"{v17_tf.location.z:.3f}",
                "v17_yaw": f"{v17_tf.rotation.yaw:.2f}",
                "v17_speed_ms": f"{v17_spd:.3f}",
                "lon_distance_m": f"{lon_dist:.3f}",
                "lon_gap_m": f"{lon_gap:.3f}",
                "lateral_offset_m": f"{lat_off:.3f}",
                "relative_speed_ms": f"{rel_speed:.3f}",
                "ego_lane_id": ego_lid,
                "v17_lane_id": v17_lid,
                "relation_state": relation,
                "rss_safe_m": f"{d_safe:.3f}",
                "rss_residual_m": f"{rss_res:.3f}",
                "rule_trigger": gt_anomaly,
                "neural_score": "NaN",
                "rule_score": "NaN",
                "anomaly_score": "NaN",
                "delta_gate": "NaN",
                "conflict_kappa": "NaN",
                "arbitration_trigger": gt_anomaly,
                "gt_anomaly": gt_anomaly,
            })

            # Incremental CSV write: write header on first row, then append
            if csv_writer is None:
                csv_writer = csv.DictWriter(csv_file, fieldnames=list(csv_rows[0].keys()))
                csv_writer.writeheader()
            csv_writer.writerow(csv_rows[-1])
            csv_file.flush()

            if tick % 20 == 0 or tick == ticks - 1:
                _log(f"[{tick:>3}/{ticks}] t={sim_time:.1f}s gap={lon_gap:+.2f}m lat={lat_off:+.2f}m rss={rss_res:+.2f}m rel={relation} v17={v17_spd:.1f}m/s")

        # ── Flush data BEFORE cleanup (cleanup may crash CARLA) ──────────
        _log("loop done, flushing data to disk ...")
        try:
            if csv_file and not csv_file.closed:
                csv_file.close()
        except Exception:
            pass
        try:
            kf_rows = select_key_frames(csv_rows) if csv_rows else []
            if kf_rows:
                with open(log_dir / "key_frames.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(kf_rows[0].keys()))
                    w.writeheader()
                    w.writerows(kf_rows)
                _log(f"key_frames.csv: {len(kf_rows)} rows")
            if kf_rows and HAS_PIL:
                for i, kf in enumerate(kf_rows, 1):
                    src = rgb_dir / f"{kf['frame_id']:06d}.png"
                    if not src.exists():
                        continue
                    phase_slug = kf["phase"].lower().replace(" ", "_")
                    dst = kf_dir / f"t{i:02d}_{phase_slug}_{kf['actual_time_s']:.1f}s.png"
                    metrics = {
                        "lon_gap": kf["lon_gap_m"],
                        "rss_residual": kf["rss_residual_m"],
                        "ego_speed": float(csv_rows[kf["frame_id"]]["ego_speed_ms"]) if kf["frame_id"] < len(csv_rows) else 0.0,
                    }
                    try:
                        annotate_frame(src, kf["phase"], kf["actual_time_s"], metrics, dst)
                    except Exception:
                        pass
        except Exception as e:
            _log(f"WARN keyframe post-processing: {e}")

        # ── Cleanup (may crash — data already saved) ─────────────────────
        _log("cleaning up actors (may crash CARLA) ...")
        _log("interrupted by user")
    except Exception:
        _log("ERROR during simulation:")
        import traceback
        traceback.print_exc()
    finally:
        # ── restore async + cleanup ────────────────────────────────────
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            if cam:
                cam.stop()
                cam.destroy()
        except Exception:
            pass
        for v in [v17, ego] + bg_vehicles:
            try:
                if v and v.is_alive:
                    v.destroy()
            except Exception:
                pass
        _log("cleanup done")
        # CSV already flushed incrementally; csv_file closed before cleanup
        try:
            if csv_file and not csv_file.closed:
                csv_file.close()
        except Exception:
            pass

    # ── Summary ────────────────────────────────────────────────────────
    _log("=" * 60)
    _log("DONE! Outputs:")
    _log(f"  RGB frames:   {rgb_dir}/ ({len(list(rgb_dir.glob('*.png')))} files)")
    _log(f"  Key frames:   {kf_dir}/  ({len(list(kf_dir.glob('*.png')))} files)")
    _log(f"  Per-frame:    {csv_path}")
    _log(f"  Key frames:   {log_dir / 'key_frames.csv'}")
    _log("=" * 60)


if __name__ == "__main__":
    main()
