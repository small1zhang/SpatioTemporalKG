"""
stk/viz/birds_eye.py  -  CARLA-style 鸟瞰图渲染器

使用 Pygame 将 FrameData 渲染为仿 CARLA 合成图像。
由 scripts/render_scenario_gif.py 驱动，独立于 Matplotlib。

依赖: pygame, Pillow (在独立的 stk_render conda 环境中)
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pygame

# ── 色彩常量 (CARLA sensor 风格) ────────────────────────────────────────
COLOR_ASPHALT      = ( 60,  60,  65)    # 路面底色
COLOR_ROAD_EDGE    = (210, 210, 210)    # 车道边线
COLOR_LANE_DASH    = (255, 255, 240)    # 车道虚线
COLOR_VEHICLE_BODY = ( 30, 130, 210)    # 车身蓝
COLOR_VEHICLE_HOOD = (220, 220, 240)    # 车头方向指示
COLOR_PEDESTRIAN   = (255, 200,  50)    # 行人黄
COLOR_TL_GREEN     = ( 50, 210,  50)
COLOR_TL_YELLOW    = (240, 210,  30)
COLOR_TL_RED       = (210,  40,  40)
COLOR_TL_OFF       = ( 80,  80,  80)
COLOR_BG           = ( 38,  40,  46)
COLOR_TEXT_INFO     = (200, 220, 240)
COLOR_FOG_OVERLAY  = (200, 210, 220)
INFO_BAR_BG        = ( 18,  20,  26)

# ── 渲染参数 ────────────────────────────────────────────────────────────
PIXELS_PER_METER  = 4.0
LANE_HALF_WIDTH_M = 2.0
LANE_DASH_M       = 1.5
LANE_GAP_M        = 2.0
VEHICLE_L_M       = 4.8
VEHICLE_W_M       = 2.0
PED_RADIUS_M      = 0.45
TL_RADIUS_M       = 0.45
MARGIN            = 60
INFO_BAR_HEIGHT   = 32


@dataclass
class RenderConfig:
    """渲染时可调的配置参数"""
    scale: float = 1.0
    show_heading_arrow: bool = True
    show_grid: bool = True
    info_bar: bool = True
    weather_effect: bool = True
    random_seed_for_weather: int = 42


# ── 坐标转换 ────────────────────────────────────────────────────────────


def _w2s(wx: float, wy: float,
         offset_x: float, offset_y: float,
         scale: float = 1.0) -> Tuple[int, int]:
    """世界坐标 (x 前, y 左) → 屏幕坐标 (右 +x, 下 +y)"""
    sx = (wx - offset_x) * PIXELS_PER_METER * scale + MARGIN
    sy = -(wy - offset_y) * PIXELS_PER_METER * scale + MARGIN
    return int(round(sx)), int(round(sy))


def _rotated_rect(cx: float, cy: float,
                  w: float, h: float, angle_rad: float) -> List[Tuple[float, float]]:
    """旋转矩形四个顶点 (屏幕坐标)"""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    half_w, half_h = w / 2, h / 2
    corners = [(-half_w, -half_h), (half_w, -half_h),
               (half_w, half_h), (-half_w, half_h)]
    out = []
    for lx, ly in corners:
        rx = cx + lx * cos_a - ly * sin_a
        ry = cy + lx * sin_a + ly * cos_a
        out.append((rx, ry))
    return out


def _draw_text(surface: pygame.Surface, text: str, x: int, y: int,
               color: Tuple[int, int, int] = COLOR_TEXT_INFO,
               size: int = 14, bold: bool = False):
    """自带阴影文字"""
    font = pygame.font.SysFont("consolas", size, bold=bold)
    shadow = font.render(text, True, (10, 10, 10))
    surface.blit(shadow, (x + 1, y + 1))
    label = font.render(text, True, color)
    surface.blit(label, (x, y))


def _draw_grid(surface: pygame.Surface,
               offset_x: float, offset_y: float,
               W: int, H: int, cfg: RenderConfig):
    """绘制 10 米参考网格"""
    spacing_m = 10
    color = (52, 55, 62)
    px_step = spacing_m * PIXELS_PER_METER * cfg.scale

    # 计算世界 x 范围
    x_min_w = offset_x - 30
    x_max_w = offset_x + (W - 2 * MARGIN) / (PIXELS_PER_METER * cfg.scale) + 30
    gx = math.floor(x_min_w / spacing_m) * spacing_m
    while gx <= x_max_w:
        p1 = _w2s(gx, offset_y - 60, offset_x, offset_y, cfg.scale)
        p2 = _w2s(gx, offset_y + (H - 2 * MARGIN) /
                  (PIXELS_PER_METER * cfg.scale) + 60, offset_x, offset_y, cfg.scale)
        pygame.draw.line(surface, color, p1, p2, 1)
        gx += spacing_m

    y_min_w = offset_y - 60
    y_max_w = offset_y + (H - 2 * MARGIN) / (PIXELS_PER_METER * cfg.scale) + 60
    gy = math.floor(y_min_w / spacing_m) * spacing_m
    while gy <= y_max_w:
        p1 = _w2s(offset_x - 60, gy, offset_x, offset_y, cfg.scale)
        p2 = _w2s(offset_x + (W - 2 * MARGIN) /
                  (PIXELS_PER_METER * cfg.scale) + 60, gy, offset_x, offset_y, cfg.scale)
        pygame.draw.line(surface, color, p1, p2, 1)
        gy += spacing_m


def _draw_lane(surface: pygame.Surface, lane: Dict,
               offset_x: float, offset_y: float, cfg: RenderConfig):
    """绘制单条车道: 路面填充 + 边线 + 中心虚线"""
    cx, cy = lane["center_x"], lane["center_y"]
    length = lane.get("length", 100.0)
    speed_limit = lane.get("speed_limit", 15.0)

    half_len = length / 2
    half_w = LANE_HALF_WIDTH_M

    p1 = _w2s(cx - half_len, cy - half_w, offset_x, offset_y, cfg.scale)
    p2 = _w2s(cx + half_len, cy - half_w, offset_x, offset_y, cfg.scale)
    p3 = _w2s(cx + half_len, cy + half_w, offset_x, offset_y, cfg.scale)
    p4 = _w2s(cx - half_len, cy + half_w, offset_x, offset_y, cfg.scale)

    pygame.draw.polygon(surface, COLOR_ASPHALT, [p1, p2, p3, p4])
    pygame.draw.line(surface, COLOR_ROAD_EDGE, p1, p2, max(2, int(3 * cfg.scale)))
    pygame.draw.line(surface, COLOR_ROAD_EDGE, p3, p4, max(2, int(3 * cfg.scale)))

    # 中心虚线
    x = cx - half_len
    while x < cx + half_len:
        x_end = min(x + LANE_DASH_M, cx + half_len)
        a = _w2s(x, cy, offset_x, offset_y, cfg.scale)
        b = _w2s(x_end, cy, offset_x, offset_y, cfg.scale)
        pygame.draw.line(surface, COLOR_LANE_DASH, a, b, max(1, int(2 * cfg.scale)))
        x = x_end + LANE_GAP_M

    # 标注速度
    if speed_limit > 0:
        sp = _w2s(cx + half_len + 1.5, cy, offset_x, offset_y, cfg.scale)
        _draw_text(surface, f"{int(speed_limit*3.6)}", sp[0] + 4, sp[1] - 6,
                   (140, 160, 180), 12)


def _draw_vehicle(surface: pygame.Surface, vehicle: Dict,
                  offset_x: float, offset_y: float, cfg: RenderConfig):
    """绘制车辆: 旋转矩形 + 方向箭头 + 速度标签"""
    vx, vy = vehicle["location_x"], vehicle["location_y"]
    heading = vehicle.get("heading_rad", 0.0)
    speed = vehicle.get("speed", 0.0)

    cx, cy = _w2s(vx, vy, offset_x, offset_y, cfg.scale)
    w_px = VEHICLE_W_M * PIXELS_PER_METER * cfg.scale
    h_px = VEHICLE_L_M * PIXELS_PER_METER * cfg.scale

    corners = _rotated_rect(cx, cy, w_px, h_px, -heading)  # pyg y-down 反向
    pygame.draw.polygon(surface, COLOR_VEHICLE_BODY, corners)
    pygame.draw.polygon(surface, (12, 60, 120), corners, max(1, int(2 * cfg.scale)))

    if cfg.show_heading_arrow:
        # 长方车头亮条 + 三角箭头
        bar_w = w_px * 0.6
        bar_h = h_px * 0.18
        bx = cx + (h_px / 2 - bar_h / 2) * math.cos(-heading)
        by = cy + (h_px / 2 - bar_h / 2) * math.sin(-heading)
        bar_corners = _rotated_rect(bx, by, bar_w, bar_h, -heading)
        pygame.draw.polygon(surface, COLOR_VEHICLE_HOOD, bar_corners)

    if speed * 3.6 >= 1:
        _draw_text(surface, f"{speed*3.6:.0f}",
                   cx - 10, int(cy + h_px * 0.42),
                   (200, 230, 250), 11)


def _draw_pedestrian(surface: pygame.Surface, ped: Dict,
                     offset_x: float, offset_y: float, cfg: RenderConfig):
    px, py = ped["location_x"], ped["location_y"]
    cx, cy = _w2s(px, py, offset_x, offset_y, cfg.scale)
    r = PED_RADIUS_M * PIXELS_PER_METER * cfg.scale
    pygame.draw.circle(surface, COLOR_PEDESTRIAN, (cx, cy), max(3, int(r)))
    pygame.draw.circle(surface, (160, 100, 0), (cx, cy), max(2, int(r)), 1)
    action = ped.get("action", "")
    if action:
        _draw_text(surface, str(action)[:6],
                   cx + max(5, int(r)), cy - 6, (220, 200, 100), 11)


def _draw_traffic_light(surface: pygame.Surface, tl: Dict,
                        offset_x: float, offset_y: float, cfg: RenderConfig):
    tx = tl.get("position_x", 0)
    ty = tl.get("position_y", 0)
    state = tl.get("state", "Green")
    cx, cy = _w2s(tx, ty, offset_x, offset_y, cfg.scale)
    r = TL_RADIUS_M * PIXELS_PER_METER * cfg.scale

    color_map = {"Green": COLOR_TL_GREEN, "Yellow": COLOR_TL_YELLOW,
                 "Red": COLOR_TL_RED, "Off": COLOR_TL_OFF}
    color = color_map.get(state, COLOR_TL_OFF)

    # 光晕
    glow_r = int(r * 2.5)
    glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
    for i in range(glow_r, 0, -1):
        alpha = max(0, 70 - i * 2)
        pygame.draw.circle(glow_surf, (*color, alpha), (glow_r, glow_r), i)
    surface.blit(glow_surf, (cx - glow_r, cy - glow_r))
    pygame.draw.circle(surface, color, (cx, cy), max(3, int(r)))
    pygame.draw.circle(surface, (20, 20, 20), (cx, cy), max(3, int(r)), 1)
    _draw_text(surface, state[:3], cx + max(6, int(r)),
               cy - 5, (220, 220, 220), 11)


def _draw_weather_overlay(surface: pygame.Surface, weather: Dict,
                          W: int, H: int, cfg: RenderConfig):
    """叠加天气效果: 雾层 / 雨 / 夜色 / 阴云"""
    fog = weather.get("fog_density", 0)
    precipitation = weather.get("precipitation", 0)
    sun_alt = weather.get("sun_altitude_angle", 60)
    cloudiness = weather.get("cloudiness", 0)

    if fog > 0:
        alpha = min(160, int(fog * 2.5))
        fog_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        fog_surf.fill((*COLOR_FOG_OVERLAY, alpha))
        surface.blit(fog_surf, (0, 0))

    if precipitation > 20:
        alpha = min(70, int(precipitation / 4))
        rng = random.Random(cfg.random_seed_for_weather)
        rain_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        for _ in range(int(precipitation * 3)):
            rx = rng.randint(0, W)
            ry = rng.randint(0, H)
            rl = rng.randint(4, 14)
            pygame.draw.line(rain_surf, (160, 190, 220, alpha),
                             (rx, ry), (rx - 2, ry + rl), 1)
        surface.blit(rain_surf, (0, 0))

    # 夜色
    brightness = (sun_alt + 10) / 90
    if brightness < 0.5:
        dark_alpha = int((0.5 - brightness) * 140)
        dark_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        dark_surf.fill((10, 10, 20, dark_alpha))
        surface.blit(dark_surf, (0, 0))

    if cloudiness > 30:
        alpha = min(50, int(cloudiness * 0.5))
        cloud_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        cloud_surf.fill((100, 100, 110, alpha))
        surface.blit(cloud_surf, (0, 0))


def render_frame_to_surface(frame_data,
                            cfg: Optional[RenderConfig] = None,
                            canvas_size: Tuple[int, int] = (820, 520)) -> pygame.Surface:
    """渲染单帧 FrameData → pygame.Surface (RGBA)"""
    if cfg is None:
        cfg = RenderConfig()
    W, H = canvas_size
    surface = pygame.Surface((W, H), pygame.SRCALPHA)
    surface.fill(COLOR_BG)

    # 收集所有元素 x/y 范围, 计算居中
    xs = ([v["location_x"] for v in frame_data.vehicles]
          + [p["location_x"] for p in frame_data.pedestrians]
          + [l["center_x"] for l in frame_data.lanes]
          + [tl.get("position_x", 0) for tl in frame_data.traffic_lights])
    ys = ([v["location_y"] for v in frame_data.vehicles]
          + [p["location_y"] for p in frame_data.pedestrians]
          + [l["center_y"] for l in frame_data.lanes]
          + [tl.get("position_y", 0) for tl in frame_data.traffic_lights])
    cx = (min(xs) + max(xs)) / 2 if xs else 0.0
    cy = (min(ys) + max(ys)) / 2 if ys else 0.0

    vw = (W - 2 * MARGIN) / (PIXELS_PER_METER * cfg.scale)
    vh = (H - 2 * MARGIN) / (PIXELS_PER_METER * cfg.scale)
    offset_x = cx - vw / 2
    offset_y = cy - vh / 2

    if cfg.show_grid:
        _draw_grid(surface, offset_x, offset_y, W, H, cfg)

    # 1) 车道 (底层)
    for lane in frame_data.lanes:
        _draw_lane(surface, lane, offset_x, offset_y, cfg)
    # 2) 信号灯
    for tl in frame_data.traffic_lights:
        _draw_traffic_light(surface, tl, offset_x, offset_y, cfg)
    # 3) 车辆
    for v in frame_data.vehicles:
        _draw_vehicle(surface, v, offset_x, offset_y, cfg)
    # 4) 行人
    for p in frame_data.pedestrians:
        _draw_pedestrian(surface, p, offset_x, offset_y, cfg)
    # 5) 天气
    if cfg.weather_effect and getattr(frame_data, 'weather', None):
        _draw_weather_overlay(surface, frame_data.weather, W, H, cfg)

    # 6) 底部信息条
    if cfg.info_bar:
        bar_surf = pygame.Surface((W, INFO_BAR_HEIGHT), pygame.SRCALPHA)
        bar_surf.fill((*INFO_BAR_BG, 210))
        surface.blit(bar_surf, (0, H - INFO_BAR_HEIGHT))

        sid = (getattr(frame_data, 'scenario_id', None)
               or getattr(frame_data, 'scene_id', None) or '')
        map_name = getattr(frame_data, 'map_name', 'Town01')
        frame_id = getattr(frame_data, 'frame_id', '?')
        elapsed = getattr(frame_data, 'elapsed_seconds', 0)

        parts = []
        if sid:
            parts.append(str(sid))
        if map_name:
            parts.append(str(map_name))
        parts.append(f"Frame {frame_id}")
        parts.append(f"t={elapsed:.1f}s")
        parts.append(f"V={len(frame_data.vehicles)}")
        parts.append(f"P={len(frame_data.pedestrians)}")
        parts.append(f"TL={len(frame_data.traffic_lights)}")
        # 天气摘要
        if frame_data.weather:
            w = frame_data.weather
            wx_parts = []
            if w.get('fog_density', 0) > 0:
                wx_parts.append(f"fog={w['fog_density']}")
            if w.get('precipitation', 0) > 0:
                wx_parts.append(f"rain={w['precipitation']}")
            if w.get('sun_altitude_angle', 60) < 0:
                wx_parts.append("night")
            if wx_parts:
                parts.append(" ".join(wx_parts))

        _draw_text(surface, "  |  ".join(parts),
                   12, H - INFO_BAR_HEIGHT + 6,
                   (210, 230, 250), 16, bold=True)

    return surface


def surface_to_pil_image(surface: pygame.Surface) -> "PIL.Image.Image":
    """将 pygame Surface 转 PIL Image (RGBA)"""
    from PIL import Image
    # 取像素 buffer
    raw = pygame.image.tostring(surface, "RGBA", False)
    w, h = surface.get_size()
    return Image.frombytes("RGBA", (w, h), raw)


def render_frame_to_png_bytes(frame_data,
                              cfg: Optional[RenderConfig] = None,
                              canvas_size: Tuple[int, int] = (820, 520)) -> bytes:
    """渲染帧 → PNG bytes (供 PIL Image.open 读取)"""
    surface = render_frame_to_surface(frame_data, cfg, canvas_size)
    img = surface_to_pil_image(surface)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
