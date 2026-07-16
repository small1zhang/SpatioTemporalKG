"""
scripts/render_scenario_gif.py  -  将 14 个场景渲染为 CARLA-style 鸟瞰动图 (GIF)

用法:
    # 渲染全部 14 个场景 (默认)
    python scripts/render_scenario_gif.py

    # 只渲染指定场景
    python scripts/render_scenario_gif.py --ids S00 S10 S20

    # 指定输出目录与帧率
    python scripts/render_scenario_gif.py --out data/rendered --fps 4 --size 820x520

依赖: pygame, Pillow (在 stk_render conda 环境中)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from typing import List, Optional

# 允许以脚本方式直接运行 (从仓库根目录)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pygame  # noqa: E402

from stk.scenario.scenario_library import all_scenarios, get_scenario  # noqa: E402
from stk.viz.birds_eye import (  # noqa: E402
    RenderConfig,
    render_frame_to_surface,
    surface_to_pil_image,
)


# ── 场景元信息 (用于在信息条上标注 scenario_id) ────────────────────────
SCENARIO_META = {
    "S00": "S00 直行跟车 (baseline)",
    "S01": "S01 信号灯交叉",
    "S02": "S02 行人横穿预警",
    "S10": "S10 急停跟车",
    "S11": "S11 车道强行并线",
    "S12": "S12 红灯右转",
    "S13": "S13 逆向超车",
    "S20": "S20 无信号T型路口",
    "S21": "S21 多车道汇入",
    "S22": "S22 紧急车辆优先",
    "S30": "S30 夜间低对照度",
    "S31": "S31 暴雨 + 雾天",
    "S32": "S32 大雾能见度受限",
    "S33": "S33 行人密集混合交通",
}


def _attach_scenario_id(frame, sid: str):
    """将 scenario_id 设置到 frame_data 上 (供信息条显示) ."""
    try:
        object.__setattr__(frame, "scenario_id", sid)
    except Exception:
        pass


def render_scenario_to_gif(
    scenario_id: str,
    out_path: str,
    cfg: Optional[RenderConfig] = None,
    canvas_size=(820, 520),
    fps: int = 4,
    duration_ms: int = 250,
    loop: int = 0,
) -> int:
    """渲染单个场景所有帧 → 一个 GIF. 返回总帧数."""
    if cfg is None:
        cfg = RenderConfig()
    frames = get_scenario(scenario_id)
    if not frames:
        return 0

    pil_frames = []
    for frame in frames:
        _attach_scenario_id(frame, scenario_id)
        surf = render_frame_to_surface(frame, cfg, canvas_size)
        pil_img = surface_to_pil_image(surf).convert("RGB")
        pil_frames.append(pil_img)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True,
        disposal=2,
    )
    return len(pil_frames)


def render_all_scenarios(
    out_dir: str = "data/rendered",
    canvas_size=(820, 520),
    fps: int = 4,
    ids: Optional[List[str]] = None,
    verbose: bool = True,
) -> dict:
    """批量渲染所有 (或指定) 场景. 返回 {scenario_id: out_path} dict."""
    pygame.init()
    try:
        pygame.display.set_mode((1, 1), pygame.HIDDEN)  # headless
    except Exception:
        # 某些环境无 display, 这里仅需要 font, surface 模块
        pass

    all_sc = all_scenarios()
    target_ids = ids if ids else sorted(all_sc.keys())
    out_dir_abs = os.path.join(_REPO_ROOT, out_dir) if not os.path.isabs(out_dir) else out_dir
    os.makedirs(out_dir_abs, exist_ok=True)

    cfg = RenderConfig()
    results = {}
    duration_ms = int(1000 / max(1, fps))

    for sid in target_ids:
        if sid not in all_sc:
            if verbose:
                print(f"[skip] unknown scenario id: {sid}")
            continue
        out_path = os.path.join(out_dir_abs, f"{sid}.gif")
        n = render_scenario_to_gif(sid, out_path, cfg, canvas_size, fps, duration_ms)
        results[sid] = out_path
        if verbose:
            print(f"[ok] {sid} -> {out_path} ({n} frames)")

    pygame.quit()
    return results


def render_scenario_to_static_png(
    scenario_id: str,
    out_path: str,
    frame_index: int = 0,
    cfg: Optional[RenderConfig] = None,
    canvas_size=(820, 520),
) -> None:
    """单帧静态 PNG 截图 (便于贴在文档中不能用 GIF 的环境)"""
    pygame.init()
    try:
        pygame.display.set_mode((1, 1), pygame.HIDDEN)
    except Exception:
        pass

    frames = get_scenario(scenario_id)
    if not frames:
        return
    if frame_index < 0 or frame_index >= len(frames):
        frame_index = 0
    _attach_scenario_id(frames[frame_index], scenario_id)
    surf = render_frame_to_surface(frames[frame_index], cfg, canvas_size)
    pil_img = surface_to_pil_image(surf).convert("RGB")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pil_img.save(out_path, format="PNG")
    pygame.quit()


def main():
    p = argparse.ArgumentParser(description="Render 14 scenarios to CARLA-style birds-eye GIFs")
    p.add_argument("--ids", nargs="*", default=None,
                   help="指定 scenario_id (默认全部 14 个)")
    p.add_argument("--out", default="data/rendered",
                   help="输出目录 (相对仓库根目录)")
    p.add_argument("--size", default="820x520",
                   help="画布大小, 格式 WxH, 默认 820x520")
    p.add_argument("--fps", type=int, default=4,
                   help="GIF 帧率, 默认 4")
    p.add_argument("--static-png", action="store_true",
                   help="除 GIF 外, 同时输出每场景首帧 PNG (便于文档贴图)")
    args = p.parse_args()

    try:
        w, h = (int(x) for x in args.size.lower().split("x"))
        canvas_size = (w, h)
    except Exception:
        canvas_size = (820, 520)

    results = render_all_scenarios(
        out_dir=args.out, canvas_size=canvas_size, fps=args.fps, ids=args.ids,
    )

    print(f"\n[done] 已生成 {len(results)} 个 GIF")
    for sid, path in results.items():
        print(f"  {sid}: {path}")

    if args.static_png:
        out_dir_abs = os.path.join(_REPO_ROOT, args.out) if not os.path.isabs(args.out) else args.out
        os.makedirs(os.path.join(out_dir_abs, "static"), exist_ok=True)
        for sid in results.keys():
            png_path = os.path.join(out_dir_abs, "static", f"{sid}_frame00.png")
            render_scenario_to_static_png(
                sid, png_path, frame_index=0, canvas_size=canvas_size,
            )
            print(f"  [png] {sid}: {png_path}")


if __name__ == "__main__":
    main()
