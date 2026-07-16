"""
scripts/render_kg_dashboard_gif.py  —  14 场景 KG 仪表板 GIF 生成

用法:
    python scripts/render_kg_dashboard_gif.py --out data/rendered/kg_dashboard
    python scripts/render_kg_dashboard_gif.py --ids S00 S02 --out data/rendered/kg_dashboard
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import List, Optional

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pygame

from stk.viz.kg_dashboard import (
    init_pygame,
    graph_from_json,
    risk_from_dict,
    render_dashboard,
    draw_text,
    _font,
)


SCENARIO_LABELS = {
    "S00": "S00 直行跟车", "S01": "S01 信号灯交叉", "S02": "S02 行人横穿",
    "S10": "S10 急停跟车", "S11": "S11 强行并线", "S12": "S12 红灯右转",
    "S13": "S13 逆向超车", "S20": "S20 无信号T型", "S21": "S21 多车道汇入",
    "S22": "S22 紧急车辆", "S30": "S30 夜间", "S31": "S31 暴雨雾天",
    "S32": "S32 大雾", "S33": "S33 行人密集",
}


def render_scenario_to_gif(
    scenario_id: str,
    replay_root: Path,
    out_dir: Path,
    canvas_size=(1200, 800),
    fps: int = 4,
    duration_ms: int = 250,
    loop: int = 0,
    headless: bool = True,
) -> int:
    """渲染单个场景所有帧 → GIF. 返回帧数."""
    scene_dir = replay_root / scenario_id
    if not scene_dir.exists():
        print(f"[err] {scenario_id}: {scene_dir} not found")
        return 0
    paths = sorted(scene_dir.glob("scene_graph_*.json"))
    if not paths:
        print(f"[err] {scenario_id}: no JSON files in {scene_dir}")
        return 0

    screen = init_pygame(canvas_size[0], canvas_size[1],
                         f"KG Dashboard - {scenario_id}", headless=headless)
    clock = pygame.time.Clock()
    history: List[str] = []
    pil_frames = []

    for idx, path in enumerate(paths):
        graph, risk_dict = graph_from_json(path)
        risk = risk_from_dict(risk_dict)
        history.append(risk.level)
        render_dashboard(screen, graph, risk, idx, clock.get_fps(), history)
        pygame.display.flip()

        # pygame surface → PIL Image
        raw = pygame.image.tostring(screen, "RGBA", False)
        from PIL import Image
        img = Image.frombytes("RGBA", canvas_size, raw).convert("RGB")
        pil_frames.append(img)
        clock.tick(fps)

    pygame.quit()

    if not pil_frames:
        return 0
    out_path = out_dir / f"{scenario_id}_kg_dashboard.gif"
    out_dir.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        out_path, save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=loop, optimize=True, disposal=2,
    )
    return len(pil_frames)


def render_scenario_static_pngs(
    scenario_id: str,
    replay_root: Path,
    out_dir: Path,
    canvas_size=(1200, 800),
    headless: bool = True,
) -> int:
    """导出每帧 PNG 静态图."""
    scene_dir = replay_root / scenario_id
    if not scene_dir.exists():
        return 0
    paths = sorted(scene_dir.glob("scene_graph_*.json"))
    if not paths:
        return 0
    screen = init_pygame(canvas_size[0], canvas_size[1],
                         f"KG Dashboard - {scenario_id}", headless=headless)
    clock = pygame.time.Clock()
    history: List[str] = []
    from PIL import Image
    for idx, path in enumerate(paths):
        graph, risk_dict = graph_from_json(path)
        risk = risk_from_dict(risk_dict)
        history.append(risk.level)
        render_dashboard(screen, graph, risk, idx, clock.get_fps(), history)
        pygame.display.flip()
        raw = pygame.image.tostring(screen, "RGBA", False)
        img = Image.frombytes("RGBA", canvas_size, raw).convert("RGB")
        png_dir = out_dir / "static" / scenario_id
        png_dir.mkdir(parents=True, exist_ok=True)
        img.save(png_dir / f"frame_{idx:02d}.png")
        clock.tick(10)
    pygame.quit()
    return len(paths)


def main():
    p = argparse.ArgumentParser(description="Render 14 scenarios to KG dashboard GIFs")
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--replay-dir", default="data/replay_json",
                   help="scene_graph JSON 根目录 (相对仓库)")
    p.add_argument("--out", default="data/rendered/kg_dashboard")
    p.add_argument("--size", default="1200x800")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--static-png", action="store_true",
                   help="同时输出每帧 PNG")
    args = p.parse_args()

    replay_root = _REPO / args.replay_dir
    out_dir = _REPO / args.out

    try:
        w, h = (int(x) for x in args.size.lower().split("x"))
        canvas = (w, h)
    except Exception:
        canvas = (1200, 800)

    target_ids = args.ids or sorted([
        d.name for d in replay_root.iterdir() if d.is_dir()
    ])

    results = {}
    for sid in target_ids:
        n = render_scenario_to_gif(
            sid, replay_root, out_dir, canvas, fps=args.fps,
        )
        results[sid] = n
        print(f"[gif] {sid}: {n} frames -> {out_dir}/{sid}_kg_dashboard.gif")

        if args.static_png:
            m = render_scenario_static_pngs(sid, replay_root, out_dir, canvas)
            print(f"  [png] {sid}: {m} static frames")

    print(f"\n[done] {len(results)} GIFs generated")
    for sid, n in results.items():
        print(f"  {sid}: {n} frames -> {out_dir}/{sid}_kg_dashboard.gif")


if __name__ == "__main__":
    main()
