# -*- coding: utf-8 -*-
"""Checkpoint 管理器: 断点续跑 (v3 阶段特性)."""
from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Checkpoint:
    scenario_id: str = ""
    last_frame: int = 0
    total_frames: int = 0
    phase: str = "extraction"

    def to_dict(self) -> dict:
        return asdict(self)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, scenario_id: str, checkpoint: Checkpoint):
        path = self._dir / f"{scenario_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, scenario_id: str) -> Optional[Checkpoint]:
        path = self._dir / f"{scenario_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Checkpoint(**data)

    def resume_from(self, scenario_id: str) -> int:
        cp = self.load(scenario_id)
        if cp is None:
            return 0
        return cp.last_frame + 1

    def clear(self, scenario_id: str):
        path = self._dir / f"{scenario_id}.json"
        if path.exists():
            path.unlink()