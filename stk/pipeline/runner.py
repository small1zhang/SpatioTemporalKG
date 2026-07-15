# -*- coding: utf-8 -*-
"""Pipeline runner: stk run pipeline."""
from __future__ import annotations
from typing import Optional
from stk.pipeline.orchestrator import PipelineOrchestrator
from stk.pipeline.checkpoint import CheckpointManager, Checkpoint


def run_pipeline(scenario_id: str = "all", max_frames: int = 6,
                 checkpoint: bool = False) -> dict:
    orchestrator = PipelineOrchestrator()
    if checkpoint:
        cm = CheckpointManager()
        start_frame = cm.resume_from(scenario_id)
        print(f"从帧 {start_frame} 恢复场景 {scenario_id}")
    if scenario_id == "all":
        return orchestrator.run_all_scenarios()
    return orchestrator.run_scenario(scenario_id, max_frames)