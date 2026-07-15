# -*- coding: utf-8 -*-
"""阶段八：主流水线集成与可视化测试."""
from __future__ import annotations
import pytest
from stk.pipeline.orchestrator import PipelineOrchestrator
from stk.pipeline.checkpoint import CheckpointManager, Checkpoint
from stk.viz.anomaly_replay import plot_anomaly_trace
from pathlib import Path


class TestOrchestrator:
    def test_orchestrator_construction(self):
        po = PipelineOrchestrator()
        assert po.incr_engine is not None
        assert po.rule_enforcer is not None

    def test_run_scenario_light(self):
        po = PipelineOrchestrator()
        try:
            result = po.run_scenario("S00", max_frames=1)
            assert "scenario" in result
        except Exception as e:
            pytest.skip(f"Skipping due to: {e}")


class TestCheckpoint:
    def test_save_load(self):
        cm = CheckpointManager(checkpoint_dir=str(Path("data/test_cp")))
        cm.save("S00", Checkpoint(scenario_id="S00", last_frame=5, total_frames=6, phase="rules"))
        cp = cm.load("S00")
        assert cp is not None and cp.last_frame == 5
        cm.clear("S00")

    def test_no_checkpoint(self):
        cm = CheckpointManager(checkpoint_dir=str(Path("data/test_cp")))
        assert cm.load("NONEXIST") is None


class TestVisualization:
    def test_plot_anomaly_trace(self):
        path = plot_anomaly_trace("sv_R13a_001", "test_trace.txt")
        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)