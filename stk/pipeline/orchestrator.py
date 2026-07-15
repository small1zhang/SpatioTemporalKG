# -*- coding: utf-8 -*-
"""Pipeline 编排器: 串联场景→行为→规则→动态 → 存储 (v3 全书)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from stk.extraction.pipeline import process_frame as extract
from stk.scenario.scenario_library import get_scenario, all_scenarios
from stk.dynamic.incremental_updater import IncrementalEngine
from stk.dynamic.snapshot_store import SnapshotStore
from stk.rules.generator import RuleEnforcer


class PipelineOrchestrator:
    def __init__(self, storage_connector=None):
        self.scenario_store: Dict[str, List] = {}
        self.incr_engine = IncrementalEngine()
        self.snapshot_store = SnapshotStore()
        self.rule_enforcer = RuleEnforcer()
        self._storage = storage_connector
        self._results: List[Dict] = []

    def run_scenario(self, scenario_id: str, max_frames: int = 6) -> Dict[str, Any]:
        """运行单场景的完整 pipeline."""
        frames = get_scenario(scenario_id)[:max_frames]
        results = []
        for f in frames:
            # 提取帧
            raw = {"frame_id": f.frame_id, "actors": [], "waypoints": [],
                   "traffic_lights": [], "events": [], "weather": {},
                   "elapsed_seconds": f.elapsed_seconds}
            # 收集 vehicles / pedestrians
            for v in f.vehicles:
                raw["actors"].append({**v, "type": "vehicle." + v.get("vehicle_type","car")})
            for p in f.pedestrians:
                raw["actors"].append({**p, "type": "walker.pedestrian"})
            # 场景层提取
            extracted = extract(raw)
            # 增量更新
            delta = self.incr_engine.process_frame(extracted)
            # 规则推理
            rule_out = self.rule_enforcer.enforce(
                frame_id=extracted["frame_id"],
                vehicles=extracted.get("vehicles", []),
                pedestrians=extracted.get("pedestrians", []),
                traffic_lights=extracted.get("traffic_lights", []),
                scene_rels=extracted.get("scene_rels", []),
            )
            # 存储快照
            self.snapshot_store.put(extracted["frame_id"], {
                "extracted": extracted, "delta": delta, "rule_out": rule_out,
            })
            results.append({
                "frame_id": extracted["frame_id"],
                "n_violations": len(rule_out["violations"]),
                "n_deltas": self.incr_engine.n_deltas,
            })
        self._results = results
        return {"scenario": scenario_id, "frames": len(frames), "results": results}

    def run_all_scenarios(self) -> Dict[str, Any]:
        all_results = {}
        for sid in ["S00","S01","S02","S10","S11","S12","S13",
                     "S20","S21","S22","S30","S31","S32","S33"]:
            all_results[sid] = self.run_scenario(sid)
        return all_results