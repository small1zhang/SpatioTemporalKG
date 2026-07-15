# -*- coding: utf-8 -*-
"""阶段五：动态更新 — 增量+反向插入+版本管理+窗口聚合 全量测试."""
from __future__ import annotations
import pytest

from stk.dynamic.diff import DeltaGraph, DiffSet, compute_delta
from stk.dynamic.incremental_updater import IncrementalEngine
from stk.dynamic.snapshot_store import SnapshotStore
from stk.dynamic.event_injector import inject_violation
from stk.dynamic.version import VersionManager, AttrVersion
from stk.dynamic.time_window import TimeWindowAggregator, SummaryEvent


# ——— T1: diff.py ————

class TestDiffSet:
    def test_added(self):
        ds = DiffSet(added={"a","b"})
        assert ds.total == 2

    def test_empty(self):
        assert not DiffSet()

    def test_bool(self):
        assert bool(DiffSet()) is False
        assert bool(DiffSet(added={"x"})) is True


class TestDeltaGraph:
    def test_empty(self):
        assert DeltaGraph().is_empty

    def test_non_empty_entities(self):
        assert not DeltaGraph(delta_entities=DiffSet(added={"x"})).is_empty

    def test_repr(self):
        assert "Δg_t" in repr(DeltaGraph())


class TestComputeDelta:
    def test_first_frame_all_added(self):
        frame = {"vehicles":[{"entity_id":"v1","entity_type":"Vehicle"}],
                 "scene_rels":[],"behavior_rels":[]}
        dg = compute_delta(frame, None)
        assert ("v1","Vehicle") in dg.delta_entities.added

    def test_no_change(self):
        f = {"vehicles":[{"entity_id":"v1","entity_type":"Vehicle","speed":10}],
             "scene_rels":[],"behavior_rels":[]}
        dg = compute_delta(f, f)
        assert dg.delta_attrs == {}

    def test_attr_change(self):
        f1 = {"vehicles":[{"entity_id":"v1","entity_type":"Vehicle","speed":10}],
              "scene_rels":[],"behavior_rels":[]}
        f2 = {"vehicles":[{"entity_id":"v1","entity_type":"Vehicle","speed":12}],
              "scene_rels":[],"behavior_rels":[]}
        dg = compute_delta(f2, f1)
        assert dg.delta_attrs["v1"]["speed"] == (10, 12)


# ——— T2: incremental_updater.py ————

class TestIncrementalEngine:
    def test_no_frame(self):
        eng = IncrementalEngine()
        assert eng.n_deltas == 0

    def test_process_first_frame(self):
        eng = IncrementalEngine()
        dg = eng.process_frame({"vehicles":[],"scene_rels":[],"behavior_rels":[]})
        assert isinstance(dg, DeltaGraph)

    def test_10_frames_9_delta_changes(self):
        eng = IncrementalEngine()
        for i in range(10):
            eng.process_frame({"vehicles":[{"entity_id":"v1","entity_type":"Vehicle",
                                            "speed":10.0+i}],
                               "scene_rels":[],"behavior_rels":[]})
        assert eng.n_deltas == 10
        changes = sum(1 for d in eng.delta_history[1:] if d.delta_attrs)
        assert changes == 9

    def test_reset(self):
        eng = IncrementalEngine()
        eng.process_frame({"vehicles":[],"scene_rels":[],"behavior_rels":[]})
        eng.reset()
        assert eng.n_deltas == 0

    def test_delta_history_isolation(self):
        eng = IncrementalEngine()
        eng.process_frame({"vehicles":[],"scene_rels":[],"behavior_rels":[]})
        h = eng.delta_history
        h.clear()
        assert eng.n_deltas == 1  # protected copy


# ——— T3: snapshot_store.py ————

class TestSnapshotStore:
    def test_put_get(self):
        ss = SnapshotStore(); ss.put(42, {"speed":10}); assert ss.get(42) == {"speed":10}

    def test_get_missing(self):
        assert SnapshotStore().get(999) is None

    def test_list_frames(self):
        ss = SnapshotStore(); ss.put(5,{}); ss.put(1,{}); ss.put(3,{})
        assert ss.list_frame_ids() == [1,3,5]

    def test_count(self):
        ss = SnapshotStore(); ss.put(0,{}); ss.put(1,{}); assert ss.count() == 2

    def test_clear(self):
        ss = SnapshotStore(); ss.put(0,{}); ss.clear(); assert ss.count() == 0


# ——— T4: event_injector.py ————

class TestEventInjector:
    def test_inject_creates_all_edge_types(self):
        result = inject_violation({"violations":[],"violation_rels":[],
                                    "evidence_rels":[],"defined_by_rels":[]},
                                  {"sv_id":"SV1","rule_code":"R1",
                                   "src_id":"A","dst_id":"B","frame_id":5,
                                   "severity":0.8,"evidence_path":["ev1","ev2"]})
        assert len(result["violations"]) == 1
        assert len(result["violation_rels"]) == 1
        assert len(result["evidence_rels"]) == 2
        assert len(result["defined_by_rels"]) == 1

    def test_inject_empty_evidence(self):
        result = inject_violation({"violations":[],"violation_rels":[],
                                    "evidence_rels":[],"defined_by_rels":[]},
                                  {"sv_id":"S","rule_code":"R","src_id":"A",
                                   "dst_id":"B","frame_id":0,"severity":0.5,
                                   "evidence_path":[]})
        assert len(result["evidence_rels"]) == 0

    def test_inject_auto_creates_keys(self):
        result = inject_violation({}, {"sv_id":"S","rule_code":"R","src_id":"A",
                                        "dst_id":"B","frame_id":0,"severity":0.5,
                                        "evidence_path":[]})
        assert "violations" in result and "violation_rels" in result


# ——— T5: version.py ————

class TestVersionManager:
    def test_record_and_get_current(self):
        vm = VersionManager(); vm.record_change("V1","speed",10.0,0)
        assert vm.get_current("V1","speed") == 10.0

    def test_version_chain(self):
        vm = VersionManager()
        vm.record_change("V1","speed",10.0,0)
        vm.record_change("V1","speed",12.0,1)
        h = vm.get_history("V1","speed")
        assert len(h) == 2 and h[0].valid_to_frame == 1

    def test_threshold_suppression(self):
        vm = VersionManager(); vm._threshold = 0.5
        vm.record_change("V1","speed",10.0,0)
        vm.record_change("V1","speed",10.3,1)  # diff<0.5
        assert len(vm.get_history("V1","speed")) == 1

    def test_close_entity(self):
        vm = VersionManager(); vm.record_change("V1","speed",10.0,0)
        vm.close_entity("V1",5)
        assert vm.get_history("V1","speed")[0].valid_to_frame == 5

    def test_reset(self):
        vm = VersionManager(); vm.record_change("V1","speed",10.0,0)
        vm.reset(); assert vm.get_current("V1","speed") is None

    def test_get_current_none(self):
        assert VersionManager().get_current("V1","speed") is None


# ——— T6: time_window.py ————

class TestTimeWindowAggregator:
    def test_empty(self):
        assert TimeWindowAggregator().summarize().violation_count == 0

    def test_one_frame(self):
        twa = TimeWindowAggregator()
        twa.add(0, [{"frame_id":0,"severity":0.7,"rule_code":"R1","src_id":"A","dst_id":"B"}])
        s = twa.summarize(0,0)
        assert s.violation_count == 1 and s.max_severity == 0.7

    def test_ten_frames(self):
        twa = TimeWindowAggregator()
        for f in range(10):
            twa.add(f, [{"frame_id":f,"severity":0.3,"rule_code":"R1","src_id":"V1","dst_id":"V2"}])
        s = twa.summarize(0,9)
        assert s.violation_count == 10

    def test_clear(self):
        twa = TimeWindowAggregator(); twa.add(0, [{}]); twa.clear()
        assert twa.summarize().violation_count == 0

    def test_summary_to_dict(self):
        s = SummaryEvent(window_start=1,window_end=5,violation_count=2,max_severity=0.8,
                         rule_codes={"R1"},involved_actors={"A"})
        d = s.to_dict()
        assert d["window_start"] == 1 and d["violation_count"] == 2


# ——— 检查点 5: 10 帧 → 9 Δg_t ———

class TestCheckpoint5:
    def test_10_frames_9_attr_deltas(self):
        eng = IncrementalEngine()
        for i in range(10):
            eng.process_frame({"vehicles":[{"entity_id":"v1","entity_type":"Vehicle",
                                            "speed":10.0+i}],
                               "scene_rels":[],"behavior_rels":[]})
        assert len(eng.delta_history) == 10
        # 第 1~9 帧各有 1 个属性变化
        assert sum(1 for d in eng.delta_history[1:] if d.delta_attrs) == 9