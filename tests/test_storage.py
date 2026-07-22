# -*- coding: utf-8 -*-
"""阶段六：Neo4j 存储 — connector/schema/serializer/writer/queries/replay 测试."""
from __future__ import annotations
import pytest
from stk.ontology.entity import BaseEntity
from stk.ontology.relation import BaseRelation
from stk.ontology.types import EntityType

from stk.storage.connector import Neo4jConnection, ConnectionPool
from stk.storage.schema import NODE_LABELS, RELATION_TYPES, get_schema_cypher
from stk.storage.serializer import (
    entity_to_cypher_params, relation_to_cypher_params,
    entity_merge_cypher, relation_merge_cypher,
    serialize_graph,
)
from stk.storage.writer import write_entity_batch, write_relation_batch
from stk.storage.queries import (
    time_slice_query, lifecycle_query, anomaly_trace_query,
    spatiotemporal_aggregate_query, spatiotemporal_subgraph_query,
    export_for_gnn_cypher, temporal_attr_query,
)
from stk.storage.replay import replay_violation, format_replay_output


# ——— T6.3 connector.py ———

class TestNeo4jConnection:
    def test_connect(self):
        c = Neo4jConnection(); c.connect()
        assert c.is_connected

    def test_close(self):
        c = Neo4jConnection(); c.connect(); c.close()
        assert not c.is_connected

    def test_run_returns_list(self):
        c = Neo4jConnection(); c.connect()
        r = c.run("MATCH (n) RETURN n LIMIT 1")
        assert isinstance(r, list)


class TestConnectionPool:
    def test_acquire(self):
        pool = ConnectionPool()
        conn = pool.acquire()
        assert conn.is_connected

    def test_close_all(self):
        pool = ConnectionPool()
        for _ in range(6):
            pool.acquire()
        pool.close_all()
        assert not any(c.is_connected for c in pool._conns)


# ——— T6.4 schema.py ———

class TestSchema:
    def test_node_labels_count(self):
        assert len(NODE_LABELS) >= 10

    def test_relation_types_count(self):
        assert len(RELATION_TYPES) >= 30

    def test_get_schema_cypher_contains_create(self):
        c = get_schema_cypher()
        assert "CREATE CONSTRAINT" in c
        assert "CREATE INDEX" in c

    def test_schema_has_vehicle_unique(self):
        c = get_schema_cypher()
        assert "vehicle_id_unique" in c


# ——— T6.5 serializer.py ———

class TestSerializer:
    def test_entity_to_params(self):
        e = BaseEntity(entity_id="V1", entity_type='Vehicle', attrs={'speed': 10.0})
        params = entity_to_cypher_params(e)
        assert params["entity_id"] == "V1"
        assert "speed" in params
        assert isinstance(params["speed"], float)

    def test_relation_to_params(self):
        r = BaseRelation(src_id="V1", dst_id="R1", relation_type="in_lane",
                         frame_id=1, valid_from=1)
        params = relation_to_cypher_params(r)
        assert params["frame_id"] == 1

    def test_entity_merge_cypher(self):
        c = entity_merge_cypher("Vehicle")
        assert "MERGE" in c
        assert "$entity_id" in c
        assert "$params" in c

    def test_relation_merge_cypher(self):
        c = relation_merge_cypher("in_lane")
        assert "MERGE" in c
        assert "$src_id" in c
        assert "$dst_id" in c
        assert "$frame_id" in c


# ——— T6.6 writer.py ———

class TestWriter:
    def test_write_entity_batch_empty(self):
        batches = write_entity_batch([])
        assert batches == []

    def test_write_entity_batch_one(self):
        e = BaseEntity(entity_id="V1", entity_type='Vehicle', attrs={'speed': 10.0})
        batches = write_entity_batch([e])
        assert len(batches) >= 1

    def test_write_relation_batch_empty(self):
        cyphers = write_relation_batch([])
        assert cyphers == []

    def test_write_relation_batch_one(self):
        r = BaseRelation(src_id="V1", dst_id="R1", relation_type="in_lane",
                         frame_id=1, valid_from=1)
        cyphers = write_relation_batch([r])
        assert len(cyphers) == 1
        assert "MERGE" in cyphers[0][0]


# ——— T6.7 queries.py ———

class TestQueries:
    def test_time_slice(self):
        q = time_slice_query(2048)
        assert "frame_id = 2048" in q

    def test_lifecycle(self):
        q = lifecycle_query("V123")
        assert "vehicle_id" in q.lower() or "V123" in q

    def test_anomaly_trace(self):
        q = anomaly_trace_query("sv_R13a_2048_v1_v2")
        assert "sv_R13a_2048_v1_v2" in q

    def test_spatiotemporal_aggregate(self):
        q = spatiotemporal_aggregate_query(2048, 4096)
        assert "2048" in q
        assert "avg(sv.severity)" in q

    def test_spatiotemporal_subgraph(self):
        q = spatiotemporal_subgraph_query(2048, 2070, road_id=5)
        assert "road_id: 5" in q or "road_id = 5" in q

    def test_export_for_gnn(self):
        q = export_for_gnn_cypher(2048, 2070, road_id=5)
        assert "OPTIONAL MATCH" in q

    def test_temporal_attr(self):
        q = temporal_attr_query("V1", 0, 100)
        assert "valid_from_frame" in q


# ——— T6.8 replay.py ———

class TestReplay:
    def test_replay_violation(self):
        r = replay_violation("sv_R13a_2048_v1_v2")
        assert r["sv_id"] == "sv_R13a_2048_v1_v2"
        assert "cypher" in r

    def test_format_replay_output(self):
        r = {"sv_id": "sv1", "cypher": "MATCH...", "nodes": {}, "edges": []}
        out = format_replay_output(r)
        assert "SafetyViolation" in out
        assert "sv1" in out


# ——— T6.5b serialize_graph coalesce mode (long-run 边压缩) ———

def _make_test_frames(n_frames: int = 5):
    """构造 N 帧合成数据: 2 车 + 1 行人 + 1 红绿灯 + 2 lane, 含所有 contains 边类型."""
    frames = []
    for fid in range(n_frames):
        frames.append({
            "frame_id": fid,
            "elapsed_seconds": fid * 0.05,
            "delta_seconds": 0.05,
            "vehicles": [
                {"entity_id": "v1", "location_x": 10.0 + fid, "location_y": 20.0,
                 "speed": 5.0, "lane_id": 1, "road_id": 5,
                 "current_lane_id": "road_5_lane_1"},
                {"entity_id": "v2", "location_x": 30.0 + fid, "location_y": 20.0,
                 "speed": 3.0, "lane_id": 1, "road_id": 5,
                 "current_lane_id": "road_5_lane_1"},
            ],
            "pedestrians": [
                {"entity_id": "p1", "location_x": 50.0, "location_y": 40.0}
            ],
            "traffic_lights": [{"entity_id": "tl1", "state": "green"}],
            "lanes": [{"entity_id": "road_5_lane_1"},
                      {"entity_id": "road_5_lane_2"}],
            "weather": {"cloudiness": 0.1, "precipitation": 0.0},
            "scene_rels": [
                {"src_id": f"scenario_frame_{fid}", "dst_id": "tl1",
                 "relation_type": "containsTrafficLight", "frame_id": fid},
                {"src_id": f"scenario_frame_{fid}", "dst_id": "road_5_lane_1",
                 "relation_type": "containsRoad", "frame_id": fid},
                {"src_id": f"scenario_frame_{fid}", "dst_id": "road_5_lane_2",
                 "relation_type": "containsRoad", "frame_id": fid},
                {"src_id": f"scenario_frame_{fid}", "dst_id": f"env_frame_{fid}",
                 "relation_type": "hasEnvironment", "frame_id": fid},
                {"src_id": f"env_frame_{fid}", "dst_id": f"scenario_frame_{fid}",
                 "relation_type": "weather_context", "frame_id": fid},
            ],
        })
    return frames


class TestSerializeGraphCoalesce:
    """coalesce_containment=True 时: 全局 scenario 节点 + 包含边合并 + frames 列表."""

    def test_default_mode_unchanged_creates_per_frame_scenario_nodes(self):
        """默认 (coalesce=False) 维持原行为: 每帧一个 scenario_frame_F 节点."""
        frames = _make_test_frames(5)
        g = serialize_graph(frames)  # 默认 coalesce_containment=False
        scenario_nodes = [n for n in g["nodes"] if n["type"] == "ScenarioSnapshot"]
        assert len(scenario_nodes) == 5  # 5 帧 → 5 个 scenario_frame_F

    def test_coalesce_creates_single_scenario_node(self):
        """coalesce=True: 只创建 1 个全局 scenario 节点."""
        frames = _make_test_frames(5)
        g = serialize_graph(frames, coalesce_containment=True)
        scenario_nodes = [n for n in g["nodes"] if n["type"] == "ScenarioSnapshot"]
        assert len(scenario_nodes) == 1
        assert scenario_nodes[0]["id"] == "scenario"
        # 全局 scenario 节点 first_frame/last_frame 覆盖全帧
        assert scenario_nodes[0]["first_frame"] == 0
        assert scenario_nodes[0]["last_frame"] == 4
        assert scenario_nodes[0]["attrs"]["frame_count"] == 5

    def test_coalesce_reduces_contains_edges(self):
        """coalesce=True: containsVehicle 从 N×V 条压到 V 条."""
        frames = _make_test_frames(5)  # 5 帧, 2 车
        g_old = serialize_graph(frames)
        g_new = serialize_graph(frames, coalesce_containment=True)
        from collections import Counter
        old_types = Counter(e["type"] for e in g_old["edges"])
        new_types = Counter(e["type"] for e in g_new["edges"])
        # 旧模式: 5 帧 × 2 车 = 10 条 containsVehicle
        assert old_types["containsVehicle"] == 10
        # 新模式: 2 车 = 2 条 (每车一条, frames=[0,1,2,3,4])
        assert new_types["containsVehicle"] == 2
        # 同理 containsPedestrian: 5 → 1
        assert old_types["containsPedestrian"] == 5
        assert new_types["containsPedestrian"] == 1
        # containsRoad: 5 × 2 = 10 → 2 (2 个 lane 各一条)
        assert old_types["containsRoad"] == 10
        assert new_types["containsRoad"] == 2
        # 总边数应显著下降
        assert len(g_new["edges"]) < len(g_old["edges"])

    def test_coalesce_edge_has_frames_attr(self):
        """coalesce=True: 合并的边 attrs 含 frames/frame_count/first_frame/last_frame."""
        frames = _make_test_frames(5)
        g = serialize_graph(frames, coalesce_containment=True)
        cv_edges = [e for e in g["edges"] if e["type"] == "containsVehicle"]
        assert len(cv_edges) == 2
        # 每条 containsVehicle 边都应有 frames=[0,1,2,3,4]
        for e in cv_edges:
            assert e["attrs"]["frames"] == [0, 1, 2, 3, 4]
            assert e["attrs"]["frame_count"] == 5
            assert e["first_frame"] == 0
            assert e["last_frame"] == 4

    def test_coalesce_in_lane_edge_accumulates_frames(self):
        """coalesce=True: in_lane 边的 frames 列表正确累积覆盖帧."""
        frames = _make_test_frames(5)
        g = serialize_graph(frames, coalesce_containment=True)
        in_lane_edges = [e for e in g["edges"] if e["type"] == "in_lane"]
        # 2 车 + 同 1 lane → 2 条 in_lane 边 (v1->lane, v2->lane)
        assert len(in_lane_edges) == 2
        for e in in_lane_edges:
            assert e["attrs"]["frames"] == [0, 1, 2, 3, 4]
            assert e["attrs"]["frame_count"] == 5

    def test_coalesce_drops_next_frame_edges(self):
        """coalesce=True: 没有逐帧 scenario 节点 => next_frame 时序边为 0."""
        frames = _make_test_frames(5)
        g_new = serialize_graph(frames, coalesce_containment=True)
        nf_edges = [e for e in g_new["edges"] if e["type"] == "next_frame"]
        assert len(nf_edges) == 0  # coalesce 模式不建 next_frame
        # 默认模式应有 4 条 (5 帧之间 4 个 next_frame)
        g_old = serialize_graph(frames)
        nf_old = [e for e in g_old["edges"] if e["type"] == "next_frame"]
        assert len(nf_old) == 4

    def test_coalesce_environment_still_per_frame(self):
        """coalesce=True: env_frame_F 节点应仍逐帧创建 (天气每帧都可能变)."""
        frames = _make_test_frames(5)
        g = serialize_graph(frames, coalesce_containment=True)
        env_nodes = [n for n in g["nodes"] if n["type"] == "EnvironmentSnapshot"]
        assert len(env_nodes) == 5  # 每帧一个 env_frame_F (env 仍逐帧)
        # hasEnvironment / weather_context 也应逐帧
        has_env = [e for e in g["edges"] if e["type"] == "hasEnvironment"]
        assert len(has_env) == 5