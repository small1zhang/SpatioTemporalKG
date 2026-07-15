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