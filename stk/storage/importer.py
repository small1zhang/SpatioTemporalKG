# -*- coding: utf-8 -*-
"""Phase5 JSON chunk → Neo4j 批量导入器.

消费 `serialize_graph()` 产出的 {nodes, edges} dict 格式
(即 viz_output/graph_0001_0_1999.json 等分片格式),
按全局 entity_id MERGE 写入 Neo4j.

用法 (通过 scripts/long_run/import_neo4j.py CLI):
    python -m scripts.long_run.import_neo4j \\
        --input viz_output/ \\
        --uri bolt://localhost:7687 --user neo4j --password stk123
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

from stk.storage.connector import Neo4jConnection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node type → Neo4j label 映射
# ---------------------------------------------------------------------------
NODE_LABEL_MAP: Dict[str, str] = {
    "Vehicle": "Vehicle",
    "Pedestrian": "Pedestrian",
    "TrafficLight": "TrafficLight",
    "RoadElement": "RoadElement",
    "EnvironmentSnapshot": "EnvironmentSnapshot",
    "ScenarioSnapshot": "ScenarioSnapshot",
    "Maneuver": "Maneuver",
    "InteractionEvent": "InteractionEvent",
    "BehaviorRelation": "BehaviorRelation",
    "SafetyViolation": "SafetyViolation",
    "ResponsibilityAssignment": "ResponsibilityAssignment",
    "Rule": "Rule",
    "Junction": "Junction",
}
ALL_NODE_LABELS: Set[str] = set(NODE_LABEL_MAP.values())

# 用于 src/dst MATCH 时统一拿 entity_id 的标签列表
# 所有节点都以 entity_id 作为主键
MATCH_LABELS_FOR_EDGES: List[str] = list(ALL_NODE_LABELS)

# ---------------------------------------------------------------------------
# 非 primitive 属性 → JSON string 序列化
# 这些类型的值 Neo4j 5.x 不支持作为 property, 必须转为 string.
# ---------------------------------------------------------------------------
PRIMITIVE_TYPES = (bool, int, float, str, type(None))


def _serialize_val(v: Any) -> Any:
    """将 Python 值序列化为 Neo4j property 安全类型.

    列表 / dict → json.dumps (string);
    float NaN/Inf → None;
    其他原样返回.
    """
    import math
    if isinstance(v, PRIMITIVE_TYPES):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, float):
            return round(v, 8)
        return v
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, default=str)
    if isinstance(v, tuple):
        return json.dumps(list(v), ensure_ascii=False, default=str)
    # fallback
    try:
        return str(v)
    except Exception:
        return None


def _pack_node_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """清洗节点 attrs, 递归序列化非 primitive 值."""
    out = {}
    for k, v in attrs.items():
        if k == "attrs" and isinstance(v, dict):
            # Maneuver / InteractionEvent 的嵌套 attrs.attrs  → 平铺为 attrs_* 前缀
            for kk, vv in v.items():
                prefixed = f"attrs_{kk}"
                out[prefixed] = _serialize_val(vv)
        else:
            out[k] = _serialize_val(v)
    return out


def _pack_edge_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """清洗边 attrs."""
    out = {}
    for k, v in attrs.items():
        if k in ("frames", "fired_frames") and isinstance(v, list):
            # 大的列表存储为 JSON string (也可考虑 gRPC / bulk export, 这里简单处理)
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = _serialize_val(v)
    return out


# ---------------------------------------------------------------------------
# Cypher 模板
# ---------------------------------------------------------------------------

NODE_MERGE_TPL = """\
UNWIND $batch AS row
MERGE (n:`{label}` {{entity_id: row.id}})
SET n.first_frame = CASE
    WHEN n.first_frame IS NULL THEN row.first_frame
    WHEN row.first_frame < n.first_frame THEN row.first_frame
    ELSE n.first_frame END,
    n.last_frame = CASE
    WHEN n.last_frame IS NULL THEN row.last_frame
    WHEN row.last_frame > n.last_frame THEN row.last_frame
    ELSE n.last_frame END
SET n += row.params
"""

# 边 MERGE — 按 (src_id, dst_id, type, first_frame, last_frame) 去重
# 注意: 调用方应先过滤掉 dangling 边 (src/dst 不在 node 集中), 否则 MATCH 会全表扫描
EDGE_MERGE_TPL = """\
UNWIND $batch AS row
MATCH (src {{entity_id: row.src_id}})
MATCH (dst {{entity_id: row.dst_id}})
WITH row, src, dst
MERGE (src)-[r:`{rel_type}` {{first_frame: row.first_frame, last_frame: row.last_frame}}]->(dst)
SET r.frame_id = CASE WHEN r.frame_id IS NULL THEN row.frame_id ELSE r.frame_id END
SET r += row.params
"""

# 强制创建 placeholder 节点的版本 (用于 --create-dangling)
EDGE_MERGE_TPL_WITH_PLACEHOLDER = """\
UNWIND $batch AS row
MERGE (src:UnknownRef {{entity_id: row.src_id}})
MERGE (dst:UnknownRef {{entity_id: row.dst_id}})
WITH row, src, dst
MERGE (src)-[r:`{rel_type}` {{first_frame: row.first_frame, last_frame: row.last_frame}}]->(dst)
SET r.frame_id = CASE WHEN r.frame_id IS NULL THEN row.frame_id ELSE r.frame_id END
SET r += row.params
"""

# ---------------------------------------------------------------------------
# GraphImporter
# ---------------------------------------------------------------------------


class GraphImporter:
    """从 phase5 JSON {nodes, edges} 导入 Neo4j.

    用法::

        conn = Neo4jConnection(uri, user, password, database)
        conn.connect()
        imp = GraphImporter(conn, batch_size=500)
        imp.ensure_schema()
        for graph in shard_graphs:
            imp.import_shard(graph)
        conn.close()
    """

    def __init__(
        self,
        conn: Neo4jConnection,
        batch_size: int = 500,
        create_dangling: bool = False,
    ):
        self.conn = conn
        self.batch_size = batch_size
        self.create_dangling = create_dangling
        # 统计
        self.node_count = 0
        self.edge_count = 0
        self.skipped_edges = 0
        self._node_types_seen: Set[str] = set()
        # 累积所有已导入 node id (跨 shard), 用于过滤 dangling 边
        self._known_node_ids: Set[str] = set()

    # ── schema ──

    def ensure_schema(self):
        """创建约束 + 索引.

        复用 stk.storage.schema.get_schema_cypher() 的输出,
        再补上几条面向 entity_id 的约束.
        """
        from stk.storage.schema import get_schema_cypher
        schema_cypher = get_schema_cypher()

        # 执行 DDL (每条可能已存在, 加 IF NOT EXISTS)
        for line in schema_cypher.split(";"):
            line = line.strip()
            if not line:
                continue
            try:
                self.conn.run_write(line)
            except Exception as exc:
                logger.warning("DDL 语句警告 (可能已存在): %s", exc)

        # 补充: entity_id 唯一约束 (用于 MERGE 主键)
        extra_ddl = [
            "CREATE CONSTRAINT entity_id_unique          IF NOT EXISTS FOR (n:Vehicle)               REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT ped_entity_id_unique       IF NOT EXISTS FOR (n:Pedestrian)             REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT rule_entity_id_unique      IF NOT EXISTS FOR (n:Rule)                   REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT env_frame_entity_id_unique  IF NOT EXISTS FOR (n:EnvironmentSnapshot)    REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT scenario_entity_id_unique   IF NOT EXISTS FOR (n:ScenarioSnapshot)       REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT sv_entity_id_unique         IF NOT EXISTS FOR (n:SafetyViolation)        REQUIRE n.entity_id IS UNIQUE",
            "CREATE INDEX idx_first_frame  IF NOT EXISTS FOR (n) ON (n.first_frame)",
            "CREATE INDEX idx_last_frame   IF NOT EXISTS FOR (n) ON (n.last_frame)",
        ]
        for ddl in extra_ddl:
            try:
                self.conn.run_write(ddl)
            except Exception as exc:
                logger.warning("额外 DDL 警告: %s", exc)

        logger.info("Schema 约束/索引就绪")

    # ── 节点导入 ──

    def import_shard(self, graph: Dict[str, Any]) -> None:
        """导入一个 shard 的全部节点和边."""
        nodes: List[Dict] = graph.get("nodes", [])
        edges: List[Dict] = graph.get("edges", [])
        logger.info(
            "开始导入 shard: %d 节点, %d 边, batch_size=%d",
            len(nodes), len(edges), self.batch_size,
        )

        # 1) 分组节点按 type
        by_label: Dict[str, List[Dict]] = {}
        for n in nodes:
            typ = n.get("type", "Unknown")
            label = NODE_LABEL_MAP.get(typ, "Unknown")
            self._node_types_seen.add(label)
            by_label.setdefault(label, []).append(n)
            # 累积已知 node id (用于边过滤)
            self._known_node_ids.add(n["id"])

        # 2) 逐类型 UNWIND MERGE
        for label, batch_nodes in by_label.items():
            rows = []
            for n in batch_nodes:
                params = _pack_node_attrs(n.get("attrs", {}))
                rows.append({
                    "id": n["id"],
                    "first_frame": n.get("first_frame", 0),
                    "last_frame": n.get("last_frame", 0),
                    "params": params,
                })
            cypher = NODE_MERGE_TPL.format(label=label)
            n_batches = self.conn.run_batch_unwind(
                cypher, rows, batch_size=self.batch_size,
            )
            self.node_count += len(batch_nodes)
            logger.debug(
                "  节点 %s: %d 条 (%d 批)", label, len(batch_nodes), n_batches,
            )

        logger.info("  节点完成: %d 条", self.node_count)

        # 3) 分组边按 type
        by_rel: Dict[str, List[Dict]] = {}
        for e in edges:
            rel_type = e.get("type", "unknown")
            by_rel.setdefault(rel_type, []).append(e)

        # 4) 逐类型 UNWIND MERGE (Python 端预先过滤 dangling 边)
        for rel_type, batch_edges in by_rel.items():
            rows = []
            skipped = 0
            for e in batch_edges:
                src_id = e["src_id"]
                dst_id = e["dst_id"]
                # 跨 shard 时, src/dst 可能在前面 shard 已导入. 用累积集合判断.
                # 如果 create_dangling=True 则不过滤, 由 placeholder 模板处理.
                if not self.create_dangling:
                    if src_id not in self._known_node_ids or dst_id not in self._known_node_ids:
                        skipped += 1
                        continue
                params = _pack_edge_attrs(e.get("attrs", {}))
                rows.append({
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "first_frame": e.get("first_frame", 0),
                    "last_frame": e.get("last_frame", 0),
                    "frame_id": e.get("frame_id", 0),
                    "params": params,
                })

            if skipped:
                self.skipped_edges += skipped
                logger.debug(
                    "  边 %s: 跳过 %d 条 (dangling src/dst)", rel_type, skipped,
                )

            if not rows:
                continue

            # 选择模板 (dangling 处理)
            if self.create_dangling:
                cypher = EDGE_MERGE_TPL_WITH_PLACEHOLDER.format(rel_type=rel_type)
            else:
                cypher = EDGE_MERGE_TPL.format(rel_type=rel_type)

            before = len(rows)
            n_batches = self.conn.run_batch_unwind(
                cypher, rows, batch_size=self.batch_size,
            )
            self.edge_count += before
            logger.debug(
                "  边 %s: %d 条 (%d 批)", rel_type, before, n_batches,
            )

        logger.info(
            "  边完成: %d 条 (累计跳过 dangling %d 条)",
            self.edge_count, self.skipped_edges,
        )

    # ── 批量目录 ──

    def import_dir(self, input_dir: str) -> None:
        """导入 viz_output/ 目录下的所有分片.

        按 phase5_kg_summary.json 中 shards 列表顺序导入.
        """
        import os
        import time

        summary_path = os.path.join(input_dir, "phase5_kg_summary.json")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(
                f"未找到 {summary_path}; 请确认 input_dir 包含 phase5 产物"
            )
        with open(summary_path) as f:
            summary = json.load(f)

        shards = summary.get("shards", [])
        logger.info("phase5_kg_summary: %d 个分片, %s", len(shards), summary.get("output_mode", ""))
        t_start = time.time()

        for shard_info in shards:
            idx = shard_info["shard_idx"]
            f_start = shard_info["frame_start"]
            f_end = shard_info["frame_end"]
            fname = f"graph_{idx:04d}_{f_start}_{f_end}.json"
            fpath = os.path.join(input_dir, fname)

            if not os.path.isfile(fpath):
                logger.warning("  分片文件缺失: %s, 跳过", fpath)
                continue

            logger.info("加载分片 %d: %s", idx, fname)
            with open(fpath) as f:
                graph = json.load(f)
            t1 = time.time()
            self.import_shard(graph)
            elapsed = time.time() - t1
            logger.info(
                "  分片 %d 完成 (%.1fs)", idx, elapsed,
            )

        total = time.time() - t_start
        logger.info("=" * 60)
        logger.info(
            "导入完成: %d 节点, %d 边 (耗时 %.1fs)",
            self.node_count, self.edge_count, total,
        )

    # ── 统计 ──

    def print_stats(self) -> None:
        """打印摘要统计."""
        print(f"Node types seen: {sorted(self._node_types_seen)}")
        print(f"Total nodes processed: {self.node_count}")
        print(f"Total edges processed: {self.edge_count}")
        if self.skipped_edges:
            print(f"Edges skipped (dangling ref): {self.skipped_edges}")
