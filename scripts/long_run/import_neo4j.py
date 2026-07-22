#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase5 shard JSON → Neo4j 一键导入 (CLI).

用法::

    # 0. 起 Neo4j
    docker compose -f docker/neo4j/docker-compose.yml up -d
    # 等 ~30s 让 Neo4j 就绪

    # 1. 跑导入
    python -m scripts.long_run.import_neo4j \\
        --input viz_output/ \\
        --uri bolt://localhost:7687 \\
        --user neo4j --password stk123

    # 2. 打开 http://localhost:7474 看图 (账号同上)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

# 让从仓库根目录 `python -m scripts.long_run.import_neo4j` 跑通
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stk.storage.connector import Neo4jConnection
from stk.storage.importer import GraphImporter


def parse_args():
    p = argparse.ArgumentParser(
        prog="import_neo4j",
        description="把 viz_output/graph_*.json phase5 分片导入 Neo4j",
    )
    p.add_argument(
        "--input", "-i", default="viz_output/",
        help="phase5 输出目录 (含 phase5_kg_summary.json + graph_*.json)",
    )
    p.add_argument("--uri", default="bolt://localhost:7687",
                   help="Neo4j bolt URI (default: %(default)s)")
    p.add_argument("--user", default="neo4j", help="Neo4j 用户名 (default: %(default)s)")
    p.add_argument("--password", default="stk123", help="Neo4j 密码 (default: %(default)s)")
    p.add_argument("--database", default="neo4j",
                   help="Neo4j 数据库 (community 版仅 'neo4j'; default: %(default)s)")
    p.add_argument("--batch-size", type=int, default=500,
                   help="UNWIND 批大小 (default: %(default)s)")
    p.add_argument(
        "--create-dangling", action="store_true",
        help="为悬挂的 sv_*_frame_* / scene_rel_*_frame_* ID 创建 :UnknownRef 占位节点 "
             "(默认跳过这些边)",
    )
    p.add_argument(
        "--drop-existing", action="store_true",
        help="导入前清空数据库所有节点和边 (慎用! 不可逆)",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="debug 日志")
    return p.parse_args()


def setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def wait_for_neo4j(conn: Neo4jConnection, retries: int = 30, interval: float = 2.0) -> bool:
    """等 Neo4j 就绪 (docker compose 起来后约 30s)."""
    for i in range(retries):
        try:
            conn.connect()
            logging.info("Neo4j 就绪 (第 %d 次尝试)", i + 1)
            return True
        except Exception as exc:
            logging.info("等待 Neo4j 启动 [%d/%d]: %s", i + 1, retries, exc)
            time.sleep(interval)
    return False


def main():
    args = parse_args()
    setup_logging(args.verbose)
    log = logging.getLogger("import_neo4j")

    conn = Neo4jConnection(
        uri=args.uri, user=args.user,
        password=args.password, database=args.database,
    )

    if not wait_for_neo4j(conn):
        log.error("Neo4j 在 %s 上未就绪, 退出", args.uri)
        sys.exit(1)

    # 可选: 清空
    if args.drop_existing:
        log.warning("⚠ 清空数据库 (MATCH (n) DETACH DELETE n) — 不可逆")
        conn.run_write("MATCH (n) DETACH DELETE n")
        log.info("清空完成")

    importer = GraphImporter(
        conn, batch_size=args.batch_size,
        create_dangling=args.create_dangling,
    )

    log.info("创建 schema 约束/索引...")
    importer.ensure_schema()
    log.info("开始导入分片...")
    t0 = time.time()
    importer.import_dir(args.input)
    elapsed = time.time() - t0

    log.info("=" * 70)
    log.info("导入完成 (用时 %.1fs)", elapsed)
    importer.print_stats()
    log.info("=" * 70)
    log.info("看图: http://localhost:7474  账号 %s/%s", args.user, args.password)
    log.info("示例查询:")
    log.info("  MATCH (n:Vehicle {entity_id:'486'})-[r]-(m) RETURN n, r, m LIMIT 50;")
    log.info("  MATCH (n) RETURN count(n);")
    log.info("  MATCH ()-[r]->() RETURN count(r);")

    conn.close()


if __name__ == "__main__":
    main()
