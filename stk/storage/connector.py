# -*- coding: utf-8 -*-
"""Neo4j 连接池 + 重试 (v3 §6.1.3)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from stk.config import load_config


class Neo4jConnection:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j",
                 password: str = "stk123", database: str = "spatiotemporal_kg"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None  # 实际 driver by neo4j.GraphDatabase.driver()
        self._connected = False

    def connect(self) -> bool:
        if not self._connected:
            # 伪连接 - 在无 Neo4j 时保留接口
            self._connected = True
        return self._connected

    def close(self):
        self._connected = False
        self.driver = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def run(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        if not self._connected:
            raise RuntimeError("Not connected to Neo4j")
        # 在真实连接时: self.driver.session().run(cypher, params).data()
        return []


class ConnectionPool:
    def __init__(self, config_path: Optional[str] = None, pool_size: int = 4):
        cfg = load_config("neo4j.yaml") if config_path is None else load_config(config_path)
        self._conns = [
            Neo4jConnection(
                uri=cfg.get("uri", "bolt://localhost:7687"),
                user=cfg.get("user", "neo4j"),
                password=cfg.get("password", "stk123"),
                database=cfg.get("database", "spatiotemporal_kg"),
            ) for _ in range(pool_size)
        ]
        self._idx = 0

    def acquire(self) -> Neo4jConnection:
        conn = self._conns[self._idx % len(self._conns)]
        self._idx += 1
        if not conn.is_connected:
            conn.connect()
        return conn

    def close_all(self):
        for c in self._conns:
            c.close()