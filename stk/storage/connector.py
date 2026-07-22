# -*- coding: utf-8 -*-
"""Neo4j 连接池 + 重试 (v3 §6.1.3).

真实 driver 包装: 之前是接口桩 (connect() 只翻 bool, run() 永远返回 []),
现在改为使用 neo4j.GraphDatabase.driver(...) 真正打通 bolt。
ConnectionPool 对外接口保持不变, 仅替换内部实现。
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 延迟 import, 避免 neo4j driver 未安装时整个 storage 包 import 失败
# (driver 已在 pyproject.toml:21 / requirements.txt:13 声明, stk 环境内已装)
_DRIVER_AVAILABLE: Optional[bool] = None
_GraphDatabase = None


def _ensure_driver():
    global _DRIVER_AVAILABLE, _GraphDatabase
    if _DRIVER_AVAILABLE is None:
        try:
            from neo4j import GraphDatabase  # noqa: WPS433
            _GraphDatabase = GraphDatabase
            _DRIVER_AVAILABLE = True
        except ImportError:  # pragma: no cover - dev 环境会有
            _DRIVER_AVAILABLE = False
            logger.warning(
                "neo4j python driver 未安装; pip install neo4j 后重试"
            )
    return _DRIVER_AVAILABLE


class Neo4jConnection:
    """单个 Neo4j 连接 (driver 实际是连接池, 这里再包一层方便 mock)."""

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j",
                 password: str = "stk123", database: str = "spatiotemporal_kg"):
        self.uri = uri
        self.user = user
        self.password = password
        # 默认 database 名 'neo4j' (community 版只有这一个); 如果传入自定义库名但不存在,
        # 写入会报错 — 我们这里宽松起见允许 None/空串, 用默认库
        self.database = database if database else None
        self.driver: Optional[Any] = None
        self._connected = False

    def connect(self) -> bool:
        """打开 driver 并验证连通性."""
        if not _ensure_driver():
            raise RuntimeError("neo4j python driver 不可用; pip install neo4j")
        if self._connected and self.driver is not None:
            return True
        self.driver = _GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        # verify_connectivity 抛异常表示连不上
        try:
            self.driver.verify_connectivity()
            self._connected = True
            logger.info("Neo4j 已连接: %s (db=%s)", self.uri, self.database or "neo4j")
            return True
        except Exception as exc:  # pragma: no cover
            self._connected = False
            self.driver = None
            raise RuntimeError(f"无法连接 Neo4j {self.uri}: {exc}") from exc

    def close(self):
        if self.driver is not None:
            try:
                self.driver.close()
            except Exception:  # pragma: no cover
                pass
        self.driver = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self.driver is not None

    def _session(self):
        if not self.is_connected:
            raise RuntimeError("Not connected to Neo4j — 请先调用 connect()")
        # database=None 时用默认 ('neo4j' community 库)
        return self.driver.session(database=self.database) if self.database else self.driver.session()

    def run(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        """执行一条 Cypher, 返回 .data() 列表."""
        with self._session() as sess:
            result = sess.run(cypher, params or {})
            try:
                return [dict(r) for r in result]
            except Exception:  # 写语句没 data(), consume 一下即可
                return []

    def run_write(self, cypher: str, params: Optional[Dict] = None) -> Optional[Any]:
        """以写事务执行 Cypher, 返回 result.consume() summary.

        适合 CREATE / MERGE / SET / CREATE INDEX 等不需要回读的场景.
        """
        def _tx_fn(tx):
            return tx.run(cypher, params or {}).consume()

        with self._session() as sess:
            try:
                return sess.execute_write(_tx_fn)
            except Exception as exc:
                logger.error("Cypher 执行失败: %s | params=%s | cypher=%s",
                             exc, _truncated(params), cypher[:200])
                raise

    def run_many(self, statements: List[Tuple[str, Dict]]) -> int:
        """批量执行多条 Cypher (写事务模式). 每条 (cypher, params) 单独跑.

        返回成功执行条数. 任一条失败立即抛异常 (默认 Neo4j 行为).
        """
        if not statements:
            return 0
        ok = 0
        with self._session() as sess:
            for cypher, params in statements:
                def _tx_fn(tx, cypher=cypher, params=params):
                    return tx.run(cypher, params or {})
                sess.execute_write(_tx_fn)
                ok += 1
        return ok

    def run_batch_unwind(self, cypher: str, batch: List[Dict], batch_size: int = 500) -> int:
        """UNWIND 批量: 把 batch 切片并以 {batch: [...]} 形式执行单条 Cypher.

        Cypher 模板需要包含 `UNWIND $batch AS row ...`.
        返回跑了几批.
        """
        if not batch:
            return 0
        n = 0
        with self._session() as sess:
            for i in range(0, len(batch), batch_size):
                slice_ = batch[i:i + batch_size]

                def _tx_fn(tx, slice_=slice_):
                    return tx.run(cypher, {"batch": slice_}).consume()

                sess.execute_write(_tx_fn)
                n += 1
        return n

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _truncated(obj, max_len: int = 200) -> str:
    s = repr(obj)
    return s if len(s) <= max_len else s[:max_len] + "..."


class ConnectionPool:
    """简单轮转连接池 — 配置默认走 config/neo4j.yaml."""

    def __init__(self, config_path: Optional[str] = None, pool_size: int = 4):
        from stk.config import load_config  # 延迟 import 避免循环依赖
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
