# -*- coding: utf-8 -*-
"""批量 MERGE 写入 (v3 §6.6)."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from stk.ontology.entity import BaseEntity
from stk.ontology.relation import BaseRelation
from stk.storage.serializer import (
    entity_merge_cypher, relation_merge_cypher,
    entity_to_cypher_params, relation_to_cypher_params,
)


def write_entity_batch(entities: List[BaseEntity],
                       batch_size: int = 500) -> List[Tuple[str, Dict[str, Any]]]:
    batches = []
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i+batch_size]
        stmts = []
        for e in batch:
            label = e.neo4j_label()
            params = entity_to_cypher_params(e)
            stmts.append((entity_merge_cypher(label), params))
        batches.extend(stmts)
    return batches


def write_relation_batch(relations: List[BaseRelation],
                         batch_size: int = 500) -> List[Tuple[str, Dict[str, Any]]]:
    batches = []
    for i in range(0, len(relations), batch_size):
        batch = relations[i:i+batch_size]
        stmts = []
        for r in batch:
            params = relation_to_cypher_params(r)
            params["src_id"] = r.src_id
            params["dst_id"] = r.dst_id
            params["frame_id"] = r.frame_id
            stmts.append((relation_merge_cypher(r.relation_type), params))
        batches.extend(stmts)
    return batches