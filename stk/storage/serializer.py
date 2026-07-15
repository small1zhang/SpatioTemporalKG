# -*- coding: utf-8 -*-
"""Entity/Relation → Cypher 参数 (v3 §6.2.4)."""
from __future__ import annotations
import json
from typing import Any, Dict, List
from stk.ontology.entity import BaseEntity
from stk.ontology.relation import BaseRelation


def entity_to_cypher_params(entity: BaseEntity) -> Dict[str, Any]:
    params = entity.to_neo4j_dict()
    for k, v in list(params.items()):
        if isinstance(v, float):
            params[k] = round(v, 6)
        elif isinstance(v, (list, dict)):
            params[k] = json.dumps(v, ensure_ascii=False)
    params["_entity_type"] = entity.entity_type
    return params


def relation_to_cypher_params(rel: BaseRelation) -> Dict[str, Any]:
    params = rel.to_neo4j_dict()
    for k, v in list(params.items()):
        if isinstance(v, float):
            params[k] = round(v, 6)
        elif isinstance(v, (list, dict)):
            params[k] = json.dumps(v, ensure_ascii=False)
    return params


def entity_merge_cypher(label: str) -> str:
    return f"MERGE (n:{label} {{entity_id: $entity_id}}) SET n += $params"


def relation_merge_cypher(rel_type: str) -> str:
    return (
        f"MATCH (src {{entity_id: $src_id}})\n"
        f"MATCH (dst {{entity_id: $dst_id}})\n"
        f"MERGE (src)-[r:{rel_type} {{frame_id: $frame_id}}]->(dst)\n"
        f"SET r += $params"
    )