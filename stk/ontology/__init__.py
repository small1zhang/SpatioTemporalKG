"""本体层：实体/关系/属性/公理的形式化定义 (§1)。"""

from .types import (
    EntityType, SceneRelationType, BehaviorRelationType,
    RuleRelationType, CrossLayerRelationType, NODE_LABELS,
    entity_type_from_label, relation_type_from_value,
)
from .entity import BaseEntity
from .relation import BaseRelation
from .temporal_triple import TemporalTriple
from .namespace import IDGenerator, NAMESPACE_PREFIXES, GLOBAL_ID_GENERATOR
from .lifecycle import NodeLifecycle, NodeLifecycleStatus
from .axioms import (
    axiom_A1_unique_id, axiom_A2_fixed_type, axiom_A3_attribute_versioned,
    axiom_A4_relation_has_temporal, axiom_A5_three_layer_evidence,
    axiom_A6_event_traceable, axiom_A7_incremental_consistency,
)

__all__ = [
    "EntityType", "SceneRelationType", "BehaviorRelationType",
    "RuleRelationType", "CrossLayerRelationType", "NODE_LABELS",
    "entity_type_from_label", "relation_type_from_value",
    "BaseEntity", "BaseRelation", "TemporalTriple",
    "IDGenerator", "NAMESPACE_PREFIXES", "GLOBAL_ID_GENERATOR",
    "NodeLifecycle", "NodeLifecycleStatus",
    "axiom_A1_unique_id", "axiom_A2_fixed_type", "axiom_A3_attribute_versioned",
    "axiom_A4_relation_has_temporal", "axiom_A5_three_layer_evidence",
    "axiom_A6_event_traceable", "axiom_A7_incremental_consistency",
]
