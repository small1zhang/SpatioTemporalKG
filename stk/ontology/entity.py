"""
Entity 基类 (v3 §1.8.2, §1.10)
e := (id, type, attrs, lifetime)
属性版本化：节点长期存在，属性按 (valid_from, valid_to) 版本记录。
"""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from stk.ontology.types import EntityType, NODE_LABELS


class BaseEntity(BaseModel):
    entity_id: str = Field(..., min_length=1)
    entity_type: EntityType = Field(...)
    valid_from: int = Field(0, ge=0)
    valid_to: Optional[int] = Field(None)
    attrs: Dict[str, Any] = Field(default_factory=dict)
    labels: list[str] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    class Config:
        use_enum_values = True
        extra = "allow"

    def neo4j_label(self) -> str:
        if isinstance(self.entity_type, str):
            for et, lbl in NODE_LABELS.items():
                if lbl == self.entity_type or et.value == self.entity_type:
                    return lbl
            return str(self.entity_type)
        return NODE_LABELS.get(self.entity_type, self.entity_type.value)

    def is_active(self, frame_id: int) -> bool:
        if frame_id < self.valid_from:
            return False
        if self.valid_to is not None and frame_id > self.valid_to:
            return False
        return True

    def to_neo4j_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "entity_id": self.entity_id,
            "valid_from": self.valid_from, "valid_to": self.valid_to,
            "labels": self.labels, "confidence": self.confidence,
        }
        for k, v in self.attrs.items():
            result[k] = v
        return result
