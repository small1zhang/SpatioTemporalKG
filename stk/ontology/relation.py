"""
Relation 基类 (v3 §1.8.3)
r := (src, dst, type, attrs, time_window, confidence)
谓词式: P(src, dst, t)
"""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class BaseRelation(BaseModel):
    src_id: str = Field(..., min_length=1)
    dst_id: str = Field(..., min_length=1)
    relation_type: str = Field(...)
    frame_id: int = Field(..., ge=0)
    valid_from: int = Field(..., ge=0)
    valid_to: Optional[int] = Field(None)
    attrs: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    class Config:
        extra = "allow"

    def predicate_str(self) -> str:
        name = self.relation_type.replace("_", " ").title().replace(" ", "")
        return f"{name}({self.src_id}, {self.dst_id}, Frame_{self.frame_id})"

    def to_neo4j_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "relation_type": self.relation_type, "frame_id": self.frame_id,
            "valid_from": self.valid_from, "valid_to": self.valid_to,
            "confidence": self.confidence,
        }
        for k, v in self.attrs.items():
            result[k] = v
        return result
