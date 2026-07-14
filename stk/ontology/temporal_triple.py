"""
时态三元组 (v3 §1.8.4)
τ := (s, p, o, t) — 基本事实表达单元
"""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class TemporalTriple(BaseModel):
    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)
    frame_id: int = Field(..., ge=0)
    valid_from: int = Field(0, ge=0)
    valid_to: Optional[int] = Field(None)
    attrs: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    def to_triple_string(self) -> str:
        """标准三元组格式: (s) -[p {t}]-> (o)"""
        return f"({self.subject}) -[{self.predicate} {{frame_id:{self.frame_id}}}]-> ({self.object})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject, "predicate": self.predicate,
            "object": self.object, "frame_id": self.frame_id,
            "valid_from": self.valid_from, "valid_to": self.valid_to,
            "attrs": self.attrs, "confidence": self.confidence,
        }
