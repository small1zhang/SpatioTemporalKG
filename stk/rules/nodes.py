"""
规则层节点定义 (v3 sec 4.4, sec 4.9, sec 4.16, sec 4.18)

本模块定义规则层 4 类核心节点：
  - RuleDefinition (Label: Rule) — 规则定义（RSS/交规均可）
  - RuleParameter (Label: Param) — RSS 参数（rho, a_max_accel 等）
  - SafetyViolation (Label: SafetyViolation) — 规则触发实例（节点+边双轨）
  - ResponsibilityAssignment (Label: Responsibility) — 责任归因节点

所有节点继承 ontology.entity.BaseEntity，entity_type 置为对应 EntityType。
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from stk.ontology.entity import BaseEntity
from stk.ontology.types import EntityType


# ============================================================
# 1. RuleDefinition — 规则定义 (v3 sec 4.4, Label: Rule)
# ============================================================

class RuleDefinition(BaseEntity):
    """规则定义节点 (v3 sec 4.4 / sec 4.9.1 / sec 4.16.1).

    Label: Rule
    rule_layer: "RSS" | "TrafficLaw"

    属性 (写入 attrs):
      rule_id          : 规则编号 (R1..R18, R13a, R14a, R15a, R15b)
      rule_name        : 规则英文名 (SafeDistanceViolation, RedLightViolation ...)
      rule_layer       : 子层 (RSS / TrafficLaw)
      predicate_name   : 谓词名 (DangerousState, RedLightViolation ...)
      formula_str      : 形式化判别式字符串
      description      : 规则中文描述
    """
    def __init__(self, entity_id, rule_id, rule_name, rule_layer,
                 predicate_name, formula_str='', description='',
                 valid_from=0, valid_to=None, confidence=1.0):
        attrs = {
            'rule_id': rule_id,
            'rule_name': rule_name,
            'rule_layer': rule_layer,
            'predicate_name': predicate_name,
            'formula_str': formula_str,
            'description': description,
        }
        super().__init__(
            entity_id=entity_id, entity_type=EntityType.RULE_DEFINITION,
            valid_from=valid_from, valid_to=valid_to,
            attrs=attrs, confidence=confidence,
        )

    @property
    def rule_id(self): return self.attrs['rule_id']
    @property
    def rule_name(self): return self.attrs['rule_name']
    @property
    def rule_layer(self): return self.attrs['rule_layer']
    @property
    def predicate_name(self): return self.attrs['predicate_name']


# ============================================================
# 2. RuleParameter — RSS 参数节点 (v3 sec 4.9.2, Label: Param)
# ============================================================

class RuleParameter(BaseEntity):
    """RSS 参数节点 (v3 sec 4.9.2 / sec 4.8 参数表).

    Label: Param
    属性 (写入 attrs):
      param_id   : 参数标识 (rho, a_max_accel, a_min_brake...)
      name       : 中文名
      value      : 参数值
      unit       : 单位 (s, m/s2, -)
    """
    def __init__(self, entity_id, param_id, name, value, unit='',
                 valid_from=0, valid_to=None):
        attrs = {
            'param_id': param_id,
            'name': name,
            'value': value,
            'unit': unit,
        }
        super().__init__(
            entity_id=entity_id, entity_type=EntityType.RULE_PARAMETER,
            valid_from=valid_from, valid_to=valid_to,
            attrs=attrs,
        )

    @property
    def param_id(self): return self.attrs['param_id']
    @property
    def value(self): return self.attrs['value']


# ============================================================
# 3. SafetyViolation — 规则触发实例节点 (v3 sec 4.3.1 / sec 4.9.3 / sec 4.16.2)
# ============================================================

class SafetyViolation(BaseEntity):
    """规则触发实例节点 (v3 sec 4.3.1).

    Label: SafetyViolation
    包含节点+边双轨表达 (v3 sec 4.3.1 + 4.3.2):
      - 节点: SafetyViolation { sv_id, rule_code, severity, predicate_str ... }
      - 边:   EntityA -[violates { ... }]-> EntityB

    属性表 (写入 attrs):
      sv_id          : 唯一 ID (sv_<rule_code>_<frame>_<src>_<dst>)
      rule_code      : 规则编号 (R1..R18, R13a...)
      rule_name      : 规则名
      rule_layer     : "RSS" / "TrafficLaw"
      frame_id       : 触发帧
      timestamp      : 触发时间 (秒)
      severity       : 严重度 [0.0, 1.0]
      predicate_str  : 谓词式字符串
      src_id         : 源实体 ID
      dst_id         : 目标实体 ID
      evidence_path  : 证据链 (依赖的场景层关系 ID 列表)
      related_actors: 相关 actor ID 列表
    """
    def __init__(self, entity_id, rule_code, rule_name, rule_layer,
                 frame_id, severity=0.5,
                 src_id='', dst_id='',
                 timestamp=0.0,
                 predicate_str='',
                 evidence_path=None,
                 related_actors=None,
                 rule_parameters=None,
                 extra_attrs=None,
                 valid_from=None, valid_to=None):
        if rule_layer not in ('RSS', 'TrafficLaw'):
            raise ValueError(f'rule_layer must be RSS/TrafficLaw, got {rule_layer}')
        attrs = {
            'sv_id': entity_id,
            'rule_code': rule_code,
            'rule_name': rule_name,
            'rule_layer': rule_layer,
            'frame_id': frame_id,
            'timestamp': timestamp,
            'severity': min(1.0, max(0.0, severity)),
            'predicate_str': predicate_str,
            'src_id': src_id,
            'dst_id': dst_id,
            'evidence_path': evidence_path or [],
            'related_actors': related_actors or [],
            'rule_parameters': rule_parameters or {},
        }
        if extra_attrs:
            attrs.update(extra_attrs)
        super().__init__(
            entity_id=entity_id, entity_type=EntityType.SAFETY_VIOLATION,
            valid_from=valid_from if valid_from is not None else frame_id,
            valid_to=valid_to,
            attrs=attrs,
        )

    @property
    def rule_code(self): return self.attrs['rule_code']
    @property
    def rule_layer(self): return self.attrs['rule_layer']
    @property
    def frame_id(self): return self.attrs['frame_id']
    @property
    def severity(self): return self.attrs['severity']
    @property
    def predicate_str(self): return self.attrs['predicate_str']
    @property
    def src_id(self): return self.attrs['src_id']
    @property
    def dst_id(self): return self.attrs['dst_id']
    @property
    def evidence_path(self): return list(self.attrs['evidence_path'])


# ============================================================
# 4. ResponsibilityAssignment — 责任归因节点 (v3 sec 4.18.1)
# ============================================================

class ResponsibilityAssignment(BaseEntity):
    """责任归因节点 (v3 sec 4.18.1).

    Label: Responsibility
    属性 (写入 attrs):
      resp_id              : 唯一 ID (resp_<sv_id>_<actor_id>)
      sv_id                : 关联的 SafetyViolation ID
      responsible_actor_id : 责任方 actor ID
      reason               : 责任理由字符串
    """
    def __init__(self, entity_id, sv_id, responsible_actor_id, reason='',
                 valid_from=0, valid_to=None):
        attrs = {
            'resp_id': entity_id,
            'sv_id': sv_id,
            'responsible_actor_id': responsible_actor_id,
            'reason': reason,
        }
        super().__init__(
            entity_id=entity_id, entity_type=EntityType.RESPONSIBILITY_ASSIGNMENT,
            valid_from=valid_from, valid_to=valid_to,
            attrs=attrs,
        )

    @property
    def sv_id(self): return self.attrs['sv_id']
    @property
    def responsible_actor_id(self): return self.attrs['responsible_actor_id']
    @property
    def reason(self): return self.attrs['reason']


# ============================================================
# 5. ID 工厂函数
# ============================================================

def make_sv_id(rule_code, frame_id, src_id, dst_id):
    """生成 SafetyViolation 唯一 ID (v3 sec 4.3.1 示例).

    sv_<rule_code>_<frame_id>_<src>_<dst>
    """
    return f'sv_{rule_code}_{frame_id}_{src_id}_{dst_id}'


def make_resp_id(sv_id, actor_id):
    """生成 ResponsibilityAssignment 唯一 ID.

    resp_<sv_id>_<actor_id>
    """
    return f'resp_{sv_id}_{actor_id}'
