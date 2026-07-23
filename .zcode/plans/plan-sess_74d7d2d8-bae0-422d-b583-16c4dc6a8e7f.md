# 第一阶段修复计划：场景层 / 行为层 / 规则层 / 增量引擎 BLOCKER 修复

## 目标
让"采集 → 行为识别 → 规则判定 → 责任归因"全链路对**异常溯因**有效。修复 8 个 BLOCKER + 接线 5 个未调用规则，全部带单元测试，最后用 2400 帧 smoke 验证 KG 含完整 ResponsibilityAssignment 节点。

## 变更总览（共 8 个文件 + 3 个新测试）

| 文件 | 变更性质 | 修复点 |
|---|---|---|
| `stk/rules/rss/model.py` | 改 4 个常量 + 接 1 个参数 | RSS-1 |
| `stk/rules/nodes.py` | `SafetyViolation.__init__` 增 1 个可选参数 `rule_parameters` | RSS-3 |
| `stk/rules/generator.py` | 重写 `_add_violation` + 接线 5 个检测器 + RSS 路径补 rule_parameters | RSS-2 + GEN-1 + RSS-3 |
| `stk/rules/traffic/rules.py` | R7/R18 调用前补 helper / R3/R4/R5 调用前补 helper | GEN-1 (接线前清理) |
| `stk/behavior/debouncer.py` | `update` 引入 `off_counter` 状态机 + `to_dict` 同步 | DEB-1 |
| `stk/behavior/detectors.py` | `detect_changing_lane` 用真横向速度 + 输出 `target_lane_id`；`detect_following` 加 heading 投影 + TTC 兜底 | DET-1 + DET-2 |
| `stk/behavior/generator.py` | `_create_relation` 把误吞 TypeError 改成 warning；`run_all_detectors` 改 changing_lane 的 dst 传 target_lane_id | DET-1 接线 |
| `stk/dynamic/incremental_updater.py` | `load_dict` 实现跳帧检测；`to_dict` 收紧异常捕获 + 数值校验 | INCR-1 |
| `tests/test_rules_regression.py` (新建) | RSS 参数 + 责任归因 + rule_parameters + 5 个规则接线 单测 | 测试 |
| `tests/test_debouncer.py` (新建) | 防抖状态机往返测试 | 测试 |
| `tests/test_incremental_resume.py` (新建) | resume 跳帧 + 字符串污染单测 | 测试 |

## 详细修改方案

### 1. `stk/rules/rss/model.py` — RSS 参数回归论文值 (RSS-1)

**L19-27** `DEFAULT_RSS_PARAMS` 字面值替换：
```python
DEFAULT_RSS_PARAMS: Dict[str, float] = {
    "rho": 0.3,                # 反应时间 (s) —— RSS 论文标准 0.3
    "a_max_accel": 0.5,        # 最大加速 (m/s^2) —— RSS 论文标准 0.5
    "a_min_brake_long": 3.0,   # 最小纵向减速 (m/s^2) —— RSS 论文标准 3.0
    "a_brake_long": 8.0,       # 前车最大减速 (m/s^2)
    "mu": 0.5,                 # 横向安全裕度 (m)
    "a_min_brake_lat": 1.5,    # 最小横向减速 (m/s^2) —— RSS 论文标准 1.5
    "a_brake_lat": 3.0,        # 目标横向减速 (m/s^2)
}
```
公式实现不动（已正确）。`check_no_proper_response` 的 `threshold=0.3, required_consecutive=3` 不改（与 ThresholdConfig 无冲突）。

### 2. `stk/rules/nodes.py` — `SafetyViolation` 加 `rule_parameters` 字段 (RSS-3)

**L102-173** `SafetyViolation.__init__` 在 evidence_path/related_actors 之后加：
```python
def __init__(self, ..., rule_parameters=None, ...):  # 新增可选参
    ...
    attrs = {
        ...
        'evidence_path': evidence_path or [],
        'related_actors': related_actors or [],
        'rule_parameters': rule_parameters or {},  # NEW: 触发时使用的 RSS/规则参数快照
    }
```
向后兼容：默认 `{}`，老代码不传不受影响。

### 3. `stk/rules/generator.py` — 三件大事 (RSS-2 + RSS-3 + GEN-1)

**3.1** `_add_violation()` (L294-323) 新增两个参数：`responsibility_reason: Optional[str]`、`rule_parameters: Optional[Dict]`，并在内部创建 `ResponsibilityAssignment` + `responsibleFor` 边：
```python
def _add_violation(rule_code, rule_name, rule_layer,
                   src_id, dst_id, evidence_id, frame_id, severity,
                   violations, violation_rels, defined_by_rels, evidence_rels,
                   responsibilities=None, resp_rels=None,           # NEW
                   responsibility_reason=None,                       # NEW
                   rule_parameters=None):                            # NEW
    ...  # 原 SafetyViolation 与 violates/definedBy/supportedByEvidence 不变
    if rule_parameters:
        sv.attrs["rule_parameters"] = rule_parameters  # 由 (2) 字段承接
    # 新建责任归因 (默认违规主体 = src_id)
    if responsibilities is not None and resp_rels is not None:
        resp_id = make_resp_id(sv_id, src_id)
        ra = ResponsibilityAssignment(
            entity_id=resp_id, sv_id=sv_id,
            responsible_actor_id=src_id,
            reason=responsibility_reason or f"{rule_code}_violation",
        )
        responsibilities.append(ra)
        resp_rels.append(responsible_for(
            resp_id=resp_id, sv_id=sv_id, frame_id=frame_id,
            valid_from=frame_id, reason=responsibility_reason or f"{rule_code}_violation",
        ))
```

**3.2** `enforce()` 9 个 TrafficLaw 调用块 (R1/R2/R8/R9/R10/R11/R13/R16/R17) 把 `responsibilities` / `resp_rels` / `responsibility_reason` / `rule_parameters` 都传给 `_add_violation`：
- 对每个规则准备 `rule_parameters={"threshold": <val>, ...}` snapshot
- `responsibility_reason` 用规则名 (e.g. `"pedestrian_priority_violation"`)
- 责任方默认为 `src_id`（违规车辆），.dst_id 受损

**3.3** `enforce()` 的 RSS 路径 (L94-167) 在 `extra_attrs={"d_min_long", "d_min_lat"}` 之外把 `rule_parameters` 也注入 (复制 `rss_params` 字典)：
```python
extra_attrs={
    "d_min_long": rss["d_min_long"],
    "d_min_lat": rss["d_min_lat"],
    "rule_parameters": dict(p),  # NEW: snapshot 实际使用的 RSS 参数
},
```

**3.4** 接线 R3/R4/R5/R7/R18 (5 个新调用块)：
- L41 import 加 `check_R3_solid_line_change, check_R4_opposite_meeting, check_R5_reversing`
- 在 enforce() 中遍历 `vehicles`，对每个车辆分别调用：
  - R3: 需要车辆在 changing_lane 状态 (查 `scene_relations` 中 `changing_lane` 关系)，传 `crossed_solid=False, is_changing_lane=<bool>`
  - R4: 双向遍历，对每对反向车调用 `check_R4_opposite_meeting(v_a, v_b, distance, is_opposite_lane=True)`
  - R5: 检查 heading 与 lane heading 反向 (依据 scene_relations 中 wrong_side_meeting 等的存在), 调用 `check_R5_reversing(v, angle_diff, duration_frames=1)`
  - R7: 对每对 (车, other) 在 in_junction 时调用，传 `other_has_priority` 由几何 / 路权规则推导
  - R18: 对每个 in_junction 车辆调用，依据 lane heading 反向触发
- 默认参数保守 (许多需要场景查询)，规则接线先打通 noop 默认；后续若数据不齐不会假阳性

### 4. `stk/rules/traffic/rules.py` — 帮 _add_violation 接线 (parser 已存在，仅 generator 侧调用)

无需改动 `rules.py` 本身（函数签名已 ok），列出来主要是接线在 generator.py 完成。

### 5. `stk/behavior/debouncer.py` — `update` 修 off_counter (DEB-1)

**L48-83** 改写：
```python
class DebounceItem:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self.on_counter: int = 0      # 连续满足帧数
        self.off_counter: int = 0    # 连续不满足帧数  ← NEW
        self.is_active: bool = False
        self.active_since: Optional[int] = None
        self.last_condition_met: Optional[bool] = None

    def update(self, condition_met, frame_id):
        if condition_met:
            self.on_counter += 1
            self.off_counter = 0
        else:
            self.off_counter += 1
            self.on_counter = 0

        if condition_met and not self.is_active and self.on_counter >= self.threshold:
            self.is_active = True
            self.active_since = frame_id
            return ("create", {"debounce_activated": frame_id, "on_counter": self.on_counter})
        elif condition_met and self.is_active:
            return ("keep", {"on_counter": self.on_counter})
        elif not condition_met and self.is_active and self.off_counter >= self.threshold:
            self.is_active = False
            self.active_since = None
            return ("delete", {"debounce_deactivated": frame_id, "off_counter": self.off_counter})
        elif not condition_met and self.is_active:
            # 已激活状态下的非首帧抖动, 维持 keep (近端抑制抖动)
            return ("keep", {"on_counter": 0, "off_counter": self.off_counter})
        else:
            return ("none", None)
```

**to_dict / from_dict** 同步镜像 on_counter/off_counter (兼容旧 schema: 兼容只有 `counter` 字段的旧 checkpoint，按 on_counter 解析)。

### 6. `stk/behavior/detectors.py` — 修 changing_lane + following (DET-1 + DET-2)

**L74-89** `detect_changing_lane` 改写：
```python
def detect_changing_lane(vehicle, scene_relations):
    vx = vehicle.get("velocity_x", 0.0)
    vy = vehicle.get("velocity_y", 0.0)
    heading = vehicle.get("heading_rad", 0.0)
    # 用 heading 把世界速度向量旋转到车体系, 横向分量 = -vx*sin + vy*cos
    lateral_speed = abs(-vx * math.sin(heading) + vy * math.cos(heading))
    # 从 scene_relations 找本车的 adjacent_lane 目标 (优先场景提供)
    target_lane_id = ""
    for r in scene_relations or []:
        if r.get("relation_type") == "adjacent_lane" and r.get("src_id") == vehicle.get("entity_id"):
            target_lane_id = r.get("dst_id", "")
            break
    if not target_lane_id:
        # 兜底: 用 vehicle.get("lane_id") 派生
        lane_id = vehicle.get("lane_id")
        if lane_id is not None:
            target_lane_id = f"road_{vehicle.get('road_id', 0)}_lane_{lane_id}"
    condition_met = lateral_speed > LANE_CHANGE_LATERAL_SPEED and bool(target_lane_id)
    return condition_met, {
        "lateral_speed": round(lateral_speed, 3),
        "target_lane_id": target_lane_id,
        "velocity_x": vx, "velocity_y": vy,
    }
```

**L97-139** `detect_following` 改写：
```python
def detect_following(vehicle, leader):
    v_x = vehicle.get("location_x", 0.0); v_y = vehicle.get("location_y", 0.0)
    l_x = leader.get("location_x", 0.0); l_y = leader.get("location_y", 0.0)
    dx = l_x - v_x; dy = l_y - v_y
    distance = math.hypot(dx, dy)
    if distance > FOLLOWING_MAX_DISTANCE or distance < 0.5:
        return (False, {"distance": distance, "relative_speed": 0.0, "ttc": None, "reason": "out_of_range"})

    v_heading = vehicle.get("heading_rad", 0.0)
    # 把 (dx,dy) 投影到 vehicle 坐标系, 前方 = 沿 heading 的正分量
    long_along = dx * math.cos(v_heading) + dy * math.sin(v_heading)
    if long_along <= 0:
        return (False, {"distance": distance, "relative_speed": 0.0, "ttc": None, "reason": "leader_not_ahead"})

    v_speed = vehicle.get("speed", 0.0); l_speed = leader.get("speed", 0.0)
    closing_speed = v_speed - l_speed  # 正值 = 后车逼近前车
    if closing_speed <= 0:
        # 后车不比前车快 -> 不构成接近性跟驰 TTC, 距离足够近才算稳定跟车
        ttc = None
        condition_met = distance < FOLLOWING_MAX_DISTANCE
    else:
        ttc = distance / closing_speed
        condition_met = distance < FOLLOWING_MAX_DISTANCE  # 主条件
    # 同车道判定 (从 lane_id 比较; 若缺失则默认 True 保留可观察)
    same_lane = (vehicle.get("lane_id") == leader.get("lane_id")) or vehicle.get("lane_id") is None
    condition_met = condition_met and same_lane
    return (condition_met, {
        "distance": round(distance, 2),
        "relative_speed": round(closing_speed, 2),
        "ttc": round(ttc, 2) if ttc is not None else None,
        "long_along": round(long_along, 2),
        "same_lane": same_lane,
    })
```

### 7. `stk/behavior/generator.py` — 噪声压制 + changing_lane 目标接通 (DET-1 接线)

- `_create_relation` (L300-338) 的 `except TypeError: return None` 改为：
  ```python
  except TypeError as e:
      # 不静默吞错: warning 给 log, 但仍返回 None 不阻塞管道
      import sys
      print(f"[behavior] _create_relation({rel_type}) TypeError: {e}", file=sys.stderr)
      return None
  ```
- `detectors.run_all_detectors` 中调用 changing_lane 的 dst 改为 `extra["target_lane_id"]` 而非 `eid`：
  ```python
  cond, extra = detect_changing_lane(v, scene_relations)
  target_lane = extra.get("target_lane_id") or ""
  add("changing_lane", eid, target_lane, cond, extra)  # dst 改成 target_lane
  ```
  注意：若 target_lane 为空，cond 已为 False，不会创建节点，安全。

### 8. `stk/dynamic/incremental_updater.py` — 跳帧检测 + 安全清理 (INCR-1)

**L18-23** `process_frame` 增加跳帧检测：
```python
def process_frame(self, frame: dict) -> DeltaGraph:
    if self._prev_frame is not None:
        prev_fid = self._prev_frame.get("frame_id")
        curr_fid = frame.get("frame_id")
        if prev_fid is not None and curr_fid is not None and curr_fid != prev_fid + 1:
            # 跳帧: 警告但不阻塞, prev 视为不连续, 让 compute_delta 按首帧处理
            import sys
            print(f"[dynamic] frame jump detected: prev={prev_fid} curr={curr_fid}, "
                  f"resetting baseline", file=sys.stderr)
            self._prev_frame = None
    dg = compute_delta(frame, self._prev_frame)
    self._delta_history.append(dg)
    self._prev_frame = frame
    return dg
```

**L39-58** `to_dict` 收紧：在 json.dumps 之外做数值类型校验，拒绝字符串数字：
```python
def to_dict(self) -> dict:
    prev = self._prev_frame
    if prev is not None:
        # 强校验: 任何 numeric 字段都不能是 str (catch default=str 字符串污染)
        try:
            json.dumps(prev)  # 不用 default=str, 严格模式
        except (TypeError, ValueError):
            prev = None
        else:
            # 二次校验: 扫一遍 attrs 里类型异常
            if not _validate_numeric_attrs(prev):
                prev = None
    return {"prev_frame": prev, "n_deltas": len(self._delta_history),
            "last_processed_frame": prev.get("frame_id", -1) if isinstance(prev, dict) else -1}
```

新增 `_validate_numeric_attrs(f: dict) -> bool` 工具函数 (在文件顶部 helper)：遍历 prev_frame 的 entities 列表，对每个 entity 的 location_x/y/z, speed, heading_rad 等数值属性检查 isinstance(x, (int, float)) 且非 bool。任何不合规返回 False。

**L60-66** `load_dict` 把占位 `pass` 替换为：检查 prev_frame 的 frame_id 是否符合预期下接的 chunk 起始帧：
```python
def load_dict(self, data: dict) -> None:
    self._prev_frame = data.get("prev_frame", None)
    if self._prev_frame is not None:
        # 数值完整性校验: 防御字符串污染型恢复
        if not _validate_numeric_attrs(self._prev_frame):
            print("[dynamic] WARNING: prev_frame numeric attrs corrupted, "
                  "resetting baseline to avoid false deltas", file=sys.stderr)
            self._prev_frame = None
    self._delta_history.clear()
```

## 单元测试 (3 个新文件，约 200 行)

### `tests/test_rules_regression.py`
1. `test_rss_params_match_paper`: assert DEFAULT_RSS_PARAMS["rho"] == 0.3, a_max_accel == 0.5, a_min_brake_long == 3.0, a_min_brake_lat == 1.5
2. `test_safety_violation_has_rule_parameters`: 构造 SafetyViolation 含 rule_parameters={"rho":0.3}, 检查 attrs["rule_parameters"]
3. `test_traffic_law_creates_responsibility`: 构造 mock frame 喂 RuleEnforcer, 触发 R1 行人优先, 验证返回节点含 ResponsibilityAssignment + responsibleFor 边
4. `test_rss_creates_responsibility_with_params`: 同上对 RSS 路径, 验证 rule_parameters 含 d_min_long + rss_params 快照
5. `test_R3_R4_R5_R7_R18_wired`: mock 数据触发这 5 个规则, 验证每个规则能在 violations 列表中找到对应 sv_id

### `tests/test_debouncer.py`
1. `test_off_counter_threshold`: 激活后连续 N 帧不满足, 第 N-1 帧仍 keep, 第 N 帧才 delete
2. `test_on_counter_threshold`: 首帧满足不激活, 连续 N 帧满足才 create
3. `test_checkpoint_round_trip`: to_dict → from_dict → update 行为一致
4. `test_string_pollution_resilience`: 旧格式 checkpoint (只有 `counter`) 仍能解析为 on_counter

### `tests/test_incremental_resume.py`
1. `test_frame_jump_resets_baseline`: prev_frame.frame_id=99, curr.frame_id=200, dg 应该按"首帧"处理 (全 added)
2. `test_string_field_pollution_dropped`: prev_frame 含 location_x="100.0" (字符串), to_dict 应 prev=None
3. `test_to_dict_rejects_non_serializable`: prev_frame 含自定义对象, to_dict 应 prev=None 而不抛异常

## 验证 (按顺序)

1. `python3 -c "import ast; ast.parse(open('<each file>').read())"` × 8
2. `PYTHONPATH=$PWD python3 -m pytest tests/test_debouncer.py tests/test_rules_regression.py tests/test_incremental_resume.py -v`
3. `PYTHONPATH=$PWD python3 -c "from stk.rules.generator import RuleEnforcer; from stk.behavior.generator import BehaviorRelationGenerator; from stk.dynamic.incremental_updater import IncrementalEngine; print('imports OK')"`
4. 端到端: `bash scripts/long_run/run_long_run.sh smoke` 跑 2400 帧, 检查 phase5_graph.json 结点数包含足够 ResponsibilityAssignment (≥300, 与 SafetyViolation 数量级匹配)
5. `git add ... && git commit -m "feat(rules+behavior): RSS params + responsibility + debouncer + detector fixes for causality tracing"`

## 不在第一阶段范围内（明确划清）

- scenario 层 build_snapshot 空壳 / compute_in_lane 几何错误 / 坐标系 left-hand 翻转 (8 个 BLOCKER) — 留到第二阶段
- 空间函数 ego_only 筛选 (B4)、scenario_library.py 物理位置错乱 — 留到第二阶段
- 移除 `eval()` in `generator.load_dict`、_active_rels 不恢复的修复 — 留到第二阶段
- 增量引擎 rule_events 状态机接线 — 留到第二阶段

理由：第一阶段聚焦"溯因能在规则层闭环"，第二阶段再处理"上游数据形态正确"。规则层的修复不依赖 scenario 层修复 (规则用 vehicle.get("lane_id") 之类的字段, 即使 upstream lane 匹配不准, 也能跑出责任归因, 满足最小可用闭环)。

## 风险与回滚

- RSS 参数从 (rho=0.1, a_max_accel=1.5) → (0.3, 0.5) 会让 d_min 显著增大 (约 3-5 倍), **历史 2400 帧 smoke 中的 SafetyViolation 数量会上升**. 这是预期行为, 不是回归错误。
- 责任归因节点数会从 ≈308 (RSS-only) 增加到 ≈所有 SafetyViolation (估计 5-10x). KG 文件可能从 111 MB 增至 ~120 MB. 可接受。
- 如出现单测失败, 按文件粒度 git checkout 单独回滚。
