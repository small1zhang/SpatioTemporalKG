// ━━━━━━━━ 异常时刻完整 KG 切片 (改 frame=1500 即可) ━━━━━━━━
// 4 层全包: 场景 + 实体 + 空间关系 + 行为 + 规则 + 违规 + 责任
// 边方向已修正: violates/responsibleFor/has_maneuver 等都从 SV/RA/Scn 出发
WITH 1500 AS frame

// 第 1 层 — 场景 + 环境 + 路网 + 红绿灯 (ScenarioSnapshot 出发)
MATCH (s:ScenarioSnapshot)-[r1]->(e)
  WHERE s.first_frame <= frame AND s.last_frame >= frame
    AND e.first_frame <= frame AND e.last_frame >= frame
    AND r1.first_frame <= frame AND r1.last_frame >= frame
    AND NOT labels(e)[0] = 'EnvironmentSnapshot'
  WITH frame, collect([s, r1, e]) AS L1

// 第 2 层 — 活跃实体之间的空间关系 (车-车, 车-路, 车-灯, 车-人, 车-Junction)
// 注意: 不走过 ScenarioSnapshot, 直接 vehicle 出发的边
WITH frame, L1
MATCH (a)-[r2]->(b)
  WHERE a.first_frame <= frame AND a.last_frame >= frame
    AND b.first_frame <= frame AND b.last_frame >= frame
    AND r2.first_frame <= frame AND r2.last_frame >= frame
    AND type(r2) IN [
      'in_lane','ahead_of','following','overtaking','opposite_direction',
      'beside','changing_lane','approaching','approaching_pedestrian',
      'nearby_pedestrian','yielding_to','blocked_view','standing_still',
      'in_junction','adjacent_lane','lane_connects'
    ]
  WITH frame, L1, collect([a, r2, b]) AS L2

// 第 3 层 — 行为 (Maneuver + InteractionEvent) - 从 ScenarioSnapshot 出发
WITH frame, L1, L2
OPTIONAL MATCH (s3:ScenarioSnapshot)-[r3]->(beh)
  WHERE s3.first_frame <= frame AND s3.last_frame >= frame
    AND beh.first_frame <= frame AND beh.last_frame >= frame
    AND r3.first_frame <= frame AND r3.last_frame >= frame
    AND type(r3) IN ['has_maneuver', 'has_interaction']
  WITH frame, L1, L2, collect([s3, r3, beh]) AS L3

// 第 4 层 a — 违规链: SV → Vehicle + SV → Rule
//   注意方向! violates: SV → Vehicle (不是 Vehicle → SV)
WITH frame, L1, L2, L3
OPTIONAL MATCH (sv4:SafetyViolation)-[r4:violates]->(v4:Vehicle)
  WHERE sv4.first_frame <= frame AND sv4.last_frame >= frame
    AND v4.first_frame <= frame AND v4.last_frame >= frame
    AND r4.first_frame <= frame AND r4.last_frame >= frame
  WITH frame, L1, L2, L3, collect([sv4, r4, v4]) AS L4a

// 第 4 层 b — SV 定义规则: 但 definedBy 边在导入时被 filter 掉 (没出现在 DB)
//   用 Rule 节点直接 match by sv.rule_code
WITH frame, L1, L2, L3, L4a
OPTIONAL MATCH (sv4b:SafetyViolation)
  WHERE sv4b.first_frame <= frame AND sv4b.last_frame >= frame
OPTIONAL MATCH (rule4b:Rule {rule_code: sv4b.rule_code})
  WITH frame, L1, L2, L3, L4a, collect([sv4b, null, rule4b]) AS L4b

// 第 4 层 c — 责任分配: RA → Vehicle (方向也是反的!)
WITH frame, L1, L2, L3, L4a, L4b
OPTIONAL MATCH (ra5:ResponsibilityAssignment)-[r5:responsibleFor]->(v5:Vehicle)
  WHERE ra5.first_frame <= frame AND ra5.last_frame >= frame
    AND v5.first_frame <= frame AND v5.last_frame >= frame
    AND r5.first_frame <= frame AND r5.last_frame >= frame
  WITH frame, L1, L2, L3, L4a, L4b, collect([ra5, r5, v5]) AS L4c

// 第 4 层 d — RA 通过 sv_id 关联 SV (ra.sv_id == sv.sv_id)
WITH frame, L1, L2, L3, L4a, L4b, L4c
OPTIONAL MATCH (ra5d:ResponsibilityAssignment)-[r5d:responsibleFor]->(v5d:Vehicle)
  WHERE ra5d.first_frame <= frame AND ra5d.last_frame >= frame
OPTIONAL MATCH (sv5d:SafetyViolation)
  WHERE sv5d.first_frame <= frame AND sv5d.last_frame >= frame
    AND sv5d.sv_id = ra5d.sv_id
  WITH frame, L1, L2, L3, L4a, L4b, L4c, collect([ra5d, null, sv5d]) AS L4d

// 拼图: UNWIND 把每层的 [[src, rel, dst], ...] 拆开
UNWIND L1 + L2 + L3 + L4a + L4b + L4c + L4d AS tuple
WITH tuple[0] AS a, tuple[1] AS r, tuple[2] AS b
WHERE a IS NOT NULL AND b IS NOT NULL
RETURN a, r, b
LIMIT 500;
