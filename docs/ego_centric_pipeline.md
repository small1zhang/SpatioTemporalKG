# Ego-Centric Filter Pipeline — 设计与使用手册

> 阶段 1-6 全部改动汇总（FE-1 ~ FE-18，19 个 commit）
> 起始 baseline: `5675a99` 之前的 master；最终落地: `b62e272`

## 0. TL;DR

| 维度 | 改动 |
|------|------|
| 核心思路 | 围绕 ego 自车的笛卡尔椭圆 ROI，过滤 / 评分 / 裁剪一切 KG 输出 |
| 影响层 | `stk/filter` (新增) · `stk/rules` · `stk/behavior` · `stk/scenario` · `stk/storage/serializer.py` · `scripts/long_run/*` · `scripts/pipeline/run_phases_1_5.py` |
| 行为开关 | `--ego-centric` (collect) / `--ego-id` (pipeline) / `--legacy-full-pairing` (config) |
| 输出字段 | `metadata.spawn_mode` · `frame_actors.csv` 追加 4 个 ego 相对字段 · `dataset_index.spawn_info` |
| 不变性 | 阶段 1-3 所有 `stk/` 子包零修改 (FE-1 ~ FE-13 commit 与阶段 3 终态 `6d141f1` 比对全无 diff) |
| 回归 | 499 passed / 6 failed (Neo4j 真连接环境失败，与本改动无关) |

---

## 1. 设计动机

旧版 KG 把场景内所有 actor 两两配对生成 `VehiclePairRelation`，20min 长跑 24000 帧会产出 ~5.4M 边、phase5_graph.json 1.3 GB。绝大多数边与 ego 无关，下游异常检测 / RSS 验证用不到。

阶段 1 起围绕 ego 自车建立笛卡尔椭圆 ROI：

```
            ╭────────────────╮
            │   front (70m)   │
            │       ego       │
       ╭────┤  ←           →  ├────╮
side  │    │       ↓         │    │  side
(50m) │    │   rear (30m)    │    │  (50m)
       ╰────┰────────────────╰────╯
```

只有 ROI 内的 actor 才参与配对、行为关系、规则评估，并按类别差异化半径（轿车 70m / 行人 30m / 自行车 50m 等）。同时引入「重要性打分 + 边稀疏化 + 静态背景外移」三个正交优化器，把 phase5 边数压缩 ≥95%。

---

## 2. 阶段总览

| 阶段 | 主题 | FE 编号 | 关键 commit |
|------|------|---------|--------------|
| 1 | 笛卡尔椭圆 ROI + EgoCentricFilter | FE-1, FE-2, FE-3 | `63a1b58`, `5675a99`, `bc1828c` |
| 2 | 按类别差异化半径 + 行人接口 | FE-5, FE-5b, FE-6, FE-7 | `2e32e9f`, `771e6c7`, `1518c15`, `583bfd5` |
| 3 | 生命周期 / 重要性 / 边稀疏 / 背景外移 | FE-8 ~ FE-13 | `7ca3e6f` → `6d141f1` |
| 4 | collect.py + run_phases_1_5.py ego-centric spawn | FE-14, FE-15, FE-16 | `007b277`, `89e06cd`, `3d831e1` |
| 5 | build_anomaly_dataset 适配 ego-centric | FE-17 | `0e73a72` |
| 6 | 端到端 runner 脚本对接 | FE-18 | `b62e272` |

---

## 3. 配置层 (config/ego_centric.yaml)

`EgoCentricConfig` 字段（见 `stk/config.py`）：

```yaml
# config/ego_centric.yaml
enabled: true
ego_id: null                    # null = 自动取第一辆 vehicle
radius_front: 70.0
radius_rear: 30.0
radius_side: 50.0
legacy_full_pairing: false      # true = 关 ego-centric, 用旧全连接

# 阶段2: 按类别差异化半径
radii_by_category:
  car:        {front: 70, rear: 30, side: 50}
  truck:      {front: 80, rear: 40, side: 55}
  motorcycle: {front: 50, rear: 25, side: 35}
  bicycle:    {front: 50, rear: 25, side: 35}
pedestrian_radius_front: 30
pedestrian_radius_rear:  15
pedestrian_radius_side:  20
```

CLI 高优先级覆盖 YAML；YAML 缺省值兼容旧版（`legacy_full_pairing=true` 时所有 ego-centric 路径短路）。

---

## 4. 数据采集层 (`scripts/long_run/collect.py`)

### 4.1 ego-centric NPC spawn (FE-14)

新增 4 个 CLI 参数：

```bash
--ego-centric                # 启用椭圆 spawn
--npc-radius-front  70.0     # 与椭圆 ROI 一致
--npc-radius-rear   30.0
--npc-radius-side    50.0
--spawn-offset 5             # ego 始 spawn_point 偏移 (可选)
```

启用后流程：
1. ego 取 `spawn_points[spawn_offset]` (offset=0 → 第一个点)
2. `_ellipse_distance_to_ego()` 筛选在椭圆内的 spawn_point
3. `spawn_vehicles_ego_centric()` 把 NPC 撒在椭圆内点；不足时从全图补
4. `spawn_walkers_ego_centric()` 同理 (用 `get_random_location_from_navigation`)
5. `metadata.spawn_mode = "ego_centric"`；失败回退 default 模式

### 4.2 bind_targets 按车道/距离筛选 (FE-15)

`anomaly_scheduler.bind_targets()` 替换原 `rng.choice` 占位实现：

| anomaly_type | 筛选策略 |
|---|---|
| `sudd_brk` / `sudd_stp` | ego 正前方 5-30m 同车道 NPC |
| `avd_col` / `avd_col_track` | ego 侧向 3-10m 同向 NPC |
| `cut_in` | 相邻车道前方 10-20m NPC |
| 其他 | 距 ego 最近 NPC |

调用方 (`collect.py`) 传入 `ego_transform` + `vehicle_waypoints: {id_str: (road_id, lane_id)}`；任一缺失时退化到 rng.choice（向后兼容）。

---

## 5. 处理层 (`scripts/long_run/pipeline.py` + `stk/storage/serializer.py`)

### 5.1 新 CLI 参数 (FE-13)

```bash
--ego-id <id>                  # 透传到 EgoCentricConfig, override YAML
--importance-threshold 0.30    # ≤0 = 关闭; 实体 score<threshold 直接 drop
--exclude-lanes                # 排除 in_lane / containsLane 等包含语义边
--prune-edges                  # 边稀疏化 (同类边按 importance top-k)
```

### 5.2 serialize_graph 集成 (FE-12)

`stk/storage/serializer.py:serialize_graph()` 新增显式三步过滤：

```
raw_graph
  → LifecycleTracker (ENTER/UPDATE/EXIT/FORGET)   # FE-8
  → BackgroundFilter (静态路网外移)                # FE-11
  → ImportanceScorer (E1-E5 打分)                  # FE-9
  → EdgePruner (按 importance top-k)               # FE-10
  → EgoCentricFilter (笛卡尔椭圆 ROI)             # FE-1
  → coalesce_containment (区间边合并)
  → 分片输出 graph_XXXX_<start>_<end>.json
```

所有过滤步骤是真函数不可变转换，可单独 ts 测试。

### 5.3 runner 脚本 (FE-18)

`run_e2e_5min.sh` 与 `run_phase5_shard.sh` 数组化参数：

```bash
EGO_CENTRIC=1
NPC_RADIUS_FRONT=70.0
IMPORTANCE_THRESHOLD=0.30
PRUNE_EDGES=1
EXCLUDE_LANES=1
# 环境变量可覆盖: RUN_DIR=... IMPORTANCE_THRESHOLD=0.10 ...
```

`run_phase5_shard.sh` 自动读 `metadata.json` 的 `ego_id` / `spawn_mode`，有 ego_id 时自动传 `--ego-id`。

---

## 6. 数据集构建 (`scripts/long_run/build_anomaly_dataset.py`, FE-17)

### 6.1 4 个 actor 新字段

`ACTOR_FIELDS` 末尾追加：

| 字段 | 含义 |
|---|---|
| `distance_to_ego` | 与 ego 欧式距离 (m) |
| `lon_to_ego` | 纵向 (ego 朝向前方为正) |
| `lat_to_ego` | 横向 (ego 左侧为正) |
| `in_ego_roi` | 1=椭圆 ROI 内, 0=外 |

每帧用 `is_ego=True` 的 actor 作原点 + `heading_rad` 旋转坐标系，向后填到 actor_rows 末尾的 dict。

### 6.2 CLI 参数

```bash
--filter-ego-roi    # 启用后只保留 in_ego_roi==1 的 actor 行
```

### 6.3 dataset_index.json 升级 v1.0 → v1.1

```json
{
  "schema_version": "1.1",
  "spawn_info": {
    "modes": {"ego_centric": 2, "default": 1},
    "ego_centric_runs": 2,
    "default_runs": 1,
    "total_runs": 3
  },
  "sources": [
    {"path": "...", "spawn_mode": "ego_centric", ...}
  ]
}
```

向后兼容：旧 run（无 `spawn_mode` 字段）默认 `"default"`，4 个 ego 字段写 0。

---

## 7. 调试配置 (`.vscode/launch.json`)

> 本地文件，被 `.gitignore` 忽略，不入库

12 条配置覆盖各阶段：

- `Collect: default spawn (baseline)`
- `Collect: ego-centric NPC spawn`
- `Collect: ego-centric + spawn-offset 5`
- `Pipeline: shard + coalesce only (baseline)`
- `Pipeline: + importance + prune + exclude-lanes`
- `Pipeline: + importance + prune (low threshold)`
- `Dataset: build (with ego fields)`
- `Dataset: build + filter-ego-roi`
- `Batch: run_phases_1_5 ego-centric`
- `Unit test: ego filter (no CARLA)`
- `Unit test: full rules (no CARLA)`
- `Unit test: all ego + filter (no CARLA)`

---

## 8. 测试矩阵

| 模块 | 测试文件 | 覆盖点 |
|------|---------|-------|
| `stk/filter/roi.py` | `tests/test_ego_centric_filter.py` | 26 条：纯函数 + EgoCentricConfig round-trip |
| `stk/filter/generator.py` | `tests/test_ego_centric_filter.py` | 含 8 条 `TestEgoCentricRSSPairs` 集成 |
| `stk/rules` | `tests/test_rules_rss.py` | ego×ROI RSS 配对，含 `legacy_full_pairing=true` 切关路径 |
| `stk/filter/importance.py` | `tests/test_importance_scorer.py` | E1-E5 各类别权重 |
| `stk/filter/edge_pruner.py` | `tests/test_edge_pruner.py` | top-k 保留 / 跨类型不混合 |
| `stk/filter/background_filter.py` | `tests/test_background_filter.py` | 静态 actor 跨帧检测 |
| `stk/filter/lifecycle_tracker.py` | `tests/test_lifecycle_tracker.py` | ENTER/UPDATE/EXIT/FORGET 状态机 |
| `stk/storage/serializer.py` | `tests/test_serializer_filtering.py` | serialize_graph 全链路 |
| `stk/behavior` | `tests/test_behavior_egocentric.py` | ego×ROI 配对 |
| `stk/scenario` | `tests/test_scenario_egocentric.py` | `compute_ahead_of` ego_id 参数 |

全套回归：`python -m pytest --ignore=tests/test_pipeline.py -q` → **499 passed / 6 failed**（Neo4j 真连接环境，与本改动无关）。

---

## 9. 一键跑通

```bash
# 端到端 (5min, ego-centric, 全优化):
bash scripts/long_run/run_e2e_5min.sh

# 已有 20min 数据, 只跑优化版 Phase5:
RUN_DIR=data/long_run/run_20260721_150239_24000f \
IMPORTANCE_THRESHOLD=0.30 \
PRUNE_EDGES=1 \
EXCLUDE_LANES=1 \
bash scripts/long_run/run_phase5_shard.sh

# 构建数据集 (含 ego 4 字段, 可选 ROI 过滤):
python scripts/long_run/build_anomaly_dataset.py \
    --run-dir data/long_run/test_5min/run_*_6000f \
    --out data/dataset/Town10HD_ecc \
    --filter-ego-roi
```

---

## 10. 完整 commit 列表

```
b62e272 feat(runner): 端到端脚本对接 ego-centric + 阶段3 新参数 (FE-18)
0e73a72 feat(dataset): build_anomaly_dataset 适配 ego-centric + 4 ego 字段 (FE-17)
89e06cd feat(scheduler): bind_targets 按车道/距离差异化筛选 (FE-15)
007b277 feat(collect): NPC 椭圆 spawn + ego_centric CLI 参数 (FE-14)
3d831e1 feat(pipeline): run_phases_1_5.py ego 标记 + ego-centric spawn (FE-16)
6d141f1 feat(pipeline): CLI 参数 --importance-threshold/--exclude-lanes/--prune-edges/--ego-id (FE-13)
60a275d feat(storage): serialize_graph 集成 ImportanceScorer+EdgePruner+BackgroundFilter (FE-12)
827a3a2 fix: select() keyward arg 'frame_id=' compatible with new signature
b6e4e36 feat(filter): BackgroundFilter 静态背景外移 (FE-11)
569787d feat(filter): EdgePruner 边稀疏化 (FE-10)
ecd0bd6 feat(filter): ImportanceScorer E1-E5 实体重要性打分 (FE-9)
7ca3e6f feat(filter): LifecycleTracker ENTER/UPDATE/EXIT/FORGET 状态机 (FE-8)
583bfd5 feat(scenario): spatial 关系 compute_ahead_of/beside 可选 ego_id (FE-7)
1518c15 feat(behavior): 行为层车辆-车辆对子 ego×ROI 过滤 (FE-6)
771e6c7 test(filter): 按类别差异化 ROI 半径 + 行人接口 10 条测试 (FE-5b)
2e32e9f feat(filter): 按类别差异化 ROI 半径 + 行人接口预留 (FE-5)
bc1828c feat(tests): TestEgoCentricRSSPairs 集成测试 8 条 (FE-3)
5675a99 feat(rules): RuleEnforcer RSS 改 ego×ROI 内他车 + legacy 切关 (FE-2)
63a1b58 feat(filter): EgoCentricFilter + EgoCentricConfig + 笛卡尔椭圆 ROI (FE-1)
```

## 11. 不变性验证

阶段 1-3 完成时 baseline = `6d141f1`，阶段 4-6 结束后比对下列路径全部零 diff：

```
✓ stk/rules
✓ stk/ontology
✓ stk/behavior
✓ stk/filter
✓ stk/storage/serializer.py
✓ stk/dynamic
✓ stk/scenario
✓ stk/config.py
✓ config/rss_rules.yaml
✓ config/traffic_rules.yaml
✓ config/ontology.yaml
✓ config/ego_centric.yaml
```

阶段 4-6 所有改动均落在 `scripts/long_run/*` 与 `scripts/pipeline/run_phases_1_5.py` 两个目录下。
