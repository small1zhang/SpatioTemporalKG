# 长时间连续采集管线 (Long-run Collection Pipeline)

## 概述

本套件实现长期连续（≥20分钟）的时空动态知识图谱数据采集和构建管线，
替代了原来的单场景分立采集模式。

### 设计原则

- **不修改现有模块内部代码** — `stk/scenario/`、`stk/behavior/`、`stk/rules/`、`stk/dynamic/`、`stk/storage/` 保持不变
- **跨分块状态持续** — `IncrementalEngine` 和 `BehaviorRelationGenerator` 实例跨 chunk 复用，状态不因分块边界重置
- **ego 视角** — ego 自车作为第一观察视角， spectator 自动跟车；异常检测以 ego 为主体
- **异常穿插在正常驾驶中** — 事件调度表按 Poisson 过程在 20 分钟内随机穿插异常，不硬性中断采集

## 文件结构

```
scripts/long_run/
├── anomaly_scheduler.py    # 异常事件调度与构造
├── collect.py              # 长时间 Phase1 采集 (ego跟随+sensors+分块)
├── pipeline.py             # 跨分块编排 Phase2→3→4→5
└── README.md               # 本文件
```

## 使用流程

### 1. 采集（Phase1）

```bash
# 启动 CARLA 服务器:
./CarlaUE4.sh -carla-rpc-port=2000 -quality-level=Low

# 运行采集脚本 (20分钟 = 24000帧 @ 20fps):
python scripts/long_run/collect.py \
    --host localhost --port 2000 \
    --town Town10HD \
    --total-frames 24000 \
    --chunk-frames 2000 \
    --vehicles 20 --walkers 8 \
    --density 2.0 \
    --out data/long_run/Town10HD_xxx
```

关键参数：
- `--total-frames 24000` — 20 分钟 @ 20fps = 24000 帧
- `--chunk-frames 2000` — 每 2000 帧写一个 chunk_XXXX.json（约每 100 秒一个）
- `--density 2.0` — 每分钟期望触发 2 次异常 (~40 次/20min)
- `--no-spectator` — 关闭 spectator 自动跟车（不启动 CARLA View 时可加速）
- `--fps 20.0` — 采集帧率（默认 20fps，可降为 10fps 降低计算负载）

### 2. 编排管线（Phase2→5）

```bash
# 跨分块编排, 输出到 run_dir/phase5/ 下:
python scripts/long_run/pipeline.py \
    --run-dir data/long_run/Town10HD_xxx/run_20260720_120000_24000f \
    --map-name Town10HD
```

可选参数：
- `--neo4j-host localhost` — 若配置了 Neo4j，自动写入图数据库
- `--out /path/to/output` — 指定输出目录（默认 = run_dir/phase5）

## 异常事件类型

| 类型 | 对应场景 | 效果 |
|------|---------|------|
| `sudd_brk` | S31 | 前车急刹 (brake=1.0) |
| `sudd_stp` | S32 | 前车急停并静止 |
| `avd_col` | S20 | 邻车紧急变道切入 |
| `jun_ny` | S21 | 路口不让行 (加速通过) |
| `rev_drive` | S22 | 逆向行驶 |
| `ped_crs` | S10 | 行人横穿 |
| `obs_blk` | S33 | 视线遮挡 |

默认权重: sudd_brk=2, jun_ny=2, 其余=1

## 输出格式

### Phase1 输出 (chunk_XXXX.json)

```json
[{
  "frame_id": 0,
  "elapsed_seconds": 0.0,
  "actors": [{"id": "123", "type": "vehicle", "location": {...}, ...}],
  "traffic_lights": [{"id": "tl_1", "state": "Green", ...}],
  "weather": {"cloudiness": 10, ...},
  "waypoints": [{"road_id": 0, "lane_id": 1, ...}],
  "events": [{"event_type": "Collision", ...}]
}]
```

### Phase5 输出 (phase5_graph.json)

`serialize_graph()` 输出的全图 JSON，包含 `{nodes, edges}` 结构。
类型与单场景 batch 输出一致，但包含更多跨帧合并的 BehaviorRelation 和 Snapshot 节点。

## 与旧管线的区别

| 维度 | run_phases_1_5.py (旧) | collect.py + pipeline.py (新) |
|------|------------------------|-------------------------------|
| 驱动模式 | 每场景一次 | 长时间同步循环 |
| 异常注入 | Scenario Runner / 预设 | 事件调度表 + CARLA 原生 API |
| 输出 | 单 json 文件 | 分块 chunk_XXXX.json |
| 状态持续 | 每次新建 engine | 跨 chunk 复用 |
| Spectator | 不跟车 | 自动跟随 ego |

## 性能参考

- 20 辆车 + 8 行人 + 20fps：~2.5x 实时 (8分钟仿真 = 约3分钟采集)
- Chunk 大小 2000 帧 (~100s)：约 15-30 MB / chunk (JSON)
- 若需要加速：`--fps 10.0` 或 `--no-spectator`
