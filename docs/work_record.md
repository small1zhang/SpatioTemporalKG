# 工作记录: 5 地图 × 20 分钟场景采集 + 知识图谱构建

## 任务清单

| 序号 | 任务 | 状态 | 产出 |
|------|------|------|------|
| 1 | 5 地图各 10 分钟数据采集 | ✅ 完成 | 各 6 chunks, 总计 ~4.5 GB |
| 2 | P0/P1 代码修改 (场景合理性与异常注入) | ✅ 完成 | 3 commits (FE-20/21/22) |
| 3 | 5 地图各 20 分钟数据采集 (含丰富性) | ✅ 完成 | 各 12 chunks, 总计 ~9.4 GB |
| 4 | Town02/Town04 同步模式 SIGSEGV 修复 | ✅ 完成 | 独立 wrapper (collect_town02_04.py) + 问题记录文档 |
| 5 | pipeline.py args 未定义 bug 修复 | ✅ 完成 | 3 处修改 (函数签名/函数体/调用处) |
| 6 | 5 地图图谱构建 (Phase2→5) | ✅ 完成 | 223k nodes, 2.4M edges, 5 × phase5_graph.json |
| 7 | 5 地图可视化分片 + 增强版 HTML | ✅ 完成 | 12 shards × 5 + 11 个交互版块 |

## 1. 采集参数与策略

### 10 分钟数据集

```
命令: collect.py --total-frames 12000 --chunk-frames 2000 --vehicles 25 --walkers 12 --density 2.0 --fps 20.0 --no-spectator
```

| 地图 | 时间 | 结果 |
|------|------|------|
| Town01 | 22:55~22:57 | 6 chunks, 844 MB ✅ |
| Town02 | 09:30~09:41 | 6 chunks, 794 MB ✅ |
| Town04 | 09:41~09:44 | 6 chunks, 748 MB ✅ |
| Town05 | 09:44~09:48 | 6 chunks, 1087 MB ✅ |
| Town10HD | 09:48~09:52 | 6 chunks, 1026 MB ✅ |

### 20 分钟数据集 (带场景丰富性)

```
差异参数: --total-frames 24000 --weather-cycle --density-ramp --spawn-offset
```

| 地图 | 车辆/行人 | 丰富性 | 结果 |
|------|----------|--------|------|
| Town01 | 30v/15w | weather + density-ramp + offset=0 | 12 chunks, 1.8G, 4570 anom ✅ |
| Town02 | 25v/12w | weather + density-ramp + offset=5 | 12 chunks, 1.8G, 5250 anom ✅ |
| Town04 | 25v/8w | weather (无 density-ramp, 防过载) | 12 chunks, 1.9G, 3800 anom ✅ |
| Town05 | 30v/15w | weather + density-ramp + offset=10 | 12 chunks, 2.3G, 5290 anom ✅ |
| Town10HD | 25v/15w | weather + density-ramp + offset=0 | 12 chunks, 2.0G, 4950 anom ✅ |

小计: **60 chunks, 9.4 GB, 23860 anomaly events**

## 2. 代码修改摘要

### P0 修复 (FE-20, commit 60fe2b7)

| 问题 | 原因 | 修法 |
|------|------|------|
| smoke_test.py 行人在车道上 spawn | 从车辆 spawn_points 取位置 + 5m 扰动 | 改用 get_random_location_from_navigation() |
| yaml 缺 vehicle/pedestrian count | 5 yaml × 14 场景多数缺失 | 按场景语义 × 地图缩放比补齐 |
| walker 重试不足 | 10 次重试大地图不够 | 30 次 + under-spawn 告警 |
| 三表字段名不一致 | batch/smoke 用 vehicles/walkers，yaml 用 vehicle_count/pedestrian_count | 全部统一为 vehicle_count/pedestrian_count |

### P1 修复 (FE-21, commit 2084e78)

| 问题 | 原因 | 修法 |
|------|------|------|
| 异常结束不恢复 autopilot | tick() 不返完成事件，collect 拿不到 | tick() 改返回 (active, completed) 二元组；车辆用 set_autopilot, prop 用 destroy |
| 椭圆朝向用 spawn_point yaw | 与 bind_targets 实时 yaw 不一致 | 加 ego_yaw_deg 参数，取 ego.get_transform().rotation.yaw |
| ped_crs 空挂 | bind_targets 当车辆绑 + apply_anomaly else [skip] | walker 绑定 + AI controller go_to_location(ego) |
| obs_blk 空挂 | 从未实现 | 懒 spawn static.prop + 事件结束 destroy |

### P1 修复 (FE-22, commit 61cd67e)

| 问题 | 原因 | 修法 |
|------|------|------|
| S22 无紧急车辆 | spawn 全用 vehicle.* 随机选 | 加 --emergency-vehicles，优先选 ambulance/police/firetruck |
| bind_targets 常降级 rng.choice | 阈值过严 + 无 waypoint 硬返回 False | 三级阈值 (strict→relaxed→nearest→rng)；无 waypoint 退化用 lat 距离 |
| crosswalk 硬编码 False | build_actor_dict 不接收 map_/carla_module | 真值化为 LaneType.Any 查询 + 修 f-string bug × 2 + PYEOF 残留 |

## 3. 采集异常与修复

### 3.1 Town02/Town04 同步模式 spawn walker → SIGSEGV

**根因**: CARLA 0.9.16 在 synchronous_mode=True 下 spawn AI walker controller (attach_to=walker)
触发 UE4 FlushRoutingGraph() → UNavigationSystemV1 NULL pointer 解引用。
Town02(1505 waypoints 小路网) / Town04(16919 waypoints 大郊区) 的 NavMesh 恰好触发。

**修法**: 写独立 wrapper `scripts/long_run/collect_town02_04.py`。
- spawn walker 阶段临时切异步模式 (异步下 controller spawn 正常)
- density-ramp 补充 walker 时同理临时切异步
- 保持 collect.py 不动，Town01/05/10HD 继续用 collect.py

**详见**: `docs/incident_town02_04_sigsegv.md`

### 3.2 pipeline.py args 未定义

**根因**: `process_chunks()` 函数体内直接引用 `args` (在 main() 中定义)，
导致 NameError: name 'args' is not defined。新增 4 个 CLI 参数 (ego_id, importance_threshold, exclude_lanes, prune_edges) 时未补到函数签名。

**修法**: 3 处改动
- L86-97: 函数签名加 4 个参数
- L138-153: `args.xxx` → 参数变量
- L682-690: 调用处补传 4 个参数

## 4. 文件清单

### 新增文件

| 文件 | 用途 |
|------|------|
| `scripts/long_run/collect_town02_04.py` | Town02/Town04 专用采集 wrapper (异步安全) |
| `docs/incident_town02_04_sigsegv.md` | SIGSEGV 根因与修复记录 |
| `docs/work_record.md` | **本文件** — 全部任务汇总 |

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `scripts/pipeline/smoke_test.py` | P0-1: walker 导航点 |
| `scripts/long_run/collect.py` | P0-6: 重试 30+警告; P1-1: tick() 返回 (active, completed); P1-2: emergency-vehicles; P1-1b: ego_yaw_live |
| `scripts/long_run/anomaly_scheduler.py` | P1-1: tick() 二元组; P1-3: ped_crs/obs_blk 实体化; P1-4: 三级阈值 |
| `scripts/pipeline/run_phases_1_5.py` | P0-6: walker 重试 30; P1-2: emergency-vehicles; P1-1b: 椭圆实时 yaw |
| `scripts/pipeline/batch_collect.py` | P0-2: 字段名统一; P1-2: emergency-vehicles 透传 |
| `scripts/carla/spawn_traffic.py` | P1-5: is_on_crosswalk 真值化; fix f-string bug × 2 + PYEOF |
| `map_configs/Town01~10HD.yaml` | P0-2: 补齐 vehicle_count/pedestrian_count |
| `scripts/long_run/pipeline.py` | process_chunks args bug 修复 (3 处)

### 数据产出目录

```
data/long_run/
├── Town01_10min/  (6 chunks, 844 MB)
├── Town02_10min/  (6 chunks, 794 MB)
├── Town04_10min/  (6 chunks, 748 MB)
├── Town05_10min/  (6 chunks, 1087 MB)
├── Town10HD_10min/ (6 chunks, 1026 MB)
├── Town01_20min/  (12 chunks, 1.8 GB)  ← 正在跑 kg
├── Town02_20min/  (12 chunks, 1.8 GB)  ← 等待 kg
├── Town04_20min/  (12 chunks, 1.9 GB)  ← 等待 kg
├── Town05_20min/  (12 chunks, 2.3 GB)  ← 等待 kg
└── Town10HD_20min/ (12 chunks, 2.0 GB) ← 等待 kg
```