# 阶段 4 实施计划：场景生成 ego 中心化

## 范围

改造范围：`collect.py` + `anomaly_scheduler.py` + `run_phases_1_5.py`（全改）

改造目标：
1. `collect.py` 的 NPC spawn 改为以 ego 为中心的环形分布（复用 `config/ego_centric.yaml` 的 `radius_front/rear/radius_side` 配置）
2. `anomaly_scheduler.py` 的 `bind_targets()` 实现真正的按车道/距离筛选（替换当前 `rng.choice`）
3. `run_phases_1_5.py` 增加 ego 标记与 ego-centric spawn 选项

## FE-14：collect.py NPC 环形 spawn

在 `scripts/long_run/collect.py` 中新增 `spawn_vehicles_ego_centric()` 函数：

```python
def spawn_vehicles_ego_centric(world, n: int, bp_lib, map_, carla_module,
                                ego_spawn_point, radius_front=70.0,
                                radius_rear=30.0, radius_side=50.0,
                                seed=42) -> List[Any]:
    """以 ego 为圆心，在前后差异化椭圆内随机撒 NPC.
    
    步骤:
      1. 获取地图所有 spawn_points
      2. 对每个 spawn_point 计算在 ego 车体坐标系下的 (lon, lat)
      3. 只保留满足椭圆方程(lon/R_long)²+(lat/R_side)²≤1.0 的点
      4. 从这些点中随机选 n-1 个 (去重), 第一个点留给 ego
      5. 如果没有足够点, 回退到最近的可用点
    """
```

### `main()` 中的改动:
- 新增 CLI 参数 `--ego-centric`、`--npc-radius-front`、`--npc-radius-rear`、`--npc-radius-side`、`--npc-count`
- 当 `--ego-centric` 启用时，走 `spawn_vehicles_ego_centric()` 而非默认的 `spawn_vehicles()`
- ego 默认用 spawn_points[0] 或 `--spawn-offset` 指定
- 写入 metadata 时标记 `"spawn_mode": "ego_centric"` 或 `"spawn_mode": "default"`

### walker 生成同样 ego 中心化:
- `spawn_walkers_ego_centric()`：用 `world.get_random_location_from_navigation()` 反复采样并过椭圆判定

### 向后兼容:
- 默认不做任何改变（`spawn_mode="default"` → 走原有随机路径）
- 只在新参数 `--ego-centric` 启用时改造

## FE-15：bind_targets 实现真正的"前后 30m"按车道筛选

在 `scripts/long_run/anomaly_scheduler.py` 中修改 `bind_targets()`：

```python
def bind_targets(events, spawned_vehicles, ego_id, seed=42):
    """把每个异常事件的 target_actor_id 绑定到 ego 前后方同车道的车。
    
    针对不同类型的 event 做差异化绑定:
      - sudd_stp / sudd_brk: 选 ego 正前方最近 NPC (同车道, 5-30m)
      - avd_col / avd_col_track: 选 ego 侧向 3-10m 的 NPC (侧方切入)
      - cut_in: 选 ego 相邻车道前方 10-20m 的 NPC
      - right_turn_conflict / jaywalker: 选 ego 前方路口处行人
      - static / others: 随机选最近的 NPC
    
    若找不到合适 NPC 则回退到随机选择 (原始行为).
    """
```

### 实现依赖:
- 获取 CARLA 地图的 `waypoint` 信息（spawn 时或 tick 中缓存的 `wp` 数据）
- 判断同车道: 通过 `vehicle.get_waypoint()` 的 `road_id` 与 `lane_id`
- 判断前后/侧向: 在 tick 循环中每帧评估位置关系

### 运行重绑:
- 在 `collect.py` 的 tick 循环中（line 727-736 附近），每帧调用 `rebind_anomaly_targets()` 
- 实时评估 ego 与 target NPC 的实际位置与车道关系，若偏离则重绑到更合适的 NPC

## FE-16：run_phases_1_5.py ego 标记 + ego-centric spawn

在 `scripts/pipeline/run_phases_1_5.py` 中：

1. **第 198 行**：首车设置 `role_name="hero"`（当前完全是 `"autopilot"`）
2. **第 305 行**：`is_ego` 正确标记为 `True`（当前全是 `False`）
3. **新增 spawn 选项**：与 collect.py 类似的 `--ego-centric` 环形 spawn 逻辑
4. **添加 CLI 参数**: `--ego-centric`, `--npc-radius-front` 等，与 collect.py 对齐
5. **添加 `spawn_mode` 字段** 到 metadata

## 提交与验收

- 3 个 FE commit（FE-14, FE-15, FE-16），每个独立可跑
- 验收：
  - `python scripts/long_run/collect.py --help` 显示全部 4 个新参数
  - 在不启用 `--ego-centric` 时全部行为不变（默认走随机路径）
  - `bind_targets` 不再纯随机：回归时原有测试不影响（bind_targets 只在 collect.py 被调用）
  - 不变量检查：所有阶段 1-3 的 499 测试保持绿色