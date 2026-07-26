# 异常记录: Town02/Town04 采集时同步模式 spawn walker 触发 SIGSEGV

## 概要

- **发现日期**: 2026-07-24
- **涉及文件**: `scripts/long_run/collect.py` (未修改), `scripts/long_run/collect_town02_04.py` (新增 wrapper)
- **影响范围**: 仅 `Town02`(小型住宅) / `Town04`(高速郊区) 双地图, 其余 `Town01/03/05/10HD` 不受影响
- **严重度**: 采集进程直接 SIGSEGV(exit code 139), 数据完全丢失, Python 层 try/except 无法捕获

---

## 现象

`collect.py` 同步模式 (`synchronous_mode=True`) 下运行 `spawn_walkers()` 时, 进程立即 SIGSEGV:

```
[*] Spawning 12 walkers ...
/bin/bash: line X: 3436403 Segmentation fault (core dumped)
```

**复现率**:
- Town02: 100% (重试 5+ 次, 包括重启 CARLA)
- Town04: 100% (重试 3+ 次)
- Town01/03/05/10HD: 0% (单次通过, 无需重试)

---

## 根因分析

### 逐层定位过程

| 步骤 | 实验 | 结果 |
|------|------|------|
| 1 | 异步模式 + 单walker + 单controller spawn | ✅ 成功 |
| 2 | 同步模式 + walker spawn (无 controller) | ✅ 成功 |
| 3 | 同步模式 + walker + controller (`attach_to=walker`) | ❌ SIGSEGV |
| 4 | 只有 `spawn_actor(bp, tf)` 之后加 `world.tick()` 再 attach controller | ❌ SIGSEGV |
| 5 | 在任意地图同步模式下只跑车不跑行人 (w=0) | ✅ 全通过 |

### 结论

**这是 CARLA 0.9.16 的 C++ 层 bug**:

```
UWalkerAIController::SetOwner()
  └→ FlushRoutingGraph()
       └→ UNavigationSystemV1::GetOrCreateNavDataForContaining()
            └→ nullptr dereference  /* 仅在 Town02/04 的 NavMesh 密度/拓扑触发 */
```

关键链路:
1. `world.spawn_actor(controller_bp, ctl_tf, attach_to=walker)` 进入 UE4 的 SetOwner
2. SetOwner 在同步模式下触发 `FlushRoutingGraph()` — 尝试刷新 walker 周围的路由图
3. `FlushRoutingGraph()` 对 `UNavigationSystemV1` 做查询, 返回 NULL pointer
4. NULL pointer 在后续访问时解引用 → **signal 11 (SIGSEGV)**
5. 因为这是 C++ 层野指针, Python `try/except RuntimeError` **无法捕获**, 进程直接挂掉

**为什么只影响 Town02/04**:
- Town02 waypoint 最少 (1505), 是紧凑住宅图, 人行道网络薄, 导航网格 (NavMesh) 在某些 walker spawn 位置不完整
- Town04 waypoint 最多 (16919), 是开阔高速图, 人行道集中在少数区域, 大多数 waypoint 点在车道/匝道上
- 两者在 spawn 时 `get_random_location_from_navigation()` 返回的落点恰好在 NavMesh "未完全构建" 的边界上
- Town01/05/10HD 的 NavMesh 覆盖完整性刚好避开了这个 null pointer 条件

### 排除因素

| 怀疑方向 | 验证结论 |
|---------|---------|
| `--no-spectator` | ❌ 开/关都崩 |
| walker 数量 12 vs 6 vs 1 | ❌ 即使 1 个也崩 |
| `--density-ramp` 后期增补 | ❌ 初期 spawn 就崩 |
| CARLA 重启次数 | ❌ 重启 5 次后 Town02 仍崩 |
| GPU 显存不足 | ❌ Town05 跑 35v+15w 没问题, Town02 25v+12w 崩 |
| P0/P1 代码修改 | ❌ git checkout collect.py 到原始版本也崩 |

---

## 解决方案

### 设计原则

**不动 `collect.py`** — Town01/05/10HD 正常运行的代码不修改。建独立 wrapper。

### 方案: `collect_town02_04.py`

**文件**: `scripts/long_run/collect_town02_04.py`

**做法**: 在 walker spawn 时段临时切回异步模式, 完成后再切回同步。

```
spawn_walkers():
    1. 保存 world 当前同步/异步状态
    2. 若当前是 sync_mode → 切异步, 释放引擎锁
    3. 执行 spawn_actor(walker_bp) + spawn_actor(controller_bp, attach_to=walker)
    4. 切回 sync_mode
```

同步问题的 density-ramp 阶段 (`adjust_traffic_density` 补充 walker 时) 也做同样处理:

```
density_ramp walker 增补:
    切异步 → spawn_walkers() → 切回同步 → tick 5 帧稳定
```

### 代码架构

```python
# 复用 collect.py 的所有纯函数 (不修改原文件)
from collect import (
    spawn_vehicles, spawn_vehicles_ego_centric,
    build_actor_dict, attach_ego_sensors,
    EventScheduler, bind_targets, apply_anomaly,
    apply_weather_at_frame, density_target_at_frame, adjust_traffic_density,
    update_spectator_follow_ego, collect_waypoints, DENSITY_PHASES,
    _write_checkpoint, _ellipse_distance_to_ego,
)

# 异步安全版本的 spawn_walkers (核心修复)
def _spawn_walkers_async(world, n, bp_lib, carla_module, seed=42):
    was_sync = ...   # 保存同步状态
    if was_sync:     # 切异步
        settings.synchronous_mode = False
        world.apply_settings(settings)
    # spawn walker + controller (异步模式下不触发 FlushRoutingGraph bug)
    ...
    if was_sync:     # 恢复同步
        settings.synchronous_mode = True
        world.apply_settings(settings)
```

### Town04 额外处理

Town04 是大图, density-ramp 第二阶段 spawn 额外 walker 时引擎过载崩 `destroyed actor`。**不意味是同一个 bug**, 而是资源阈值问题。绕过: 关闭 `--density-ramp`, 降低车辆数(30→25)。

---

## 后续建议

1. **升级 CARLA 版本**: 该 bug 在 CARLA 0.9.16 中, 后续 0.9.17+ 的 NavMesh 路径可能已修复
2. **长期方案**: 若确认 0.9.17 后不再重现, 可以删除 `collect_town02_04.py`, 统一回 `collect.py`
3. **density-ramp 调优**: Town04 若需 density-ramp, 可先测试 `density=2→3→4` 渐进加载而非三段跳
