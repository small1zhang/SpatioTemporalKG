# CARLA Cut-in Safety-Critical Scenario for Paper

生成论文定性分析用的 CARLA 切入场景序列。

## 场景

```
V17（相邻车道）→ 切入 → V17 减速 → 不安全跟车 → RSS 违反 → 恢复
```

## 架构

| 角色 | 控制方式 | 说明 |
|---|---|---|
| **Ego** | Traffic Manager 自动驾驶 | 巡航 13 m/s；恢复期减速至 7 m/s |
| **V17** | **手动 VehicleControl** | 切入/减速/加速全部由脚本精确控制，不依赖 TM |
| **背景车** | Traffic Manager | 6 辆，随机种子固定 |

V17 不走 TM 路径规划，**100% 确定性**，不会跑偏到岔路。

## 使用步骤

### 1. 启动 CARLA

```bash
cd /home/aisecurity/Carla
./CarlaUE4.sh -quality-level=Low -carla-rpc-port=2000 -RenderOffScreen &
```

### 2. 运行脚本

```bash
cd /home/aisecurity/01_ZHB/SpatioTemporalKG/scripts/carla/cut_in_paper

# Town04（稳定，推荐）
conda activate stk
python cut_in_scenario.py --host localhost --port 2000 --map Town04

# 或 Town10HD
python cut_in_scenario.py --host localhost --port 2000 --map Town10HD
```

### 3. 查看产出

```
output/
├── rgb/            220 帧全程 RGB (1600×900)
│   ├── 000000.png
│   └── 000219.png
├── keyframes/      4 张关键帧（带文字标注）
│   ├── t01_normal_7.6s.png
│   ├── t02_cut-in_8.8s.png
│   ├── t03_unsafe_following_12.5s.png
│   └── t04_recovery_15.0s.png
└── logs/
    ├── per_frame.csv    逐帧状态（含 RSS 残差）
    └── key_frames.csv   关键帧汇总
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `localhost` | CARLA 地址 |
| `--port` | `2000` | CARLA 端口 |
| `--map` | `Town04` | 地图 |
| `--seed` | `2026` | 随机种子 |
| `--out` | `output` | 输出根目录 |
| `--bg` | `6` | 背景车辆数 |

## 时序设计（tick = 0.1s）

| 时段 (s) | tick | 阶段 | V17 行为 |
|---|---|---|---|
| 0.0–8.0 | 0–80 | 正常跟车 | 相邻车道巡航 12 m/s |
| 8.0–9.5 | 80–95 | 切入 | 车道横移，速度略降 |
| 9.5–10.5 | 95–105 | 稳定同一车道 | 8 m/s，纵向间距缩小 |
| 10.5–14.5 | 105–145 | 不安全跟车 | 持续低速，RSS 残差负值 |
| 14.5–18.5 | 145–185 | 恢复 | 加速至14 m/s，Ego 减速 |
| 18.5–22.0 | 185–219 | 稳定跟车 | 各指标回归正常 |

## 关键帧挑选规则

在目标时间 ±0.3s 窗口内：

- **Normal (~7.6s)**: 横向偏移最大（V17 最明显在邻道）
- **Cut-in (~8.8s)**: 横向偏移 ≈ 1.75m（车道分界线）
- **Unsafe Following (~12.5s)**: 纵向间距最小
- **Recovery (~15.0s)**: 纵向间距最大（恢复最多）

## per_frame.csv 字段

| 字段 | 说明 |
|---|---|
| `frame_id` | 帧号 (0–219) |
| `sim_time` | 仿真时间 (s) |
| `ego_speed_ms` / `v17_speed_ms` | 速度 (m/s) |
| `lon_gap_m` | 保险杠-保险杠纵向间距 |
| `lateral_offset_m` | 横向偏移 (正=左) |
| `rss_safe_m` | RSS 安全距离 |
| `rss_residual_m` | RSS 残差 (负=违反) |
| `gt_anomaly` | 地面真值 (1=RSS 违反) |
| `neural_score` / `anomaly_score` / `delta_gate` / `conflict_kappa` | **NaN**（待模型补全） |

## 重跑

每次运行前脚本自动清理 world + 重新 spawn，**不需要重启 CARLA**。
