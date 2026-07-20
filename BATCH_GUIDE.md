# SpatioTemporalKG 批量采集指南

## 快速开始

```bash
cd /home/aisecurity/01_ZHB/SpatioTemporalKG

# 1. 冒烟测试 (验证管线)
bash run_smoke.sh                    # 默认 Town10HD + S00
bash run_smoke.sh Town01 S10         # 指定地图+场景

# 2. 批量采集 (全部 70 任务)
bash run_batch.sh

# 3. 从断点继续 (跳过已完成)
bash run_batch.sh --resume
```

## 自定义运行

```bash
# 只跑指定地图
bash run_batch.sh --maps Town10HD,Town01,Town02

# 只跑指定场景
bash run_batch.sh --scenarios S00,S10,S30

# 指定 GPU 和并行数
bash run_batch.sh --gpus 1,2,3 --parallel 3

# 组合使用
bash run_batch.sh --maps Town10HD --scenarios S10,S30,S33 --gpus 2
```

## 场景列表

| 组别 | 场景 | 描述 | 预期违规 |
|------|------|------|---------|
| **A 基线** | S00 | 直行跟车基线 | 0 |
| | S01 | 信号路口正常通行 | 0 |
| | S02 | 行人远距避让 | 0 |
| **B 风险** | S10 | 行人鬼探头 | sv_001 |
| | S11 | 无信号左转冲突 | sv_002 |
| | S12 | 红灯抢行 | sv_003 |
| | S13 | 跟车过近/追尾风险 | sv_004 |
| **C 复杂** | S20 | 汇入主路冲突 | sv_005 |
| | S21 | 三车路口无信号 | sv_006 |
| | S22 | 应急车辆通行权 | sv_007 |
| **D 环境** | S30 | 夜间+行人鬼探头 | sv_008 |
| | S31 | 雨天跨线盲变 | sv_009 |
| | S32 | 施工路段绕行 | sv_010 |
| | S33 | 路口逆光+多行人 | sv_011 |

## 输出结构

```
data/runs/batch/
├── Town10HD/
│   ├── S00/
│   │   ├── phases_XXXXXXXX_100f/
│   │   │   ├── phase1_extraction.json
│   │   │   ├── phase2_scenario.json
│   │   │   ├── phase3_behavior.json
│   │   │   ├── phase3_rules.json
│   │   │   ├── phase4_deltas.json
│   │   │   ├── phase5_graph.json
│   │   │   └── phase5_kg_summary.json
│   │   └── result.json
│   ├── S01/
│   └── ...
├── Town01/
├── Town02/
├── Town04/
├── Town05/
└── batch_summary_XXXXXXXXXX.json
```

## 结果查看

```bash
# 查看批量汇总
cat data/runs/batch/batch_summary_*.json | python -m json.tool

# 查看单任务结果
cat data/runs/batch/Town10HD/S00/result.json | python -m json.tool

# 统计通过率
cat data/runs/batch/batch_summary_*.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Pass: {d[\"passed\"]}/{d[\"total_tasks\"]}')
"
```

## 注意事项

- 每个任务会 cold-boot CARLA (进程级隔离)，约需 20-30s
- 默认使用 GPU 1,2,3 (避开 GPU 0 上的训练任务)
- 支持断点续传，中断后重新运行即可自动跳过已完成任务
- 每个任务输出约 5-15 MB，70 个任务总计约 500 MB - 1 GB
