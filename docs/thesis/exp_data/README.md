# 第 6 章实验数据目录

本目录存放第 6 章实验所用的原始结果 JSON，**体量较小（~4.5 KB/文件）且是论文表/图的直接来源**，因此将其纳入版本控制以便复现和引用。

## 文件清单

| 文件 | 对应论文章节 | 来源 | 描述 |
|------|------------|------|------|
| `cross_town_eval_full.json` | §6.7 表 6-18 | `scripts/long_run/eval_cross_town.py` | 跨 Town OOD 泛化评估：Town01/02/04/05 零样本 FPR + Town10HD 域内 F1 |

## 计划添加

| 文件 | 对应论文章节 | 来源 | 状态 |
|------|------------|------|------|
| `pr_curve_v6_41K.json` | §6.4.4 表 6-13 续 2 | `scripts/long_run/pr_curve_scan.py` | 📝 待复制 |
| `ablation_results.json` | §6.4.3 表 6-13 续 | `scripts/long_run/run_ablation.py` | 📝 待复制 |
| `results.json` (v6 41K) | §6.4.2 表 6-13 | `scripts/long_run/exp_realdata.py` | 📝 待复制 |

## 生成方式

所有结果均可通过对应脚本用以下命令复现：

```bash
# 跨 Town 评估
python scripts/long_run/eval_cross_town.py \
    --threshold 0.15 --max-actors 30 \
    --output docs/thesis/exp_data/cross_town_eval_full.json

# PR 曲线扫描
python scripts/long_run/pr_curve_scan.py \
    --all --device cuda:0 \
    --output docs/thesis/exp_data/pr_curve_v6_41K.json

# 消融实验 (每个消融约 15 min)
python scripts/long_run/exp_realdata.py --max-frames 2000 --epochs 20 [--oversample-pos] \
    [--focal-gamma 3.0] [--alpha-cap 500] \
    --out-dir docs/thesis/exp_data/ablations/_[subdir]/
```

## 注意

- `exp_results/realdata/` 下的原始训练产物（`model.pt` 132 MB, `history.json` 9 KB, 训练日志等）**不纳入版本控制**，可通过重跑脚本复现
- 只复制**论文表/图的直接数据来源**，保持仓库轻量
