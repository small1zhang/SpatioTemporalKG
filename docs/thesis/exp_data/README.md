# 第 6 章实验数据归档

本目录存放第 6 章全部实验的原始结果文件，作为论文表/图的直接数据来源。**仅复制论文引用所需的 JSON 与日志**（约 180 KB）；训练产物（`model.pt` 132 MB 等）保留在 `exp_results/` 不入版本。

## 目录结构

```
docs/thesis/exp_data/
├── README.md                      # 本文件
├── main_results/                 # §6.4.2 主结果 v6 41K 全量训练
│   ├── results_v6_41K.json          → 表 6-13 (P/R/F1/TP/FP/FN/TN)
│   └── history_v6_41K.json          → 训练曲线（每 epoch 的 loss/F1）
├── ablation/                     # §6.4.3 真实数据集消融实验
│   ├── A_no_oversample/             → 消融 A: 去 oversample (F1=0.842)
│   │   ├── results.json
│   │   └── history.json
│   ├── B_no_skipconn/               → 消融 B: 去 skip connection (F1=0.021)
│   ├── C_gamma2/                    → 消融 C: γ=3→2 (F1=0.842)
│   └── D_alpha100/                  → 消融 D: α_cap=500→100 (F1=1.000)
├── pr_curve/                     # §6.4.4 PR 曲线分析
│   └── pr_curve_v6_41K.json         → 表 6-13 续 2 (38 阈值扫描)
├── cross_town/                    # §6.7 跨 Town OOD 泛化
│   ├── cross_town_eval_full.json    → 表 6-18 (1200 OOD 帧 + 4110 ID)
│   └── cross_town_eval_subset.json  → 早期 100 帧/城镇快速评估
└── logs/                         # 训练/评估日志
    ├── train_v6_41K.log             → v6 41K 全量训练主日志
    ├── train_v6_2000f_oversample.log → v6 2000 帧子集（含 oversample）开发日志
    ├── ablation_A.log               → 消融 A 训练日志
    ├── ablation_B.log               → 消融 B 训练日志（崩溃现象现场）
    ├── ablation_C.log               → 消融 C 训练日志
    ├── ablation_D.log               → 消融 D 训练日志
    └── pr_curve_v6_41K.log          → PR 曲线扫描日志
```

## 数据 ↔ 论文表/图对应

| 论文位置 | 表/图 | 数据来源 | 关键指标 |
|---------|------|---------|---------|
| §6.4.2 | 表 6-13 | `main_results/results_v6_41K.json` | F1=1.000，TP=1050，FP=0，FN=0 |
| §6.4.2 | 表 6-13 | `main_results/history_v6_41K.json` | 20 epoch 训练曲线（loss 收敛轨迹） |
| §6.4.3 | 表 6-13 续 | `ablation/{A,B,C,D}/results.json` | 4 组消融对比 |
| §6.4.4 | 图 6-1 + 表 6-13 续 2 | `pr_curve/pr_curve_v6_41K.json` | 38 阈值 P/R/F1 |
| §6.7 | 表 6-18 | `cross_town/cross_town_eval_full.json` | OOD FPR=4.43% |

## 复现命令

所有结果可用以下命令在 `data/dataset/` 真实数据上复现：

```bash
# 1. 主结果（v6 41K 全量训练，约 15 min）
python scripts/long_run/exp_realdata.py \
    --max-frames 41150 --epochs 20 \
    --oversample-pos 5 --focal-gamma 3.0 --alpha-cap 500 --threshold 0.15 \
    --out-dir exp_results/realdata/

# 2. 消融实验（每条约 15 min）
# A (去 oversample)
python scripts/long_run/exp_realdata.py --max-frames 2000 --epochs 20 \
    --focal-gamma 3.0 --alpha-cap 500 \
    --out-dir exp_results/ablations/A_no_oversample/
# B (去 skip conn) — 需修改 stk/gnn/k_hstgan.py 移除 h_temporal += h_spatial 后训练
# C (γ=2)    --focal-gamma 2.0
# D (α=100)  --alpha-cap 100

# 3. PR 曲线扫描（约 1 min，依已训练模型）
python scripts/long_run/pr_curve_scan.py --all \
    --output exp_results/realdata/pr_curve_scan_v6_41K.json

# 4. 跨 Town OOD 评估（约 2.5 min，依已训练模型）
python scripts/long_run/eval_cross_town.py --threshold 0.15 --max-actors 30 \
    --output exp_results/realdata/cross_town_eval_full.json
```

## 时间戳 & commit

| 提交 SHA | 内容 | 日期 |
|---------|------|------|
| `f2b3c0e` | 主训练 + 消融脚本与结果 | 2026-07-29 |
| `bb8e46f` | 论文 §6.4.2/§6.4.3/§6.4.4 内容更新 | 2026-07-29 |
| `5b905b8` | 跨 Town 评估脚本与结果 + §6.7 更新 | 2026-07-29 |
| `683ea15` | 本目录建立 + .gitignore 例外 | 2026-07-29 |
| _(本次提交)_ | 归档结构补全 + README 详化 | 2026-07-29 |

## 备注

- **训练产物未入库**：`model.pt`（550 KB–132 MB）、`history.json` 中的 batch-level 梯度等数据量大，保留在 `exp_results/` 工作目录中，可通过重跑命令复现
- **标注一致性**：论文表 6-13 续中的 P/R/F1 节选自 `ablation/*/results.json` 的 `final_metrics` 字段
- **跨 Town 评估的 OOD 帧全部为正常帧**（n_pos=0），因此 cross_town 表中 P/R/F1 无意义，主要看 **FPR**（假阳性率，越低越好）
