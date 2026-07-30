# 第 6 章实验数据 — 快捷索引

> ⚠️ **本目录内容已迁移至 `exp_results/`**
>
> 实验结果数据的唯一权威源已移至项目根目录的 `exp_results/` 仓，结构更完整（含模型权重、
> checkpoint、训练日志、审查脚本与 37 项自动验证报告）。
>
> 本目录 `docs/thesis/exp_data/` **仅保留论文引用级 JSON 子集的快捷索引**，
> 所有数据以 `exp_results/` 内容为准。

## 目录结构

本目录保留 6 类文件的"轻量快照"，便于论文评审直接阅读（无需浏览模型权重）。

```
docs/thesis/exp_data/
├── README.md                      # 本文件
├── main_results/                 # §6.4.2 主结果 v6 41K 全量训练
│   ├── results_v6_41K.json          → 表 6-13 (P/R/F1/TP/FP/FN/TN)
│   └── history_v6_41K.json          → 训练曲线（每 epoch 的 loss/F1）
├── ablation/                     # §6.4.3 真实数据集消融实验
│   ├── A_no_oversample/             → 消融 A: 去 oversample (F1=0.842)
│   ├── B_no_skipconn/               → 消融 B: 去 skip connection (F1=0.021)
│   ├── C_gamma2/                    → 消融 C: γ=3→2 (F1=0.842)
│   └── D_alpha100/                  → 消融 D: α_cap=500→100 (F1=1.000)
├── pr_curve/                     # §6.4.4 PR 曲线分析
│   └── pr_curve_v6_41K.json         → 表 6-13 续 2 (21 阈值扫描，覆盖 [0.01, 0.46])
├── cross_town/                    # §6.7 跨 Town OOD 泛化
│   ├── cross_town_eval_full.json    → 表 6-18 (1200 OOD 帧 + 4110 ID)
│   └── cross_town_eval_subset.json  → 早期 100 帧/城镇快速评估
└── logs/                         # 训练/评估日志
    ├── train_v6_41K.log
    ├── train_v6_2000f_oversample.log
    ├── ablation_A.log … ablation_D.log
    └── pr_curve_v6_41K.log
```

## 权威数据位置（推荐）

| 你想要的 | 对应路径（项目根目录 relative） |
|---------|-------------------------------|
| **完整数据 + 模型权重 + 审查脚本** | [`exp_results/`](/exp_results/README.md) |
| 主实验（§6.4.2） F1=1.000 | `exp_results/main_v6/checkpoint/results.json` |
| 4 组消融（§6.4.3） A/B/C/D | `exp_results/ablation/{A,B,C,D}/results.json` |
| PR 曲线 21 阈值（§6.4.4） | `exp_results/pr_curve/scan_v6_41K.json` |
| 跨 Town OOD（§6.7） FPR=4.43% | `exp_results/cross_town/eval_full.json` |
| 37 项数据自检报告 | `exp_results/_verify_report.txt` |
| 一键验证脚本 | `exp_results/verify_data.py` |

## 数据 ↔ 论文表/图对应

| 论文位置 | 表/图 | 数据来源（exp_results/） | 关键指标 |
|---------|------|-------------------------|---------|
| §6.4.2 | 表 6-13 | `main_v6/checkpoint/results.json` | F1=1.000，TP=1050，FP=0，FN=0 |
| §6.4.2 | 表 6-13 | `main_v6/checkpoint/history.json` | 20 epoch 训练曲线 |
| §6.4.3 | 表 6-13 续 | `ablation/{A,B,C,D}/results.json` | 4 组消融对比 |
| §6.4.4 | 图 6-2 + 表 6-13 续 2 | `pr_curve/scan_v6_41K.json` | 21 阈值 P/R/F1 (16 个阈值 F1=1.000) |
| §6.7 | 表 6-18 | `cross_town/eval_full.json` | OOD FPR=4.43% |

## 复现命令

所有结果可用以下命令在 `data/dataset/` 真实数据上复现：

```bash
# 1. 主结果（v6 41K 全量训练，约 15 min）
python scripts/long_run/exp_realdata.py \
    --max-frames 41150 --epochs 20 \
    --oversample-pos 5 --focal-gamma 3.0 --alpha-cap 500 --threshold 0.15 \
    --out-dir exp_results/main_v6/

# 2. 消融实验（每条约 15 min）
# A (去 oversample)
python scripts/long_run/exp_realdata.py --max-frames 2000 --epochs 20 \
    --focal-gamma 3.0 --alpha-cap 500 \
    --out-dir exp_results/ablation/A_no_oversample/
# B (去 skip conn) — 需修改 stk/gnn/k_hstgan.py 移除 h_temporal += h_spatial 后训练
# C (γ=2)    --focal-gamma 2.0
# D (α=100)  --alpha-cap 100

# 3. PR 曲线扫描（约 1 min，依已训练模型）
python scripts/long_run/pr_curve_scan.py --all \
    --output exp_results/pr_curve/scan_v6_41K.json

# 4. 跨 Town OOD 评估（约 2.5 min，依已训练模型）
python scripts/long_run/eval_cross_town.py --threshold 0.15 --max-actors 30 \
    --output exp_results/cross_town/eval_full.json
```

## 时间戳 & commit

| 提交 SHA | 内容 | 日期 |
|---------|------|------|
| `f2b3c0e` | 主训练 + 消融脚本与结果 | 2026-07-29 |
| `bb8e46f` | 论文 §6.4.2/§6.4.3/§6.4.4 内容更新 | 2026-07-29 |
| `5b905b8` | 跨 Town 评估脚本与结果 + §6.7 更新 | 2026-07-29 |
| `683ea15` | 本目录建立 + .gitignore 例外 | 2026-07-29 |
| `b1ed920` | `exp_results/` 仓建立（36 个文件 / 1.3 MB，含模型权重 + 审查脚本 + 37 项 PASS 报告） | 2026-07-30 |
| _(本次提交)_ | `exp_results/` 为权威源；本目录降级为"快捷索引 + 轻量 JSON 快照" | 2026-07-30 |

## 备注

- **权威性**：`exp_results/` 含模型权重 + checkpoint + 训练日志 + 37 项 PASS 自动审查报告，
  是评审/复现的一手来源。本目录保留的 21 个 JSON/log 是论文表/图直接引用的最小子集。
- **标注一致性**：论文表 6-13 续中的 P/R/F1 节选自 `ablation/*/results.json` 的 `final_metrics` 字段
- **跨 Town 评估的 OOD 帧全部为正常帧**（n_pos=0），因此 cross_town 表中 P/R/F1 无意义，主要看 **FPR**（假阳性率，越低越好）
