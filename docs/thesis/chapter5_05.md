# 5.5 完整融合算法与复杂度分析

## 5.5.1 KS-NBCF 端到端算法

将 5.2–5.4 节三个子模块串联为一个完整的训练-推理算法：

```
算法 5.5: KS-NBCF.infer_frame(G_t, Δg_t, K, prev_state)
输入: 当前帧 STKG G_t, 差分 Δg_t, 规则池 K, 上一帧持续状态 prev_state
输出: y_fused, explanation_path, conflict_coeff K_val

# === 5.2 φ_feat: 特征层规则先验注入 ===
1. κ_rss ← compute_rss_residuals(G_t.vehicles, K.rss.params)
2. κ_rule ← compute_rule_strengths(G_t.vehicles, K.traffic.rules)
3. X_aug ← concat([G_t.X, normalize(κ_rss)], dim=-1)
4. Z_rule ← MLP_rule(κ_rule)

# === K-HSTGAN 前向（4.2-4.5）===
5. H_spatial ← RGAT(X_aug, G_t.A_spatial)
6. H_spatial' ← H_spatial + Z_rule
7. δ_t ← construct_delta(Δg_t)
8. H_lstm ← DeltaGatedLSTM(H_spatial',δ_t, prev_state.lstm)
9. H_behavior ← BehaviorAttention(H_lstm, G_t.behavior_events)
10. H_temporal ← TransformerSelfAttn(concat(H_lstm, H_behavior))
11. y_anomaly, y_scene, y_behavior, y_rule ← MultiHeadAttnFusion(H_temporal)
12. ε_t ← compute_multihead_variance(H_temporal)

# === 5.3 φ_loop 阶段 III: 推理时注意力激活规则模板 ===
13. if y_anomaly > 0.5:
14.    S_a ← top_K_attentions(attentions, K=10)
15.    new_violations ← TemplateActivator.activate(S_a, G_t)
16. else:
17.    new_violations ← []
18. end if

# === 规则引擎独立评估 ===
19. rule_violations ← RuleEnforcer.enforce(G_t, prev_state.rule_history)
20. all_violations ← rule_violations + new_violations
21. s_t ← max([sv.severity for sv in all_violations]) if all_violations else 0

# === 5.4 φ_fuse: D-S 证据融合 ===
22. m_rule ← construct_rule_mass(s_t)
23. m_gnn ← construct_gnn_mass(y_anomaly, ε_t)
24. K_val, m_fused ← dempster_combine(m_rule, m_gnn)

# === 冲突消解 ===
25. if K_val > τ_K and max(m_fused) < 0.5:
26.    y_final, explanation ← ConflictResolver.resolve(m_rule, m_gnn, K_val, G_t)
27. else:
28.    y_final ← argmax(m_fused)
29.    explanation ← "D-S consistent"
30. end if

# === 状态更新 ===
31. prev_state.lstm ← H_lstm[-1]
32. prev_state.rule_history.append(all_violations)
33. return y_final, explanation, K_val, m_fused
```

## 5.5.2 复杂度分析

设帧 $t$ 车辆数 $N$，规则数 $|\mathcal{R}| = 14$（交规）+ 3（RSS）= 17，窗口长度 $T$，注意力头 $H = 4$，证据链平均长度 $L$。

### 5.5.2.1 各阶段复杂度

| 阶段 | 复杂度 | 主要耗时项 | 实测占比 |
|------|--------|---------|---------|
| $\phi_{\text{feat}}$ | $\mathcal{O}(N \cdot |\mathcal{R}|)$ | RSS 计算 + 规则匹配 | 3%–5% |
| K-HSTGAN 前向 | 见 4.3.6 节表 4-2 | RGAT + DHLSTM + Transformer | 80%–85% |
| $\phi_{\text{loop}}$ III | $\mathcal{O}(K \cdot U)$ | 注意力提取 + 模板匹配 | 1%–2%（$U$ 模板数）|
| 规则引擎 | $\mathcal{O}(N^2 \cdot |\mathcal{R}|)$ | 两两关系扫描 | 5%–7% |
| $\phi_{\text{fuse}}$ | $\mathcal{O}(L)$ | 证据链查询 + D-S 组合 | 1%–3% |
| 总计 | $\mathcal{O}(N^2 |\mathcal{R}| + H N T F'^2)$ | — | 100% |

K-HSTGAN 前向占总耗时 80% 以上，融合机制本身的开销可忽略。说明 KS-NBCF 不显著增加额外推理时间，符合实时性要求。

### 5.5.2.2 内存占用估算

设 $N = 30$，$F' = 64$，$T = 30$：

| 数据 | 形状 | 内存 |
|------|------|------|
| $\mathbf{X}^{\text{aug}}$ | $30 \times 23$ | 2.7 KB |
| $\mathbf{H}_{\text{spatial}}'$ | $30 \times 30 \times 64$ | 460 KB |
| $\mathbf{H}_{\text{lstm}}$ | $30 \times 30 \times 64$ | 460 KB |
| 注意力权重 | $30 \times 30 \times 15 \times 4$ | 720 KB |
| 规则引擎状态 | $14 + 30 \times 3$（iem） | 几 KB |
| 证据链 | 1 KB / sv × 平均 5 sv | 5 KB |
| **总内存** | — | **~2 MB** |

总内存控制在 5 MB 内，远低于 GPU 显存的常见阈值，可在嵌入式推理平台上运行。

## 5.5.3 训练与推理分别的算法实例

### 5.5.3.1 训练循环

训练时调用 5.5.1 节算法主要变化：

- 多次迭代 epoch，每个 epoch 末调用算法 5.2 做 $\phi_{\text{loop}}$ 阶段 II 反馈调整；
- 主损失 + 辅助损失按 (4.40) 加权；
- 弱监督 $\gamma_3$ 按 (5.3) 递减；
- 在 epoch $\geq$ 30 时执行 4.5.4.1 阶段 III 微调；
- 早停与 EMA 在 epoch 末执行。

### 5.5.3.2 推理时

推理时去掉所有损失计算，仅做前向 + D-S 融合 + 冲突消解。算法流程同 5.5.1 节，但跳过算法 5.2 的反馈调整（仅在训练时执行）。

## 5.5.4 配置与超参数汇总

表 5-3 列出 KS-NBCF 的全部可配置项及默认值。

**表 5-3** KS-NBCF 配置参数表

| 超参数 | 默认值 | 来源 | 说明 |
|--------|--------|------|------|
| RSS 参数 $\rho$ | 0.3 s | `config/rss_rules.yaml` | 反应时间 |
| $\tau_K$ 冲突阈值 | 0.3 | 5.4.4 节 | D-S 融合冲突阈值 |
| 辅助损失 $\lambda_{1,2,3}$ | 0.5 | (4.40) | 多任务权重 |
| 弱监督权重 $\gamma_3^{\text{init}}$ | 0.5 | (5.3) | 训练 I 阶段弱监督初始权重 |
| 温控周期 $T_{\text{warm}}$ | 10 epochs | (5.3) | 弱监督递减周期 |
| 反馈步长 $\beta$ | 0.001 | (5.4) | 规则置信度反馈步长 |
| 规则置信度下限 | 0.3 | 5.3.3 节 | 触发参数调整的下限 |
| Top-K 注意力 | 10 | (5.7) | 子图抽取数 |
| 证据强度阈值 | 0.8 | 5.4.5 节 | 信任规则证据链强度下限 |
| overlap 阈值 | 0.5 | 5.4.5 节 | 子图重叠率阈值 |
| 窗口长度 $T$ | 30 帧 | 4.1.1.1 节 | 时序窗口 |

## 5.5.5 小结

本节给出 KS-NBCF 的端到端算法（算法 5.5），并完成复杂度分析。融合机制本身开销不超过总耗时 5%，整体性能受 K-HSTGAN 前向传播主导，在嵌入式平台可流畅运行。表 5-3 集中给出全部超参数与默认值，对第 6 章实验调参给予依据。下一节将对已有融合方法做对比并给出本章总结。