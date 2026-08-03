# 四大框架图 — 视觉预览（Mermaid 对照版）

> 仅为**布局预览**，正式论文图请使用同目录下的 `figs_framework.tex`（科技蓝渐变 + 光泽感 LaTeX TikZ 源码）。
> Mermaid 用于在设计阶段快速校对结构；论文正文插图请编译 `.tex` 文件。

---

## 图 1 — 创新点① STKG 四层本体构建框架（§3）

```mermaid
flowchart LR
    classDef ember fill:#FFE4D1,stroke:#FF7A45,color:#0B1F3A,stroke-width:1.5px
    classDef glass fill:#A8D8EA,stroke:#1B4F8C,color:#0B1F3A
    classDef core fill:#2E86DE,stroke:#0B1F3A,color:#fff
    classDef violet fill:#5A4FCF,stroke:#5A4FCF,color:#fff

    CARLA["CARLA 0.9.16<br/>5 地图·20min·24K 帧"]:::ember
    Frame["FrameData<br/>actor/TL/env"]:::ember
    Delta["DeltaGraph Δg_t"]:::ember

    CHAP["四层本体<br/>O=(E,A,R,T,P)"]:::core
    DYN["横向动态<br/>版本化·窗口聚合"]:::core

    L1["场景层<br/>6 类节点·15 种空间关系"]:::glass
    L2["行为层<br/>11 检测器·13 行为关系"]:::glass
    L3["规则层<br/>RSS+14 交规+4 扩充"]:::glass
    L4["动态层<br/>增量引擎·版本化"]:::glass

    MB1["manifestsAs"]:::violet
    MB2["violates"]:::violet
    MB3["definedBy"]:::violet
    MB4["supportedByEvidence"]:::violet

    NEO["Neo4j<br/>持久化"]:::ember
    PYG["PyG Data<br/>→ K-HSTGAN"]:::ember

    CARLA --> Frame --> CHAP
    Delta --> L4
    CHAP --> L1 --> L2 --> L3 --> L4
    L1 -.- MB1 -.->|跨层桥接| L3
    L2 -.- MB2 -.-> L3
    L3 -.- MB3 -.-> L1
    L4 -.- MB4
    L4 --> NEO --> PYG

    DYN -.激活.-> L1
    DYN -.激活.-> L4
```

---

## 图 2 — 创新点② K-HSTGAN 三层时空图注意力网络（§4）

```mermaid
flowchart LR
    classDef ember fill:#FFE4D1,stroke:#FF7A45,color:#0B1F3A
    classDef core fill:#2E86DE,stroke:#0B1F3A,color:#fff
    classDef glass fill:#A8D8EA,stroke:#1B4F8C,color:#0B1F3A
    classDef violet fill:#5A4FCF,stroke:#5A4FCF,color:#fff

    Xt["X_t ∈ ℝ^(N×18)<br/>节点物理特征"]:::ember
    At["A_t^(15)<br/>场景关系"]:::ember
    Bt["B_t^(13)<br/>行为关系"]:::ember
    Delta["Δg_t<br/>差分图"]:::ember
    Rule["κ_rss, κ_rule<br/>知识先验"]:::ember

    H["<b>K-HSTGAN</b>"]:::core
    S1["<b>空间编码层</b><br/>RGAT §4.2"]:::core
    S2["<b>时序编码层</b><br/>DH-LSTM+Attn §4.3"]:::core
    S3["<b>知识注入层</b><br/>§4.4"]:::core

    S1a["关系感知注意力 α_ij"]:::glass
    S1b["15 种关系先验 γ_k"]:::glass
    S1c["多头聚合 H=4"]:::glass

    S2a["帧级 LSTM"]:::glass
    S2b["差分门控 g_t=σ·"]:::glass
    S2c["Scene Transformer"]:::glass

    S3a["RSS 公式编码 (5 维)"]:::glass
    S3b["交规 Embedding (14 维)"]:::glass
    S3c["弱监督标签 ŷ_rule"]:::glass

    F1["<b>多模态多任务融合头</b> §4.5"]:::glass

    H1["p_scene ∈ ℝ^3<br/>场景异常"]:::violet
    H2["p_behavior ∈ ℝ^7<br/>行为异常"]:::violet
    H3["p_rule ∈ ℝ^24<br/>规则异常"]:::violet

    ANOM["<b>p_anomaly ∈ [0,1]</b><br/>异常分数"]:::ember

    Xt --> S1
    At --> S1
    Bt --> S1
    Delta --> S2b
    Rule --> S3

    H --> S1
    S1 --> S1a
    S1 --> S1b
    S1 --> S1c
    S1 --> S2 --> S2a
    S2 --> S2b
    S2 --> S2c
    S2 --> S3 --> S3a
    S3 --> S3b
    S3 --> S3c

    S2 --> F1
    S3 --> F1
    F1 --> H1
    F1 --> H2
    F1 --> H3
    H1 --> ANOM
    H2 --> ANOM
    H3 --> ANOM

    S2b -.Dempster反传.-> H1
```

---

## 图 3 — 创新点③ KS-NBCF 符号-神经双向闭环融合（§5）

```mermaid
flowchart TB
    classDef ember fill:#FFE4D1,stroke:#FF7A45,color:#0B1F3A
    classDef core fill:#2E86DE,stroke:#0B1F3A,color:#fff
    classDef glass fill:#A8D8EA,stroke:#1B4F8C,color:#0B1F3A
    classDef violet fill:#5A4FCF,stroke:#5A4FCF,color:#fff
    classDef night fill:#1A1B3A,stroke:#0B1F3A,color:#fff

    RE["<b>规则引擎</b><br/>RuleEnforcer §3"]:::ember
    GNN["<b>K-HSTGAN</b><br/>§4"]:::ember
    rss["RSS 算子"]:::glass
    r14["14 交规"]:::glass
    rssExt["4 扩充规则<br/>CUTIN/CZ ADAPT"]:::glass
    emb["空间编码 RGAT"]:::glass
    lstm["差分门控 LSTM"]:::glass
    ki["知识注入"]:::glass

    RE --> rss
    RE --> r14
    RE --> rssExt
    GNN --> emb
    GNN --> lstm
    GNN --> ki

    P1["<b>φ_feat</b><br/>特征层注入"]:::core
    P2["<b>φ_loop</b><br/>双向闭环训练"]:::core
    P3["<b>φ_fuse</b><br/>D-S 证据融合"]:::core

    feat1["h_v^(0)=[x_v‖κ_rss‖κ_rule]"]:::glass
    feat2["37 维初始特征"]:::glass
    loop1["Stage I 弱监督"]:::glass
    loop2["Stage II 反馈"]:::glass
    loop3["Stage III 模板"]:::glass
    fuse1["m_rule, m_GNN"]:::glass
    fuse2["Dempster 组合"]:::glass

    CONFLICT{"冲突系数 K > τ?"}:::violet
    RESOLVE["<b>KG 证据链回溯</b><br/>overlap 仲裁"]:::violet
    FINAL["<b>ŷ, m_fused, ExpPath</b>"]:::night

    rss -.-> P1
    r14 --> P1
    rssExt -.-> P1
    emb --> P1
    ki --> P1
    P1 --> feat1
    P1 --> feat2

    P1 ==> P2
    P2 --> loop1
    P2 --> loop2
    P2 --> loop3
    P2 <-.置信度调整.-> RE

    P2 ==> P3
    P3 --> fuse1
    P3 --> fuse2
    RE -.->|s_v 触发强度| P3
    GNN -.->|p_v 异常概率| P3

    P3 --> CONFLICT
    CONFLICT -- K>τ --> RESOLVE
    CONFLICT -- K≤τ --> FINAL
    RESOLVE --> FINAL
    fuse1 --> FINAL
    fuse2 --> FINAL
```

---

## 图 4 — 实验与结果验证框架 §6（RQ1→RQ5）

```mermaid
flowchart TB
    classDef ember fill:#FFE4D1,stroke:#FF7A45,color:#0B1F3A
    classDef core fill:#2E86DE,stroke:#0B1F3A,color:#fff
    classDef glass fill:#A8D8EA,stroke:#1B4F8C,color:#0B1F3A
    classDef moss fill:#9BD7B8,stroke:#3FA17A,color:#0B1F3A
    classDef viol fill:#C8C0F0,stroke:#5A4FCF,color:#0B1F3A
    classDef night fill:#1A1B3A,color:#fff

    DATA["<b>数据基础设施</b><br/>41,150 帧·5 地图·20min"]:::ember
    fl["frame_labels.csv (41K)"]:::glass
    al["anomaly_log.json (4,926)"]:::glass
    gt["CARLA GT 提取器"]:::glass

    DATA --> fl
    DATA --> al
    DATA --> gt

    RQ1["<b>RQ1</b> 图谱构建质量"]:::core
    RQ2["<b>RQ2</b> 流式性能"]:::core
    RQ3["<b>RQ3</b> 异常检测"]:::core
    RQ4["<b>RQ4</b> 消融实验"]:::core
    RQ5["<b>RQ5</b> 融合+消解"]:::core
    fl --> RQ1
    al --> RQ1
    gt --> RQ1
    RQ1 ==> RQ2 ==> RQ3 ==> RQ4 ==> RQ5

    RQ1a["场景关系 F1<br/>表 6-4"]:::glass
    RQ1b["行为检测 F1<br/>表 6-5"]:::glass
    RQ1c["规则 DR/FAR<br/>表 6-6"]:::glass
    RQ1 --> RQ1a
    RQ1 --> RQ1b
    RQ1 --> RQ1c

    RQ2a["吞吐/延迟<br/>表 6-8"]:::glass
    RQ2b["内存/长时<br/>表 6-9/10"]:::glass
    RQ2c["增量 vs 全量<br/>表 6-11"]:::glass
    RQ2 --> RQ2a
    RQ2 --> RQ2b
    RQ2 --> RQ2c

    RQ3a["<b>K-HSTGAN 主结果 F1=1.0</b><br/>表 6-13"]:::glass
    RQ3b["PR 曲线 38 阈值<br/>表 6-13 续"]:::glass
    RQ3c["跨 Town OOD<br/>表 6-18"]:::glass
    RQ3 --> RQ3a
    RQ3 --> RQ3b
    RQ3 --> RQ3c

    RQ4a["架构消融<br/>表 6-13 续"]:::glass
    RQ4b["融合消融<br/>表 6-15"]:::glass
    RQ4c["系统级<br/>表 6-10 续"]:::glass
    RQ4 --> RQ4a
    RQ4 --> RQ4b
    RQ4 --> RQ4c

    RQ5a["冲突消解矩阵<br/>表 6-?"]:::glass
    RQ5b["Case Study<br/>证据链可视化"]:::glass
    RQ5 --> RQ5a
    RQ5 --> RQ5b

    REAL["<b>已真实化</b><br/>表 6-13/6-18/6-3"]:::moss
    EST["<b>预估待跑</b><br/>表 6-4/5/6 + 6-8→11"]:::ember
    MANUAL["<b>需人工评审</b><br/>表 6-16/17"]:::viol

    RQ1c -.-> EST
    RQ2a -.-> EST
    RQ3a ==> REAL
    RQ3c ==> REAL
    RQ5b -.-> MANUAL

    CONCLUSION["<b>→ 第 7 章 总结与展望</b>"]:::night
    REAL --> CONCLUSION
    EST --> CONCLUSION
    MANUAL --> CONCLUSION
```

---

## 编译说明

```bash
cd docs/thesis/figures/
pdflatex -shell-escape figs_framework.tex
# 输出 figs_framework.pdf（含 4 个独立页）
# 若需分图：将每个 tikzpicture 块拆为独立 tex
```

## 在论文正文中引用示例

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{figures/fig_stkg_framework.pdf}
  \caption{STKG 四层本体构建框架}
  \label{fig:stkg_framework}
\end{figure}
```
