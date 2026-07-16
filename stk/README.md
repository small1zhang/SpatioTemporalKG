# stk/  —  SpatioTemporalKG 核心库

## 模块

- ontology/    本体层(实体/关系/类型/命名空间/公理)
- scenario/    场景层(场景库/FrameData/空间关系)
- behavior/    行为层(行为生成/检测/防抖/跨层桥接)
- rules/       规则层(RSS + 交规R1-R18)
- extraction/  数据提取(Actor/红绿灯/路网/天气)
- dynamic/     动态更新(增量引擎/快照/差异图)
- pipeline/    流水线编排(编排器/检查点)
- storage/     Neo4j存储(连接器/序列化/写入/查询/回放)
- viz/         可视化(鸟瞰图/KG Dashboard)

## 数据流

CARLA → extraction → scenario → behavior → rules → dynamic → storage → phase5_graph.json
