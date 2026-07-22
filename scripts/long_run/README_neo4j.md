# Neo4j 可视化导入

把 phase5 的 3 个分片 (`viz_output/graph_0001_*.json` 等) 转入 Neo4j, 用 Browser 看图.

## 一键流程

```bash
# 步骤 0: 起 Neo4j (一次性 ~30s)
docker compose -f docker/neo4j/docker-compose.yml up -d

# 步骤 1: 跑导入 (~3-5 分钟, 38 万边)
python -m scripts.long_run.import_neo4j \
    --input viz_output/ \
    --uri bolt://localhost:7687 \
    --user neo4j --password stk123

# 步骤 2: 浏览器打开 http://localhost:7474
#          账号 neo4j / 密码 stk123
```

## 选项

| 选项 | 默认 | 说明 |
|---|---|---|
| `--input` | `viz_output/` | phase5 输出目录 (含 `phase5_kg_summary.json` + `graph_*.json`) |
| `--uri` | `bolt://localhost:7687` | Neo4j bolt URI |
| `--user` | `neo4j` | 用户名 |
| `--password` | `stk123` | 密码 (与 `config/neo4j.yaml` 对齐) |
| `--database` | `neo4j` | 数据库名 (community 版仅 `neo4j`) |
| `--batch-size` | `500` | UNWIND 批大小 |
| `--create-dangling` | off | 为悬挂的 `sv_*_frame_*` / `scene_rel_*_frame_*` ID 创建占位节点 |
| `--drop-existing` | off | 导入前 `MATCH (n) DETACH DELETE n` 清空 (慎用) |
| `--verbose` | off | debug 日志 |

## 数据规模预期

3 分片总计:
- 13607 节点 / 386107 边 (原始)
- 经过 `entity_id` MERGE 合并跨 shard 相同实体后, 实际数据库节点 ~11000
- 54553 条 `definedBy` + `supportedByEvidence` 边里 ~80% (43604 条) src id 不在节点集中,
  默认会被静默跳过. 想保留这些就加 `--create-dangling`

## 停 / 重启 Neo4j

```bash
# 停 (保留数据)
docker compose -f docker/neo4j/docker-compose.yml down

# 重启
docker compose -f docker/neo4j/docker-compose.yml up -d

# 彻底清空数据 (不可逆)
docker compose -f docker/neo4j/docker-compose.yml down -v
```

## Browser 验证查询

打开 http://localhost:7474 后贴入 Cypher:

```cypher
// 1. 总数统计
MATCH (n) RETURN count(n);
MATCH ()-[r]->() RETURN count(r);

// 2. 自车 + 邻居 (1 跳)
MATCH (n:Vehicle {entity_id: "486"})-[r]-(m)
RETURN n, r, m LIMIT 100;

// 3. ego 与其他车的空间关系类型分布
MATCH (e:Vehicle {is_ego: true})-[r]->(o:Vehicle)
RETURN type(r) AS relation, count(*) AS n
ORDER BY n DESC;

// 4. 一段时间内违反的规则
MATCH (v:Vehicle)-[r:violates]->(sv:SafetyViolation)-[:definedBy]->(rule:Rule)
WHERE r.first_frame >= 0 AND r.last_frame <= 2000
RETURN v.entity_id AS vehicle, sv.rule_code AS code,
       rule.rule_name AS rule, r.fired_frames AS fired
LIMIT 50;

// 5. 道路与车辆包含关系 (coalesce 后的 frames 数组)
MATCH (s:ScenarioSnapshot)-[r:containsVehicle]->(v:Vehicle)
WHERE v.is_ego = true
RETURN s, r, v;

// 6. 时间切片: frame 1000-1100 期间发生的事件
MATCH (n) WHERE n.first_frame <= 1100 AND n.last_frame >= 1000
RETURN labels(n)[0] AS type, count(*) AS n
ORDER BY n DESC;

// 7. 自车的所有 maneuver 链
MATCH (v:Vehicle {entity_id: "486"})-[:has_maneuver]->(m:Maneuver)
RETURN m.first_frame, m.last_frame, m.labels
ORDER BY m.first_frame;
```

## 故障排除

### `Neo4j 在 bolt://localhost:7687 上未就绪`
- 检查容器状态: `docker ps | grep stk-neo4j`
- 看日志: `docker logs stk-neo4j --tail 50`
- 重试等就绪: 脚本内置 30 次重试, 每次间隔 2s

### `ConstraintAlreadyExists` 警告
- 正常, `ensure_schema()` 用了 `IF NOT EXISTS`, 但部分老 Neo4j 版本会仍报 warn
- 实际 schema 创建成功, 不影响导入

### 边大量"丢失"
- `definedBy` / `supportedByEvidence` 54553 条边里, 80% 指向不带节点集的 ID
- 默认行为: 静默跳过这些边
- 想保留 → 加 `--create-dangling`, 会创建 `:UnknownRef` 占位节点

### 内存不足 (大图)
- docker-compose.yml 已配 2GB heap + 1GB pagecache
- 还是溢出就调大: `NEO4J_server_memory_heap_max__size=4G`

## 文件清单

| 文件 | 作用 |
|---|---|
| `docker/neo4j/docker-compose.yml` | Neo4j 容器定义 |
| `stk/storage/connector.py` | 真 Neo4j driver 包装 (替换原桩代码) |
| `stk/storage/importer.py` | `GraphImporter` 类, 处理 JSON dict 形态 |
| `scripts/long_run/import_neo4j.py` | CLI 入口 |
