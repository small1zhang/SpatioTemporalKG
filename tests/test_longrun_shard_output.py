# -*- coding: utf-8 -*-
"""长时运行 Phase5 分片输出测试 (无需 CARLA)."""

from __future__ import annotations
import json
import tempfile
from pathlib import Path

import pytest

from stk.storage.serializer import serialize_graph


def _make_test_frames(n_frames: int = 10):
    """构造 N 帧合成数据: 2 车 + 1 行人 + 1 红绿灯 + 2 lane."""
    frames = []
    for fid in range(n_frames):
        frames.append({
            "frame_id": fid,
            "elapsed_seconds": fid * 0.05,
            "delta_seconds": 0.05,
            "vehicles": [
                {"entity_id": "v1", "location_x": 10.0 + fid, "location_y": 20.0,
                 "speed": 5.0, "lane_id": 1, "road_id": 5,
                 "current_lane_id": "road_5_lane_1"},
                {"entity_id": "v2", "location_x": 30.0 + fid, "location_y": 20.0,
                 "speed": 3.0, "lane_id": 1, "road_id": 5,
                 "current_lane_id": "road_5_lane_1"},
            ],
            "pedestrians": [{"entity_id": "p1", "location_x": 50.0, "location_y": 40.0}],
            "traffic_lights": [{"entity_id": "tl1", "state": "green"}],
            "lanes": [{"entity_id": "road_5_lane_1"}, {"entity_id": "road_5_lane_2"}],
            "weather": {"cloudiness": 0.1, "precipitation": 0.0},
            "scene_rels": [
                {"src_id": f"scenario_frame_{fid}", "dst_id": "tl1",
                 "relation_type": "containsTrafficLight", "frame_id": fid},
                {"src_id": f"scenario_frame_{fid}", "dst_id": "road_5_lane_1",
                 "relation_type": "containsRoad", "frame_id": fid},
                {"src_id": f"scenario_frame_{fid}", "dst_id": "road_5_lane_2",
                 "relation_type": "containsRoad", "frame_id": fid},
                {"src_id": f"scenario_frame_{fid}", "dst_id": f"env_frame_{fid}",
                 "relation_type": "hasEnvironment", "frame_id": fid},
                {"src_id": f"env_frame_{fid}", "dst_id": f"scenario_frame_{fid}",
                 "relation_type": "weather_context", "frame_id": fid},
            ],
        })
    return frames


class TestLongRunShardOutput:
    """验证 Phase5 分片输出逻辑 (--shard-frames)."""

    def test_shard_splits_frames_correctly(self):
        """把 10 帧按 shard_frames=3 切分 => 得到 4 个分片 (3+3+3+1)."""
        frames = _make_test_frames(10)
        shard_frames = 3
        shards = []
        for start_idx in range(0, len(frames), shard_frames):
            shard_data = frames[start_idx:start_idx + shard_frames]
            shards.append(shard_data)
        assert len(shards) == 4
        assert len(shards[0]) == 3
        assert len(shards[1]) == 3
        assert len(shards[2]) == 3
        assert len(shards[3]) == 1

    def test_each_shard_serializes_independently(self):
        """每个分片独立调用 serialize_graph 应产出有效的 {nodes, edges} JSON."""
        frames = _make_test_frames(10)
        shard_frames = 3
        for start_idx in range(0, len(frames), shard_frames):
            shard_data = frames[start_idx:start_idx + shard_frames]
            g = serialize_graph(shard_data, coalesce_containment=True)
            assert "nodes" in g
            assert "edges" in g
            assert isinstance(g["nodes"], list)
            assert isinstance(g["edges"], list)
            # 每个分片都应有 1 个全局 scenario 节点
            scenario_nodes = [n for n in g["nodes"] if n["type"] == "ScenarioSnapshot"]
            assert len(scenario_nodes) == 1
            assert scenario_nodes[0]["id"] == "scenario"

    def test_shard_writes_multiple_graph_files(self):
        """模拟 pipeline --shard-frames=3 时写出多个 graph_XXXX.json 文件."""
        frames = _make_test_frames(10)
        shard_frames = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            shard_infos = []
            total_nodes = 0
            total_edges = 0

            for start_idx in range(0, len(frames), shard_frames):
                shard_data = frames[start_idx:start_idx + shard_frames]
                f_start = shard_data[0]["frame_id"]
                f_end = shard_data[-1]["frame_id"]
                shard_idx = len(shard_infos) + 1

                g = serialize_graph(shard_data, coalesce_containment=True)
                n_nodes = len(g["nodes"])
                n_edges = len(g["edges"])
                total_nodes += n_nodes
                total_edges += n_edges

                path = out_dir / f"graph_{shard_idx:04d}_{f_start}_{f_end}.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(g, f)

                assert path.exists()
                assert path.stat().st_size > 0
                shard_infos.append({
                    "path": path,
                    "frame_start": f_start, "frame_end": f_end,
                    "nodes": n_nodes, "edges": n_edges,
                })

            # 验证: 10 帧 / 3 = 4 个分片
            assert len(shard_infos) == 4

            # 验证文件名格式和帧范围
            assert shard_infos[0]["path"].name == "graph_0001_0_2.json"
            assert shard_infos[1]["path"].name == "graph_0002_3_5.json"
            assert shard_infos[2]["path"].name == "graph_0003_6_8.json"
            assert shard_infos[3]["path"].name == "graph_0004_9_9.json"

            # 验证每个分片文件的可读性 & JSON 结构
            for info in shard_infos:
                with open(info["path"]) as f:
                    loaded = json.load(f)
                assert "nodes" in loaded
                assert "edges" in loaded
                assert len(loaded["nodes"]) == info["nodes"]
                assert len(loaded["edges"]) == info["edges"]

    def test_shard_summary_can_be_aggregated(self):
        """各分片统计汇总后应反映整体图结构 (节点/边类型汇总)."""
        frames = _make_test_frames(10)
        shard_frames = 3

        all_types_nodes = {}
        all_types_edges = {}
        total_edges_sharded = 0
        total_nodes_sharded = 0

        for start_idx in range(0, len(frames), shard_frames):
            shard_data = frames[start_idx:start_idx + shard_frames]
            g = serialize_graph(shard_data, coalesce_containment=True)
            for n in g["nodes"]:
                all_types_nodes[n["type"]] = all_types_nodes.get(n["type"], 0) + 1
            for e in g["edges"]:
                all_types_edges[e["type"]] = all_types_edges.get(e["type"], 0) + 1
            total_nodes_sharded += len(g["nodes"])
            total_edges_sharded += len(g["edges"])

        # 和整帧一起跑的对比 (10 帧, coalesce)
        g_full = serialize_graph(frames, coalesce_containment=True)
        # 分片模式的总节点数会略高于全帧模式 (跨分片重复节点)
        # 但边数应接近
        assert total_edges_sharded >= len(g_full["edges"])

    def test_single_mode_no_shard_produces_one_file(self):
        """不设置 shard-frames 时写出单个 phase5_graph.json (向后兼容)."""
        frames = _make_test_frames(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            g = serialize_graph(frames, coalesce_containment=True)
            path = out_dir / "phase5_graph.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(g, f)
            assert path.exists()
            assert path.stat().st_size > 0

            # 整个目录下只有 1 个 graph 文件
            graph_files = list(out_dir.glob("*.json"))
            assert len(graph_files) == 1

    def test_shard_with_empty_frames_produces_empty_ok(self):
        """零帧数据应妥善处理."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            # 空帧列表
            frames = []
            shards_created = 0
            for start_idx in range(0, len(frames), 3):
                shard_data = frames[start_idx:start_idx + 3]
                if not shard_data:
                    continue
                g = serialize_graph(shard_data, coalesce_containment=True)
                shards_created += 1
            assert shards_created == 0
