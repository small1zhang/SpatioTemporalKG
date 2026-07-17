"""场景库测试 — 验证 14 个复杂场景的工厂完整性 (v3 §2 附录 A2)

测试维度：
    TestScenarioRegistry: 注册表完整性和元数据正确性
    TestScenarioBaselines: A 档基线场景的字段断言
    TestScenarioAnomalies: B 档异常场景的异常触发条件
    TestScenarioComplexity: C 档多车冲突场景的冲突关系
    TestScenarioCoupling: D 档跨层联动场景的环境耦合
    TestScenarioBuildSnapshot: 场景库产出可被 build_snapshot() 直接消费
"""
from __future__ import annotations

import pytest

from stk.scenario import (
    list_scenarios, get_scenario, get_scenario_meta, all_scenarios,
    total_frames, SCENARIO_REGISTRY, SCENARIO_FACTORIES,
    build_snapshot,
    make_S00_baseline_following, make_S01_normal_signalized_intersection,
    make_S02_pedestrian_far_avoidance,
    make_S10_pedestrian_sudden_crossing,
    make_S11_unprotected_left_turn_conflict,
    make_S12_red_light_running,
    make_S13_too_close_following,
    make_S20_merging_conflict,
    make_S21_three_way_unsignalized,
    make_S22_emergency_vehicle_yielding,
    make_S30_night_pedestrian_sudden,
    make_S31_rainy_lane_change_blind,
    make_S32_construction_detour,
    make_S33_glare_multi_pedestrian,
)


# ─── 注册表 ────────────────────────────────────────────────

class TestScenarioRegistry:
    def test_list_scenarios_returns_14(self):
        # 14 个场景全部注册 - A:3 + B:4 + C:3 + D:4
        sids = list_scenarios()
        assert len(sids) == 14, f"应有14个场景，实际 {len(sids)}"
        expected = ["S00", "S01", "S02", "S10", "S11", "S12", "S13",
                    "S20", "S21", "S22", "S30", "S31", "S32", "S33"]
        assert sids == expected, f"场景顺序不对: {sids}"

    def test_registry_factories_match_registry(self):
        # SCENARIO_REGISTRY 和 SCENARIO_FACTORIES 必须一一对应
        for sid in SCENARIO_REGISTRY:
            assert sid in SCENARIO_FACTORIES, f"{sid} 在 registry 但不在 factories"
        for sid in SCENARIO_FACTORIES:
            assert sid in SCENARIO_REGISTRY, f"{sid} 在 factories 但不在 registry"

    def test_each_meta_has_required_keys(self):
        # 每个元数据必须有 tier/category/expected_rules/expected_sv/expected_behaviors
        required_keys = {"tier", "category", "expected_rules",
                          "expected_sv", "expected_behaviors"}
        for sid in list_scenarios():
            meta = get_scenario_meta(sid)
            assert required_keys.issubset(meta.keys()), \
                f"{sid} 元数据缺少字段: {set(meta.keys()) ^ required_keys}"

    def test_tier_distribution(self):
        tiers = [get_scenario_meta(sid)["tier"] for sid in list_scenarios()]
        assert tiers.count("A") == 3, f"A档应为3, 实际 {tiers.count('A')}"
        assert tiers.count("B") == 4, f"B档应为4, 实际 {tiers.count('B')}"
        assert tiers.count("C") == 3, f"C档应为3, 实际 {tiers.count('C')}"
        assert tiers.count("D") == 4, f"D档应为4, 实际 {tiers.count('D')}"

    def test_baseline_no_rules(self):
        # A 档基线不应触发任何规则
        for sid in ["S00", "S01", "S02"]:
            meta = get_scenario_meta(sid)
            assert meta["expected_rules"] == [], \
                f"{sid} (A档基线) 不应触发规则，实际 {meta['expected_rules']}"
            assert meta["expected_sv"] == "-", \
                f"{sid} (A档基线) 不应产生 sv"

    def test_total_frames_84(self):
        # 14 个场景 × 6 帧 = 84 帧
        assert total_frames() == 84, f"应84帧，实际 {total_frames()}"

    def test_each_scenario_6_frames(self):
        # 每个场景固定 6 帧 (itemporal ablation W/T 需要)
        for sid in list_scenarios():
            frames = get_scenario(sid)
            assert len(frames) == 6, f"{sid} 应6帧，实际 {len(frames)}"

    def test_unknown_scenario_raises(self):
        with pytest.raises(KeyError, match="Unknown scenario"):
            get_scenario("S99")
        with pytest.raises(KeyError, match="Unknown scenario"):
            get_scenario_meta("S99")


# ─── A 基线 ────────────────────────────────────────────────

class TestScenarioBaselines:
    def test_S00_baseline_following(self):
        frames = make_S00_baseline_following()
        assert len(frames) == 6
        # 每帧 2 车匀速
        for f in frames:
            assert len(f.vehicles) == 2
            v0, v1 = f.vehicles
            assert v0["speed"] == 8.0 and v1["speed"] == 8.0
        # 车距恒定(前车-后车=16m)
        for f in frames:
            assert abs(f.vehicles[0]["location_x"] - f.vehicles[1]["location_x"]) == 16.0

    def test_S01_normal_signalized_intersection(self):
        frames = make_S01_normal_signalized_intersection()
        assert len(frames) == 6
        # 信号灯一直绿色
        for f in frames:
            assert f.traffic_lights[0]["state"] == "Green"
        # 单车通过路口
        for f in frames:
            assert len(f.vehicles) == 1

    def test_S02_pedestrian_far_avoidance(self):
        frames = make_S02_pedestrian_far_avoidance()
        assert len(frames) == 6
        # 行人距离>=20m (v3 §2.9.2 不应触发 nearby)
        for f in frames:
            veh = f.vehicles[0]
            ped = f.pedestrians[0]
            dist = ((veh["location_x"] - ped["location_x"])**2 +
                    (veh["location_y"] - ped["location_y"])**2) ** 0.5
            assert dist >= 20.0, f"应>=20m, 实际 {dist:.2f}"


# ─── B 异常 ────────────────────────────────────────────────

class TestScenarioAnomalies:
    def test_S10_pedestrian_sudden_crossing(self):
        frames = make_S10_pedestrian_sudden_crossing()
        assert len(frames) == 6
        # frame3起行人进入车道(y近0)
        for i, f in enumerate(frames):
            ped = f.pedestrians[0]
            if i < 3:
                assert ped["location_y"] < -2.0, f"frame{i} 行人应隐藏, y={ped['location_y']}"
            else:
                assert ped["location_y"] > -1.5, f"frame{i} 行人应走出, y={ped['location_y']}"
        # 公交车静止
        for f in frames:
            assert f.vehicles[1]["speed"] == 0.0, "公交车应静止"

    def test_S11_unprotected_left_turn_conflict(self):
        frames = make_S11_unprotected_left_turn_conflict()
        assert len(frames) == 6
        # 两车朝向相反 (本车0°, 对向180°)
        for f in frames:
            assert f.vehicles[0]["heading_rad"] == 0.0
            assert abs(f.vehicles[1]["heading_rad"] - 3.14159) < 0.01
        # 两车逐渐接近(对向x递减)
        for i in range(1, len(frames)):
            assert frames[i].vehicles[1]["location_x"] < frames[i-1].vehicles[1]["location_x"]

    def test_S12_red_light_running(self):
        frames = make_S12_red_light_running()
        assert len(frames) == 6
        # 信号灯前 2 帧 Green, 后 4 帧 Red
        states = [f.traffic_lights[0]["state"] for f in frames]
        assert states[0] == "Green" and states[2] == "Red"
        # 后 4 帧 violation (车在红灯时仍前进)
        for i in [3, 4, 5]:
            assert states[i] == "Red"
            assert frames[i].vehicles[0]["speed"] > 0, "红灯状态车不应停车"

    def test_S13_too_close_following(self):
        frames = make_S13_too_close_following()
        assert len(frames) == 6
        # THW = gap / follower_speed, gap=3m / speed=8m/s = 0.375s < 1.0s
        for f in frames:
            gap = f.vehicles[0]["location_x"] - f.vehicles[1]["location_x"]
            thw = gap / f.vehicles[1]["speed"]
            assert thw < 1.0, f"THW应<1.0s, 实际 {thw:.3f}s"


# ─── C 多车冲突 ────────────────────────────────────────────────

class TestScenarioComplexity:
    def test_S20_merging_conflict(self):
        frames = make_S20_merging_conflict()
        assert len(frames) == 6
        # 3 车同帧
        for f in frames:
            assert len(f.vehicles) == 3, "应有3车(主路+匝道+主路后车)"
        # 匝道车逐渐向主道并线(y递减)
        ys = [f.vehicles[1]["location_y"] for f in frames]
        assert ys[-1] < ys[0], "匝道车应向主道并线(y递减)"

    def test_S21_three_way_unsignalized(self):
        frames = make_S21_three_way_unsignalized()
        assert len(frames) == 6
        # 3 车, 不同朝向
        for f in frames:
            assert len(f.vehicles) == 3
            headings = [v["heading_rad"] for v in f.vehicles]
            assert len(set(headings)) == 3, f"3车朝向应不同: {headings}"

    def test_S22_emergency_vehicle_yielding(self):
        frames = make_S22_emergency_vehicle_yielding()
        assert len(frames) == 6
        # 救护车(veh_221)有 is_emergency 字段
        for f in frames:
            assert f.vehicles[1].get("is_emergency", False) is True, \
                "救护车应有 is_emergency=True 标识"
        # 救护车从后方逼近：与自车的绝对纵向距离应保持大于0且逐渐走完
        # 关键不在于"超过"，而在于第1帧救护车在自车后方（amb_x < ego_x）
        assert frames[0].vehicles[1]["location_x"] < frames[0].vehicles[0]["location_x"], \
            "frame0 救护车应在自车后方"
        # 救护车速度大于自车（逼近条件）
        assert frames[0].vehicles[1]["speed"] > frames[0].vehicles[0]["speed"], \
            "救护车应比自车快，形成逼近态势"
        # 救护车to自车的差距应递减(因为救护车更快)
        deltas = [f.vehicles[0]["location_x"] - f.vehicles[1]["location_x"]
                  for f in frames]
        # deltas[0]应为正(后车在后)，deltas[-1]应小于deltas[0]
        assert deltas[0] > 0, f"frame0 救护车应在自车后方, deltas={deltas[0]}"
        assert deltas[-1] < deltas[0], \
            f"救护车应缩短自车纵向距离, deltas[0]={deltas[0]}, deltas[-1]={deltas[-1]}"


# ─── D 跨层联动 ────────────────────────────────────────────────

class TestScenarioCoupling:
    def test_S30_night_pedestrian_sudden(self):
        frames = make_S30_night_pedestrian_sudden()
        assert len(frames) == 6
        # 夜间 (sun_altitude<0)
        for f in frames:
            assert f.weather["sun_altitude_angle"] < 0, "应夜间照明"
        # 高 fog_density
        for f in frames:
            assert f.weather["fog_density"] == 15, "夜间应有低视距 fog"
        # 行人 frame3起走出
        for i, f in enumerate(frames):
            ped = f.pedestrians[0]
            if i < 3:
                assert ped["location_y"] < -2.0
            else:
                assert ped["location_y"] > -1.0

    def test_S31_rainy_lane_change_blind(self):
        frames = make_S31_rainy_lane_change_blind()
        assert len(frames) == 6
        # 大雨 (precipitation >50)
        for f in frames:
            assert f.weather["precipitation"] == 80, "应大雨"
            assert f.weather["wetness"] == 90, "应高湿滑"
        # 邻车道逐渐并入(y递减)
        ys = [f.vehicles[1]["location_y"] for f in frames]
        assert ys[-1] < ys[0], "邻车应逐渐并道"

    def test_S32_construction_detour(self):
        frames = make_S32_construction_detour()
        assert len(frames) == 6
        # frame2 起出现路障锥桶
        for i, f in enumerate(frames):
            if i < 2:
                assert len(f.pedestrians) == 0, f"frame{i} 不应有路障"
            else:
                assert len(f.pedestrians) > 0, f"frame{i} 应出现施工锥桶"
                for c in f.pedestrians:
                    assert "ConstructionCone" in c.get("labels", []), \
                        "锥桶应有 ConstructionCone 标签"

    def test_S33_glare_multi_pedestrian(self):
        frames = make_S33_glare_multi_pedestrian()
        assert len(frames) == 6
        # 朝向太阳 (低角度)
        for f in frames:
            assert f.weather["sun_altitude_angle"] == 5, "应低角度逆光"
        # 每帧 3 行人横穿
        for f in frames:
            assert len(f.pedestrians) == 3, "应有3行人同步横穿"
        # 行人速度不同
        speeds = set(f.pedestrians[0]["speed"] for f in frames)
        assert len(speeds) == 1  # 全程同一行人集 (起始速度恒定)
        sps = [p["speed"] for p in frames[0].pedestrians]
        assert len(set(sps)) == 3, "3个行人应有3种不同速度"


# ─── 与 build_snapshot 的衔接 ─────────────────────────────

class TestScenarioBuildSnapshot:
    def test_all_scenarios_can_be_built(self):
        # 关键：每个场景的每帧都能被 build_snapshot() 消费
        for sid in list_scenarios():
            frames = get_scenario(sid)
            for f in frames:
                snap, env = build_snapshot(f)
                assert snap is not None and env is not None
                # 字段在 attrs 字典中 (Pydantic extra="allow")
                assert snap.attrs["frame_id"] == f.frame_id
                assert snap.attrs["n_vehicles"] == len(f.vehicles)
                assert snap.attrs["n_pedestrians"] == len(f.pedestrians)

    def test_all_scenarios_dict(self):
        all_scn = all_scenarios()
        assert len(all_scn) == 14
        assert set(all_scn.keys()) == set(list_scenarios())

    def test_environment_carried_through(self):
        # D 场景必须把天气耦合保留在 EnvironmentSnapshot
        for sid in ["S30", "S31", "S32", "S33"]:
            frames = get_scenario(sid)
            for f in frames:
                _, env = build_snapshot(f)
                # 字段在 attrs 字典中 (Pydantic extra="allow")
                if sid == "S30":
                    assert env.attrs["sun_altitude_angle"] < 0
                if sid == "S31":
                    assert env.attrs["precipitation"] > 50
