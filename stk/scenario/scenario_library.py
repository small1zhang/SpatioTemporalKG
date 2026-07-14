"""
场景库 (v3 §2 扩充) — 14 个预置复杂交通场景工厂

每个场景对应一个 make_Sxx_* 函数，返回 List[FrameData]，
可直接喂入 build_snapshot() 得到完整实体+关系图。

场景类型：
    A  基线 (S00–S02)     — 正常场景，不触发规则
    B  单点异常 (S10–S13)  — 单个异常因素
    C  多车冲突 (S20–S22)  — 路口/汇入冲突
    D  跨层联动 (S30–S33)  — 环境×交通耦合

对接 v3 章节：
    §2 场景层：产出 FrameData / SceneRelationType
    §3 行为层：预声明 ManeuverNode / InteractionEvent 标签
    §4 规则层：预声明每条 SafetyViolation 触发条件
    §7 提取层：帧结构对齐 CARLA tick 节奏（近似 400ms/帧）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from stk.scenario.snapshot_builder import FrameData, build_snapshot
from stk.scenario.nodes import (
    VehicleEntity, PedestrianEntity, TrafficLightEntity,
    RoadElementEntity, EnvironmentSnapshot, ScenarioSnapshot,
)
from stk.scenario.relations import (
    build_relation, in_lane, on_road, in_junction,
    adjacent_lane, lane_connects, ahead_of, beside,
    nearby_pedestrian, controlled_by,
    contains_vehicle, contains_pedestrian, contains_traffic_light,
    contains_road, has_environment, weather_context,
)

# ─── A 基线 ────────────────────────────────────────────────────────

def make_S00_baseline_following() -> List[FrameData]:
    """S00: 直行跟车基线

    同车道 2 车匀速行驶，前车 8m/s、后车 8m/s（车距≈16m），无任何异常。
    预期行为层输出: following
    预期规则触发: 无（THW≈2.0s > 1.0s）→ 属于安全场景。

    场景层产出: 2 车道 + 2 车
    关系: in_lane × 2, ahead_of × 1
    """
    frames: List[FrameData] = []
    for f_id in range(6):
        t = f_id * 0.4  # elapsed seconds, ~400ms per frame
        # 车道不变
        lanes = [
            {
                "entity_id": "road_5_lane_1", "road_id": 5, "lane_id": 1,
                "center_x": 0.0, "center_y": 0.0, "length": 200.0, "speed_limit": 15.0,
            },
            {
                "entity_id": "road_5_lane_2", "road_id": 5, "lane_id": 2,
                "center_x": 0.0, "center_y": 4.0, "length": 200.0, "speed_limit": 15.0,
            },
        ]
        base_x = 100.0 + t * 8.0  # 前车位置
        vehicles = [
            {
                "entity_id": "veh_010",
                "vehicle_type": "vehicle.audi.tt",
                "location_x": base_x, "location_y": 0.3, "location_z": 0.5,
                "speed": 8.0, "heading_rad": 0.0,
            },
            {
                "entity_id": "veh_011",
                "vehicle_type": "vehicle.tesla.model3",
                "location_x": base_x - 16.0, "location_y": 0.3, "location_z": 0.5,
                "speed": 8.0, "heading_rad": 0.0,
            },
        ]
        frames.append(FrameData(
            frame_id=f_id, elapsed_seconds=t, delta_seconds=0.4,
            vehicles=vehicles, pedestrians=[], traffic_lights=[], lanes=lanes,
            weather={"fog_density": 0, "cloudiness": 0, "precipitation": 0,
                     "wetness": 0, "sun_altitude_angle": 80, "wind_intensity": 0},
            map_name="Town01", random_seed=0, traffic_density=2,
        ))
    return frames


def make_S01_normal_signalized_intersection() -> List[FrameData]:
    """S01: 信号路口正常通行

    车流在绿灯时通过十字路口，无冲突。
    预期行为层输出: following, approaching_intersection
    预期规则触发: 无

    场景层产出: 1 信号灯 + 1 车 + 4 车道(双向)
    关系: in_lane, in_junction, controlled_by, contains_traffic_light
    """
    frames: List[FrameData] = []
    for f_id in range(6):
        t = f_id * 0.4
        lanes = [
            {"entity_id": "road_1_lane_1", "road_id": 1, "lane_id": 1, "center_x": 0.0, "center_y": 0.0, "length": 100.0, "speed_limit": 10.0},
            {"entity_id": "road_1_lane_2", "road_id": 1, "lane_id": 2, "center_x": 0.0, "center_y": 4.0, "length": 100.0, "speed_limit": 10.0},
            {"entity_id": "road_2_lane_1", "road_id": 2, "lane_id": 1, "center_x": 100.0, "center_y": 0.0, "length": 100.0, "speed_limit": 10.0},
            {"entity_id": "road_2_lane_2", "road_id": 2, "lane_id": 2, "center_x": 100.0, "center_y": 4.0, "length": 100.0, "speed_limit": 10.0},
            {"entity_id": "junction_center", "road_id": 0, "lane_id": 0, "center_x": 50.0, "center_y": 2.0, "length": 20.0, "speed_limit": 8.0},
        ]
        base_x = -20.0 + t * 6.0
        vehicles = [
            {
                "entity_id": "veh_020",
                "vehicle_type": "vehicle.audi.tt",
                "location_x": base_x, "location_y": 0.3, "location_z": 0.5,
                "speed": 6.0, "heading_rad": 0.0,
            },
        ]
        tls = [
            {"entity_id": "tl_050", "state": "Green", "position_x": 50.0, "position_y": 2.0},
        ]
        frames.append(FrameData(
            frame_id=f_id, elapsed_seconds=t, delta_seconds=0.4,
            vehicles=vehicles, pedestrians=[], traffic_lights=tls, lanes=lanes,
            weather={"fog_density": 0, "cloudiness": 0, "precipitation": 0,
                     "wetness": 0, "sun_altitude_angle": 80, "wind_intensity": 0},
            map_name="Town01", random_seed=0, traffic_density=1,
        ))
    return frames


def make_S02_pedestrian_far_avoidance() -> List[FrameData]:
    """S02: 行人远距避让

    行人距车道>20m，车辆无避险需要。正常减速避让。
    预期行为层输出: approaching_pedestrian
    预期规则触发: 无（distance≈25m > 20m 阈值）

    场景层产出: 1 车 + 1 行人
    关系: nearby_pedestrian
    """
    frames: List[FrameData] = []
    for f_id in range(6):
        t = f_id * 0.4
        base_x = 10.0 + t * 8.0
        vehicles = [
            {
                "entity_id": "veh_030",
                "vehicle_type": "vehicle.audi.tt",
                "location_x": base_x, "location_y": 0.3, "location_z": 0.5,
                "speed": 8.0, "heading_rad": 0.0,
            },
        ]
        pedestrians = [
            {
                "entity_id": "ped_030",
                "location_x": base_x + 5.0, "location_y": 22.0, "location_z": 0.0,
                "speed": 0.0, "action": "Standing",
            },
        ]
        lanes = [
            {"entity_id": "road_3_lane_1", "road_id": 3, "lane_id": 1, "center_x": 0.0, "center_y": 0.0, "length": 200.0, "speed_limit": 10.0},
        ]
        frames.append(FrameData(
            frame_id=f_id, elapsed_seconds=t, delta_seconds=0.4,
            vehicles=vehicles, pedestrians=pedestrians, traffic_lights=[], lanes=lanes,
            weather={"fog_density": 0, "cloudiness": 20, "precipitation": 0,
                     "wetness": 0, "sun_altitude_angle": 70, "wind_intensity": 0},
            map_name="Town01", random_seed=0, traffic_density=1,
        ))
    return frames

# ─── B 单点异常 ────────────────────────────────────────────────────

def make_S10_pedestrian_sudden_crossing() -> List[FrameData]:
    """S10: 行人鬼探头

    自车行驶中，行人从公交车后方突然横穿。距离<5m 时行人进入车道。
    预期行为层输出: approaching_pedestrian, blocked_view
    预期规则触发: R4 (pedestrian proximity) + RSS ped_proximity
    预期 SafetyViolation: sv_001

    场景层产出: 自车 + 公交(停) + 行人
    关系: ahead_of (自车→公交), nearby_pedestrian, beside (行人→公交)
    """
    frames: List[FrameData] = []
    bus_center_y = -2.0  # 公交车占用车道 - bus 宽度
    for f_id in range(6):
        t = f_id * 0.4
        bus_x = 20.0
        # 自车位置：逐渐接近公交
        ego_x = bus_x - 25.0 + t * 8.0  # 25→... 逐渐拉近距离
        # 行人位置：frame3 之前在人行道上，frame3 后进入车道
        if f_id < 3:
            ped_x = bus_x + 2.0
            ped_y = -5.0  # 在公交后方人行道
        else:
            ped_x = bus_x + 5.0
            ped_y = -0.5  # 突然进入车道
        vehicles = [
            {
                "entity_id": "veh_100",
                "vehicle_type": "vehicle.audi.tt",
                "location_x": ego_x, "location_y": 0.3, "location_z": 0.5,
                "speed": 8.0 if f_id < 4 else 4.0, "heading_rad": 0.0,
            },
            {
                "entity_id": "veh_101",
                "vehicle_type": "vehicle.mercedes.coupe",
                "location_x": bus_x, "location_y": bus_center_y, "location_z": 0.5,
                "speed": 0.0, "heading_rad": 0.0,  # 公交车静止
            },
        ]
        pedestrians = [
            {
                "entity_id": "ped_100",
                "location_x": ped_x, "location_y": ped_y, "location_z": 0.0,
                "speed": 2.5, "action": "Crossing",
            },
        ]
        lanes = [
            {"entity_id": "road_10_lane_1", "road_id": 10, "lane_id": 1, "center_x": 0.0, "center_y": 0.0, "length": 100.0, "speed_limit": 10.0},
            {"entity_id": "road_10_lane_2", "road_id": 10, "lane_id": 2, "center_x": 0.0, "center_y": 4.0, "length": 100.0, "speed_limit": 10.0},
        ]
        frames.append(FrameData(
            frame_id=f_id, elapsed_seconds=t, delta_seconds=0.4,
            vehicles=vehicles, pedestrians=pedestrians, traffic_lights=[], lanes=lanes,
            weather={"fog_density": 0, "cloudiness": 10, "precipitation": 0,
                     "wetness": 0, "sun_altitude_angle": 60, "wind_intensity": 0},
            map_name="Town01", random_seed=0, traffic_density=2,
        ))
    return frames


def make_S11_unprotected_left_turn_conflict() -> List[FrameData]:
    """S11: 无信号左转冲突

    对向直行+本车左转，无信号灯控制路口。
    预期行为层输出: yielding_to, wrong_side_meeting
    预期规则触发: R5 (intersection priority) + RSS junction_priority
    预期 SafetyViolation: sv_002

    场景层产出: 2 车 + 1 路口
    关系: adjacent_lane, ahead_of
    """
    frames: List[FrameData] = []
    for f_id in range(6):
        t = f_id * 0.4
        # 本车(左转)：从坐标原点驶向路口
        ego_x = 0.0 + t * 3.0
        ego_y = 2.0
        # 对向直行(从路口上方下来)
        opp_x = 28.0 - t * 5.0
        opp_y = 2.5
        vehicles = [
            {
                "entity_id": "veh_110",
                "vehicle_type": "vehicle.audi.tt",
                "location_x": ego_x, "location_y": ego_y, "location_z": 0.5,
                "speed": 3.0, "heading_rad": 0.0,
            },
            {
                "entity_id": "veh_111",
                "vehicle_type": "vehicle.tesla.model3",
                "location_x": opp_x, "location_y": opp_y, "location_z": 0.5,
                "speed": 5.0, "heading_rad": 3.14159,  # 正向驶来（180°）
            },
        ]
        lanes = [
            {"entity_id": "road_11_lane_1", "road_id": 11, "lane_id": 1, "center_x": 0.0, "center_y": 0.0, "length": 50.0, "speed_limit": 10.0},
            {"entity_id": "road_11_lane_2", "road_id": 11, "lane_id": 2, "center_x": 0.0, "center_y": 4.0, "length": 50.0, "speed_limit": 10.0},
            {"entity_id": "junction_11", "road_id": 0, "lane_id": 0, "center_x": 15.0, "center_y": 2.0, "length": 30.0, "speed_limit": 8.0},
        ]
        frames.append(FrameData(
            frame_id=f_id, elapsed_seconds=t, delta_seconds=0.4,
            vehicles=vehicles, pedestrians=[], traffic_lights=[], lanes=lanes,
            weather={"fog_density": 0, "cloudiness": 0, "precipitation": 0,
                     "wetness": 0, "sun_altitude_angle": 80, "wind_intensity": 0},
            map_name="Town01", random_seed=0, traffic_density=2,
        ))
    return frames

def make_S12_red_light_running() -> List[FrameData]:
    """S12: 红灯抢行

    自车在红灯时冲过停止线，信号灯状态从 Green→Red。
    预期行为层输出: approaching_intersection
    预期规则触发: R1 (traffic signal) + RSS red_light
    预期 SafetyViolation: sv_003

    场景层产出: 1 信号灯 + 1 车
    关系: in_junction, controlled_by
    """
    frames: List[FrameData] = []
    tl_state = ["Green", "Green", "Red", "Red", "Red", "Red"]
    for f_id in range(6):
        t = f_id * 0.4
        ego_x = -5.0 + t * 8.0  # 第3帧到达路口 (x≈15)
        vehicles = [
            {
                "entity_id": "veh_120",
                "vehicle_type": "vehicle.audi.tt",
                "location_x": ego_x, "location_y": 0.3, "location_z": 0.5,
                "speed": 8.0, "heading_rad": 0.0,
            },
        ]
        tls = [
            {"entity_id": "tl_120", "state": tl_state[f_id], "position_x": 20.0, "position_y": 0.0},
        ]
        lanes = [
            {"entity_id": "road_12_lane_1", "road_id": 12, "lane_id": 1, "center_x": 0.0, "center_y": 0.0, "length": 60.0, "speed_limit": 10.0},
            {"entity_id": "junction_12", "road_id": 0, "lane_id": 0, "center_x": 15.0, "center_y": 0.0, "length": 20.0, "speed_limit": 8.0},
        ]
        frames.append(FrameData(
            frame_id=f_id, elapsed_seconds=t, delta_seconds=0.4,
            vehicles=vehicles, pedestrians=[], traffic_lights=tls, lanes=lanes,
            weather={"fog_density": 0, "cloudiness": 0, "precipitation": 0,
                     "wetness": 0, "sun_altitude_angle": 80, "wind_intensity": 0},
            map_name="Town01", random_seed=0, traffic_density=1,
        ))
    return frames


def make_S13_too_close_following() -> List[FrameData]:
    """S13: 跟车过近（追尾风险）

    后车 THW < 1.0s（前车 6m/s，后车 8m/s，间距≈3m→0.5s）。
    预期行为层输出: following, blocked_view
    预期规则触发: R7 (safe distance) + RSS THW
    预期 SafetyViolation: sv_004

    场景层产出: 2 车
    关系: ahead_of
    """
    frames: List[FrameData] = []
    for f_id in range(6):
        t = f_id * 0.4
        leader_x = 20.0 + t * 6.0
        follower_x = leader_x - 3.0  # 仅3m车距
        vehicles = [
            {
                "entity_id": "veh_130",
                "vehicle_type": "vehicle.mercedes.coupe",
                "location_x": leader_x, "location_y": 0.3, "location_z": 0.5,
                "speed": 6.0, "heading_rad": 0.0,
            },
            {
                "entity_id": "veh_131",
                "vehicle_type": "vehicle.audi.tt",
                "location_x": follower_x, "location_y": 0.3, "location_z": 0.5,
                "speed": 8.0, "heading_rad": 0.0,
            },
        ]
        lanes = [
            {"entity_id": "road_13_lane_1", "road_id": 13, "lane_id": 1, "center_x": 0.0, "center_y": 0.0, "length": 100.0, "speed_limit": 10.0},
        ]
        frames.append(FrameData(
            frame_id=f_id, elapsed_seconds=t, delta_seconds=0.4,
            vehicles=vehicles, pedestrians=[], traffic_lights=[], lanes=lanes,
            weather={"fog_density": 0, "cloudiness": 0, "precipitation": 0,
                     "wetness": 0, "sun_altitude_angle": 80, "wind_intensity": 0},
            map_name="Town01", random_seed=0, traffic_density=2,
        ))
    return frames

# ─── C 多车冲突 ────────────────────────────────────────────────────

def make_S20_merging_conflict() -> List[FrameData]:
    """S20: 汇入主路冲突

    匝道（车道1）车辆加速并线, 主路（车道2）车辆未让行。
    预期行为层输出: changing_lane, yielding_to
    预期规则触发: R5 (intersection/junction priority) + RSS lane_change_safe
    预期 SafetyViolation: sv_005

    场景层产出: 3 车 + 2 道路(主路+匝道)
    关系: adjacent_lane, in_lane
    """
    frames: List[FrameData] = []
    for f_id in range(6):
        t = f_id * 0.4
        # 主路车
        main_x = 15.0 + t * 8.0
        # 匝道车：逐渐靠近主车道
        ramp_x = 0.0 + t * 10.0  # 匝道车加速
        ramp_y = 6.0 - t * 0.5   # 逐渐减少横向偏移（并线轨迹）
        vehicles = [
            {
                "entity_id": "veh_200",
                "vehicle_type": "vehicle.tesla.model3",
                "location_x": main_x, "location_y": 2.3, "location_z": 0.5,
                "speed": 8.0, "heading_rad": 0.0,
            },
            {
                "entity_id": "veh_201",
                "vehicle_type": "vehicle.audi.tt",
                "location_x": ramp_x, "location_y": max(ramp_y, 2.0), "location_z": 0.5,
                "speed": 10.0, "heading_rad": 0.05,
            },
            {
                "entity_id": "veh_202",
                "vehicle_type": "vehicle.mercedes.coupe",
                "location_x": main_x - 10.0, "location_y": 2.3, "location_z": 0.5,
                "speed": 8.0, "heading_rad": 0.0,
            },
        ]
        lanes = [
            {"entity_id": "road_20_main", "road_id": 20, "lane_id": 1, "center_x": 0.0, "center_y": 2.0, "length": 80.0, "speed_limit": 15.0},
            {"entity_id": "road_20_ramp", "road_id": 20, "lane_id": 2, "center_x": 0.0, "center_y": 6.0, "length": 60.0, "speed_limit": 10.0},
        ]
        frames.append(FrameData(
            frame_id=f_id, elapsed_seconds=t, delta_seconds=0.4,
            vehicles=vehicles, pedestrians=[], traffic_lights=[], lanes=lanes,
            weather={"fog_density": 0, "cloudiness": 30, "precipitation": 0,
                     "wetness": 0, "sun_altitude_angle": 60, "wind_intensity": 0},
            map_name="Town01", random_seed=0, traffic_density=3,
        ))
    return frames


def make_S21_three_way_unsignalized() -> List[FrameData]:
    """S21: 三车交叉路口无信号

    三方向车辆同时到达无信号路口，让行规则争议。
    预期行为层输出: yielding_to, approaching_intersection
    预期规则触发: R5 + RSS first_come_first_serve
    预期 SafetyViolation: sv_006

    场景层产出: 3 车 + 1 路口
    关系: in_junction, ahead_of
    """
    frames: List[FrameData] = []
    angles = [0.0, 2.094, 4.189]  # 0°, 120°, 240°
    for f_id in range(6):
        t = f_id * 0.4
        vehicles = []
        for i in range(3):
            rad = angles[i]
            speed = 4.0
            dist = 25.0 - t * speed
            vehicles.append({
                "entity_id": f"veh_21{i}",
                "vehicle_type": "vehicle.audi.tt",
                "location_x": dist * -1,  # 从路口外驶入, 不同方向经旋转计算
                "location_y": 0.0 if i == 0 else (-30 if i == 1 else 30),
                "location_z": 0.5,
                "speed": speed, "heading_rad": rad,
            })
        lanes = [
            {"entity_id": "road_21_lane_1", "road_id": 21, "lane_id": 1, "center_x": 0.0, "center_y": 0.0, "length": 60.0, "speed_limit": 7.0},
            {"entity_id": "road_21_lane_2", "road_id": 21, "lane_id": 2, "center_x": 0.0, "center_y": 4.0, "length": 60.0, "speed_limit": 7.0},
            {"entity_id": "road_21_lane_3", "road_id": 21, "lane_id": 3, "center_x": 0.0, "center_y": -4.0, "length": 60.0, "speed_limit": 7.0},
            {"entity_id": "junction_21", "road_id": 0, "lane_id": 0, "center_x": 0.0, "center_y": 0.0, "length": 20.0, "speed_limit": 5.0},
        ]
        frames.append(FrameData(
            frame_id=f_id, elapsed_seconds=t, delta_seconds=0.4,
            vehicles=vehicles, pedestrians=[], traffic_lights=[], lanes=lanes,
            weather={"fog_density": 0, "cloudiness": 20, "precipitation": 0,
                     "wetness": 0, "sun_altitude_angle": 70, "wind_intensity": 0},
            map_name="Town01", random_seed=0, traffic_density=3,
        ))
    return frames


def make_S22_emergency_vehicle_yielding() -> List[FrameData]:
    """S22: 应急车辆通行权警报

    后方救护车鸣笛逼近，前车未让行。
    预期行为层输出: yielding_to, emergency
    预期规则触发: R6 (emergency priority) + RSS emergency_priority
    预期 SafetyViolation: sv_007

    场景层产出: 1 救护车 + 1 自车
    关系: ahead_of (反向 → 后车有紧急标识), beside
    """
    frames: List[FrameData] = []
    for f_id in range(6):
        t = f_id * 0.4
        ego_x = 10.0 + t * 4.0
        amb_x = (10.0 - 8.0) + t * 10.0  # 救护车从后方 8m 起以 10 m/s 追前
        vehicles = [
            {
                "entity_id": "veh_220",
                "vehicle_type": "vehicle.audi.tt",
                "location_x": ego_x, "location_y": 0.3, "location_z": 0.5,
                "speed": 4.0, "heading_rad": 0.0,
                "is_emergency": False,
            },
            {
                "entity_id": "veh_221",
                "vehicle_type": "vehicle.mercedes.coupe",
                "location_x": amb_x, "location_y": 0.7, "location_z": 0.5,
                "speed": 10.0, "heading_rad": 0.0,
                "is_emergency": True,
            },
        ]
        lanes = [
            {"entity_id": "road_22_lane_1", "road_id": 22, "lane_id": 1, "center_x": 0.0, "center_y": 0.0, "length": 100.0, "speed_limit": 10.0},
        ]
        frames.append(FrameData(
            frame_id=f_id, elapsed_seconds=t, delta_seconds=0.4,
            vehicles=vehicles, pedestrians=[], traffic_lights=[], lanes=lanes,
            weather={"fog_density": 0, "cloudiness": 0, "precipitation": 0,
                     "wetness": 0, "sun_altitude_angle": 80, "wind_intensity": 0},
            map_name="Town01", random_seed=0, traffic_density=2,
        ))
    return frames

# ─── D 跨层联动 ────────────────────────────────────────────────────

def make_S30_night_pedestrian_sudden() -> List[FrameData]:
    """S30: 夜间 + 行人鬼探头

    低照度（sun_altitude_angle=-10）+ 行人从停放车辆后方横穿。
    环境耦合放大异常：亮度↓ → TTC 感知延迟↑ → 制动距离有效延后。
    预期行为层输出: approaching_pedestrian, blocked_view
    预期规则触发: R4 (pedestrian proximity) + R9 (environment adaptation)
                + RSS ped_proximity
    预期 SafetyViolation: sv_008

    场景层产出: 自车 + 1 行人 + 1 停靠车
    关系: nearby_pedestrian, has_environment, weather_context
    """
    frames: List[FrameData] = []
    for f_id in range(6):
        t = f_id * 0.4
        ego_x = 0.0 + t * 6.0
        parked_x = 25.0  # 停靠车挡视线
        if f_id < 3:
            ped_x = parked_x + 2.0
            ped_y = -5.0
        else:
            ped_x = parked_x + 5.0
            ped_y = 0.5
        vehicles = [
            {
                "entity_id": "veh_300",
                "vehicle_type": "vehicle.audi.tt",
                "location_x": ego_x, "location_y": 0.3, "location_z": 0.5,
                "speed": 6.0, "heading_rad": 0.0,
            },
            {
                "entity_id": "veh_301",
                "vehicle_type": "vehicle.tesla.model3",
                "location_x": parked_x, "location_y": -2.0, "location_z": 0.5,
                "speed": 0.0, "heading_rad": 0.0,
            },
        ]
        pedestrians = [
            {
                "entity_id": "ped_300",
                "location_x": ped_x, "location_y": ped_y, "location_z": 0.0,
                "speed": 2.0, "action": "Crossing",
            },
        ]
        lanes = [
            {"entity_id": "road_30_lane_1", "road_id": 30, "lane_id": 1, "center_x": 0.0, "center_y": 0.0, "length": 80.0, "speed_limit": 10.0},
            {"entity_id": "road_30_lane_2", "road_id": 30, "lane_id": 2, "center_x": 0.0, "center_y": 4.0, "length": 80.0, "speed_limit": 10.0},
        ]
        frames.append(FrameData(
            frame_id=f_id, elapsed_seconds=t, delta_seconds=0.4,
            vehicles=vehicles, pedestrians=pedestrians, traffic_lights=[], lanes=lanes,
            weather={"fog_density": 15, "cloudiness": 80, "precipitation": 0,
                     "wetness": 30, "sun_altitude_angle": -10, "wind_intensity": 0},
            map_name="Town01", random_seed=0, traffic_density=2,
        ))
    return frames


def make_S31_rainy_lane_change_blind() -> List[FrameData]:
    """S31: 雨天跨线盲变

    大雨(降水=80, 湿滑=90, 光照低) + 邻车突然变道。
    环境耦合放大异常：雨天→ACC目标丢失→变道预警响应时间延长。
    预期行为层输出: changing_lane, wrong_side_meeting
    预期规则触发: R5 (lane change safe) + R10 (adverse weather)
                + RSS lane_change_safe
    预期 SafetyViolation: sv_009

    场景层产出: 自车 + 1 邻车
    关系: adjacent_lane, beside
    """
    frames: List[FrameData] = []
    for f_id in range(6):
        t = f_id * 0.4
        ego_x = 5.0 + t * 7.0
        # 邻车：从邻车道逐渐侵入本车道
        other_x = ego_x + 2.0
        other_y = 4.5 - f_id * 0.6  # 每帧横向入侵 0.6m
        if other_y < 1.5:
            other_y = 1.5  # 侵入到本车道3/4处
        vehicles = [
            {
                "entity_id": "veh_310",
                "vehicle_type": "vehicle.audi.tt",
                "location_x": ego_x, "location_y": 0.3, "location_z": 0.5,
                "speed": 7.0, "heading_rad": 0.0,
            },
            {
                "entity_id": "veh_311",
                "vehicle_type": "vehicle.tesla.model3",
                "location_x": other_x, "location_y": round(other_y, 2), "location_z": 0.5,
                "speed": 7.5, "heading_rad": 0.0,
            },
        ]
        lanes = [
            {"entity_id": "road_31_lane_1", "road_id": 31, "lane_id": 1, "center_x": 0.0, "center_y": 0.0, "length": 60.0, "speed_limit": 10.0},
            {"entity_id": "road_31_lane_2", "road_id": 31, "lane_id": 2, "center_x": 0.0, "center_y": 4.0, "length": 60.0, "speed_limit": 10.0},
        ]
        frames.append(FrameData(
            frame_id=f_id, elapsed_seconds=t, delta_seconds=0.4,
            vehicles=vehicles, pedestrians=[], traffic_lights=[], lanes=lanes,
            weather={"fog_density": 60, "cloudiness": 95, "precipitation": 80,
                     "wetness": 90, "sun_altitude_angle": 20, "wind_intensity": 15},
            map_name="Town01", random_seed=0, traffic_density=2,
        ))
    return frames


def make_S32_construction_detour() -> List[FrameData]:
    """S32: 施工路段绕行

    车道收窄（部分车道用锥桶封闭）→ 车辆需绕行。
    预期行为层输出: changing_lane, following
    预期规则触发: R11 (construction zone) + RSS safe_coridor
    预期 SafetyViolation: sv_010

    场景层产出: 自车 + 1 前车 + 1 锥桶路障(用 PedestrianEntity 模拟)
    关系: in_lane, adjacent_lane
    """
    frames: List[FrameData] = []
    for f_id in range(6):
        t = f_id * 0.4
        # 自车、前车同车道；锥桶在车道2标记封闭
        veh_x = 10.0 + t * 5.0
        lead_x = veh_x + 8.0
        # 锥桶：从 frame 2 开始出现施工区域
        cones = []
        if f_id >= 2:
            for j in range(4):
                cones.append({
                    "entity_id": f"ped_32_cone_{j}",
                    "location_x": 28.0 + j * 2.0, "location_y": 4.2,
                    "location_z": 0.0, "speed": 0.0, "action": "Obstacle",
                    "labels": ["ConstructionCone"],
                })
        vehicles = [
            {
                "entity_id": "veh_320",
                "vehicle_type": "vehicle.audi.tt",
                "location_x": veh_x, "location_y": 0.3, "location_z": 0.5,
                "speed": 5.0, "heading_rad": 0.0,
            },
            {
                "entity_id": "veh_321",
                "vehicle_type": "vehicle.mercedes.coupe",
                "location_x": lead_x, "location_y": 0.3, "location_z": 0.5,
                "speed": 5.0, "heading_rad": 0.0,
            },
        ]
        lanes = [
            {"entity_id": "road_32_lane_1", "road_id": 32, "lane_id": 1, "center_x": 0.0, "center_y": 0.0, "length": 80.0, "speed_limit": 8.0},
            {"entity_id": "road_32_lane_2", "road_id": 32, "lane_id": 2, "center_x": 0.0, "center_y": 4.0, "length": 80.0, "speed_limit": 8.0},
        ]
        frames.append(FrameData(
            frame_id=f_id, elapsed_seconds=t, delta_seconds=0.4,
            vehicles=vehicles, pedestrians=cones, traffic_lights=[], lanes=lanes,
            weather={"fog_density": 10, "cloudiness": 30, "precipitation": 0,
                     "wetness": 0, "sun_altitude_angle": 60, "wind_intensity": 5},
            map_name="Town01", random_seed=0, traffic_density=2,
        ))
    return frames


def make_S33_glare_multi_pedestrian() -> List[FrameData]:
    """S33: 路口逆光 + 多行人横穿

    朝向太阳（sun_altitude_angle=5°，低角度逆光）+ 路口多行人横穿。
    环境耦合放大异常：逆光→相机/人眼感应延迟→TTC 高估。
    预期行为层输出: approaching_pedestrian, blocked_view
    预期规则触发: R4 (pedestrian proximity) + R9 (environment adaptation)
                + RSS ped_proximity
    预期 SafetyViolation: sv_011

    场景层产出: 自车 + 3 行人 + 1 路口
    关系: nearby_pedestrian, weather_context
    """
    frames: List[FrameData] = []
    for f_id in range(6):
        t = f_id * 0.4
        ego_x = -5.0 + t * 5.0
        # 三个行人横穿，速度不同
        speeds = [1.0, 1.5, 2.0]
        pedestrians = []
        for i, sp in enumerate(speeds):
            ped_x = 20.0 + i * 3.0
            ped_y = 8.0 - t * sp  # 从马路对面横穿
            pedestrians.append({
                "entity_id": f"ped_33{i}",
                "location_x": ped_x, "location_y": round(max(ped_y, -1.0), 2),
                "location_z": 0.0, "speed": sp, "action": "Crossing",
            })
        vehicles = [
            {
                "entity_id": "veh_330",
                "vehicle_type": "vehicle.audi.tt",
                "location_x": ego_x, "location_y": 0.3, "location_z": 0.5,
                "speed": 7.0, "heading_rad": 0.0,
            },
        ]
        lanes = [
            {"entity_id": "road_33_lane_1", "road_id": 33, "lane_id": 1, "center_x": 0.0, "center_y": 0.0, "length": 60.0, "speed_limit": 10.0},
            {"entity_id": "road_33_lane_2", "road_id": 33, "lane_id": 2, "center_x": 0.0, "center_y": 4.0, "length": 60.0, "speed_limit": 10.0},
            {"entity_id": "junction_33", "road_id": 0, "lane_id": 0, "center_x": 20.0, "center_y": 2.0, "length": 30.0, "speed_limit": 7.0},
        ]
        frames.append(FrameData(
            frame_id=f_id, elapsed_seconds=t, delta_seconds=0.4,
            vehicles=vehicles, pedestrians=pedestrians, traffic_lights=[], lanes=lanes,
            weather={"fog_density": 0, "cloudiness": 5, "precipitation": 0,
                     "wetness": 0, "sun_altitude_angle": 5, "wind_intensity": 0},
            map_name="Town01", random_seed=0, traffic_density=1,
        ))
    return frames


# ─── 注册表 / 总入口 ─────────────────────────────────────────────

# 14 个场景的工厂 + 元数据：
#   tier: A/B/C/D 基线/异常/多车/跨层
#   category: 短标签 - 用于自动验证
#   expected_rules: 预期规则层应命中的规则 ID 列表
#   expected_sv: 预期 SafetyViolation ID（"-" 表示无）
#   expected_behaviors: 预期行为层应识别出的 maneuver 标签集合
# 14 个场景的工厂 + 元数据（已对齐 v3 §4.14 R1-R18 + §4.9 RSS 子类 R13a/R14a/R15a/b）：
#   tier: A/B/C/D  基线/单点异常/多车冲突/跨层联动
#   category: 短标签 - 用于自动验证
#   expected_rules: 预期 v3 规则层应命中的规则 ID 列表
#                   规则 ID 严格对齐 v3 §4.14.2 (R1-R18) 与 v3 §4.9.3 (RSS 子类)
#   expected_sv: 预期 SafetyViolation ID（"-" 表示无）
#   expected_behaviors: 预期行为层应识别出的 maneuver 标签集合
SCENARIO_REGISTRY: Dict[str, Dict[str, Any]] = {
    "S00": {"tier": "A", "category": "baseline_following",
            "expected_rules": [], "expected_sv": "-",
            "expected_behaviors": ["following"]},
    "S01": {"tier": "A", "category": "baseline_signal_intersection",
            "expected_rules": [], "expected_sv": "-",
            "expected_behaviors": ["following", "approaching_intersection"]},
    "S02": {"tier": "A", "category": "baseline_pedestrian_far",
            "expected_rules": [], "expected_sv": "-",
            "expected_behaviors": ["approaching_pedestrian"]},
    # B 单点异常 (v3 §4.14 + §4.9 对齐)
    "S10": {"tier": "B", "category": "anomaly_pedestrian_sudden",
            # v3 R1 = 行人优先（人行横道）；R13a = RSS SafeDistanceViolation
            "expected_rules": ["R1", "RSS_R13a"], "expected_sv": "sv_001",
            "expected_behaviors": ["approaching_pedestrian", "blocked_view"]},
    "S11": {"tier": "B", "category": "anomaly_unprotected_left",
            # v3 R4 = 对向会车违规；R7 = 路口未让行
            "expected_rules": ["R4", "R7"], "expected_sv": "sv_002",
            "expected_behaviors": ["yielding_to", "wrong_side_meeting"]},
    "S12": {"tier": "B", "category": "anomaly_red_light_run",
            # v3 R2 = 闯红灯
            "expected_rules": ["R2"], "expected_sv": "sv_003",
            "expected_behaviors": ["approaching_intersection"]},
    "S13": {"tier": "B", "category": "anomaly_too_close",
            # v3 R13a = RSS SafeDistanceViolation (纵向); R15a = NoProperResponse
            "expected_rules": ["RSS_R13a", "RSS_R15a"], "expected_sv": "sv_004",
            "expected_behaviors": ["following", "blocked_view"]},
    # C 多车冲突
    "S20": {"tier": "C", "category": "complexity_merging",
            # v3 R7 = 路口未让行（这里扩展包含汇入主路的让行场景）
            "expected_rules": ["R7"], "expected_sv": "sv_005",
            "expected_behaviors": ["changing_lane", "yielding_to"]},
    "S21": {"tier": "C", "category": "complexity_three_way",
            "expected_rules": ["R7"], "expected_sv": "sv_006",
            "expected_behaviors": ["yielding_to", "approaching_intersection"]},
    "S22": {"tier": "C", "category": "complexity_emergency",
            # v3 R7=路口未让行；R8=弱势参与者保护（扩展到应急车辆优先权）
            "expected_rules": ["R7", "R8"], "expected_sv": "sv_007",
            "expected_behaviors": ["yielding_to", "emergency"]},
    # D 跨层联动 (环境耦合)
    "S30": {"tier": "D", "category": "coupling_night_ped",
            # v3 R1 = 行人优先；R8 = 弱势参与者保护；R13a = RSS 纵向距离
            "expected_rules": ["R1", "R8", "RSS_R13a"], "expected_sv": "sv_008",
            "expected_behaviors": ["approaching_pedestrian", "blocked_view"]},
    "S31": {"tier": "D", "category": "coupling_rainy_lc",
            # v3 R11 = 恶劣天气限速；R17 = 不按规定车道；R14a = LateralDangerousState
            "expected_rules": ["R11", "R17", "RSS_R14a"], "expected_sv": "sv_009",
            "expected_behaviors": ["changing_lane", "wrong_side_meeting"]},
    "S32": {"tier": "D", "category": "coupling_construction",
            # v3 R14 = 违反交通标志；R17 = 不按规定车道
            "expected_rules": ["R14", "R17"], "expected_sv": "sv_010",
            "expected_behaviors": ["changing_lane", "following"]},
    "S33": {"tier": "D", "category": "coupling_glare_multi_ped",
            # v3 R1 = 行人优先；R8 = 弱势参与者保护；R13a = RSS 纵向距离
            "expected_rules": ["R1", "R8", "RSS_R13a"], "expected_sv": "sv_011",
            "expected_behaviors": ["approaching_pedestrian", "blocked_view"]},
}

SCENARIO_FACTORIES = {
    "S00": make_S00_baseline_following,
    "S01": make_S01_normal_signalized_intersection,
    "S02": make_S02_pedestrian_far_avoidance,
    "S10": make_S10_pedestrian_sudden_crossing,
    "S11": make_S11_unprotected_left_turn_conflict,
    "S12": make_S12_red_light_running,
    "S13": make_S13_too_close_following,
    "S20": make_S20_merging_conflict,
    "S21": make_S21_three_way_unsignalized,
    "S22": make_S22_emergency_vehicle_yielding,
    "S30": make_S30_night_pedestrian_sudden,
    "S31": make_S31_rainy_lane_change_blind,
    "S32": make_S32_construction_detour,
    "S33": make_S33_glare_multi_pedestrian,
}


def list_scenarios() -> List[str]:
    """返回所有可用场景 ID（已排序）。

    >>> list_scenarios()
    ['S00', 'S01', 'S02', 'S10', 'S11', 'S12', 'S13',
     'S20', 'S21', 'S22', 'S30', 'S31', 'S32', 'S33']
    """
    return sorted(SCENARIO_FACTORIES.keys())


def get_scenario(scenario_id: str) -> List[FrameData]:
    """按场景 ID 取出整套帧序列。

    Args:
        scenario_id: 'S00'..'S33'

    Returns:
        List[FrameData]，可直接喂入 build_snapshot()
    """
    if scenario_id not in SCENARIO_FACTORIES:
        raise KeyError(f"Unknown scenario: {scenario_id}. "
                       f"Available: {list_scenarios()}")
    return SCENARIO_FACTORIES[scenario_id]()


def get_scenario_meta(scenario_id: str) -> Dict[str, Any]:
    """取出场景元数据（tier/category/expected_rules/expected_sv）。"""
    if scenario_id not in SCENARIO_REGISTRY:
        raise KeyError(f"Unknown scenario: {scenario_id}")
    return dict(SCENARIO_REGISTRY[scenario_id])


def all_scenarios() -> Dict[str, List[FrameData]]:
    """把全部 14 个场景各跑一遍，按 {id: [frames]} 返回。

    用于阶段 9 完整跑批 / 消融实验 / baseline 对照的统一输入。
    """
    return {sid: get_scenario(sid) for sid in list_scenarios()}


def total_frames(scenario_id: Optional[str] = None) -> int:
    """统计帧数。

    Args:
        scenario_id: 指定则只数该场景；None 则数全部场景
    """
    if scenario_id is not None:
        return len(get_scenario(scenario_id))
    return sum(len(get_scenario(sid)) for sid in list_scenarios())
