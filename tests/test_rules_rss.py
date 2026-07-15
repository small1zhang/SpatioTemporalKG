# -*- coding: utf-8 -*-
"""T2: RSS model tests (v3 sec 4.8)."""
from __future__ import annotations
import math
import pytest
from stk.rules.rss.model import (
    DEFAULT_RSS_PARAMS,
    check_lateral_dangerous_state,
    check_no_proper_response,
    check_responsible_agent,
    check_safe_distance_longitudinal,
    compute_dmin_lat,
    compute_dmin_long,
)


class TestDefaultRSSParams:
    REQUIRED_KEYS = ("rho", "a_max_accel", "a_min_brake_long", "a_brake_long",
                     "mu", "a_min_brake_lat", "a_brake_lat")

    def test_all_required_keys_present(self):
        for k in self.REQUIRED_KEYS:
            assert k in DEFAULT_RSS_PARAMS, f"missing {k}"

    def test_all_positive_values(self):
        for k, v in DEFAULT_RSS_PARAMS.items():
            assert v > 0

    def test_rho_leq_one(self):
        assert 0 < DEFAULT_RSS_PARAMS["rho"] <= 1.0


class TestComputeDminLong:
    def test_both_zero_gives_expected(self):
        d = compute_dmin_long(0.0, 0.0)
        rho = DEFAULT_RSS_PARAMS["rho"]; amax = DEFAULT_RSS_PARAMS["a_max_accel"]
        aminb = DEFAULT_RSS_PARAMS["a_min_brake_long"]
        expected = max(0.0, 0.5*amax*rho*rho + (amax*rho)**2 / (2*aminb))
        assert math.isclose(d, expected, rel_tol=1e-9)

    def test_monotonic_in_v_A(self):
        ds = [compute_dmin_long(v, 0.0) for v in (0.0, 5.0, 10.0, 20.0)]
        for a, b in zip(ds, ds[1:]):
            assert b > a

    def test_monotonic_decreasing_in_v_B(self):
        ds = [compute_dmin_long(20.0, v) for v in (0.0, 5.0, 10.0, 18.0)]
        for a, b in zip(ds, ds[1:]):
            assert b < a

    def test_floor_zero(self):
        assert compute_dmin_long(0.0, 50.0) >= 0.0

    def test_custom_params_override(self):
        custom = {**DEFAULT_RSS_PARAMS, "rho": 0.5}
        assert compute_dmin_long(10.0, 0.0, custom) > compute_dmin_long(10.0, 0.0)


class TestComputeDminLat:
    def test_both_zero_gives_mu(self):
        assert math.isclose(compute_dmin_lat(0.0, 0.0), DEFAULT_RSS_PARAMS["mu"], rel_tol=1e-9)

    def test_monotonic_in_v_lat_A(self):
        ds = [compute_dmin_lat(v, 0.0) for v in (0.0, 0.5, 1.0, 2.0)]
        for a, b in zip(ds, ds[1:]):
            assert b > a

    def test_monotonic_decreasing_in_v_lat_B(self):
        ds = [compute_dmin_lat(2.0, v) for v in (0.0, 0.5, 1.0, 1.5)]
        for a, b in zip(ds, ds[1:]):
            assert b < a


class TestCheckSafeDistanceLongitudinal:
    def test_at_boundary_no_violation(self):
        d_min = compute_dmin_long(20.0, 10.0)
        is_v, d_act, d_min2 = check_safe_distance_longitudinal(d_min, 20.0, 10.0)
        assert is_v is False
        assert math.isclose(d_act, d_min, rel_tol=1e-9)
        assert math.isclose(d_min2, d_min, rel_tol=1e-9)

    def test_below_boundary_violation(self):
        d_min = compute_dmin_long(20.0, 10.0)
        assert check_safe_distance_longitudinal(d_min - 1e-3, 20.0, 10.0)[0] is True

    def test_above_boundary_no_violation(self):
        d_min = compute_dmin_long(20.0, 10.0)
        assert check_safe_distance_longitudinal(d_min + 1.0, 20.0, 10.0)[0] is False

    def test_zero_distance_zero_vB_violation(self):
        is_v, _, d_min = check_safe_distance_longitudinal(0.0, 10.0, 0.0)
        assert is_v is True
        assert d_min > 0


class TestCheckLateralDangerousState:
    def test_at_boundary_no_violation(self):
        d_min = compute_dmin_lat(1.0, 0.0)
        is_v, d_act, d_min2 = check_lateral_dangerous_state(d_min, 1.0, 0.0)
        assert is_v is False
        assert math.isclose(d_act, d_min2, rel_tol=1e-9)

    def test_below_boundary_violation(self):
        d_min = compute_dmin_lat(1.0, 0.0)
        assert check_lateral_dangerous_state(d_min - 1e-3, 1.0, 0.0)[0] is True

    def test_above_boundary_no_violation(self):
        d_min = compute_dmin_lat(1.0, 0.0)
        assert check_lateral_dangerous_state(d_min + 1.0, 1.0, 0.0)[0] is False


class TestCheckNoProperResponse:
    def test_all_brakes_above_threshold_not_violation(self):
        assert check_no_proper_response([0.5, 0.6, 0.7, 0.8]) is False

    def test_three_consecutive_below_threshold_violation(self):
        assert check_no_proper_response([0.1, 0.2, 0.2]) is True

    def test_three_consecutive_at_threshold_not_violation(self):
        assert check_no_proper_response([0.3, 0.3, 0.3]) is False

    def test_custom_required_consecutive(self):
        assert check_no_proper_response([0.1, 0.2], required_consecutive=2) is True
        assert check_no_proper_response([0.1, 0.5], required_consecutive=2) is False

    def test_insufficient_samples_not_violation(self):
        assert check_no_proper_response([0.1, 0.2]) is False


class TestCheckResponsibleAgent:
    def test_ego_no_response_other_compliant_responsible(self):
        assert check_responsible_agent(True, True) is True

    def test_ego_no_response_other_non_compliant_not_responsible(self):
        assert check_responsible_agent(True, False) is False

    def test_ego_has_response_other_compliant_not_responsible(self):
        assert check_responsible_agent(False, True) is False

    def test_both_bad_not_responsible(self):
        assert check_responsible_agent(False, False) is False