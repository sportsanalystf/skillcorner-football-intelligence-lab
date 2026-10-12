"""The tracking engine: smoothing, differentiation, gating and effort detection."""

import numpy as np
import pandas as pd
import pytest

from src.config import (DT, MAX_PLAUSIBLE_ACCEL_MS2, MAX_PLAUSIBLE_SPEED_MS,
                        SPRINT_THRESHOLD_MS)
from src.tracking.kinematics import (_count_bouts, add_kinematics, detect_efforts,
                                     segment_ids, smooth_and_differentiate, speed_zone,
                                     summarise_player_physical)


def _track(n=300, v=6.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    return v * t + rng.normal(0, noise, n), np.zeros(n) + rng.normal(0, noise, n)


def test_constant_velocity_is_recovered_from_a_noisy_track():
    x, y = _track(v=6.0, noise=0.05)
    k = smooth_and_differentiate(x, y, np.ones(x.size, bool))
    assert np.nanmean(k["speed"]) == pytest.approx(6.0, abs=0.05)
    # And it is recovered *precisely*: raw differencing on this signal has a
    # standard deviation about ten times larger.
    assert np.nanstd(k["speed"]) < 0.10
    raw = np.hypot(np.diff(x), np.diff(y)) / DT
    assert np.std(raw) > 5 * np.nanstd(k["speed"])


def test_constant_acceleration_is_recovered():
    a = 2.5
    t = np.arange(200) * DT
    x = 0.5 * a * t**2
    y = np.zeros_like(t)
    k = smooth_and_differentiate(x, y, np.ones(t.size, bool))
    mid = slice(20, -20)
    assert np.nanmean(k["accel"][mid]) == pytest.approx(a, abs=0.05)


def test_heading_points_along_the_direction_of_travel():
    t = np.arange(100) * DT
    k = smooth_and_differentiate(3 * t, 3 * t, np.ones(t.size, bool))
    assert np.nanmean(k["heading"][10:-10]) == pytest.approx(np.pi / 4, abs=0.02)


def test_a_hard_turn_at_constant_speed_is_not_an_acceleration():
    # Circular motion: |v| constant, so tangential acceleration is ~0 even
    # though dv/dt is large.
    t = np.arange(300) * DT
    r, w = 10.0, 0.6
    x, y = r * np.cos(w * t), r * np.sin(w * t)
    k = smooth_and_differentiate(x, y, np.ones(t.size, bool))
    mid = slice(30, -30)
    assert np.nanmean(np.abs(k["accel"][mid])) < 0.05
    assert np.nanmean(k["accel_mag"][mid]) == pytest.approx(r * w**2, rel=0.1)


def test_implausible_values_are_masked_not_clipped():
    x = np.zeros(60)
    x[30:] = 500.0  # a 500 m jump
    k = smooth_and_differentiate(x, np.zeros(60), np.ones(60, bool))
    bad = k["speed"] > MAX_PLAUSIBLE_SPEED_MS
    assert not bad.any(), "gated speeds must be NaN, never a clipped ceiling value"
    assert np.isnan(k["speed"]).any()


def test_gate_thresholds_are_configurable_for_the_ball():
    t = np.arange(80) * DT
    x = 25.0 * t  # 25 m/s, impossible for a player, ordinary for a struck ball
    k_player = smooth_and_differentiate(x, np.zeros_like(x), np.ones(x.size, bool))
    k_ball = smooth_and_differentiate(x, np.zeros_like(x), np.ones(x.size, bool),
                                      max_speed=45.0, max_accel=400.0)
    assert np.isnan(k_player["speed"]).all()
    assert np.nanmean(k_ball["speed"]) == pytest.approx(25.0, abs=0.2)


def test_segment_ids_bridge_short_gaps_and_split_long_ones():
    present = np.array([1, 1, 0, 1, 1] + [0] * 15 + [1, 1], dtype=bool)
    seg = segment_ids(present, max_gap=10)
    assert seg[0] == seg[3] == 0, "a 1-frame gap stays in the same segment"
    assert seg[-1] == 1, "a 15-frame gap starts a new segment"
    assert (seg[5:20] == -1).all(), "the long gap itself belongs to no segment"


def test_no_velocity_is_produced_across_a_long_gap():
    x = np.concatenate([np.zeros(40), np.full(40, 300.0)])
    present = np.concatenate([np.ones(40, bool), np.zeros(30, bool), np.ones(10, bool)])
    x = np.where(present, x[: present.size], np.nan)
    k = smooth_and_differentiate(np.nan_to_num(x), np.zeros(present.size), present)
    assert np.isnan(k["speed"][40:70]).all()


def test_speed_zone_boundaries_and_nan_handling():
    got = list(speed_zone(np.array([0.0, 1.49, 1.5, 3.49, 3.5, 5.49, 5.5, 6.99, 7.0, np.nan])))
    assert got == ["walking", "walking", "jogging", "jogging", "running", "running",
                   "hsr", "hsr", "sprint", "unknown"]


def test_effort_detection_requires_the_threshold_to_be_held():
    speed = np.zeros(100)
    speed[10] = 9.0                 # one frame -- an artefact
    speed[50:60] = 9.0              # one second -- a real effort
    efforts = detect_efforts(speed, np.arange(100), SPRINT_THRESHOLD_MS)
    assert len(efforts) == 1
    assert efforts[0]["start_frame"] == 50
    assert efforts[0]["duration_s"] == pytest.approx(1.0)
    assert efforts[0]["peak_speed"] == pytest.approx(9.0)


def test_count_bouts_ignores_flickers():
    m = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
    assert _count_bouts(m, min_frames=3) == 1


def test_add_kinematics_treats_periods_as_hard_boundaries():
    df = pd.DataFrame({
        "player_id": [1] * 40,
        "frame": list(range(20)) + list(range(1000, 1020)),
        "period": [1] * 20 + [2] * 20,
        "x": list(np.arange(20) * 0.5) + list(np.arange(20) * 0.5 + 60),
        "y": [0.0] * 40,
        "is_detected": [True] * 40,
    })
    out = add_kinematics(df)
    assert len(out) == 40
    # The 60 m half-time jump must not create a step.
    assert out["step_m"].max() < 1.0


def test_summarise_player_physical_totals_are_consistent():
    n = 600
    df = pd.DataFrame({
        "player_id": [7] * n, "frame": np.arange(n), "period": [1] * n,
        "x": np.arange(n) * 0.4, "y": np.zeros(n), "is_detected": [True] * n,
    })
    trk = add_kinematics(df)
    s = summarise_player_physical(trk)
    # 0.4 m per 0.1 s = 4 m/s for 60 s -> 240 m, all in the "running" band.
    assert s["total_distance_m"] == pytest.approx(240, rel=0.02)
    assert s["distance_running_m"] == pytest.approx(s["total_distance_m"], rel=0.05)
    assert s["sprint_count"] == 0
    zone_sum = sum(s[f"distance_{z}_m"] for z in
                   ["walking", "jogging", "running", "hsr", "sprint"])
    assert zone_sum == pytest.approx(s["total_distance_m"], rel=1e-6)
    assert s["m_per_min"] == pytest.approx(s["total_distance_m"] / s["tracked_minutes"])
