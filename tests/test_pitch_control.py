"""The pitch-control model: symmetry, monotonicity and sane extremes."""

import numpy as np
import pandas as pd
import pytest

from src.tracking.pitch_control import control_summary, control_surface
from src.utils.pitch import STANDARD_PITCH


def _players(home_xy, away_xy, vel=None):
    rows = []
    for i, (x, y) in enumerate(home_xy):
        v = (vel or {}).get(("h", i), (0.0, 0.0))
        rows.append({"player_id": i, "team_id": 1, "x_std": x, "y_std": y,
                     "vx": v[0], "vy": v[1]})
    for i, (x, y) in enumerate(away_xy):
        v = (vel or {}).get(("a", i), (0.0, 0.0))
        rows.append({"player_id": 100 + i, "team_id": 2, "x_std": x, "y_std": y,
                     "vx": v[0], "vy": v[1]})
    return pd.DataFrame(rows)


def test_a_mirrored_pitch_splits_control_exactly_in_half():
    home = [(-20, y) for y in (-20, 0, 20)]
    away = [(20, y) for y in (-20, 0, 20)]
    res = control_surface(_players(home, away), home_team_id=1)
    assert res.home_share == pytest.approx(0.5, abs=1e-9)


def test_control_is_antisymmetric_under_swapping_the_teams():
    home = [(-30, 0), (-10, 12), (5, -20)]
    away = [(25, 4), (12, -14), (-2, 22)]
    a = control_surface(_players(home, away), home_team_id=1)
    b = control_surface(_players(home, away), home_team_id=2)
    assert np.allclose(a.p_home, 1 - b.p_home, atol=1e-9)


def test_a_cell_next_to_a_lone_player_is_almost_entirely_theirs():
    res = control_surface(_players([(-40, 0)], [(40, 0)]), home_team_id=1)
    xi = int(np.argmin(np.abs(res.xs + 40)))
    yi = int(np.argmin(np.abs(res.ys - 0)))
    assert res.p_home[yi, xi] > 0.99
    xi2 = int(np.argmin(np.abs(res.xs - 40)))
    assert res.p_home[yi, xi2] < 0.01


def test_velocity_shifts_control_ahead_of_the_player():
    still = control_surface(_players([(0, 0)], [(30, 0)]), home_team_id=1)
    moving = control_surface(_players([(0, 0)], [(30, 0)], vel={("h", 0): (7.0, 0.0)}),
                             home_team_id=1)
    xi = int(np.argmin(np.abs(still.xs - 12)))
    yi = int(np.argmin(np.abs(still.ys - 0)))
    assert moving.p_home[yi, xi] > still.p_home[yi, xi]


def test_probabilities_stay_in_range_and_the_grid_is_the_requested_shape():
    res = control_surface(_players([(-10, 5), (0, 0)], [(10, -5)]), home_team_id=1,
                          nx=30, ny=20)
    assert res.p_home.shape == (20, 30)
    assert res.p_home.min() >= 0.0 and res.p_home.max() <= 1.0


def test_missing_a_whole_team_returns_an_even_split_rather_than_failing():
    res = control_surface(_players([(0, 0)], []), home_team_id=1)
    assert np.allclose(res.p_home, 0.5)


def test_control_summary_reports_each_team_in_its_own_terms():
    # Home camped in the +x half and attacking +x should own its attacking third.
    home = [(35, y) for y in (-15, 0, 15)]
    away = [(-35, y) for y in (-15, 0, 15)]
    res = control_surface(_players(home, away), home_team_id=1)
    s = control_summary(res, home_attacks_positive_x=True)
    assert s["home_control_att_third"] > 0.95
    assert s["away_control_att_third"] > 0.95   # their attacking third is -x
    assert s["home_control"] + s["away_control"] == pytest.approx(1.0)


def test_share_in_region_matches_a_manual_mean():
    res = control_surface(_players([(-20, 0), (0, 10)], [(20, -5)]), home_team_id=1)
    m = (res.xs >= 0) & (res.xs <= 52.5)
    assert res.share_in_region(0, 52.5) == pytest.approx(float(res.p_home[:, m].mean()))
