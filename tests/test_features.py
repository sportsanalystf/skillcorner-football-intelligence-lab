"""Team shape, passing networks, possessions and pose geometry."""

import numpy as np
import pandas as pd
import pytest

from src.features.networks import build_network
from src.features.team_shape import add_opponent_context
from tests.conftest import needs_data


def test_add_opponent_context_flips_the_opponents_frame():
    tf = pd.DataFrame({
        "frame": [1, 1], "team_id": [10, 20],
        "def_line_x": [-30.0, -25.0], "att_line_x": [10.0, 5.0],
        "centroid_x": [-8.0, -6.0], "compactness_m": [12.0, 14.0],
        "width_m": [40.0, 38.0], "hull_area_m2": [800.0, 700.0],
        "n_own_def_third": [4, 5], "n_own_att_third": [2, 1],
    })
    out = add_opponent_context(tf)
    assert len(out) == 2
    a = out[out["team_id"] == 10].iloc[0]
    # Team 20's defensive line is at -25 in *their* frame, i.e. +25 in team 10's.
    assert a["opp_def_line_x"] == pytest.approx(25.0)
    assert a["space_between_lines_m"] == pytest.approx(25.0 - 10.0)
    # Team 20's defensive third holds 5 players; that is team 10's attacking third.
    assert a["n_opp_in_our_att_third"] == 5
    assert a["superiority_att_third"] == 2 - 5


def test_build_network_places_nodes_at_mean_tracked_position():
    tracking = pd.DataFrame({
        "frame": [1, 2, 1, 2], "player_id": [1, 1, 2, 2], "team_id": [7] * 4,
        "x_std": [0.0, 10.0, -20.0, -20.0], "y_std": [0.0, 0.0, 5.0, 5.0],
        "attack_sign": [1] * 4, "position": ["CM", "CM", "CB", "CB"],
    })
    passes = pd.DataFrame({
        "team_id": [7, 7, 7], "frame_start": [1, 2, 2],
        "player_id": [1, 1, 2], "receiver_id": [2.0, 2.0, 1.0],
        "player_name": ["A", "A", "B"], "pass_outcome": ["successful"] * 3,
        "event_uid": ["x1", "x2", "x3"], "pass_epv_total": [0.01, 0.02, -0.01],
        "progression_m": [3.0, 4.0, -2.0], "fil_line_breaking": [True, False, False],
    })
    nodes, edges = build_network(passes, tracking, team_id=7, min_edge=1)
    a = nodes[nodes["player_id"] == 1].iloc[0]
    assert a["x"] == pytest.approx(5.0)   # mean of 0 and 10, not the pass origin
    assert set(nodes["involvement"]) == {3, 3}
    ab = edges[(edges["player_id"] == 1) & (edges["receiver_id"] == 2)].iloc[0]
    assert ab["weight"] == 2
    assert ab["x0"] == pytest.approx(5.0) and ab["x1"] == pytest.approx(-20.0)


def test_build_network_drops_incomplete_passes_by_default():
    tracking = pd.DataFrame({
        "frame": [1, 1], "player_id": [1, 2], "team_id": [7, 7],
        "x_std": [0.0, 5.0], "y_std": [0.0, 0.0], "attack_sign": [1, 1],
        "position": ["CM", "CB"],
    })
    passes = pd.DataFrame({
        "team_id": [7], "frame_start": [1], "player_id": [1], "receiver_id": [2.0],
        "player_name": ["A"], "pass_outcome": ["unsuccessful"], "event_uid": ["x"],
        "pass_epv_total": [0.0], "progression_m": [1.0], "fil_line_breaking": [False],
    })
    nodes, edges = build_network(passes, tracking, team_id=7, min_edge=1)
    assert edges.empty


# --------------------------------------------------------------------------- #
# Pose geometry
# --------------------------------------------------------------------------- #

def _pose_row(facing_deg: float) -> pd.DataFrame:
    """A synthetic player whose shoulders and hips face ``facing_deg``."""
    th = np.radians(facing_deg)
    fx, fy = np.cos(th), np.sin(th)
    # Left shoulder is 90 degrees anticlockwise from the facing direction.
    lx, ly = -fy, fx
    half = 0.2
    return pd.DataFrame([{
        "lShoulder_x": lx * half, "lShoulder_y": ly * half, "lShoulder_z": 1.4,
        "rShoulder_x": -lx * half, "rShoulder_y": -ly * half, "rShoulder_z": 1.4,
        "lHip_x": lx * 0.1, "lHip_y": ly * 0.1, "lHip_z": 0.9,
        "rHip_x": -lx * 0.1, "rHip_y": -ly * 0.1, "rHip_z": 0.9,
        "neck_x": 0.0, "neck_y": 0.0, "neck_z": 1.5,
        "midHip_x": 0.0, "midHip_y": 0.0, "midHip_z": 0.9,
        "lAnkle_x": 0.1, "lAnkle_y": 0.0, "lAnkle_z": 0.1,
        "rAnkle_x": -0.1, "rAnkle_y": 0.0, "rAnkle_z": 0.1,
        "x": 0.0, "y": 0.0, "ball_x": 10.0, "ball_y": 0.0,
    }])


@pytest.mark.parametrize("deg", [0, 45, 90, 135, 180, -90])
def test_orientation_recovers_the_facing_angle_it_was_built_from(deg):
    from src.features.pose import orientation

    out = orientation(_pose_row(deg))
    got = np.degrees(float(out["facing_rad"].iloc[0]))
    diff = (got - deg + 180) % 360 - 180
    assert abs(diff) < 1e-6, f"expected {deg}, got {got}"


def test_orientation_sign_convention_is_not_inverted():
    """A player facing +x must read 0 degrees to a goal at +x, not 180."""
    from src.features.pose import orientation

    out = orientation(_pose_row(0.0))
    assert float(out["facing_x"].iloc[0]) == pytest.approx(1.0)
    assert float(out["ball_bearing_deg"].iloc[0]) == pytest.approx(0.0, abs=1e-6)
    back = orientation(_pose_row(180.0))
    assert float(back["ball_bearing_deg"].iloc[0]) == pytest.approx(180.0, abs=1e-6)


def test_trunk_and_stance_geometry():
    from src.features.pose import orientation

    out = orientation(_pose_row(0.0)).iloc[0]
    assert out["trunk_length_m"] == pytest.approx(0.6)
    assert out["trunk_lean_deg"] == pytest.approx(0.0, abs=1e-6)
    assert out["shoulder_width_m"] == pytest.approx(0.4)
    assert out["hip_width_m"] == pytest.approx(0.2)
    assert out["stance_width_m"] == pytest.approx(0.2)
    assert out["shoulder_hip_twist_deg"] == pytest.approx(0.0, abs=1e-6)


@needs_data
def test_real_pose_orientation_agrees_with_movement_at_speed():
    """The empirical check that the construction is right way round.

    Runs against whatever pose source is present. If the shoulder rotation were
    inverted this median would sit near 180 degrees instead of near 7.
    """
    from src.config import DERIVED_DIR
    from src.preprocessing.build_tracking import load_tracking

    o = pd.read_parquet(DERIVED_DIR / "pose_orientation.parquet")
    o = o[o["orientation_valid"]]
    mid = int(o["match_id"].iloc[0])
    trk = load_tracking(mid, columns=["frame", "player_id", "heading", "speed"])
    o = o.assign(frame=o["tracking_frame"].round().astype(int))
    m = o.merge(trk, on=["frame", "player_id"], how="inner")
    fast = m[m["speed"] > 4.0]
    assert len(fast) > 50
    d = np.degrees(np.abs(np.arctan2(np.sin(fast["facing_rad"] - fast["heading"]),
                                     np.cos(fast["facing_rad"] - fast["heading"]))))
    assert np.median(d) < 20.0
    assert (d < 45).mean() > 0.9
