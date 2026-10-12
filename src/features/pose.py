"""Body-pose features: skeletons, orientation and limb geometry.

What is claimed here
--------------------
Body orientation from a shoulder line and a hip line is a **geometric
construction**, not a biomechanical measurement. It says which way the torso
was square to at the instant the joints were resolved. It says nothing about
head or gaze direction, and it inherits every error in the joint estimates --
each of which carries its own ``p90_mae_cm``, typically 4-15 cm here.

At a 40 cm shoulder half-width, a 6 cm error on each shoulder is up to ~12
degrees of orientation error. That is fine for separating "facing forward" from
"facing his own goal"; it is not fine for a claim about 10-degree differences,
and none is made.

SkillCorner also states that z is accurate relative to the player's centroid
and not in the pitch frame. Trunk lean is therefore reported as an angle
*within the pose*, never as an absolute posture relative to the ground plane.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import DERIVED_DIR, POSE_FPS
from src.data.pose_loading import JOINTS, iter_pose, pose_to_tracking_frame

log = logging.getLogger(__name__)

JOINTS_PATH = DERIVED_DIR / "pose_joints.parquet"
ORIENTATION_PATH = DERIVED_DIR / "pose_orientation.parquet"

#: Joints needed for the orientation construction. A player-frame missing any
#: of them produces no orientation row rather than a guess.
ORIENT_REQUIRED = ("lShoulder", "rShoulder", "lHip", "rHip", "neck", "midHip")


def stream_to_frames(match_id: int, frame_range=None, stride: int = 1):
    """Yield (meta, joints_dict) per player-frame from the pose stream."""
    for i, rec in enumerate(iter_pose(match_id, frame_range)):
        if stride > 1 and i % stride:
            continue
        base = {
            "match_id": match_id,
            "pose_frame": rec["frame"],
            "tracking_frame": pose_to_tracking_frame(rec["frame"]),
            "period": rec.get("period"),
            "timestamp": rec.get("timestamp"),
            "ball_x": (rec.get("ball_data") or {}).get("x"),
            "ball_y": (rec.get("ball_data") or {}).get("y"),
            "ball_z": (rec.get("ball_data") or {}).get("z"),
            "possession_group": (rec.get("possession") or {}).get("group"),
            "possession_player_id": (rec.get("possession") or {}).get("player_id"),
        }
        for p in rec.get("player_data") or []:
            meta = dict(base)
            meta.update(
                player_id=p.get("player_id"), x=p.get("x"), y=p.get("y"),
                is_detected=p.get("is_detected"),
            )
            yield meta, p.get("joints")


def joints_table(match_id: int, frame_range=None, stride: int = 1) -> pd.DataFrame:
    """Long table: one row per joint per player per pose frame."""
    rows = []
    for meta, joints in stream_to_frames(match_id, frame_range, stride):
        if not joints:
            continue
        for name, j in joints.items():
            xyz = j.get("xyz") or [np.nan] * 3
            rows.append(
                {**meta, "joint": name, "jx": xyz[0], "jy": xyz[1], "jz": xyz[2],
                 "p90_mae_cm": j.get("p90_mae_cm")}
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["joint"] = pd.Categorical(df["joint"], categories=list(JOINTS))
    return df


def wide_joints(joints: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long joint table to one row per player-frame."""
    if joints.empty:
        return joints
    # NOTE: the index must contain no nullable column. ``pivot_table`` drops
    # index rows containing NaN, and ``possession_player_id`` is null on most
    # frames -- including it here silently discarded 77% of player-frames.
    idx = ["match_id", "pose_frame", "tracking_frame", "period", "player_id",
           "x", "y", "is_detected", "ball_x", "ball_y"]
    assert joints[idx].notna().all().all(), "pivot index columns must be complete"
    wide = joints.pivot_table(index=idx, columns="joint",
                              values=["jx", "jy", "jz"], observed=True)
    wide.columns = [f"{b}_{a[1:]}" for a, b in wide.columns]
    err = joints.groupby(idx, observed=True)["p90_mae_cm"].agg(["mean", "max"])
    err.columns = ["joint_err_mean_cm", "joint_err_max_cm"]
    out = wide.join(err).reset_index()
    extra = (
        joints[idx + ["possession_group", "possession_player_id", "timestamp"]]
        .drop_duplicates(subset=idx)
    )
    return out.merge(extra, on=idx, how="left")


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #

def _unit(vx, vy):
    n = np.hypot(vx, vy)
    n = np.where(n < 1e-6, np.nan, n)
    return vx / n, vy / n


def orientation(wide: pd.DataFrame) -> pd.DataFrame:
    """Torso and hip facing directions, plus trunk geometry.

    The facing vector is the shoulder line rotated 90 degrees anticlockwise:
    for a player facing +x, the left shoulder sits at +y and the right at -y,
    so ``rShoulder - lShoulder`` points at -y and ``R(90)`` maps it back to +x.
    Hips give a second, independent estimate; the two are averaged when they
    agree to within 90 degrees and the shoulder estimate is used otherwise
    (a twisted torso is real, and the shoulders are what a receiving player
    turns first).
    """
    if wide.empty:
        return wide
    d = wide.copy()
    sx = d["lShoulder_x"] - d["rShoulder_x"]
    sy = d["lShoulder_y"] - d["rShoulder_y"]
    # facing = R(+90) applied to (rShoulder - lShoulder) = (-(-sy), -sx) = (sy, -sx)
    fx_s, fy_s = _unit(sy, -sx)

    hx = d["lHip_x"] - d["rHip_x"]
    hy = d["lHip_y"] - d["rHip_y"]
    fx_h, fy_h = _unit(hy, -hx)

    agree = (fx_s * fx_h + fy_s * fy_h) > 0
    fx = np.where(agree, (fx_s + fx_h) / 2, fx_s)
    fy = np.where(agree, (fy_s + fy_h) / 2, fy_s)
    fx, fy = _unit(fx, fy)

    d["facing_x"], d["facing_y"] = fx, fy
    d["facing_rad"] = np.arctan2(fy, fx)
    d["shoulder_width_m"] = np.hypot(sx, sy)
    d["hip_width_m"] = np.hypot(hx, hy)
    d["shoulder_hip_twist_deg"] = np.degrees(
        np.abs(np.arctan2(fy_s, fx_s) - np.arctan2(fy_h, fx_h))
    )
    d["shoulder_hip_twist_deg"] = np.minimum(
        d["shoulder_hip_twist_deg"], 360 - d["shoulder_hip_twist_deg"]
    )

    # Trunk: neck relative to mid-hip. Reported as an angle within the pose.
    tx = d["neck_x"] - d["midHip_x"]
    ty = d["neck_y"] - d["midHip_y"]
    tz = d["neck_z"] - d["midHip_z"]
    d["trunk_length_m"] = np.sqrt(tx**2 + ty**2 + tz**2)
    d["trunk_lean_deg"] = np.degrees(np.arctan2(np.hypot(tx, ty), np.abs(tz)))
    d["trunk_lean_dir_rad"] = np.arctan2(ty, tx)

    # Stance
    d["stance_width_m"] = np.hypot(d["lAnkle_x"] - d["rAnkle_x"],
                                   d["lAnkle_y"] - d["rAnkle_y"])
    d["ankle_height_diff_m"] = (d["lAnkle_z"] - d["rAnkle_z"]).abs()

    # Where the ball is relative to the player, and how open the body is to it.
    bx = d["ball_x"] - d["x"]
    by = d["ball_y"] - d["y"]
    ux, uy = _unit(bx, by)
    d["ball_distance_m"] = np.hypot(bx, by)
    d["ball_bearing_deg"] = np.degrees(
        np.arccos(np.clip(fx * ux + fy * uy, -1, 1))
    )
    d["facing_ball"] = d["ball_bearing_deg"] < 60
    valid = d[list(f"{p}_{a}" for p in ORIENT_REQUIRED for a in ("x", "y"))].notna().all(axis=1)
    d["orientation_valid"] = valid & d["facing_rad"].notna()
    return d


def add_attacking_orientation(d: pd.DataFrame, attack_sign_by_period: dict[int, int]) -> pd.DataFrame:
    """Angle between facing direction and the goal the player is attacking."""
    sign = d["period"].map(attack_sign_by_period).astype("float")
    goal_x = sign  # unit vector towards the attacked goal, in pitch x
    cos = d["facing_x"] * goal_x
    d["facing_goal_deg"] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    d["body_orientation_class"] = pd.cut(
        d["facing_goal_deg"], [-0.1, 60, 120, 180.1],
        labels=["forward", "sideways", "backward"],
    )
    d["openness"] = np.cos(np.radians(d["facing_goal_deg"]))
    return d


def build(match_id: int | None = None, stride: int = 1) -> dict[str, int]:
    """Build the pose tables from whatever pose source is available."""
    from src.data.loaders import load_match_meta
    from src.data.pose_loading import SAMPLE_MATCH_ID, available_sources

    src = available_sources()
    mid = match_id or (src["full_matches"][0] if src["full_matches"] else SAMPLE_MATCH_ID)
    joints = joints_table(mid, stride=stride)
    if joints.empty:
        log.warning("no pose rows for %s", mid)
        return {"joints": 0, "orientation": 0}
    wide = wide_joints(joints)
    orient = orientation(wide)
    meta = load_match_meta(mid)
    squad = meta.players_frame()[["player_id", "team_id", "side", "display_name", "position"]]
    orient = orient.merge(squad, on="player_id", how="left")
    signs = {
        p: (1 if meta.home_attacks_positive_x.get(p, True) else -1)
        for p in meta.home_attacks_positive_x
    }
    # attack sign is per team; store home-frame sign and flip for away rows.
    orient["home_attack_sign"] = orient["period"].map(signs)
    orient["attack_sign"] = np.where(
        orient["side"] == "home", orient["home_attack_sign"], -orient["home_attack_sign"]
    )
    orient["facing_goal_deg"] = np.degrees(
        np.arccos(np.clip(orient["facing_x"] * orient["attack_sign"], -1, 1))
    )
    orient["body_orientation_class"] = pd.cut(
        orient["facing_goal_deg"], [-0.1, 60, 120, 180.1],
        labels=["forward", "sideways", "backward"],
    )
    orient["openness"] = np.cos(np.radians(orient["facing_goal_deg"]))
    orient["source"] = "full_match" if mid in src["full_matches"] else "sample_phase"

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    joints.to_parquet(JOINTS_PATH, index=False, compression="zstd")
    orient.to_parquet(ORIENTATION_PATH, index=False, compression="zstd")
    return {"joints": len(joints), "orientation": len(orient), "match_id": mid}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(build())
