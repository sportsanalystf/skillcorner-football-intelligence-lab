"""Velocity, acceleration and running-load metrics from broadcast tracking.

Why not finite differences
--------------------------
Broadcast tracking positions carry sub-metre noise from the homography and the
detector. A first difference at 10 Hz turns a 0.3 m wobble into 3 m/s, and a
second difference turns it into 60 m/s^2. Differentiating the raw signal is the
single easiest way to fabricate sprints that never happened.

What this module does instead
-----------------------------
1. **Densify.** Each player's track is placed on the match's full frame grid,
   with gaps as NaN.
2. **Bridge short gaps.** Gaps up to ``max_gap_frames`` (default 1.0 s) are
   filled by linear interpolation; anything longer splits the track into
   separate segments, because interpolating across a 20 s absence invents
   motion.
3. **Smooth and differentiate in one step.** A Savitzky-Golay filter of order 2
   over a 1.3 s window is applied to x and y. Velocity is taken as the
   filter's analytic first derivative rather than differencing the smoothed
   series -- the two are close but the analytic form is better conditioned at
   segment edges.
4. **Gate.** Speeds above 12 m/s and accelerations above 8 m/s^2 are masked,
   not clipped. Masking keeps an artefact out of a distance total; clipping
   would quietly add a plausible-looking 12 m/s to it.

Everything is computed on the *raw* match pitch in metres. Rescaling to the
standard pitch happens downstream and does not change speeds materially
(the largest rescale in this dataset is 105/106 = 0.9%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.config import (
    DT,
    HI_ACCEL_THRESHOLD,
    HSR_THRESHOLD_MS,
    MAX_PLAUSIBLE_ACCEL_MS2,
    MAX_PLAUSIBLE_SPEED_MS,
    MAX_PLAUSIBLE_STEP_M,
    MED_ACCEL_THRESHOLD,
    MIN_EFFORT_DURATION_S,
    SG_POLYORDER,
    SG_WINDOW_FRAMES,
    SPEED_ZONES,
    SPRINT_THRESHOLD_MS,
    TRACKING_FPS,
)

__all__ = [
    "segment_ids",
    "smooth_and_differentiate",
    "add_kinematics",
    "speed_zone",
    "detect_efforts",
    "summarise_player_physical",
]


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #

def segment_ids(present: np.ndarray, max_gap: int) -> np.ndarray:
    """Label contiguous runs of observation, bridging gaps of ``<= max_gap``.

    Args:
        present: Boolean array, True where the player has a position.
        max_gap: Longest run of missing frames that still counts as the same
            segment.

    Returns:
        Integer array; -1 where the frame belongs to no segment (a long gap),
        otherwise the segment index.
    """
    n = present.size
    out = np.full(n, -1, dtype="int64")
    if not present.any():
        return out
    idx = np.flatnonzero(present)
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    for k, (s, e) in enumerate(zip(starts, ends)):
        out[s : e + 1] = k
    return out


def _interp_within(values: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Linearly fill NaNs inside each segment; leave outside-segment NaN."""
    out = values.astype("float64").copy()
    for k in np.unique(seg[seg >= 0]):
        m = seg == k
        idx = np.flatnonzero(m)
        v = out[idx]
        good = ~np.isnan(v)
        if good.sum() < 2:
            continue
        v[~good] = np.interp(idx[~good], idx[good], v[good])
        out[idx] = v
    return out


# --------------------------------------------------------------------------- #
# Core smoothing + differentiation
# --------------------------------------------------------------------------- #

def smooth_and_differentiate(
    x: np.ndarray,
    y: np.ndarray,
    present: np.ndarray,
    *,
    window: int = SG_WINDOW_FRAMES,
    polyorder: int = SG_POLYORDER,
    max_gap: int = TRACKING_FPS,
    dt: float = DT,
    max_speed: float = MAX_PLAUSIBLE_SPEED_MS,
    max_accel: float = MAX_PLAUSIBLE_ACCEL_MS2,
) -> dict[str, np.ndarray]:
    """Smooth one player's track and return position, velocity, acceleration.

    Returns a dict of equal-length arrays: ``xs``, ``ys`` (smoothed position),
    ``vx``, ``vy``, ``speed``, ``accel`` (tangential, signed), ``accel_mag``,
    ``heading`` (radians, atan2(vy, vx)) and ``segment``.

    ``max_speed`` / ``max_accel`` set the plausibility gate. The defaults are
    the human ones; a ball needs much wider bounds and passes its own.
    """
    n = x.size
    seg = segment_ids(present, max_gap)
    xs = _interp_within(np.where(present, x, np.nan), seg)
    ys = _interp_within(np.where(present, y, np.nan), seg)

    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    ax = np.full(n, np.nan)
    ay = np.full(n, np.nan)
    sx = np.full(n, np.nan)
    sy = np.full(n, np.nan)

    for k in np.unique(seg[seg >= 0]):
        m = seg == k
        idx = np.flatnonzero(m)
        L = idx.size
        if L < 3:
            continue
        w = min(window, L if L % 2 == 1 else L - 1)
        po = min(polyorder, w - 1)
        px, py = xs[idx], ys[idx]
        sx[idx] = savgol_filter(px, w, po, deriv=0, mode="interp")
        sy[idx] = savgol_filter(py, w, po, deriv=0, mode="interp")
        vx[idx] = savgol_filter(px, w, po, deriv=1, delta=dt, mode="interp")
        vy[idx] = savgol_filter(py, w, po, deriv=1, delta=dt, mode="interp")
        if po >= 2:
            ax[idx] = savgol_filter(px, w, po, deriv=2, delta=dt, mode="interp")
            ay[idx] = savgol_filter(py, w, po, deriv=2, delta=dt, mode="interp")

    speed = np.hypot(vx, vy)
    # Tangential acceleration: the component of dv/dt along the direction of
    # travel. This is what "accelerating" and "decelerating" mean in a physical
    # report; |dv/dt| would call a hard turn at constant pace an acceleration.
    with np.errstate(invalid="ignore", divide="ignore"):
        accel = np.where(speed > 1e-6, (vx * ax + vy * ay) / speed, np.nan)
    accel_mag = np.hypot(ax, ay)
    heading = np.arctan2(vy, vx)

    bad = (speed > max_speed) | (np.abs(accel) > max_accel)
    speed = np.where(bad, np.nan, speed)
    accel = np.where(bad, np.nan, accel)
    vx = np.where(bad, np.nan, vx)
    vy = np.where(bad, np.nan, vy)

    return {
        "xs": sx, "ys": sy, "vx": vx, "vy": vy,
        "speed": speed, "accel": accel, "accel_mag": accel_mag,
        "heading": heading, "segment": seg,
    }


def add_kinematics(
    df: pd.DataFrame,
    *,
    id_col: str = "player_id",
    frame_col: str = "frame",
    period_col: str = "period",
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """Attach kinematics to a long tracking frame.

    The input must be one row per (entity, frame). Periods are treated as hard
    segment boundaries so the half-time jump never produces a velocity.
    """
    df = df.sort_values([id_col, frame_col], ignore_index=True)
    pieces: list[pd.DataFrame] = []

    for (ent, per), grp in df.groupby([id_col, period_col], sort=False, dropna=False):
        frames = grp[frame_col].to_numpy()
        f0, f1 = int(frames.min()), int(frames.max())
        full = np.arange(f0, f1 + 1)
        pos = pd.DataFrame({frame_col: full}).merge(grp, on=frame_col, how="left")
        present = pos[x_col].notna().to_numpy()
        k = smooth_and_differentiate(
            pos[x_col].to_numpy(dtype="float64"),
            pos[y_col].to_numpy(dtype="float64"),
            present,
        )
        for key, arr in k.items():
            pos[key] = arr
        pos[id_col] = ent
        pos[period_col] = per
        pieces.append(pos.loc[present].copy())

    out = pd.concat(pieces, ignore_index=True)

    # Step distance from the *smoothed* track, gated on a plausible step.
    out = out.sort_values([id_col, frame_col], ignore_index=True)
    dx = out.groupby(id_col, sort=False)["xs"].diff()
    dy = out.groupby(id_col, sort=False)["ys"].diff()
    dframe = out.groupby(id_col, sort=False)[frame_col].diff()
    step = np.hypot(dx, dy)
    step = step.where((dframe == 1) & (step <= MAX_PLAUSIBLE_STEP_M), 0.0)
    out["step_m"] = step.fillna(0.0)
    out["speed_zone"] = speed_zone(out["speed"].to_numpy())
    return out


def speed_zone(speed: np.ndarray) -> np.ndarray:
    """Bucket speeds into the configured intensity zones."""
    speed = np.asarray(speed, dtype="float64")
    labels = np.array([z[0] for z in SPEED_ZONES] + ["unknown"])
    edges = [z[1] for z in SPEED_ZONES][1:]
    idx = np.digitize(np.nan_to_num(speed, nan=-1.0), edges)
    idx = np.where(np.isnan(speed), len(SPEED_ZONES), idx)
    return labels[idx]


# --------------------------------------------------------------------------- #
# Efforts
# --------------------------------------------------------------------------- #

def detect_efforts(
    speed: np.ndarray,
    frames: np.ndarray,
    threshold: float,
    *,
    min_duration_s: float = MIN_EFFORT_DURATION_S,
    dt: float = DT,
) -> list[dict]:
    """Find runs where speed stays above ``threshold`` for long enough.

    A sprint is not "one frame above 7 m/s". Requiring the threshold to hold
    for ``min_duration_s`` removes almost all of the single-frame artefacts
    that survive smoothing.
    """
    above = np.nan_to_num(speed, nan=0.0) >= threshold
    if not above.any():
        return []
    d = np.diff(above.astype("int8"), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    min_len = max(1, int(round(min_duration_s / dt)))
    out = []
    for s, e in zip(starts, ends):
        if e - s < min_len:
            continue
        seg = speed[s:e]
        out.append(
            {
                "start_frame": int(frames[s]),
                "end_frame": int(frames[e - 1]),
                "duration_s": (e - s) * dt,
                "peak_speed": float(np.nanmax(seg)),
                "mean_speed": float(np.nanmean(seg)),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Per-player summaries
# --------------------------------------------------------------------------- #

def summarise_player_physical(track: pd.DataFrame) -> pd.Series:
    """Physical summary for one player's tracked frames in one match.

    ``track`` must already carry the columns added by :func:`add_kinematics`.
    """
    speed = track["speed"].to_numpy()
    accel = track["accel"].to_numpy()
    frames = track["frame"].to_numpy()
    step = track["step_m"].to_numpy()

    tracked_s = len(track) * DT
    out: dict[str, float] = {
        "frames_tracked": len(track),
        "tracked_minutes": tracked_s / 60.0,
        "detected_frac": float(track["is_detected"].mean()) if "is_detected" in track else np.nan,
        "total_distance_m": float(np.nansum(step)),
        "top_speed_ms": float(np.nanmax(speed)) if np.isfinite(speed).any() else np.nan,
        "psv99_ms": float(np.nanpercentile(speed, 99)) if np.isfinite(speed).any() else np.nan,
        "mean_speed_ms": float(np.nanmean(speed)),
    }
    for name, lo, hi in SPEED_ZONES:
        m = (speed >= lo) & (speed < hi)
        out[f"distance_{name}_m"] = float(np.nansum(np.where(m, step, 0.0)))
        out[f"time_{name}_s"] = float(m.sum() * DT)

    out["hsr_distance_m"] = out["distance_hsr_m"] + out["distance_sprint_m"]
    out["hi_distance_m"] = out["hsr_distance_m"]
    out["sprint_distance_m"] = out["distance_sprint_m"]

    hsr = detect_efforts(speed, frames, HSR_THRESHOLD_MS)
    spr = detect_efforts(speed, frames, SPRINT_THRESHOLD_MS)
    out["hsr_count"] = len(hsr)
    out["sprint_count"] = len(spr)
    out["max_effort_duration_s"] = max((e["duration_s"] for e in spr), default=0.0)

    acc = np.nan_to_num(accel, nan=0.0)
    out["high_accel_count"] = _count_bouts(acc >= HI_ACCEL_THRESHOLD)
    out["med_accel_count"] = _count_bouts((acc >= MED_ACCEL_THRESHOLD) & (acc < HI_ACCEL_THRESHOLD))
    out["high_decel_count"] = _count_bouts(acc <= -HI_ACCEL_THRESHOLD)
    out["med_decel_count"] = _count_bouts((acc <= -MED_ACCEL_THRESHOLD) & (acc > -HI_ACCEL_THRESHOLD))
    out["max_accel_ms2"] = float(np.nanmax(accel)) if np.isfinite(accel).any() else np.nan
    out["max_decel_ms2"] = float(np.nanmin(accel)) if np.isfinite(accel).any() else np.nan

    out["m_per_min"] = out["total_distance_m"] / max(out["tracked_minutes"], 1e-9)
    return pd.Series(out)


def _count_bouts(mask: np.ndarray, min_frames: int = 3) -> int:
    """Count runs of ``True`` at least ``min_frames`` long (0.3 s default)."""
    if not mask.any():
        return 0
    d = np.diff(mask.astype("int8"), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int(np.sum((ends - starts) >= min_frames))
