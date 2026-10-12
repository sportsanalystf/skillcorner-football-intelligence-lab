"""Pitch geometry and coordinate transforms.

SkillCorner's native frame is metres, origin at the centre spot, +x along the
long axis, +y along the short axis. Match pitches in this dataset are 104-106 m
long and 68 m wide, so any cross-match comparison has to be rescaled first.

Three coordinate spaces are used throughout the project:

``raw``
    As delivered: metres on that match's own pitch, in broadcast orientation
    (which goal is at -x depends on the period).
``standard``
    Metres rescaled onto a 105 x 68 reference pitch. Comparable across matches,
    still in broadcast orientation.
``attacking``
    Standard metres, flipped so the team of interest always attacks towards +x.
    This is the space every tactical metric is computed in.

A fourth convenience space, ``normalised``, maps the standard pitch onto
0-100 x 0-100 for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import (
    CENTRE_CIRCLE_R,
    GOAL_WIDTH,
    PENALTY_AREA_LENGTH,
    PENALTY_AREA_WIDTH,
    PENALTY_SPOT,
    SIX_YARD_LENGTH,
    SIX_YARD_WIDTH,
    STD_PITCH_LENGTH,
    STD_PITCH_WIDTH,
)


@dataclass(frozen=True)
class Pitch:
    """A pitch of a given size, centred on the origin."""

    length: float = STD_PITCH_LENGTH
    width: float = STD_PITCH_WIDTH

    @property
    def half_length(self) -> float:
        return self.length / 2.0

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (-self.half_length, self.half_length, -self.half_width, self.half_width)

    @property
    def area(self) -> float:
        return self.length * self.width


STANDARD_PITCH = Pitch()


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

def to_standard(x, y, pitch: Pitch):
    """Rescale native match metres onto the 105 x 68 reference pitch."""
    sx = STD_PITCH_LENGTH / pitch.length
    sy = STD_PITCH_WIDTH / pitch.width
    return np.asarray(x, dtype="float64") * sx, np.asarray(y, dtype="float64") * sy


def to_attacking(x, y, sign):
    """Flip so the team attacks towards +x.

    ``sign`` is +1 when the team already attacks towards +x and -1 otherwise;
    it may be a scalar or an array broadcastable against ``x``.
    """
    sign = np.asarray(sign, dtype="float64")
    return np.asarray(x, dtype="float64") * sign, np.asarray(y, dtype="float64") * sign


def to_normalised(x, y, pitch: Pitch = STANDARD_PITCH):
    """Map metres to 0-100 in both axes (x: own goal -> opposition goal)."""
    nx = (np.asarray(x, dtype="float64") + pitch.half_length) / pitch.length * 100.0
    ny = (np.asarray(y, dtype="float64") + pitch.half_width) / pitch.width * 100.0
    return nx, ny


def from_normalised(nx, ny, pitch: Pitch = STANDARD_PITCH):
    x = np.asarray(nx, dtype="float64") / 100.0 * pitch.length - pitch.half_length
    y = np.asarray(ny, dtype="float64") / 100.0 * pitch.width - pitch.half_width
    return x, y


# --------------------------------------------------------------------------- #
# Zones
# --------------------------------------------------------------------------- #

def thirds(x, pitch: Pitch = STANDARD_PITCH) -> np.ndarray:
    """Defensive / middle / attacking third, for coordinates in attacking space."""
    x = np.asarray(x, dtype="float64")
    edges = [-pitch.half_length, -pitch.length / 6, pitch.length / 6, pitch.half_length]
    idx = np.digitize(x, edges[1:-1])
    return np.array(["defensive_third", "middle_third", "attacking_third"])[idx]


def channels(y, pitch: Pitch = STANDARD_PITCH) -> np.ndarray:
    """Five vertical channels, SkillCorner-style, for attacking-space y.

    Boundaries follow the penalty-area edges: the central channel is the width
    of the penalty area, the half-spaces run from there to the six-yard-box
    projection lines, and the wide channels are what remains.
    """
    y = np.asarray(y, dtype="float64")
    inner = PENALTY_AREA_WIDTH / 2 * 0.45   # ~9.07 m -> centre channel
    outer = PENALTY_AREA_WIDTH / 2          # ~20.16 m -> half-space edge
    idx = np.digitize(y, [-outer, -inner, inner, outer])
    return np.array(["wide_right", "half_space_right", "centre", "half_space_left", "wide_left"])[idx]


def in_penalty_area(x, y, pitch: Pitch = STANDARD_PITCH, attacking: bool = True) -> np.ndarray:
    """Whether a point is in the penalty area being attacked (or defended)."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    edge = pitch.half_length - PENALTY_AREA_LENGTH
    if attacking:
        return (x >= edge) & (np.abs(y) <= PENALTY_AREA_WIDTH / 2)
    return (x <= -edge) & (np.abs(y) <= PENALTY_AREA_WIDTH / 2)


def distance_to_goal(x, y, pitch: Pitch = STANDARD_PITCH) -> np.ndarray:
    """Straight-line distance to the centre of the goal being attacked."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    return np.hypot(pitch.half_length - x, y)


def angle_to_goal(x, y, pitch: Pitch = STANDARD_PITCH) -> np.ndarray:
    """Angle in radians subtended by the goalmouth from a point (shot angle)."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    gx = pitch.half_length
    a = np.hypot(gx - x, y - GOAL_WIDTH / 2)
    b = np.hypot(gx - x, y + GOAL_WIDTH / 2)
    cos = (a**2 + b**2 - GOAL_WIDTH**2) / (2 * a * b)
    return np.arccos(np.clip(cos, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# Drawing primitives
# --------------------------------------------------------------------------- #

def pitch_line_segments(pitch: Pitch = STANDARD_PITCH) -> list[tuple[list[float], list[float]]]:
    """Line segments that draw a pitch, as ``(xs, ys)`` pairs.

    Renderer-agnostic on purpose: Plotly and matplotlib both consume this.
    """
    hl, hw = pitch.half_length, pitch.half_width
    segs: list[tuple[list[float], list[float]]] = []

    segs.append(([-hl, hl, hl, -hl, -hl], [-hw, -hw, hw, hw, -hw]))     # touchlines
    segs.append(([0, 0], [-hw, hw]))                                     # halfway

    for s in (-1, 1):
        pa_x = s * (hl - PENALTY_AREA_LENGTH)
        segs.append(
            (
                [s * hl, pa_x, pa_x, s * hl],
                [-PENALTY_AREA_WIDTH / 2, -PENALTY_AREA_WIDTH / 2,
                 PENALTY_AREA_WIDTH / 2, PENALTY_AREA_WIDTH / 2],
            )
        )
        sy_x = s * (hl - SIX_YARD_LENGTH)
        segs.append(
            (
                [s * hl, sy_x, sy_x, s * hl],
                [-SIX_YARD_WIDTH / 2, -SIX_YARD_WIDTH / 2,
                 SIX_YARD_WIDTH / 2, SIX_YARD_WIDTH / 2],
            )
        )
        segs.append(
            (
                [s * hl, s * (hl + 2.0), s * (hl + 2.0), s * hl],
                [-GOAL_WIDTH / 2, -GOAL_WIDTH / 2, GOAL_WIDTH / 2, GOAL_WIDTH / 2],
            )
        )

    th = np.linspace(0, 2 * np.pi, 120)
    segs.append((list(CENTRE_CIRCLE_R * np.cos(th)), list(CENTRE_CIRCLE_R * np.sin(th))))

    # Penalty arcs, clipped to the part outside the box.
    for s in (-1, 1):
        spot = s * (hl - PENALTY_SPOT)
        edge = s * (hl - PENALTY_AREA_LENGTH)
        a = np.linspace(0, 2 * np.pi, 400)
        cx = spot + CENTRE_CIRCLE_R * np.cos(a)
        cy = CENTRE_CIRCLE_R * np.sin(a)
        keep = (cx < edge) if s > 0 else (cx > edge)
        if keep.any():
            order = np.argsort(cy[keep])
            segs.append((list(cx[keep][order]), list(cy[keep][order])))

    return segs


def penalty_spots(pitch: Pitch = STANDARD_PITCH) -> list[tuple[float, float]]:
    hl = pitch.half_length
    return [(-(hl - PENALTY_SPOT), 0.0), (hl - PENALTY_SPOT, 0.0), (0.0, 0.0)]


def grid(nx: int, ny: int, pitch: Pitch = STANDARD_PITCH):
    """Cell-centre coordinate grid over the pitch."""
    xs = np.linspace(-pitch.half_length, pitch.half_length, nx + 1)
    ys = np.linspace(-pitch.half_width, pitch.half_width, ny + 1)
    cx = (xs[:-1] + xs[1:]) / 2
    cy = (ys[:-1] + ys[1:]) / 2
    return cx, cy, np.meshgrid(cx, cy)
