"""Central configuration for Football Intelligence Lab.

Every path, constant and tunable used by the pipeline lives here so that the
provenance of a number is a single grep away.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(os.environ.get("FIL_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DERIVED_DIR = DATA_DIR / "derived"

RAW_MATCHES_DIR = RAW_DIR / "matches"
RAW_AGGREGATES_DIR = RAW_DIR / "aggregates"
RAW_BODYPOSE_DIR = RAW_DIR / "bodypose"
MATCHES_INDEX_JSON = RAW_DIR / "matches.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

WAREHOUSE_PATH = PROCESSED_DIR / "fil.duckdb"

for _d in (PROCESSED_DIR, DERIVED_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Upstream data source
# --------------------------------------------------------------------------- #

OPENDATA_REPO = "SkillCorner/opendata"
OPENDATA_REF = "master"
#: GitHub serves Git-LFS payloads (the tracking JSONL files) from media.*, not raw.*
LFS_MEDIA_BASE = f"https://media.githubusercontent.com/media/{OPENDATA_REPO}/{OPENDATA_REF}"
RAW_BASE = f"https://raw.githubusercontent.com/{OPENDATA_REPO}/{OPENDATA_REF}"
BODYPOSE_HF_BASE = (
    "https://huggingface.co/datasets/SkillCorner/opendata-bodypose/resolve/main"
)

#: Matches for which SkillCorner publishes 3D body pose.
BODYPOSE_MATCH_IDS: tuple[int, ...] = (1925299, 1996435)

# --------------------------------------------------------------------------- #
# Frame rates and clocks
# --------------------------------------------------------------------------- #

TRACKING_FPS = 10
POSE_FPS = 25
DT = 1.0 / TRACKING_FPS  # seconds between tracking frames

# --------------------------------------------------------------------------- #
# Pitch
# --------------------------------------------------------------------------- #

#: Reference pitch used whenever matches are compared with one another.
STD_PITCH_LENGTH = 105.0
STD_PITCH_WIDTH = 68.0

PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.32
SIX_YARD_LENGTH = 5.5
SIX_YARD_WIDTH = 18.32
CENTRE_CIRCLE_R = 9.15
GOAL_WIDTH = 7.32
PENALTY_SPOT = 11.0

# --------------------------------------------------------------------------- #
# Kinematics
# --------------------------------------------------------------------------- #

#: Savitzky-Golay smoothing applied to raw XY before differentiating.
#: 1.3 s window at 10 Hz. Long enough to kill single-frame jitter from the
#: broadcast homography, short enough to preserve a real acceleration burst.
SG_WINDOW_FRAMES = 13
SG_POLYORDER = 2

#: Physiological plausibility gates. Anything above is treated as a tracking
#: artefact and masked, not clipped, so it cannot silently inflate a total.
MAX_PLAUSIBLE_SPEED_MS = 12.0        # ~43.2 km/h; the men's 100 m record pace is 12.4
MAX_PLAUSIBLE_ACCEL_MS2 = 8.0
MAX_PLAUSIBLE_STEP_M = 1.6           # 16 m/s over one 0.1 s frame

#: Speed zones, m/s. Boundaries follow SkillCorner's published physical
#: glossary bands so that per-match numbers are comparable with the
#: season aggregates: HSR >= 5.5 m/s, sprint >= 7.0 m/s.
SPEED_ZONES: tuple[tuple[str, float, float], ...] = (
    ("walking", 0.0, 1.5),
    ("jogging", 1.5, 3.5),
    ("running", 3.5, 5.5),
    ("hsr", 5.5, 7.0),
    ("sprint", 7.0, float("inf")),
)
HSR_THRESHOLD_MS = 5.5
SPRINT_THRESHOLD_MS = 7.0
HI_ACCEL_THRESHOLD = 3.0    # m/s^2, "high" acceleration
MED_ACCEL_THRESHOLD = 1.5   # m/s^2, "medium" acceleration
#: A qualifying effort must be sustained for at least this long.
MIN_EFFORT_DURATION_S = 0.7

# --------------------------------------------------------------------------- #
# Pitch control model
# --------------------------------------------------------------------------- #

PC_GRID_NX = 60
PC_GRID_NY = 40
PC_REACTION_TIME_S = 0.7
PC_MAX_SPEED_MS = 8.0
PC_TTI_SIGMA = 0.45        # logistic steepness on time-to-intercept difference

# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #

#: Chart palette. The categorical hues are the validated dark-mode steps of the
#: reference eight-hue theme, checked with the palette validator against this
#: app's surface (#141A22): all six slots pass the lightness band, chroma floor,
#: adjacent CVD separation (worst dE 8.4), normal-vision floor (19.3) and the
#: 3:1 contrast gate. Do not substitute a hue without re-running that check.
SERIES = (
    "#3987e5",  # 1 blue     -- home team
    "#d95926",  # 2 orange   -- away team
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#9085e9",  # 6 violet
)

TEAM_COLOURS = {"home": SERIES[0], "away": SERIES[1]}

PALETTE = {
    "bg": "#0E1319",
    "surface": "#141A22",
    "panel": "#1A212B",
    "panel_alt": "#212A36",
    "line": "#2B3644",
    "grid": "#222C38",
    "text": "#EAF0F7",
    "text_secondary": "#A9B6C6",
    "muted": "#7A8798",
    "accent": SERIES[0],
    "accent2": SERIES[1],
    "accent3": SERIES[3],
    "good": "#199e70",
    "warn": "#c98500",
    "bad": "#e66767",
    "pitch": "#111821",
    "pitch_line": "#33404F",
}

#: Diverging ramp for pitch control: away (orange) <-> neutral <-> home (blue).
#: The poles are the two teams' identity colours, so control never has to be
#: decoded twice, and the midpoint is a true neutral against the pitch.
SEQ_CONTROL = [
    [0.00, "#d95926"],
    [0.25, "#8a4430"],
    [0.50, "#2E3742"],
    [0.75, "#2c5f9c"],
    [1.00, "#3987e5"],
]

#: Single-hue sequential ramp (blue, light -> dark reversed for dark surface).
SEQ_MAGNITUDE = [
    [0.00, "#141A22"],
    [0.25, "#184f95"],
    [0.50, "#256abf"],
    [0.75, "#3987e5"],
    [1.00, "#9ec5f4"],
]
