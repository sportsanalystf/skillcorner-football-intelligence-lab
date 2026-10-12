"""Readers for the raw SkillCorner files.

Nothing here changes the meaning of the data. Parsing, typing and the one
piece of arithmetic every downstream table needs -- the match clock -- happen
here; anything opinionated happens in :mod:`src.preprocessing`.

Clock convention, established empirically in ``notebooks/01_data_exploration``
and asserted in ``tests/test_clock.py``:

    period 1:  match_clock_s = (frame - period_1_start_frame) / 10
    period 2:  match_clock_s = 2700 + (frame - period_2_start_frame) / 10

which reproduces the ``timestamp`` string in the tracking file and the
``time_start``/``minute_start`` columns in the event and phase files exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.config import (
    MATCHES_INDEX_JSON,
    RAW_AGGREGATES_DIR,
    RAW_MATCHES_DIR,
    TRACKING_FPS,
)

PERIOD_OFFSET_S = {1: 0.0, 2: 2700.0, 3: 5400.0, 4: 6300.0}


# --------------------------------------------------------------------------- #
# Match metadata
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PitchDims:
    length: float
    width: float


@dataclass
class MatchMeta:
    """Everything about a match that is not a time series."""

    match_id: int
    raw: dict[str, Any] = field(repr=False)

    # -- identity ---------------------------------------------------------- #
    @property
    def date(self) -> str:
        return str(self.raw.get("date_time", ""))[:10]

    @property
    def home_team_id(self) -> int:
        return int(self.raw["home_team"]["id"])

    @property
    def away_team_id(self) -> int:
        return int(self.raw["away_team"]["id"])

    @property
    def home_team_name(self) -> str:
        return self.raw["home_team"].get("short_name") or self.raw["home_team"]["name"]

    @property
    def away_team_name(self) -> str:
        return self.raw["away_team"].get("short_name") or self.raw["away_team"]["name"]

    @property
    def score(self) -> tuple[int, int]:
        return int(self.raw["home_team_score"]), int(self.raw["away_team_score"])

    @property
    def label(self) -> str:
        h, a = self.score
        return f"{self.home_team_name} {h}-{a} {self.away_team_name}"

    @property
    def pitch(self) -> PitchDims:
        return PitchDims(float(self.raw["pitch_length"]), float(self.raw["pitch_width"]))

    # -- clocks ------------------------------------------------------------ #
    @property
    def period_start_frames(self) -> dict[int, int]:
        return {int(p["period"]): int(p["start_frame"]) for p in self.raw["match_periods"]}

    @property
    def period_end_frames(self) -> dict[int, int]:
        return {int(p["period"]): int(p["end_frame"]) for p in self.raw["match_periods"]}

    def clock_seconds(self, frame, period):
        """Vectorised frame -> match clock in seconds."""
        import numpy as np

        frame = np.asarray(frame, dtype="float64")
        period = np.asarray(period, dtype="float64")
        out = np.full(frame.shape, np.nan)
        for p, start in self.period_start_frames.items():
            m = period == p
            out[m] = PERIOD_OFFSET_S.get(p, 0.0) + (frame[m] - start) / TRACKING_FPS
        return out

    # -- orientation ------------------------------------------------------- #
    @property
    def home_attacks_positive_x(self) -> dict[int, bool]:
        """Per period, True when the home team attacks towards +x.

        ``home_team_side`` is a list ordered by period; ``left_to_right`` means
        the home team's goal is at negative x and it attacks towards +x.
        """
        sides = self.raw.get("home_team_side") or []
        return {i + 1: (s == "left_to_right") for i, s in enumerate(sides)}

    def attack_sign(self, team_id: int, period: int) -> int:
        """+1 when ``team_id`` attacks towards +x in ``period``, else -1."""
        home_pos = self.home_attacks_positive_x.get(period, True)
        is_home = team_id == self.home_team_id
        return 1 if (home_pos == is_home) else -1

    # -- squads ------------------------------------------------------------ #
    def players_frame(self) -> pd.DataFrame:
        rows = []
        for p in self.raw["players"]:
            role = p.get("player_role") or {}
            pt = (p.get("playing_time") or {}).get("total") or {}
            rows.append(
                {
                    "match_id": self.match_id,
                    "player_id": int(p["id"]),
                    "team_id": int(p["team_id"]),
                    "side": "home" if int(p["team_id"]) == self.home_team_id else "away",
                    "shirt_number": p.get("number"),
                    "first_name": (p.get("first_name") or "").strip(),
                    "last_name": (p.get("last_name") or "").strip(),
                    "short_name": (p.get("short_name") or "").strip(),
                    "birthday": p.get("birthday"),
                    "position": role.get("acronym"),
                    "position_name": role.get("name"),
                    "position_group": role.get("position_group"),
                    "is_starter": bool(
                        pt.get("start_frame") is not None
                        and pt.get("start_frame") <= self.period_start_frames.get(1, 0) + 50
                    ),
                    "minutes_played": pt.get("minutes_played"),
                    "minutes_tip": pt.get("minutes_tip"),
                    "minutes_otip": pt.get("minutes_otip"),
                    "start_frame": pt.get("start_frame"),
                    "end_frame": pt.get("end_frame"),
                    "goals": p.get("goal"),
                    "own_goals": p.get("own_goal"),
                    "yellow_card": p.get("yellow_card"),
                    "red_card": p.get("red_card"),
                    "trackable_object": p.get("trackable_object"),
                }
            )
        df = pd.DataFrame(rows)
        df["display_name"] = df["short_name"].where(
            df["short_name"].str.len() > 0,
            (df["first_name"] + " " + df["last_name"]).str.strip(),
        )
        return df


@lru_cache(maxsize=64)
def load_match_meta(match_id: int) -> MatchMeta:
    path = RAW_MATCHES_DIR / str(match_id) / f"{match_id}_match.json"
    return MatchMeta(match_id=int(match_id), raw=json.loads(path.read_text()))


def match_index() -> pd.DataFrame:
    """The repository's ``matches.json`` as a frame, plus what is on disk."""
    records = json.loads(MATCHES_INDEX_JSON.read_text())
    df = pd.DataFrame(
        [
            {
                "match_id": r["id"],
                "date_time": r["date_time"],
                "home_team_id": r["home_team"]["id"],
                "home_team_name": r["home_team"]["short_name"],
                "away_team_id": r["away_team"]["id"],
                "away_team_name": r["away_team"]["short_name"],
                "competition_id": r.get("competition_id"),
                "season_id": r.get("season_id"),
            }
            for r in records
        ]
    ).sort_values("date_time", ignore_index=True)
    df["has_tracking"] = [tracking_path(m).exists() for m in df.match_id]
    return df


def available_match_ids() -> list[int]:
    return sorted(int(p.name) for p in RAW_MATCHES_DIR.iterdir() if p.is_dir())


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #

def tracking_path(match_id: int) -> Path:
    return RAW_MATCHES_DIR / str(match_id) / f"{match_id}_tracking_extrapolated.jsonl"


def iter_tracking(match_id: int) -> Iterator[dict[str, Any]]:
    """Stream the tracking file one frame at a time.

    A whole match is ~90 MB of JSON and ~1.3 M player-frames; streaming keeps
    the ingest step at roughly constant memory.
    """
    path = tracking_path(match_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No tracking for match {match_id} at {path}. Run `python -m src.data.download`."
        )
    try:
        import orjson as _json

        loads = _json.loads
    except ImportError:  # pragma: no cover
        loads = json.loads
    with path.open("rb") as fh:
        for line in fh:
            if line.strip():
                yield loads(line)


# --------------------------------------------------------------------------- #
# Dynamic events and phases of play
# --------------------------------------------------------------------------- #

def events_path(match_id: int) -> Path:
    return RAW_MATCHES_DIR / str(match_id) / f"{match_id}_dynamic_events.csv"


def phases_path(match_id: int) -> Path:
    return RAW_MATCHES_DIR / str(match_id) / f"{match_id}_phases_of_play.csv"


def load_events_raw(match_id: int) -> pd.DataFrame:
    return pd.read_csv(events_path(match_id), low_memory=False)


def load_phases_raw(match_id: int) -> pd.DataFrame:
    return pd.read_csv(phases_path(match_id), low_memory=False)


# --------------------------------------------------------------------------- #
# Season aggregates
# --------------------------------------------------------------------------- #

AGG_FILES = {
    "physical": "aus1league_physicalaggregates_20242025.csv",
    "passing": "aus1league_passingaggregates_20242025.csv",
    "obr": "aus1league_obraggregates_20242025.csv",
}


def load_aggregate(kind: str) -> pd.DataFrame:
    if kind not in AGG_FILES:
        raise KeyError(f"unknown aggregate '{kind}'; choose from {sorted(AGG_FILES)}")
    return pd.read_csv(RAW_AGGREGATES_DIR / AGG_FILES[kind], low_memory=False)
