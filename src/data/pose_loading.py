"""Streaming readers for SkillCorner 3D body pose.

Pose is the one part of this dataset that does not fit in memory: a full match
is ~3.3 GB of JSON and roughly 45 million joint observations. Everything here
streams, and the only thing ever materialised is the flattened Parquet the rest
of the project reads.

Two sources are supported and the code does not care which it gets:

``sample``
    ``data/raw/bodypose/sample_1925299_phase406.jsonl.gz`` -- one 12.3 s phase
    of play, committed to the SkillCorner repository. 309 pose frames.
``archive``
    ``data/raw/bodypose/{match_id}.jsonl.zip`` -- a full match, if
    ``python -m src.data.download --bodypose`` was able to reach the Hugging
    Face Hub. Streamed straight out of the zip.

Frame rates differ: pose is 25 fps and tracking is 10 fps, so
``pose_frame = 2.5 * tracking_frame`` and only even tracking frames land
exactly on a pose frame.
"""

from __future__ import annotations

import gzip
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.config import BODYPOSE_MATCH_IDS, POSE_FPS, RAW_BODYPOSE_DIR, TRACKING_FPS

SAMPLE_PATH = RAW_BODYPOSE_DIR / "sample_1925299_phase406.jsonl.gz"
SAMPLE_MATCH_ID = 1925299

JOINTS: tuple[str, ...] = (
    "nose", "neck", "lEye", "rEye", "lEar", "rEar",
    "lShoulder", "rShoulder", "lElbow", "rElbow", "lWrist", "rWrist",
    "lThumb", "rThumb", "lPinky", "rPinky",
    "midHip", "lHip", "rHip", "lKnee", "rKnee", "lAnkle", "rAnkle",
    "lHeel", "rHeel", "lBigToe", "rBigToe", "lSmallToe", "rSmallToe",
)

SKELETON: tuple[tuple[str, str], ...] = (
    ("nose", "neck"), ("nose", "lEye"), ("nose", "rEye"),
    ("lEye", "lEar"), ("rEye", "rEar"),
    ("neck", "lShoulder"), ("neck", "rShoulder"), ("neck", "midHip"),
    ("lShoulder", "lElbow"), ("lElbow", "lWrist"),
    ("rShoulder", "rElbow"), ("rElbow", "rWrist"),
    ("lWrist", "lThumb"), ("lWrist", "lPinky"),
    ("rWrist", "rThumb"), ("rWrist", "rPinky"),
    ("midHip", "lHip"), ("midHip", "rHip"),
    ("lHip", "lKnee"), ("lKnee", "lAnkle"),
    ("rHip", "rKnee"), ("rKnee", "rAnkle"),
    ("lAnkle", "lHeel"), ("lAnkle", "lBigToe"), ("lBigToe", "lSmallToe"),
    ("rAnkle", "rHeel"), ("rAnkle", "rBigToe"), ("rBigToe", "rSmallToe"),
)


def tracking_to_pose_frame(tracking_frame: int | float) -> float:
    return tracking_frame * (POSE_FPS / TRACKING_FPS)


def pose_to_tracking_frame(pose_frame: int | float) -> float:
    return pose_frame * (TRACKING_FPS / POSE_FPS)


def archive_path(match_id: int) -> Path:
    return RAW_BODYPOSE_DIR / f"{match_id}.jsonl.zip"


def has_full_match(match_id: int) -> bool:
    p = archive_path(match_id)
    return p.exists() and p.stat().st_size > 100_000_000


def available_sources() -> dict[str, Any]:
    """What pose data this checkout actually has."""
    return {
        "sample": SAMPLE_PATH.exists(),
        "sample_match_id": SAMPLE_MATCH_ID,
        "full_matches": [m for m in BODYPOSE_MATCH_IDS if has_full_match(m)],
    }


def iter_sample() -> Iterator[dict[str, Any]]:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"pose sample missing at {SAMPLE_PATH}")
    with gzip.open(SAMPLE_PATH, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def iter_archive(match_id: int, frame_range: tuple[int, int] | None = None) -> Iterator[dict[str, Any]]:
    path = archive_path(match_id)
    if not path.exists():
        raise FileNotFoundError(
            f"no full pose archive for {match_id}; run "
            "`python -m src.data.download --bodypose` (needs huggingface.co)"
        )
    lo, hi = frame_range or (-1, 1 << 62)
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".jsonl"))
        with zf.open(name) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                fr = rec.get("frame", -1)
                if fr < lo:
                    continue
                if fr > hi:
                    break
                yield rec


def iter_pose(match_id: int, frame_range: tuple[int, int] | None = None) -> Iterator[dict[str, Any]]:
    """Best available pose stream for a match."""
    if has_full_match(match_id):
        yield from iter_archive(match_id, frame_range)
    elif match_id == SAMPLE_MATCH_ID:
        lo, hi = frame_range or (-1, 1 << 62)
        for rec in iter_sample():
            if lo <= rec.get("frame", -1) <= hi:
                yield rec
    else:
        raise FileNotFoundError(f"no pose available for match {match_id}")
