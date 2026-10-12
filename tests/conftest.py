import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DERIVED_DIR, PROCESSED_DIR  # noqa: E402
from src.data.loaders import available_match_ids  # noqa: E402

REF_MATCH = 1874553


def _built() -> bool:
    return (PROCESSED_DIR / "events.parquet").exists() and (
        PROCESSED_DIR / "tracking" / f"match_id={REF_MATCH}" / "tracking.parquet"
    ).exists()


needs_data = pytest.mark.skipif(
    not _built(), reason="processed tables not built; run `python -m src.pipeline`"
)


@pytest.fixture(scope="session")
def ref_match() -> int:
    return REF_MATCH


@pytest.fixture(scope="session")
def match_ids() -> list[int]:
    return available_match_ids()
