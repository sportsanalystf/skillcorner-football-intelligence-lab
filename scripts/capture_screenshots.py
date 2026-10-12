"""Capture real screenshots of the running application.

    python -m streamlit run app/main.py &
    python scripts/capture_screenshots.py

Uses Playwright against a locally running server, so the images in the README
are the application as it actually renders, not mock-ups.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playwright.sync_api import sync_playwright

from src.config import DOCS_DIR

OUT = DOCS_DIR / "screenshots"
BASE = "http://localhost:8501"

PAGES = [
    ("", "01-overview"),
    ("match_explorer", "02-match-explorer"),
    ("tactical", "03-tactical"),
    ("possession_epv", "04-possession-epv"),
    ("passing", "05-passing"),
    ("off_ball", "06-off-ball"),
    ("physical", "07-physical"),
    ("player_explorer", "08-player-explorer"),
    ("body_pose", "09-body-pose"),
    ("research", "10-research"),
    ("data_quality", "11-data-quality"),
]


def main(width: int = 1680, height: int = 1150, settle: float = 9.0) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": width, "height": height},
                                device_scale_factor=2)
        for path, name in PAGES:
            page.goto(f"{BASE}/{path}", wait_until="networkidle", timeout=120_000)
            # Streamlit streams its result; wait for the spinner to clear.
            for _ in range(60):
                if page.locator('[data-testid="stStatusWidget"]').count() == 0:
                    break
                time.sleep(0.5)
            time.sleep(settle)
            page.mouse.move(5, 5)
            page.screenshot(path=str(OUT / f"{name}.png"))
            print(f"captured {name}")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
