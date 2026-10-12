"""Every application page renders, and the interactive paths render too.

These are smoke tests, not visual tests: they assert that a page runs to
completion against the real built tables with no uncaught exception, including
the branches behind the main selectors. Streamlit executes the body of every
`st.tabs` tab, so a single run covers all tabs on a page.
"""

from pathlib import Path

import pytest

from tests.conftest import needs_data

pytestmark = needs_data

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = [
    "home", "match_explorer", "tactical", "possession_epv", "passing",
    "off_ball", "physical", "player_explorer", "body_pose", "research",
    "data_quality",
]


def _run(path, timeout=300, **state):
    """Run a page from scratch, optionally with widget state pre-seeded.

    Widget state is set through ``session_state`` on a fresh AppTest rather
    than by ``set_value().run()``, because re-running a page whose widget tree
    depends on an earlier selection confuses AppTest's widget bookkeeping.
    Every page's top-level selectors carry explicit keys for this reason.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(path), default_timeout=timeout)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, f"{path.name} with {state}:\n{_describe(at)}"
    return at


def _describe(at) -> str:
    return "\n".join(
        getattr(e, "value", str(e)) + "\n" + "".join(getattr(e, "stack_trace", []) or [])
        for e in at.exception
    )


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = _run(APP / "pages" / f"{page}.py")
    assert len(at.markdown) > 0


def test_entrypoint_builds_navigation():
    import ast

    src = (APP / "main.py").read_text()
    tree = ast.parse(src)
    assert "st.navigation" in src
    pages = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "Page"]
    assert len(pages) == len(PAGES)
    for node in pages:
        target = node.args[0].value
        assert (APP / target).exists(), f"main.py points at a missing page: {target}"


def test_match_explorer_free_scrub_branch():
    _run(APP / "pages" / "match_explorer.py", me_mode="Free scrub")


@pytest.mark.parametrize(
    "kind",
    ["all", "led to a shot", "progressive pass", "under high pressure",
     "line-breaking pass", "highest EPV"],
)
def test_match_explorer_possession_filters_all_run(kind):
    _run(APP / "pages" / "match_explorer.py", me_filter=kind)


def test_possession_page_single_match_scope():
    _run(APP / "pages" / "possession_epv.py", pe_scope="One match")


@pytest.mark.parametrize(
    "window",
    ["Until first substitution", "Full match", "First half", "Second half"],
)
def test_passing_page_windows_all_run(window):
    _run(APP / "pages" / "passing.py", pass_window=window)


@pytest.mark.parametrize(
    "show",
    ["All passes", "Completed only", "Progressive", "Line-breaking",
     "Under high pressure", "Into the box", "Led to a shot"],
)
def test_passing_page_filters_all_run(show):
    _run(APP / "pages" / "passing.py", pm_show=show)


def test_tactical_page_away_team_branch():
    at = _run(APP / "pages" / "tactical.py")
    team = [s for s in at.selectbox if (s.label or "") == "Team"]
    assert team
    _run(APP / "pages" / "tactical.py", tac_team=team[0].options[1])


def test_body_pose_page_reports_its_source():
    at = _run(APP / "pages" / "body_pose.py")
    text = " ".join(m.value for m in at.markdown if isinstance(m.value, str))
    assert "sample" in text.lower() or "full match" in text.lower()


@pytest.mark.parametrize("page,key", [
    ("possession_epv", "pe_match"),
    ("match_explorer", "me_match"),
    ("tactical", "tac_match"),
])
def test_a_match_without_epv_still_renders(page, key):
    """Four matches carry no EPV or pressure at all; nothing may fall over."""
    seed = {"pe_scope": "One match"} if page == "possession_epv" else {}
    at = _run(APP / "pages" / f"{page}.py", **seed)
    sel = [s for s in at.selectbox if s.key == key]
    assert sel, f"{page} has no {key} selector"
    target = [o for o in sel[0].options if "Newcastle" in o]
    assert target, "expected match 1886347 (Auckland v Newcastle) in the list"
    _run(APP / "pages" / f"{page}.py", **{**seed, key: target[0]})
