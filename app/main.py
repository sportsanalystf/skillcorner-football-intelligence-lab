"""Football Intelligence Lab -- Streamlit entry point.

Run with:  streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Football Intelligence Lab",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.components.ui import inject_css  # noqa: E402
from src.visualization import theme  # noqa: E402,F401  (registers the Plotly template)

inject_css()

PAGES = [
    st.Page("pages/home.py", title="Overview", icon=":material/home:", default=True),
    st.Page("pages/match_explorer.py", title="Match Explorer", icon=":material/sports_soccer:"),
    st.Page("pages/tactical.py", title="Tactical Analysis", icon=":material/dashboard:"),
    st.Page("pages/possession_epv.py", title="Possession & EPV", icon=":material/timeline:"),
    st.Page("pages/passing.py", title="Passing", icon=":material/hub:"),
    st.Page("pages/off_ball.py", title="Off-Ball & Pressing", icon=":material/directions_run:"),
    st.Page("pages/physical.py", title="Physical", icon=":material/monitor_heart:"),
    st.Page("pages/player_explorer.py", title="Player Explorer", icon=":material/person:"),
    st.Page("pages/body_pose.py", title="3D Body Pose", icon=":material/accessibility_new:"),
    st.Page("pages/research.py", title="Research", icon=":material/science:"),
    st.Page("pages/data_quality.py", title="Data Quality", icon=":material/rule:"),
]

with st.sidebar:
    st.markdown(
        '<div style="padding:0.2rem 0 0.9rem 0">'
        '<div style="font-size:1.05rem;font-weight:660;letter-spacing:-0.01em;color:#EAF0F7">'
        "Football Intelligence Lab</div>"
        '<div style="font-size:0.76rem;color:#7A8798;margin-top:2px">'
        "SkillCorner Open Data · A-League 2024/25</div></div>",
        unsafe_allow_html=True,
    )

nav = st.navigation(PAGES)

with st.sidebar:
    st.divider()
    st.caption(
        "Broadcast tracking, dynamic events, phases of play, season physical "
        "aggregates and 3D body pose, from "
        "[SkillCorner Open Data](https://github.com/SkillCorner/opendata)."
    )

nav.run()
