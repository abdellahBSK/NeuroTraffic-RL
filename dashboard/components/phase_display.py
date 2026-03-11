"""Phase display component for the NeuroTraffic-RL dashboard.

Shows the current active traffic light phase with a colour indicator.
"""

from __future__ import annotations

import streamlit as st

_PHASE_INFO = {
    0: ("NS_GREEN",  "🟢", "#00b300"),
    1: ("NS_YELLOW", "🟡", "#e6b800"),
    2: ("EW_GREEN",  "🟢", "#00b300"),
    3: ("EW_YELLOW", "🟡", "#e6b800"),
}


def render_phase_display(current_phase: int, phase_elapsed: float = 0.0) -> None:
    """Render the current active traffic light phase indicator.

    Args:
        current_phase: Phase index (0–3).
        phase_elapsed: Seconds elapsed in the current phase.
    """
    name, icon, color = _PHASE_INFO.get(current_phase, ("UNKNOWN", "⬜", "#888888"))
    st.markdown(
        f"""
        <div style="text-align:center; padding:16px; border-radius:12px;
                    background:{color}22; border: 2px solid {color};">
            <span style="font-size:2.5rem;">{icon}</span><br/>
            <span style="font-size:1.4rem; font-weight:600; color:{color};">{name}</span><br/>
            <span style="font-size:0.9rem; color:#888;">Phase elapsed: {phase_elapsed:.1f}s</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
