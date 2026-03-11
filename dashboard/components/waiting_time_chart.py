"""Waiting time timeline component for the NeuroTraffic-RL dashboard.

Renders a rolling 60-step line chart of total waiting time.
"""

from __future__ import annotations

from typing import List

import streamlit as st


def render_waiting_time_chart(history: List[float], window: int = 60) -> None:
    """Render a rolling line chart of total waiting time over the last ``window`` steps.

    Args:
        history: Ordered list of ``total_waiting`` values (oldest first).
        window:  Number of recent steps to display.
    """
    import pandas as pd

    recent = history[-window:] if len(history) > window else history
    df = pd.DataFrame({"Total Waiting Time (s)": recent})
    st.line_chart(df, use_container_width=True)
