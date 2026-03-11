"""Queue chart component for the NeuroTraffic-RL dashboard.

Renders a colour-coded bar chart of per-arm queue lengths.
"""

from __future__ import annotations

from typing import Dict, List

import streamlit as st


_PHASE_NAMES = {0: "NS_GREEN", 1: "NS_YELLOW", 2: "EW_GREEN", 3: "EW_YELLOW"}
_PHASE_COLORS = {0: "🟢", 1: "🟡", 2: "🟢", 3: "🟡"}


def render_queue_chart(history: List[float], arm_labels: List[str] = None) -> None:
    """Render a live bar chart of current queue lengths per arm.

    Colour coding:
        Green  → queue < 5
        Orange → 5 ≤ queue < 10
        Red    → queue ≥ 10

    Args:
        history:    Rolling history of ``total_waiting`` (used as queue proxy).
        arm_labels: Optional list of arm names for the x-axis.
    """
    import pandas as pd

    if arm_labels is None:
        arm_labels = ["North", "South", "East", "West"]

    # Use the last snapshot as the "current" queue value
    current_total = history[-1] if history else 0.0
    # Distribute evenly as a simple proxy (real per-arm data from MetricsStore in future)
    per_arm = [current_total / len(arm_labels)] * len(arm_labels)

    colors = []
    for v in per_arm:
        if v < 5:
            colors.append("🟢")
        elif v < 10:
            colors.append("🟠")
        else:
            colors.append("🔴")

    df = pd.DataFrame({
        "Arm": [f"{colors[i]} {arm_labels[i]}" for i in range(len(arm_labels))],
        "Queue (veh)": per_arm,
    })

    st.bar_chart(df.set_index("Arm"), use_container_width=True)
