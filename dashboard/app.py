"""NeuroTraffic-RL Real-Time KPI Dashboard.

Launch with::

    streamlit run dashboard/app.py

The dashboard auto-refreshes every 2 seconds, reading from MetricsStore
(a JSON file written by the training process).  The simulation and the
dashboard are fully decoupled — the dashboard works even when training
is paused.
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.components.phase_display import render_phase_display
from dashboard.components.queue_chart import render_queue_chart
from dashboard.components.waiting_time_chart import render_waiting_time_chart
from dashboard.metrics_store import MetricsStore

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroTraffic-RL Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for KPI cards
st.markdown(
    """
    <style>
    .kpi-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        border: 1px solid #313244;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #cdd6f4;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #6c7086;
        margin-top: 4px;
    }
    .kpi-delta-pos { color: #a6e3a1; font-size: 0.9rem; }
    .kpi-delta-neg { color: #f38ba8; font-size: 0.9rem; }
    h1, h2, h3 { color: #cdd6f4 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🚦 NeuroTraffic-RL — Real-Time Dashboard")
st.caption("Bd Mohammed VI × Bd Abou Chouaib Doukkali · Casablanca, Morocco")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
REFRESH_SECONDS = 2

store = MetricsStore()
data = store.read()
history = data.get("history", [])
latest = data.get("latest", {})

# ---------------------------------------------------------------------------
# KPI Summary Cards
# ---------------------------------------------------------------------------
st.subheader("📊 KPI Summary")
col1, col2, col3, col4 = st.columns(4)

avg_wait = float(history[-1].get("total_waiting", 0.0)) if history else 0.0
throughput_sum = sum(s.get("throughput", 0) for s in history[-60:]) if history else 0
current_phase = latest.get("current_phase", 0)
episode_reward = latest.get("episode_reward", 0.0)

with col1:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value">{avg_wait:.1f}s</div>'
        '<div class="kpi-label">⏱ Avg Waiting Time</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value">{throughput_sum}</div>'
        '<div class="kpi-label">🚗 Throughput (last 60 steps)</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    phase_names = {0: "NS 🟢", 1: "NS 🟡", 2: "EW 🟢", 3: "EW 🟡"}
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value">'
        f'{phase_names.get(current_phase, "—")}</div>'
        '<div class="kpi-label">🚦 Active Phase</div></div>',
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value">{episode_reward:+.2f}</div>'
        '<div class="kpi-label">🏆 Episode Reward</div></div>',
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Charts Row 1: Queue bars + Phase indicator
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([3, 1])

with col_left:
    st.subheader("🚦 Queue Length by Arm")
    wait_history = store.get_history("total_waiting")
    render_queue_chart(wait_history or [0.0])

with col_right:
    st.subheader("Phase")
    render_phase_display(current_phase=current_phase)

st.divider()

# ---------------------------------------------------------------------------
# Charts Row 2: Waiting time timeline + Reward curve
# ---------------------------------------------------------------------------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("⏳ Waiting Time (last 60 steps)")
    render_waiting_time_chart(wait_history or [0.0], window=60)

with col_b:
    st.subheader("📈 Episode Reward Curve")
    reward_history = store.get_history("episode_reward")
    if reward_history:
        import pandas as pd
        df = pd.DataFrame({"Episode Reward": reward_history[-300:]})
        st.line_chart(df, use_container_width=True)
    else:
        st.info("No reward data yet — start training to populate this chart.")

st.divider()

# ---------------------------------------------------------------------------
# Reward breakdown
# ---------------------------------------------------------------------------
if latest.get("reward_breakdown"):
    st.subheader("🔍 Reward Component Breakdown (last step)")
    breakdown = latest["reward_breakdown"]
    cols = st.columns(len(breakdown))
    for i, (k, v) in enumerate(breakdown.items()):
        with cols[i]:
            label = k.replace("_term", "").replace("_", " ").title()
            delta_cls = "kpi-delta-pos" if v >= 0 else "kpi-delta-neg"
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-value {delta_cls}">'
                f'{v:+.4f}</div><div class="kpi-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Footer + auto-refresh
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(f"Last updated: step {latest.get('step', '—')} · auto-refresh every {REFRESH_SECONDS}s")

# Streamlit auto-rerun (polling pattern)
time.sleep(REFRESH_SECONDS)
st.rerun()
