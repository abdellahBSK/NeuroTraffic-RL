"""KPI computation utilities for NeuroTraffic-RL.

Provides stateless functions for computing traffic performance metrics
from episode data collected during training and evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-episode KPIs
# ---------------------------------------------------------------------------

def compute_p95_wait(waiting_times: List[float]) -> float:
    """Compute the 95th-percentile waiting time from a list of per-step values.

    Args:
        waiting_times: List of cumulative waiting time snapshots (seconds).

    Returns:
        95th-percentile value, or 0.0 if the list is empty.
    """
    if not waiting_times:
        return 0.0
    return float(np.percentile(waiting_times, 95))


def compute_mean_queue(queue_lengths: List[float]) -> float:
    """Compute mean queue length over an episode.

    Args:
        queue_lengths: Per-step total queue length values.

    Returns:
        Mean queue length, or 0.0 if the list is empty.
    """
    if not queue_lengths:
        return 0.0
    return float(np.mean(queue_lengths))


def compute_throughput_rate(total_arrived: int, duration_seconds: float) -> float:
    """Compute throughput in vehicles per minute.

    Args:
        total_arrived:    Number of vehicles that completed their trips.
        duration_seconds: Episode duration in seconds.

    Returns:
        Vehicles per minute, or 0.0 if duration is zero.
    """
    if duration_seconds <= 0:
        return 0.0
    return total_arrived / (duration_seconds / 60.0)


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def compute_improvement_pct(baseline: float, agent: float) -> float:
    """Compute percentage improvement of the agent over baseline.

    Improvement is positive when the agent has a *lower* value (for
    metrics where lower is better, such as waiting time).

    Args:
        baseline: Baseline metric value.
        agent:    Agent metric value.

    Returns:
        Percentage improvement in [−∞, 100].  Returns 0.0 if baseline is 0.
    """
    if baseline == 0:
        return 0.0
    return (baseline - agent) / baseline * 100.0


def format_comparison_table(
    ppo_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
) -> str:
    """Format a side-by-side comparison table as a string.

    Args:
        ppo_metrics:      Metric dict for the PPO agent.
        baseline_metrics: Metric dict for the FixedCycle baseline.

    Returns:
        Multi-line string table.
    """
    header = (
        f"{'Metric':<30} {'FixedCycle':>12} {'PPO':>12} {'Δ%':>8}\n"
        + "─" * 66
    )
    rows = [header]

    lower_is_better = {"avg_wait", "max_wait", "p95_wait", "avg_queue"}

    for key in sorted(set(ppo_metrics) | set(baseline_metrics)):
        b_val = baseline_metrics.get(key, float("nan"))
        p_val = ppo_metrics.get(key, float("nan"))

        if key in lower_is_better:
            delta = compute_improvement_pct(b_val, p_val)
            symbol = "✓" if delta >= 0 else "✗"
        else:
            # Higher is better (e.g. throughput)
            delta = compute_improvement_pct(b_val, p_val) * -1
            symbol = "✓" if delta >= 0 else "✗"

        rows.append(f"{key:<30} {b_val:>12.2f} {p_val:>12.2f} {delta:>+7.1f}% {symbol}")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Running statistics helper
# ---------------------------------------------------------------------------

class RunningStats:
    """Incrementally compute mean and standard deviation.

    Uses Welford's online algorithm.

    Example::

        stats = RunningStats()
        for val in episode_rewards:
            stats.update(val)
        print(stats.mean, stats.std)
    """

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    def update(self, value: float) -> None:
        """Add a new observation to the running statistics.

        Args:
            value: New scalar observation.
        """
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._M2 += delta * delta2

    @property
    def mean(self) -> float:
        """Current running mean."""
        return self._mean

    @property
    def variance(self) -> float:
        """Current running variance (Bessel-corrected)."""
        if self._n < 2:
            return 0.0
        return self._M2 / (self._n - 1)

    @property
    def std(self) -> float:
        """Current running standard deviation."""
        return float(np.sqrt(self.variance))

    @property
    def count(self) -> int:
        """Number of observations."""
        return self._n
