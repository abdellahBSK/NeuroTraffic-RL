"""Unit tests for the RewardCalculator.

No SUMO dependency — pure Python tests.
Run with: pytest tests/test_reward.py -v
"""

from __future__ import annotations

import pytest

from env.reward_calculator import RewardCalculator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calc() -> RewardCalculator:
    """Default reward calculator with spec coefficients."""
    return RewardCalculator(alpha=1.0, beta=0.5, gamma=0.3, delta=0.1,
                            max_wait_cap=300.0, max_queue_cap=20.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRewardCalculator:

    def test_zero_waiting_zero_throughput_gives_negative_reward(self, calc):
        """With no waiting and no throughput, reward should be 0 (no bonus, no penalty)."""
        reward = calc.calculate(
            lane_waiting_times={"N2J_0": 0.0, "S2J_0": 0.0},
            throughput_delta=0,
            current_action=0,
            prev_action=None,
        )
        assert reward == 0.0

    def test_high_waiting_gives_negative_reward(self, calc):
        """Large waiting times must produce a negative reward."""
        reward = calc.calculate(
            lane_waiting_times={"N2J_0": 300.0, "S2J_0": 300.0},
            throughput_delta=0,
            current_action=0,
            prev_action=None,
        )
        assert reward < 0.0

    def test_throughput_bonus_is_positive(self, calc):
        """Throughput bonus should contribute positively."""
        reward_no_tp = calc.calculate(
            lane_waiting_times={}, throughput_delta=0,
            current_action=0, prev_action=None)
        reward_with_tp = calc.calculate(
            lane_waiting_times={}, throughput_delta=10,
            current_action=0, prev_action=None)
        assert reward_with_tp > reward_no_tp

    def test_phase_change_penalty_applied(self, calc):
        """Switching from NS_GREEN (0) to EW_GREEN (2) incurs a penalty."""
        reward_keep = calc.calculate(
            lane_waiting_times={}, throughput_delta=0,
            current_action=0, prev_action=0)  # Same phase, no penalty
        reward_switch = calc.calculate(
            lane_waiting_times={}, throughput_delta=0,
            current_action=2, prev_action=0)  # Switch, penalty applied
        assert reward_keep > reward_switch

    def test_phase_change_penalty_not_applied_on_first_step(self, calc):
        """No phase-change penalty when prev_action is None (first step)."""
        reward = calc.calculate(
            lane_waiting_times={}, throughput_delta=0,
            current_action=2, prev_action=None)
        assert reward == 0.0  # No wait, no throughput, no penalty

    def test_max_queue_penalty_applied(self, calc):
        """Very high waiting time in one lane triggers starvation penalty."""
        reward_balanced = calc.calculate(
            lane_waiting_times={"N2J_0": 10.0, "S2J_0": 10.0},
            throughput_delta=0, current_action=0, prev_action=None)
        reward_starved = calc.calculate(
            lane_waiting_times={"N2J_0": 300.0, "S2J_0": 0.0},
            throughput_delta=0, current_action=0, prev_action=None)
        assert reward_starved < reward_balanced

    def test_breakdown_has_all_components(self, calc):
        """get_component_breakdown() must include all 4 terms + total."""
        calc.calculate(
            lane_waiting_times={"N2J_0": 50.0},
            throughput_delta=3, current_action=0, prev_action=None)
        breakdown = calc.get_component_breakdown()
        assert "wait_term" in breakdown
        assert "throughput_term" in breakdown
        assert "max_queue_term" in breakdown
        assert "phase_change_term" in breakdown
        assert "total" in breakdown

    def test_breakdown_total_matches_return(self, calc):
        """Breakdown total must match the returned reward."""
        reward = calc.calculate(
            lane_waiting_times={"N2J_0": 100.0, "E2J_0": 50.0},
            throughput_delta=5, current_action=2, prev_action=0)
        breakdown = calc.get_component_breakdown()
        assert abs(breakdown["total"] - reward) < 1e-9

    def test_from_config_factory(self):
        """from_config() must correctly parse all coefficient keys."""
        cfg = {"alpha": 2.0, "beta": 1.0, "gamma": 0.5, "delta": 0.2,
               "max_wait_cap": 200.0, "max_queue_cap": 15.0}
        calc = RewardCalculator.from_config(cfg)
        assert calc.alpha == 2.0
        assert calc.beta == 1.0
        assert calc.gamma == 0.5
        assert calc.delta == 0.2
        assert calc.max_wait_cap == 200.0
        assert calc.max_queue_cap == 15.0

    def test_reward_bounded(self, calc):
        """Reward should be finite for extreme inputs."""
        import math
        reward = calc.calculate(
            lane_waiting_times={"N2J_0": 1e9},
            throughput_delta=0, current_action=0, prev_action=None)
        assert math.isfinite(reward)
