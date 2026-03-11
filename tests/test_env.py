"""Unit tests for the SumoIntersectionEnv.

All TraCI calls are mocked — no real SUMO installation required.
Run with: pytest tests/test_env.py -v
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_configs() -> tuple:
    """Return minimal intersection, sim, and training config dicts."""
    intersection_cfg = {
        "intersection_id": "test_intersection",
        "tl_id": "J0",
        "incoming_lanes": ["N2J_0", "N2J_1", "S2J_0", "S2J_1",
                           "E2J_0", "E2J_1", "W2J_0", "W2J_1"],
        "incoming_edges": ["N2J", "S2J", "E2J", "W2J"],
        "observation_arms": [
            {"id": "north", "lanes": ["N2J_0", "N2J_1"]},
            {"id": "south", "lanes": ["S2J_0", "S2J_1"]},
            {"id": "east",  "lanes": ["E2J_0", "E2J_1"]},
            {"id": "west",  "lanes": ["W2J_0", "W2J_1"]},
        ],
        "phases": [
            {"id": 0, "name": "NS_GREEN",  "min_duration": 5,  "max_duration": 30},
            {"id": 1, "name": "NS_YELLOW", "fixed_duration": 3},
            {"id": 2, "name": "EW_GREEN",  "min_duration": 5,  "max_duration": 30},
            {"id": 3, "name": "EW_YELLOW", "fixed_duration": 3},
        ],
        "step_seconds": 5,
        "max_episode_seconds": 100,
    }
    sim_cfg = {
        "sumo": {
            "binary": "sumo",
            "gui_binary": "sumo-gui",
            "config_file": "sumo/networks/casablanca_intersection.sumocfg",
            "port_start": 8813,
            "port_end": 8899,
            "no_warnings": True,
            "no_step_log": True,
            "time_to_teleport": -1,
        },
        "baseline": {"green_duration": 30, "yellow_duration": 3},
    }
    train_cfg = {
        "reward": {
            "alpha": 1.0, "beta": 0.5, "gamma": 0.3, "delta": 0.1,
            "max_wait_cap": 300.0, "max_queue_cap": 20.0,
            "max_speed": 13.89, "max_density": 0.1,
        },
    }
    return intersection_cfg, sim_cfg, train_cfg


def _mock_traci() -> MagicMock:
    """Return a fully mocked traci module."""
    mock = MagicMock()
    mock.simulation.getTime.return_value = 0.0
    mock.simulation.getArrivedNumber.return_value = 0
    mock.trafficlight.getPhase.return_value = 0
    mock.trafficlight.setPhase.return_value = None
    mock.lane.getLastStepHaltingNumber.return_value = 0
    mock.lane.getLastStepMeanSpeed.return_value = 10.0
    mock.lane.getLastStepOccupancy.return_value = 20.0
    mock.lane.getWaitingSum.return_value = 0.0
    mock.lane.getLastStepVehicleNumber.return_value = 2
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSumoIntersectionEnv:
    """Unit tests for SumoIntersectionEnv with mocked TraCI."""

    def _make_env(self):
        """Create a SumoIntersectionEnv with mocked SUMO internals."""
        from env.sumo_env import SumoIntersectionEnv

        icfg, scfg, tcfg = _make_configs()
        env = SumoIntersectionEnv(icfg, scfg, tcfg, use_gui=False, label="test")

        # Inject mock traci directly so _start_sumo() is bypassed
        mock_t = _mock_traci()
        env._traci = mock_t

        # Build sub-components manually
        from env.observation_builder import ObservationBuilder
        from env.phase_manager import PhaseManager
        from env.reward_calculator import RewardCalculator

        env._reward_calc = RewardCalculator.from_config(tcfg["reward"])
        env._obs_builder = ObservationBuilder(
            traci_module=mock_t,
            intersection_cfg=icfg,
        )
        env._phase_manager = PhaseManager(
            traci_module=mock_t,
            tl_id="J0",
            phase_configs=icfg["phases"],
        )
        env._phase_manager.reset(sim_time=0.0)
        return env, mock_t

    def test_observation_space_shape(self):
        """Observation space must have shape (26,)."""
        from env.sumo_env import SumoIntersectionEnv
        icfg, scfg, tcfg = _make_configs()
        env = SumoIntersectionEnv(icfg, scfg, tcfg)
        assert env.observation_space.shape == (26,), (
            f"Expected (26,), got {env.observation_space.shape}"
        )

    def test_action_space(self):
        """Action space must be Discrete(2)."""
        from env.sumo_env import SumoIntersectionEnv
        from gymnasium import spaces
        icfg, scfg, tcfg = _make_configs()
        env = SumoIntersectionEnv(icfg, scfg, tcfg)
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 2

    def test_collect_state_returns_valid_observation(self):
        """_collect_state() must return float32 array of shape (26,) with finite values."""
        env, _ = self._make_env()
        obs = env._collect_state()
        assert obs.shape == (26,), f"Expected (26,), got {obs.shape}"
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs)), "Observation contains non-finite values."

    def test_step_returns_correct_structure(self):
        """step() must return (obs, reward, terminated, truncated, info)."""
        env, mock_t = self._make_env()
        mock_t.simulation.getTime.return_value = 5.0

        obs, reward, terminated, truncated, info = env.step(0)

        assert obs.shape == (26,)
        assert obs.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "step" in info
        assert "reward_breakdown" in info

    def test_terminated_is_always_false(self):
        """terminated should always be False (we use truncated for episode limits)."""
        env, _ = self._make_env()
        _, _, terminated, _, _ = env.step(0)
        assert not terminated

    def test_episode_truncates_at_max_steps(self):
        """truncated must become True once step_count >= max_episode_seconds."""
        env, mock_t = self._make_env()
        env._step_count = 95  # 100 - 5 = one step away from truncation
        mock_t.simulation.getTime.return_value = 95.0

        _, _, _, truncated, _ = env.step(0)
        assert truncated, "Episode should truncate at max_episode_seconds=100."

    def test_illegal_yellow_action_raises(self):
        """PhaseManager must raise ValueError if agent selects phase 1 or 3."""
        from env.phase_manager import PhaseManager
        mock_t = _mock_traci()
        pm = PhaseManager(
            traci_module=mock_t,
            tl_id="J0",
            phase_configs=_make_configs()[0]["phases"],
        )
        pm.reset(sim_time=0.0)

        with pytest.raises(ValueError, match="YELLOW"):
            pm.apply_phase(action=1, sim_time=20.0)  # NS_YELLOW

        with pytest.raises(ValueError, match="YELLOW"):
            pm.apply_phase(action=3, sim_time=20.0)  # EW_YELLOW

    def test_get_valid_actions(self):
        """get_valid_actions() should return [0, 2] when not in yellow."""
        from env.phase_manager import PhaseManager
        mock_t = _mock_traci()
        pm = PhaseManager(
            traci_module=mock_t,
            tl_id="J0",
            phase_configs=_make_configs()[0]["phases"],
        )
        pm.reset(sim_time=0.0)
        assert pm.get_valid_actions() == [0, 2]

    def test_close_does_not_raise(self):
        """close() must not raise even if SUMO was never started."""
        from env.sumo_env import SumoIntersectionEnv
        icfg, scfg, tcfg = _make_configs()
        env = SumoIntersectionEnv(icfg, scfg, tcfg)
        env.close()  # Should not raise

    def test_intersection_id_property(self):
        """intersection_id must return the config value."""
        from env.sumo_env import SumoIntersectionEnv
        icfg, scfg, tcfg = _make_configs()
        env = SumoIntersectionEnv(icfg, scfg, tcfg)
        assert env.intersection_id == "test_intersection"
