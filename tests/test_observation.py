"""Unit tests for the ObservationBuilder.

All TraCI calls are mocked — no real SUMO required.
Run with: pytest tests/test_observation.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_traci(
    halting: int = 2,
    mean_speed: float = 8.0,
    occupancy: float = 30.0,   # raw % (will be /100)
    waiting: float = 40.0,
    vehicles: int = 4,
    phase: int = 0,
) -> MagicMock:
    mock = MagicMock()
    mock.lane.getLastStepHaltingNumber.return_value = halting
    mock.lane.getLastStepMeanSpeed.return_value = mean_speed
    mock.lane.getLastStepOccupancy.return_value = occupancy
    mock.lane.getWaitingSum.return_value = waiting
    mock.lane.getLastStepVehicleNumber.return_value = vehicles
    mock.trafficlight.getPhase.return_value = phase
    return mock


def _make_intersection_cfg() -> dict:
    return {
        "tl_id": "J0",
        "incoming_lanes": ["N2J_0", "N2J_1", "S2J_0", "S2J_1",
                           "E2J_0", "E2J_1", "W2J_0", "W2J_1"],
        "observation_arms": [
            {"id": "north", "lanes": ["N2J_0", "N2J_1"]},
            {"id": "south", "lanes": ["S2J_0", "S2J_1"]},
            {"id": "east",  "lanes": ["E2J_0", "E2J_1"]},
            {"id": "west",  "lanes": ["W2J_0", "W2J_1"]},
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestObservationBuilder:

    def _make_builder(self, traci_mock=None):
        from env.observation_builder import ObservationBuilder
        if traci_mock is None:
            traci_mock = _make_mock_traci()
        return ObservationBuilder(
            traci_module=traci_mock,
            intersection_cfg=_make_intersection_cfg(),
        ), traci_mock

    def test_observation_size_property(self):
        """observation_size must equal 26."""
        builder, _ = self._make_builder()
        assert builder.observation_size == 26

    def test_build_returns_correct_shape(self):
        """build() must return array of shape (26,)."""
        builder, _ = self._make_builder()
        obs = builder.build(phase_elapsed=5.0, sim_time=100.0)
        assert obs.shape == (26,)

    def test_build_returns_float32(self):
        """build() must return float32 dtype."""
        builder, _ = self._make_builder()
        obs = builder.build(phase_elapsed=5.0, sim_time=100.0)
        assert obs.dtype == np.float32

    def test_normalised_features_in_range(self):
        """Normalised features (indices 1,2,3,4 per arm + global) must be in [0,1]."""
        builder, _ = self._make_builder()
        obs = builder.build(phase_elapsed=5.0, sim_time=100.0)

        # Features 1-4 of each 5-feature arm block are normalised to [0,1]
        for arm_idx in range(4):
            base = arm_idx * 5
            for feat_offset in [1, 2, 3, 4]:
                val = obs[base + feat_offset]
                assert 0.0 <= val <= 1.0, (
                    f"Feature at index {base + feat_offset} = {val} out of [0,1]"
                )

    def test_phase_one_hot_encoding(self):
        """Phase one-hot (indices 20-23) must have exactly one 1.0."""
        for phase in range(4):
            mock_t = _make_mock_traci(phase=phase)
            builder, _ = self._make_builder(traci_mock=mock_t)
            obs = builder.build(phase_elapsed=0.0, sim_time=0.0)
            one_hot = obs[20:24]
            assert one_hot.sum() == pytest.approx(1.0), (
                f"Phase {phase}: one-hot sum={one_hot.sum()}, expected 1.0"
            )
            assert one_hot[phase] == pytest.approx(1.0), (
                f"Phase {phase}: one-hot[{phase}]={one_hot[phase]}, expected 1.0"
            )

    def test_time_encoding_in_range(self):
        """Time sin/cos (indices 24, 25) must be in [-1, 1]."""
        builder, _ = self._make_builder()
        obs = builder.build(phase_elapsed=0.0, sim_time=43200.0)  # noon
        assert -1.0 <= obs[24] <= 1.0
        assert -1.0 <= obs[25] <= 1.0

    def test_safe_defaults_on_empty_lane(self):
        """build() must not raise even if all TraCI calls return 0."""
        mock_t = _make_mock_traci(halting=0, mean_speed=0.0, occupancy=0.0,
                                  waiting=0.0, vehicles=0, phase=0)
        builder, _ = self._make_builder(traci_mock=mock_t)
        obs = builder.build(phase_elapsed=0.0, sim_time=0.0)
        assert obs.shape == (26,)
        assert np.all(np.isfinite(obs))

    def test_high_values_are_capped(self):
        """Extreme raw values must be clipped to [0,1]."""
        mock_t = _make_mock_traci(mean_speed=1000.0, occupancy=9999.0, waiting=1e6)
        builder, _ = self._make_builder(traci_mock=mock_t)
        obs = builder.build(phase_elapsed=999.0, sim_time=0.0)
        # All normalised slots must stay ≤ 1.0
        for arm_idx in range(4):
            base = arm_idx * 5
            for feat_offset in [1, 2, 3, 4]:
                assert obs[base + feat_offset] <= 1.0 + 1e-6
