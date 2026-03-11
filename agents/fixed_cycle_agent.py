"""Fixed-cycle baseline agent for NeuroTraffic-RL.

Implements a deterministic 30 s green / 3 s yellow cycle — the classical
fixed-time traffic light controller used as a benchmark.

This agent satisfies the same ``BaseAgent`` interface as PPOAgent, so both
can be evaluated with the identical harness in ``training/evaluate.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class FixedCycleAgent(BaseAgent):
    """Deterministic fixed-time traffic signal controller.

    Alternates between NS_GREEN (phase 0) and EW_GREEN (phase 2) on a
    fixed schedule.  Yellow transitions are handled by the environment's
    PhaseManager, not by this agent.

    The agent always returns a green phase; yellow is never returned
    directly (env auto-injects it).

    Args:
        green_duration:    Number of RL steps to hold each green phase.
        communication_bus: Unused in Phase 1 (required by BaseAgent).
    """

    def __init__(
        self,
        green_duration: int = 6,  # 6 steps × 5 s/step = 30 s
        communication_bus: Optional[Any] = None,
    ) -> None:
        super().__init__(communication_bus=communication_bus)
        self._green_duration = green_duration
        self._step_count: int = 0
        self._phases = [0, 2]  # NS_GREEN, EW_GREEN
        self._phase_idx: int = 0

        logger.info(
            "FixedCycleAgent initialised — green=%d steps.", green_duration
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def predict(self, observation: np.ndarray) -> int:  # noqa: ARG002
        """Return the next phase based purely on the fixed schedule.

        The observation is ignored — this agent is purely time-driven.

        Args:
            observation: Current environment observation (ignored).

        Returns:
            Phase ID (0 = NS_GREEN, 2 = EW_GREEN).
        """
        if self._step_count >= self._green_duration:
            self._phase_idx = (self._phase_idx + 1) % len(self._phases)
            self._step_count = 0

        self._step_count += 1
        return self._phases[self._phase_idx]

    def reset(self) -> None:
        """Reset the internal cycle counter for a new episode."""
        self._step_count = 0
        self._phase_idx = 0

    def train(self, total_timesteps: int, callbacks: Optional[Any] = None) -> None:
        """No-op: FixedCycleAgent does not learn.

        Args:
            total_timesteps: Ignored.
            callbacks:       Ignored.
        """
        logger.info("FixedCycleAgent.train() called — nothing to learn.")

    def evaluate(self, n_episodes: int) -> Dict[str, float]:
        """Placeholder evaluation — actual evaluation run via training/evaluate.py.

        Args:
            n_episodes: Number of evaluation episodes.

        Returns:
            Empty dict (evaluation is handled externally for both agents).
        """
        logger.info(
            "FixedCycleAgent.evaluate() — run training/evaluate.py for full report."
        )
        return {}

    def save(self, path: str) -> None:
        """No-op: FixedCycleAgent has no learnable parameters.

        Args:
            path: Ignored.
        """
        logger.debug("FixedCycleAgent.save() — no-op (no learnable params).")

    def load(self, path: str) -> None:
        """No-op: FixedCycleAgent has no learnable parameters.

        Args:
            path: Ignored.
        """
        logger.debug("FixedCycleAgent.load() — no-op.")
