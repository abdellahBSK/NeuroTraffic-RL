"""Abstract base agent for NeuroTraffic-RL.

All agents (PPO, FixedCycle, future DQN/SAC) must implement this interface.
This guarantees interchangeability in the evaluation harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseAgent(ABC):
    """Contract for all traffic signal control agents.

    Designed with Phase 2 in mind: the ``communication_bus`` parameter
    allows agents to share observations with neighbours without modifying
    the core interface.

    Args:
        communication_bus: Optional message bus for multi-agent coordination
                           (Phase 2).  Pass ``None`` in Phase 1.
    """

    def __init__(self, communication_bus: Optional[Any] = None) -> None:
        self._communication_bus = communication_bus

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def predict(self, observation: np.ndarray) -> int:
        """Select an action given the current observation.

        Args:
            observation: Float32 numpy array of shape ``(obs_size,)``.

        Returns:
            Integer action in [0, num_actions).
        """
        ...

    @abstractmethod
    def train(self, total_timesteps: int, callbacks: Optional[Any] = None) -> None:
        """Train the agent for the specified number of timesteps.

        Args:
            total_timesteps: Total environment steps to train for.
            callbacks:       Optional SB3-compatible callback or callback list.
        """
        ...

    @abstractmethod
    def evaluate(self, n_episodes: int) -> Dict[str, float]:
        """Run deterministic evaluation for ``n_episodes`` and return metrics.

        Args:
            n_episodes: Number of episodes to evaluate.

        Returns:
            Dict with at minimum: ``avg_reward``, ``avg_wait``, ``throughput``.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the agent (policy weights, config) to disk.

        Args:
            path: File or directory path to save to.
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously saved agent from disk.

        Args:
            path: File or directory path to load from.
        """
        ...
