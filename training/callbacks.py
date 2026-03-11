"""Custom SB3 callbacks for NeuroTraffic-RL.

Provides:
    - ``MetricsCallback``: Writes per-step and per-episode metrics to
      MetricsStore so the Streamlit dashboard stays up to date during training.
    - ``BestModelCallback``: Saves the model whenever a new best mean reward
      is achieved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCallback(BaseCallback):
    """Write training metrics to MetricsStore at each step.

    This callback is non-fatal: a MetricsStore failure never interrupts
    training.

    Args:
        log_freq:    Write to MetricsStore every ``log_freq`` steps.
        verbose:     SB3 verbosity level.
    """

    def __init__(self, log_freq: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._log_freq = log_freq
        self._metrics_store: Optional[Any] = None

    def _on_training_start(self) -> None:
        """Initialise MetricsStore connection."""
        try:
            from dashboard.metrics_store import MetricsStore
            self._metrics_store = MetricsStore()
            logger.debug("MetricsCallback connected to MetricsStore.")
        except Exception as exc:
            logger.warning("MetricsCallback: MetricsStore unavailable (%s).", exc)

    def _on_step(self) -> bool:
        """Log metrics every ``log_freq`` timesteps."""
        if self.n_calls % self._log_freq != 0:
            return True

        if self._metrics_store is None:
            return True

        try:
            infos = self.locals.get("infos", [{}])
            info = infos[0] if infos else {}
            rewards = self.locals.get("rewards", [0.0])
            reward = float(rewards[0]) if rewards else 0.0

            self._metrics_store.update(
                step=self.num_timesteps,
                total_waiting=float(info.get("total_waiting", 0.0)),
                throughput=int(info.get("throughput_delta", 0)),
                current_phase=int(info.get("current_phase", 0)),
                episode_reward=float(info.get("episode_reward", reward)),
                reward_breakdown=info.get("reward_breakdown", {}),
            )
        except Exception as exc:
            logger.debug("MetricsCallback write failed (non-fatal): %s", exc)

        return True


class BestModelCallback(BaseCallback):
    """Save the model whenever mean episode reward improves.

    Args:
        save_path:      Directory to save best model checkpoints.
        eval_freq:      Evaluation frequency in timesteps.
        n_eval_episodes: Number of episodes per evaluation.
        verbose:         SB3 verbosity level.
    """

    def __init__(
        self,
        save_path: str = "models/",
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self._save_path = Path(save_path)
        self._save_path.mkdir(parents=True, exist_ok=True)
        self._eval_freq = eval_freq
        self._n_eval_episodes = n_eval_episodes
        self._best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq != 0:
            return True

        # Access the mean episode reward logged by SB3
        ep_reward_mean = self.logger.name_to_value.get("rollout/ep_rew_mean")
        if ep_reward_mean is None:
            return True

        if ep_reward_mean > self._best_mean_reward:
            self._best_mean_reward = ep_reward_mean
            best_path = self._save_path / "best_model"
            self.model.save(str(best_path))
            if self.verbose >= 1:
                logger.info(
                    "New best model saved — mean_reward=%.4f → %s",
                    ep_reward_mean, best_path,
                )

        return True
