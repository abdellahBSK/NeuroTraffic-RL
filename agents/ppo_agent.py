"""PPO agent wrapper for NeuroTraffic-RL.

Wraps Stable-Baselines3 PPO with project-specific defaults, structured
logging, and a communication_bus hook for Phase 2 multi-agent extension.

Usage::

    import yaml
    from env import SumoIntersectionEnv
    from agents import PPOAgent

    with open("configs/training.yaml") as f:
        train_cfg = yaml.safe_load(f)

    env = SumoIntersectionEnv(icfg, scfg, train_cfg)
    agent = PPOAgent(env, config=train_cfg)
    agent.train(total_timesteps=500_000)
    agent.save("models/best_model")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class PPOAgent(BaseAgent):
    """Stable-Baselines3 PPO wrapped to the BaseAgent interface.

    Args:
        env:               Gymnasium-compatible environment.
        config:            Parsed training YAML dict (``configs/training.yaml``).
        communication_bus: Phase 2 message bus (currently a no-op, passed
                           through to BaseAgent).

    Example::

        agent = PPOAgent(env, config=train_cfg)
        agent.train(total_timesteps=500_000, callbacks=[my_callback])
        metrics = agent.evaluate(n_episodes=5)
        agent.save("models/ppo_casablanca")
    """

    def __init__(
        self,
        env: Any,
        config: Dict[str, Any],
        communication_bus: Optional[Any] = None,
    ) -> None:
        super().__init__(communication_bus=communication_bus)

        try:
            from stable_baselines3 import PPO as SB3PPO
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required. Install it with: "
                "pip install stable-baselines3"
            ) from exc

        self._env = env
        self._config = config
        hp = config.get("hyperparameters", {})
        log_path = config.get("tensorboard_log", "logs/tensorboard/")

        self._model = SB3PPO(
            policy=config.get("policy", "MlpPolicy"),
            env=env,
            learning_rate=hp.get("learning_rate", 3e-4),
            n_steps=hp.get("n_steps", 2048),
            batch_size=hp.get("batch_size", 64),
            n_epochs=hp.get("n_epochs", 10),
            gamma=hp.get("gamma", 0.99),
            gae_lambda=hp.get("gae_lambda", 0.95),
            clip_range=hp.get("clip_range", 0.2),
            ent_coef=hp.get("ent_coef", 0.01),
            vf_coef=hp.get("vf_coef", 0.5),
            max_grad_norm=hp.get("max_grad_norm", 0.5),
            verbose=config.get("verbose", 1),
            tensorboard_log=log_path,
            seed=config.get("seed", 42),
        )

        logger.info(
            "PPOAgent initialised — lr=%.2e, n_steps=%d, batch=%d",
            hp.get("learning_rate", 3e-4),
            hp.get("n_steps", 2048),
            hp.get("batch_size", 64),
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def predict(self, observation: np.ndarray) -> int:
        """Run deterministic policy inference.

        Args:
            observation: Current observation vector.

        Returns:
            Integer action selected by the policy.
        """
        action, _ = self._model.predict(observation, deterministic=True)
        return int(action)

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callbacks: Optional[Any] = None,
    ) -> None:
        """Train the PPO agent.

        Args:
            total_timesteps: Total env steps to train for.  Defaults to the
                             value in ``configs/training.yaml``.
            callbacks:       SB3-compatible callback or callback list.
        """
        steps = total_timesteps or self._config.get("total_timesteps", 500_000)
        logger.info("PPOAgent.train() starting — total_timesteps=%d.", steps)
        self._model.learn(
            total_timesteps=steps,
            callback=callbacks,
            reset_num_timesteps=False,
        )
        logger.info("PPOAgent.train() complete.")

    def evaluate(self, n_episodes: int = 5) -> Dict[str, float]:
        """Evaluate the policy deterministically for ``n_episodes``.

        Returns a summary metrics dict.  Full comparison with FixedCycle
        is handled by ``training/evaluate.py``.

        Args:
            n_episodes: Number of evaluation episodes.

        Returns:
            Dict with ``avg_reward``, ``std_reward``.
        """
        from stable_baselines3.common.evaluation import evaluate_policy

        mean_reward, std_reward = evaluate_policy(
            self._model,
            self._env,
            n_eval_episodes=n_episodes,
            deterministic=True,
        )
        metrics = {"avg_reward": float(mean_reward), "std_reward": float(std_reward)}
        logger.info(
            "Evaluation over %d episodes — avg_reward=%.3f ± %.3f",
            n_episodes, mean_reward, std_reward,
        )
        return metrics

    def save(self, path: str) -> None:
        """Save the PPO model to disk.

        Args:
            path: File path (without .zip extension) or directory.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(save_path))
        logger.info("PPOAgent saved to %s.", save_path)

    def load(self, path: str) -> None:
        """Load a saved PPO model from disk.

        Args:
            path: File path to the saved .zip model.
        """
        from stable_baselines3 import PPO as SB3PPO

        self._model = SB3PPO.load(path, env=self._env)
        logger.info("PPOAgent loaded from %s.", path)
