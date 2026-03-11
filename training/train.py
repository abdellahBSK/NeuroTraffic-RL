"""Main training script for NeuroTraffic-RL.

Trains a PPO agent on the Casablanca intersection (or any configured
intersection) and saves the best model for later evaluation.

Usage::

    # Minimal (uses defaults from configs/)
    python training/train.py

    # Custom intersection config
    python training/train.py --intersection configs/intersection_casablanca.yaml

    # Resume a previous run
    python training/train.py --resume models/best_model.zip

    # With SUMO GUI visualisation
    python training/train.py --gui

    # Override timesteps
    python training/train.py --timesteps 1000000
"""

import argparse
import sys
from pathlib import Path

import yaml

# Make project root importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.ppo_agent import PPOAgent
from env.sumo_env import SumoIntersectionEnv
from training.callbacks import BestModelCallback, MetricsCallback
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _build_env(
    intersection_cfg: dict,
    sim_cfg: dict,
    train_cfg: dict,
    use_gui: bool = False,
    label: str = "train",
) -> SumoIntersectionEnv:
    return SumoIntersectionEnv(
        intersection_cfg=intersection_cfg,
        sim_cfg=sim_cfg,
        train_cfg=train_cfg,
        use_gui=use_gui,
        label=label,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="NeuroTraffic-RL — Train a PPO agent on a SUMO intersection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--intersection",
        default="configs/intersection_casablanca.yaml",
        metavar="PATH",
        help="Path to the intersection config YAML.",
    )
    parser.add_argument(
        "--config",
        default="configs/training.yaml",
        metavar="PATH",
        help="Path to the training hyperparameters YAML.",
    )
    parser.add_argument(
        "--sim-config",
        default="configs/simulation.yaml",
        metavar="PATH",
        help="Path to the SUMO simulation config YAML.",
    )
    parser.add_argument(
        "--resume",
        metavar="PATH",
        default=None,
        help="Path to a .zip model checkpoint to resume training from.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        metavar="N",
        help="Total training timesteps (overrides training.yaml).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch SUMO with GUI (slower, useful for debugging).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for training."""
    args = _parse_args()

    logger.info("=" * 60)
    logger.info("NeuroTraffic-RL — Training started")
    logger.info("  Intersection config : %s", args.intersection)
    logger.info("  Training config     : %s", args.config)
    logger.info("  Simulation config   : %s", args.sim_config)
    logger.info("  Resume              : %s", args.resume or "No")
    logger.info("  SUMO GUI            : %s", args.gui)
    logger.info("=" * 60)

    # Load configs
    i_cfg = _load_yaml(args.intersection)
    t_cfg = _load_yaml(args.config)
    s_cfg = _load_yaml(args.sim_config)

    # Create environment
    env = _build_env(i_cfg, s_cfg, t_cfg, use_gui=args.gui, label="train")

    # Create agent
    agent = PPOAgent(env=env, config=t_cfg)

    # Optionally resume from checkpoint
    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        agent.load(args.resume)

    # Callbacks
    save_path = t_cfg.get("save_path", "models/")
    eval_freq = t_cfg.get("eval_freq", 10_000)

    callbacks = [
        MetricsCallback(log_freq=100),
        BestModelCallback(
            save_path=save_path,
            eval_freq=eval_freq,
            n_eval_episodes=t_cfg.get("n_eval_episodes", 5),
        ),
    ]

    # Train
    total_timesteps = args.timesteps or t_cfg.get("total_timesteps", 500_000)
    logger.info("Training for %d timesteps…", total_timesteps)

    try:
        agent.train(total_timesteps=total_timesteps, callbacks=callbacks)
    finally:
        env.close()

    # Save final model
    final_path = Path(save_path) / "final_model"
    agent.save(str(final_path))
    logger.info("Final model saved to %s.", final_path)

    # Quick evaluation report
    logger.info("Running quick post-training evaluation…")
    eval_env = _build_env(i_cfg, s_cfg, t_cfg, use_gui=False, label="eval_post")
    eval_agent = PPOAgent(env=eval_env, config=t_cfg)
    eval_agent.load(str(Path(save_path) / "best_model"))
    metrics = eval_agent.evaluate(n_episodes=t_cfg.get("n_eval_episodes", 5))
    eval_env.close()

    logger.info("=" * 60)
    logger.info("Post-training evaluation:")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)
    logger.info("=" * 60)
    logger.info("Training complete. Run `make eval` for full comparison.")


if __name__ == "__main__":
    main()
