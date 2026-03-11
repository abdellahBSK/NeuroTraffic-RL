"""Evaluation script for NeuroTraffic-RL.

Compares a trained PPOAgent against the FixedCycleAgent baseline over
multiple episodes and saves a JSON report plus a formatted table.

Usage::

    python training/evaluate.py --model models/best_model.zip
    python training/evaluate.py --model models/best_model.zip --episodes 10
    python training/evaluate.py --help
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.fixed_cycle_agent import FixedCycleAgent
from agents.ppo_agent import PPOAgent
from env.sumo_env import SumoIntersectionEnv
from utils.logger import get_logger
from utils.metrics import compute_p95_wait, compute_throughput_rate, format_comparison_table

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _run_episode(
    env: SumoIntersectionEnv,
    agent,
    fixed_cycle: bool = False,
) -> Dict[str, float]:
    """Run one evaluation episode and return episode metrics.

    Args:
        env:         Environment instance (already reset externally).
        agent:       Agent to evaluate (PPOAgent or FixedCycleAgent).
        fixed_cycle: If True, call ``agent.reset()`` before the episode.

    Returns:
        Dict with keys: total_reward, avg_queue, total_waiting, throughput,
        n_steps, p95_wait.
    """
    if fixed_cycle and hasattr(agent, "reset"):
        agent.reset()

    obs, _ = env.reset()
    done = False

    total_reward = 0.0
    waiting_snapshots: List[float] = []
    queue_snapshots: List[float] = []
    total_throughput = 0
    step_count = 0

    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += float(reward)
        waiting_snapshots.append(float(info.get("total_waiting", 0.0)))
        queue_snapshots.append(float(info.get("total_waiting", 0.0)))  # proxy
        total_throughput += int(info.get("throughput_delta", 0))
        step_count += 1

    sim_duration = float(step_count * env._intersection_cfg.get("step_seconds", 5))

    return {
        "total_reward": total_reward,
        "avg_wait": float(np.mean(waiting_snapshots)) if waiting_snapshots else 0.0,
        "max_wait": float(np.max(waiting_snapshots)) if waiting_snapshots else 0.0,
        "p95_wait": compute_p95_wait(waiting_snapshots),
        "avg_queue": float(np.mean(queue_snapshots)) if queue_snapshots else 0.0,
        "throughput": total_throughput,
        "throughput_rate": compute_throughput_rate(total_throughput, sim_duration),
        "n_steps": step_count,
    }


def _aggregate(episodes: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate episode metrics via mean across episodes.

    Args:
        episodes: List of per-episode metric dicts.

    Returns:
        Dict of mean metric values.
    """
    if not episodes:
        return {}
    keys = episodes[0].keys()
    return {k: float(np.mean([ep[k] for ep in episodes])) for k in keys}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="NeuroTraffic-RL — Evaluate PPO agent vs FixedCycle baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="models/best_model.zip",
        metavar="PATH",
        help="Path to the trained PPO model (.zip).",
    )
    parser.add_argument(
        "--intersection",
        default="configs/intersection_casablanca.yaml",
        metavar="PATH",
        help="Intersection configuration YAML.",
    )
    parser.add_argument(
        "--config",
        default="configs/training.yaml",
        metavar="PATH",
        help="Training configuration YAML (for reward coefficients etc.).",
    )
    parser.add_argument(
        "--sim-config",
        default="configs/simulation.yaml",
        metavar="PATH",
        help="Simulation configuration YAML.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        metavar="N",
        help="Number of evaluation episodes per agent.",
    )
    parser.add_argument(
        "--report-dir",
        default="logs/",
        metavar="DIR",
        help="Directory to save the JSON evaluation report.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for evaluation."""
    args = _parse_args()

    logger.info("=" * 60)
    logger.info("NeuroTraffic-RL — Evaluation")
    logger.info("  Model      : %s", args.model)
    logger.info("  Episodes   : %d per agent", args.episodes)
    logger.info("=" * 60)

    i_cfg = _load_yaml(args.intersection)
    t_cfg = _load_yaml(args.config)
    s_cfg = _load_yaml(args.sim_config)

    # ------------------------------------------------------------------
    # Evaluate FixedCycle baseline
    # ------------------------------------------------------------------
    logger.info("Evaluating FixedCycle baseline…")
    baseline_agent = FixedCycleAgent(
        green_duration=int(
            s_cfg.get("baseline", {}).get("green_duration", 30)
            / i_cfg.get("step_seconds", 5)
        )
    )
    baseline_episodes = []
    for ep in range(1, args.episodes + 1):
        env = SumoIntersectionEnv(i_cfg, s_cfg, t_cfg, use_gui=False, label=f"baseline_{ep}")
        metrics = _run_episode(env, baseline_agent, fixed_cycle=True)
        env.close()
        baseline_episodes.append(metrics)
        logger.info("  FixedCycle ep %d: reward=%.2f, avg_wait=%.1fs",
                    ep, metrics["total_reward"], metrics["avg_wait"])

    baseline_agg = _aggregate(baseline_episodes)

    # ------------------------------------------------------------------
    # Evaluate PPO agent
    # ------------------------------------------------------------------
    ppo_episodes = []
    if Path(args.model).exists():
        logger.info("Evaluating PPO agent from %s…", args.model)
        for ep in range(1, args.episodes + 1):
            env = SumoIntersectionEnv(i_cfg, s_cfg, t_cfg, use_gui=False, label=f"ppo_{ep}")
            ppo_agent = PPOAgent(env=env, config=t_cfg)
            ppo_agent.load(args.model)
            metrics = _run_episode(env, ppo_agent, fixed_cycle=False)
            env.close()
            ppo_episodes.append(metrics)
            logger.info("  PPO ep %d: reward=%.2f, avg_wait=%.1fs",
                        ep, metrics["total_reward"], metrics["avg_wait"])
    else:
        logger.warning("PPO model %s not found. Skipping PPO evaluation.", args.model)

    ppo_agg = _aggregate(ppo_episodes)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    logger.info("\n" + format_comparison_table(ppo_agg, baseline_agg))

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": args.model,
        "n_episodes": args.episodes,
        "ppo": ppo_agg,
        "baseline": baseline_agg,
    }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"evaluation_{timestamp}.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    logger.info("Evaluation report saved to %s", report_path)


if __name__ == "__main__":
    main()
