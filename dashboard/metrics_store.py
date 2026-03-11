"""Shared metrics state for the NeuroTraffic-RL dashboard.

``MetricsStore`` provides a file-backed (JSON) key-value store that the
training process writes to and the Streamlit dashboard reads from.
The API is designed to be drop-in replaceable with a Redis backend
for production deployments (Phase 2+).

Design:
    - Writer (training loop):  calls ``update()`` at every log step.
    - Reader (Streamlit):      calls ``read()`` on auto-refresh.
    - No shared in-process state: file on disk = single source of truth.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEFAULT_PATH = Path(os.getenv("METRICS_FILE", "logs/metrics.json"))
_HISTORY_SIZE = 300   # rolling window kept in file


class MetricsStore:
    """File-backed store for live training metrics.

    Thread-safe via a file-level lock (``threading.Lock``).

    Args:
        path: Path to the JSON metrics file. Defaults to ``logs/metrics.json``
              or the ``METRICS_FILE`` environment variable.

    Example (writer side, training loop)::

        store = MetricsStore()
        store.update(step=1200, total_waiting=45.3, throughput=7, ...)

    Example (reader side, dashboard)::

        store = MetricsStore()
        data = store.read()
        print(data["history"][-1]["total_waiting"])
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else _DEFAULT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def update(
        self,
        step: int,
        total_waiting: float = 0.0,
        throughput: int = 0,
        current_phase: int = 0,
        episode_reward: float = 0.0,
        reward_breakdown: Optional[Dict[str, float]] = None,
    ) -> None:
        """Append a new metrics snapshot to the store.

        Args:
            step:             Current simulation step (int).
            total_waiting:    Total cumulative waiting time this step (s).
            throughput:       Vehicles that arrived at their destination.
            current_phase:    Active traffic light phase ID (0–3).
            episode_reward:   Cumulative episode reward so far.
            reward_breakdown: Per-component reward dict from RewardCalculator.
        """
        snapshot = {
            "ts": time.time(),
            "step": step,
            "total_waiting": round(total_waiting, 3),
            "throughput": throughput,
            "current_phase": current_phase,
            "episode_reward": round(episode_reward, 4),
            "reward_breakdown": reward_breakdown or {},
        }

        with self._lock:
            data = self._load_safe()
            history: List[dict] = data.get("history", [])
            history.append(snapshot)
            # Keep only the most recent snapshots
            history = history[-_HISTORY_SIZE:]
            data["history"] = history
            data["latest"] = snapshot
            self._save(data)

    def reset_episode(self) -> None:
        """Clear history and reset the store for a new episode."""
        with self._lock:
            self._save({"history": [], "latest": {}})

    # ------------------------------------------------------------------
    # Read API (dashboard side)
    # ------------------------------------------------------------------

    def read(self) -> Dict[str, Any]:
        """Return the full metrics store contents.

        Returns:
            Dict with keys ``history`` (list of snapshots) and ``latest``
            (most recent snapshot dict).  Returns empty structure on error.
        """
        with self._lock:
            return self._load_safe()

    def get_history(self, key: str) -> List[float]:
        """Return a list of values for ``key`` across all history snapshots.

        Args:
            key: Metric key (e.g. ``"total_waiting"``, ``"episode_reward"``).

        Returns:
            Ordered list of float values, oldest first.
        """
        data = self.read()
        return [float(snap.get(key, 0.0)) for snap in data.get("history", [])]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_safe(self) -> Dict[str, Any]:
        """Load JSON from disk, returning empty structure on any error."""
        try:
            if self._path.exists():
                with open(self._path, "r") as fh:
                    return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
        return {"history": [], "latest": {}}

    def _save(self, data: Dict[str, Any]) -> None:
        """Atomically write data to disk."""
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2)
        tmp.replace(self._path)
