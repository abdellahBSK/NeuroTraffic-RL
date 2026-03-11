"""TraCI helper utilities for NeuroTraffic-RL.

Provides safe wrappers around raw TraCI calls and SUMO environment
detection utilities. This is the *only* place that should directly
reference ``traci`` — all other modules call these helpers.
"""

import os
import shutil
import socket
import sys
from contextlib import closing
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SUMO environment setup
# ---------------------------------------------------------------------------

def setup_sumo_path() -> bool:
    """Ensure the SUMO tools directory is on ``sys.path``.

    Reads ``SUMO_HOME`` from the environment, falling back to common Linux
    installation paths.

    Returns:
        True if SUMO tools were successfully added to ``sys.path``.
    """
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        tools_path = os.path.join(sumo_home, "tools")
        if tools_path not in sys.path:
            sys.path.append(tools_path)
        logger.debug("SUMO tools path set from SUMO_HOME: %s", tools_path)
        return True

    # Common fallback paths
    fallbacks = [
        "/usr/share/sumo/tools",
        "/usr/local/share/sumo/tools",
        "/opt/sumo/tools",
    ]
    for path in fallbacks:
        if os.path.isdir(path):
            if path not in sys.path:
                sys.path.append(path)
            logger.debug("SUMO tools path set via fallback: %s", path)
            return True

    logger.warning(
        "SUMO_HOME not set and no fallback SUMO installation found. "
        "Set the SUMO_HOME environment variable."
    )
    return False


def find_sumo_binary(preferred: str = "sumo") -> str:
    """Locate the SUMO executable on the system PATH.

    Args:
        preferred: Binary name to search for (``"sumo"`` or ``"sumo-gui"``).

    Returns:
        Absolute path to the SUMO binary.

    Raises:
        FileNotFoundError: If the binary cannot be found.
    """
    path = shutil.which(preferred)
    if path:
        logger.debug("Found SUMO binary: %s", path)
        return path

    # Try SUMO_HOME-relative path
    sumo_home = os.environ.get("SUMO_HOME", "")
    candidate = os.path.join(sumo_home, "bin", preferred)
    if os.path.isfile(candidate):
        logger.debug("Found SUMO binary via SUMO_HOME: %s", candidate)
        return candidate

    raise FileNotFoundError(
        f"Could not find SUMO binary '{preferred}'. "
        "Install SUMO and set the SUMO_HOME environment variable."
    )


# ---------------------------------------------------------------------------
# Port management for parallel TraCI connections
# ---------------------------------------------------------------------------

def find_free_port(start: int = 8813, end: int = 8899) -> int:
    """Find an available TCP port in the given range.

    Args:
        start: First port to check (inclusive).
        end:   Last port to check (inclusive).

    Returns:
        An available port number.

    Raises:
        RuntimeError: If no free port is found in the range.
    """
    for port in range(start, end + 1):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            result = sock.connect_ex(("127.0.0.1", port))
            if result != 0:  # Port is not in use
                logger.debug("Found free port: %d", port)
                return port

    raise RuntimeError(
        f"No free TCP port found in range [{start}, {end}]. "
        "Reduce the number of parallel environments."
    )


# ---------------------------------------------------------------------------
# Safe TraCI data accessors
# ---------------------------------------------------------------------------

def safe_get_halting_number(traci_module, lane_id: str) -> int:
    """Return the number of halted vehicles on ``lane_id``, defaulting to 0.

    Args:
        traci_module: The active traci (or libsumo) module.
        lane_id: SUMO lane identifier.

    Returns:
        Number of halted vehicles, or 0 on error.
    """
    try:
        return int(traci_module.lane.getLastStepHaltingNumber(lane_id))
    except Exception as exc:
        logger.debug("safe_get_halting_number(%s) failed: %s", lane_id, exc)
        return 0


def safe_get_mean_speed(traci_module, lane_id: str) -> float:
    """Return mean vehicle speed on ``lane_id`` in m/s, defaulting to 0.0.

    Args:
        traci_module: The active traci (or libsumo) module.
        lane_id: SUMO lane identifier.

    Returns:
        Mean speed in m/s, or 0.0 on error.
    """
    try:
        return float(traci_module.lane.getLastStepMeanSpeed(lane_id))
    except Exception as exc:
        logger.debug("safe_get_mean_speed(%s) failed: %s", lane_id, exc)
        return 0.0


def safe_get_occupancy(traci_module, lane_id: str) -> float:
    """Return lane occupancy ratio [0, 1], defaulting to 0.0.

    Args:
        traci_module: The active traci (or libsumo) module.
        lane_id: SUMO lane identifier.

    Returns:
        Occupancy ratio in [0, 1], or 0.0 on error.
    """
    try:
        return float(traci_module.lane.getLastStepOccupancy(lane_id)) / 100.0
    except Exception as exc:
        logger.debug("safe_get_occupancy(%s) failed: %s", lane_id, exc)
        return 0.0


def safe_get_waiting_sum(traci_module, lane_id: str) -> float:
    """Return cumulative waiting time on ``lane_id`` in seconds.

    Args:
        traci_module: The active traci (or libsumo) module.
        lane_id: SUMO lane identifier.

    Returns:
        Cumulative waiting time in seconds, or 0.0 on error.
    """
    try:
        return float(traci_module.lane.getWaitingSum(lane_id))
    except Exception as exc:
        logger.debug("safe_get_waiting_sum(%s) failed: %s", lane_id, exc)
        return 0.0


def safe_get_vehicle_count(traci_module, lane_id: str) -> int:
    """Return the number of vehicles currently on ``lane_id``.

    Args:
        traci_module: The active traci (or libsumo) module.
        lane_id: SUMO lane identifier.

    Returns:
        Vehicle count, or 0 on error.
    """
    try:
        return int(traci_module.lane.getLastStepVehicleNumber(lane_id))
    except Exception as exc:
        logger.debug("safe_get_vehicle_count(%s) failed: %s", lane_id, exc)
        return 0


def get_lane_metrics(
    traci_module,
    lane_ids: List[str],
    lane_length: float = 200.0,
) -> Dict[str, Dict[str, float]]:
    """Batch-collect key metrics for all lanes in a single call.

    Args:
        traci_module: The active traci (or libsumo) module.
        lane_ids:     List of SUMO lane IDs to query.
        lane_length:  Nominal lane length in metres (for density computation).

    Returns:
        Mapping of ``lane_id → {queue, speed, occupancy, waiting_time, density}``.
    """
    metrics: Dict[str, Dict[str, float]] = {}
    for lane_id in lane_ids:
        queue = safe_get_halting_number(traci_module, lane_id)
        speed = safe_get_mean_speed(traci_module, lane_id)
        occupancy = safe_get_occupancy(traci_module, lane_id)
        waiting = safe_get_waiting_sum(traci_module, lane_id)
        count = safe_get_vehicle_count(traci_module, lane_id)
        density = count / lane_length if lane_length > 0 else 0.0

        metrics[lane_id] = {
            "queue": float(queue),
            "speed": speed,
            "occupancy": occupancy,
            "waiting_time": waiting,
            "density": density,
        }
    return metrics
