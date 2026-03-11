# NeuroTraffic-RL Environment Specification

## State Space

**observation_space:** `Box(26,)` float32

See `env/observation_builder.py` for the full implementation.

### Feature Layout

```
Indices 0–19 : Per-arm features (4 arms × 5 features)
    [arm*5 + 0] queue_length   : total halted vehicles (raw int, not normalised)
    [arm*5 + 1] avg_speed      : mean speed ÷ max_speed → [0, 1]
    [arm*5 + 2] density        : veh/m ÷ max_density → [0, 1]
    [arm*5 + 3] waiting_time   : total wait (s) / max_wait_cap → [0, 1]
    [arm*5 + 4] phase_elapsed  : seconds in phase / max_phase_dur → [0, 1]

Indices 20–25 : Global features
    [20..23]    current_phase_one_hot : 4-dim one-hot
    [24]        time_of_day_sin
    [25]        time_of_day_cos
```

### Arm Order
0 = North, 1 = South, 2 = East, 3 = West  
(matches `observation_arms` list in `configs/intersection_casablanca.yaml`)

---

## Action Space

**action_space:** `Discrete(4)`

| Action | Name | Description |
|---|---|---|
| 0 | `NS_GREEN` | North-South gets green |
| 1 | `NS_YELLOW` | ⚠️ Agent must NOT select — auto-injected |
| 2 | `EW_GREEN` | East-West gets green |
| 3 | `EW_YELLOW` | ⚠️ Agent must NOT select — auto-injected |

`PhaseManager.get_valid_actions()` returns `[0, 2]` when not in a yellow transition.

---

## Reward Function

```
r = -α · norm_total_wait
  + β · throughput_delta
  - γ · norm_max_queue
  - δ · phase_change_penalty
```

| Coefficient | Default | YAML key |
|---|---|---|
| α (wait weight) | 1.0 | `reward.alpha` |
| β (throughput) | 0.5 | `reward.beta` |
| γ (starvation) | 0.3 | `reward.gamma` |
| δ (oscillation) | 0.1 | `reward.delta` |

**Normalisation caps** (from `configs/training.yaml`):
- `max_wait_cap`: 300 s
- `max_queue_cap`: 20 vehicles

---

## Episode Termination

- `terminated`: Always `False` (no hard terminal state).
- `truncated`: `True` when `step_count >= max_episode_seconds` (default: 3600 s).

---

## Info Dict (step output)

| Key | Type | Description |
|---|---|---|
| `step` | int | Simulation second |
| `throughput_delta` | int | Vehicles arrived this step |
| `total_waiting` | float | Sum of lane waiting times (s) |
| `episode_reward` | float | Cumulative episode reward |
| `reward_breakdown` | dict | Per-component reward (for debugging) |
| `current_phase` | int | Active TL phase (0–3) |
