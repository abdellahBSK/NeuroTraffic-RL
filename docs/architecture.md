# NeuroTraffic-RL System Architecture

## Overview

NeuroTraffic-RL is a reinforcement learning pipeline for intelligent traffic signal control at a single 4-way intersection. The system learns to minimise vehicle waiting time and maximise throughput, beating the fixed-cycle baseline.

## Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Training Loop                                │
│                                                                      │
│   configs/         ──▶  SumoIntersectionEnv  ◀──────────────────┐  │
│   (YAML)                       │                                 │  │
│                                │ step(action)                    │  │
│                          ┌─────▼──────┐                          │  │
│   SUMO process  ◀──TraCI──┤ObsBuilder  │──▶ obs(26,) ──▶ PPOAgent│  │
│   (subprocess)  ──TraCI──▶│PhaseManager│                          │  │
│                          │RewardCalc  │──▶ reward ──▶ SB3 PPO   │  │
│                          └─────┬──────┘                          │  │
│                                │ emit_metrics()                   │  │
│                          MetricsStore (JSON)                      │  │
│                                │                                 │  │
│                         Streamlit Dashboard ◀────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

| Module | Layer | Responsibility |
|---|---|---|
| `utils/logger.py` | Utils | Structured logging (all modules) |
| `utils/traci_utils.py` | Utils | Safe TraCI accessors + port management |
| `utils/metrics.py` | Utils | KPI computation (p95, throughput rate) |
| `env/base_env.py` | Env | Abstract Gymnasium base + Phase 2 bus hook |
| `env/observation_builder.py` | Env | 26-feature state vector from TraCI data |
| `env/reward_calculator.py` | Env | 4-component reward function |
| `env/phase_manager.py` | Env | Phase transitions + YELLOW injection |
| `env/sumo_env.py` | Env | SumoIntersectionEnv (full Gym contract) |
| `agents/base_agent.py` | Agents | Abstract agent interface |
| `agents/ppo_agent.py` | Agents | SB3 PPO wrapper |
| `agents/fixed_cycle_agent.py` | Agents | Fixed-time baseline |
| `communication/message_bus.py` | Comms | Phase 2 scaffold (no-op in Phase 1) |
| `training/train.py` | Training | CLI training script |
| `training/evaluate.py` | Training | PPO vs baseline comparison |
| `training/callbacks.py` | Training | SB3 callbacks |
| `dashboard/` | Dashboard | Real-time Streamlit KPI display |

## State Space (Formal Specification)

**Vector dimension:** $d = 4 \times 5 + 6 = 26$

### Per-arm features (4 arms × 5 = 20)

| Index | Feature | Formula | Range |
|---|---|---|---|
| $5k+0$ | queue length | $\sum_{\ell \in arm_k} \text{halting}(\ell)$ | $[0, \infty)$ |
| $5k+1$ | avg speed | $\bar{v}_k \;/\; v_{max}$ | $[0, 1]$ |
| $5k+2$ | density | $\bar{\rho}_k \;/\; \rho_{max}$ | $[0, 1]$ |
| $5k+3$ | waiting time | $\min(W_k, W_{cap}) \;/\; W_{cap}$ | $[0, 1]$ |
| $5k+4$ | phase elapsed | $t_{elapsed} \;/\; t_{max}$ | $[0, 1]$ |

### Global features (6)

| Index | Feature | Formula |
|---|---|---|
| 20..23 | phase one-hot | $\mathbf{e}_{p}$ where $p \in \{0,1,2,3\}$ |
| 24 | time sin | $\sin(2\pi t / 86400)$ |
| 25 | time cos | $\cos(2\pi t / 86400)$ |

## Action Space

$\mathcal{A} = \text{Discrete}(4)$: $\{0=NS\_GREEN,\; 1=NS\_YELLOW,\; 2=EW\_GREEN,\; 3=EW\_YELLOW\}$

> **Agent constraint:** The agent may only select phases 0 or 2. Yellow phases (1, 3) are automatically injected by `PhaseManager` during green→green transitions. Selecting phase 1 or 3 raises `ValueError`.

## Reward Function

$$r_t = -\alpha \cdot \bar{W}_t + \beta \cdot \Delta T_t - \gamma \cdot Q_{max,t} - \delta \cdot \mathbb{1}[\text{unnecessary switch}]$$

| Term | Symbol | Default | Meaning |
|---|---|---|---|
| Waiting penalty | $\alpha$ | 1.0 | Normalised total wait per step |
| Throughput bonus | $\beta$ | 0.5 | Vehicles completing trips |
| Starvation penalty | $\gamma$ | 0.3 | Max per-lane queue (normalised) |
| Switch penalty | $\delta$ | 0.1 | Green→Green phase oscillation |

## Adding a New Intersection

1. Create `configs/intersection_<name>.yaml` (copy from `intersection_casablanca.yaml`)
2. Create `sumo/networks/<name>.net.xml` and `<name>.rou.xml`
3. Run: `python training/train.py --intersection configs/intersection_<name>.yaml`
4. No Python code changes required.

## Phase 2 Extension Points

| Hook | Location | Phase 2 action |
|---|---|---|
| `connect_communication_bus(bus)` | `BaseTrafficEnv` | Pass `RedisMessageBus()` |
| `communication_bus` param | `PPOAgent.__init__` | Subscribe to neighbour observations |
| `NoOpMessageBus` | `communication/message_bus.py` | Replace with real implementation |
