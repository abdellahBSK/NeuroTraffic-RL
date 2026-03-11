<div align="center">

# 🚦 NeuroTraffic-RL

**Research-Grade Intelligent Traffic Signal Control Systems via Deep Reinforcement Learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![SUMO 1.18.0+](https://img.shields.io/badge/SUMO-1.18.0+-green.svg)](https://sumo.dlr.de/docs/Downloads.php)
[![Stable Baselines3](https://img.shields.io/badge/RL-Stable%20Baselines3-purple.svg)](https://stable-baselines3.readthedocs.io/en/master/)
[![Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-orange.svg)](https://gymnasium.farama.org/)

</div>---

NeuroTraffic-RL is a clean, robust, and extensible reinforcement learning pipeline designed to optimize traffic signal control. Built on top of the world-class SUMO traffic simulator, it allows AI agents to learn optimal control policies to minimize congestion, reduce waiting times, and maximize throughput.

Currently in **Phase 1**, the system implements a highly optimized Proximal Policy Optimization (PPO) agent controlling a single complex 4-way intersection (Casablanca), beating fixed-cycle baselines.

---

## ✨ Key Features (Phase 1)

*   **Clean Architecture:** Strict separation of concerns. SUMO logic, Gym environments, RL agents, and visualizations are entirely decoupled.
*   **State-of-the-Art RL:** Native integration with Gymnasium and Stable-Baselines3 (PPO).
*   **Intelligent State Representation:** 26-dimensional feature vector per intersection combining queue lengths, densities, wait times, and temporal data.
*   **Custom Reward Shaping:** Configurable 4-component reward function balancing wait times, throughput, starvation, and phase-oscillation penalties.
*   **Real-time Dashboard:** A live Streamlit dashboard tracking KPIs (Queue Lengths, Waiting Time, Phases) during training and evaluation.
*   **Headless & GUI Modes:** Train blindingly fast in the background, or watch the agent learn in the SUMO GUI.

---

## 🚀 Quick Start

Get up and running with NeuroTraffic-RL in under 5 minutes.

### 1. Prerequisites

You need Python 3.10+ and the SUMO Traffic Simulator installed.

**Install SUMO (Ubuntu/Debian):**
```bash
bash scripts/setup_sumo.sh
source ~/.bashrc
```
*(For MacOS/Windows, see the [official SUMO installation guide](https://sumo.dlr.de/docs/Installing.html))*

**Install Python Dependencies:**
```bash
git clone https://github.com/abdellahBSK/NeuroTraffic-RL.git
cd NeuroTraffic-RL
make setup
```

### 2. Configure Environment Variables
Copy the example environment file.
```bash
cp .env.example .env
```
*(Ensure `SUMO_HOME` is correctly set in your `.env` or system path if the setup script didn't catch it).*

### 3. Training the Agent

Train the PPO agent headless (fastest):
```bash
make train
```

Train while watching the agent in the SUMO GUI:
```bash
make train-gui
```

### 4. Real-time Monitoring

To watch the live metrics (queue lengths, rewards, phase allocations) while the agent trains, open a new terminal tab and run:
```bash
make dashboard
```
This will launch a Streamlit dashboard at `http://localhost:8501`.

### 5. Evaluate Performance

Compare your trained PPO agent against the baseline Fixed-Cycle traffic lights:
```bash
make eval
```
This will output a statistical comparison report and save it to the `logs/` directory.

---

## 🧠 System Architecture

NeuroTraffic-RL is built for extensibility:

1.  **Environment (`env/`)**: Translates messy SUMO/TraCI data into clean Numpy arrays. `ObservationBuilder` handles the 26-feature state, `RewardCalculator` computes the complex 4-part reward, and `PhaseManager` safely handles yellow-light transitions.
2.  **Agents (`agents/`)**: Wraps RL implementations. Phase 1 provides `PPOAgent` and `FixedCycleAgent`.
3.  **Configs (`configs/`)**: 100% parameter-driven. Change intersection layouts, max phase times, or reward weights without touching Python code.
4.  **Dashboard (`dashboard/`)**: Decoupled, asynchronous Streamlit UI reading from a shared metric store.

For a deep dive into the math and architecture, see [docs/architecture.md](docs/architecture.md) and [docs/environment.md](docs/environment.md).

---

## 🛠️ Makefile Commands Reference

| Command | Description |
| :--- | :--- |
| `make setup` | Install Python dependencies. |
| `make train` | Run PPO training headless. |
| `make train-gui` | Run PPO training with SUMO GUI. |
| `make eval` | Evaluate PPO agent vs FixedCycle baseline. |
| `make dashboard` | Launch the real-time Streamlit dashboard (`localhost:8501`). |
| `make test` | Run the Python unit test suite (no SUMO required). |
| `make clean` | Expunge generated models, logs, and caches. |

---

## 🗺️ Roadmap

- [x] **Phase 1: Single Intersection.** Robust PPO control, Gym environment, Baseline comparisons.
- [ ] **Phase 2: Multi-Intersection (In Progress).** Inter-agent communication via Message Bus (Redis scaffolding ready).
- [ ] **Phase 3: Graph Neural Networks.** Implement GNNs to process arbitrary grid configurations.
- [ ] **Phase 4: Real-world OSM Import.** Direct import pipeline from OpenStreetMap.

See [docs/advancement.md](docs/advancement.md) for the current development standing and immediate next tasks.

---

*NeuroTraffic-RL — Because stopping at red lights is a sub-optimal policy.*