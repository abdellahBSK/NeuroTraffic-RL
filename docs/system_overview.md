# NeuroTraffic-RL: System Architecture & Documentation

## 1. Introduction
NeuroTraffic-RL is an intelligent traffic signal control system designed to optimize traffic flow and alleviate congestion at urban intersections using Reinforcement Learning (RL). Traditional localized traffic light systems rely on fixed-time phases or simple induction loop triggers, leading to suboptimal throughput during varying traffic conditions. NeuroTraffic-RL solves this by dynamically adapting traffic light phases in real-time. Reinforcement Learning is employed because it excels at sequential decision-making in complex environments, allowing the system to learn optimal control policies from interaction rather than relying on hardcoded heuristics.

## 2. System Overview
The architecture of NeuroTraffic-RL follows a Clean Architecture approach to decouple the Reinforcement Learning agent from the traffic simulation platform.
- **Agent**: The RL decision-maker (PPO) that selects traffic light phases based on current observations.
- **Environment**: A custom Gym-like interface mapping RL concepts (observations, actions, rewards) into traffic simulation concepts.
- **SUMO (Simulation of Urban MObility)**: The underlying microscopic traffic simulator that models the physical vehicles, roads, and traffic lights.

### Data Flow
1. **SUMO** simulates a traffic step.
2. **Environment** extracts raw state data (e.g., vehicle positions, speeds) from SUMO via TraCI.
3. **Environment** processes raw state data into a normalized **Observation vector** and calculates the **Reward** from the previous step.
4. **Agent** receives the Observation and computes an **Action** (the next traffic light phase).
5. **Environment** translates the Action into a TraCI command to update SUMO's traffic lights.

```text
+-------------------+                 +-------------------+                 +-------------------+
|                   |   Observation   |                   |    Raw State    |                   |
|     RL Agent      | <-------------- |    Environment    | <-------------- |  SUMO Simulator   |
|  (PPO algorithm)  |     Reward      |  (Gym Interface)  |                 |                   |
|                   | --------------> |                   | --------------> |                   |
+-------------------+      Action     +-------------------+ TraCI Commands  +-------------------+
```

## 3. Environment Description
The environment is a custom `gym.Env`-like implementation designed to standardize the interaction between the agent and SUMO.

### Observation Space
The observation space is a continuous vector composed of real-time traffic metrics extracted from SUMO. It includes:
- `queue_length`: The number of vehicles halted at the intersection lanes.
- `avg_speed`: The average speed of vehicles approaching the intersection.
- `density`: The percentage of road space occupied by vehicles.
- `waiting_time`: The cumulative time vehicles have spent stopped.
- `phase_elapsed`: The time elapsed since the current traffic light phase started.

**Normalization:** All observations are normalized to a `[0, 1]` or `[-1, 1]` range (e.g., dividing `queue_length` by the maximum lane capacity). This ensures stable and efficient Neural Network training without dominating scale discrepancies.

### Action Space
The action space is `Discrete(4)`, representing the selection of the next traffic light phase.
- `0`: `NS_GREEN` (North-South goes straight/right)
- `1`: `NS_YELLOW` (North-South warning before transition)
- `2`: `EW_GREEN` (East-West goes straight/right)
- `3`: `EW_YELLOW` (East-West warning before transition)

### Reward Function
The reward function is formulated to minimize congestion and maximize throughput. At each step, the reward $R_t$ is computed based on:
- **Waiting time penalty:** Negative reward proportional to the increase in total waiting time across all vehicles.
- **Throughput bonus:** Positive reward for vehicles successfully passing the intersection.
- **Switching penalty:** A small negative penalty applied when switching phases to avoid rapid oscillations (flickering).

## 4. SUMO Integration (TraCI)
**SUMO** is an open-source, highly portable, microscopic and continuous road traffic simulation package. 
**TraCI** (Traffic Control Interface) is a Python API that allows external scripts to interact with a running SUMO simulation in real-time.

The agent interacts with SUMO by:
1. **Starting the simulation:** Launching the `sumo` or `sumo-gui` binary with the network and route files.
2. **Stepping the simulation:** Advancing the simulation by a fixed time step using `traci.simulationStep()`.
3. **Extracting data:** Querying sensors and metrics via functions like `traci.lane.getLastStepVehicleNumber()` and `traci.lane.getWaitingTime()`.

**TraCI Usage Example:**
```python
import traci

# Connect to SUMO
traci.start(["sumo", "-c", "casablanca_intersection.sumocfg"])

while traci.simulation.getMinExpectedNumber() > 0:
    # Get current queue length for a lane
    queue = traci.lane.getLastStepHaltingNumber("lane_N2S_0")
    
    # Change traffic light phase
    traci.trafficlight.setPhase("J0", 2) # Switch to phase 2 (EW_GREEN)
    
    # Step the simulation
    traci.simulationStep()

traci.close()
```

## 5. Reinforcement Learning Agent (PPO)

### 5.1 Why PPO
We use Proximal Policy Optimization (PPO) because it strikes a strong balance between sample efficiency, stability, and ease of tuning. Unlike DQN, which requires managing target networks and replay buffers for discrete spaces, PPO directly optimizes the policy while using a clipping mechanism to prevent destructively large, unbounded policy updates.

### 5.2 PPO Algorithm Explanation
PPO maintains two main components:
- **Policy ($\pi_\theta$)**: Determines the probability of taking an action given the state.
- **Value Function ($V_\phi$)**: Estimates the expected return (cumulative discounted reward) from a given state.

PPO uses Generalized Advantage Estimation (GAE) to compute the **Advantage ($A_t$)**, which evaluates how much better an action was compared to the average expected value. 
To prevent destabilizing updates, PPO defines a **Clipped Objective Function**:

$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right] $$

Where $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ is the probability ratio between the new and old policy.

### 5.3 Agent Architecture
The architecture comprises a feed-forward Neural Network using Multi-Layer Perceptrons (MLPs).
- **Input**: The normalized continuous observation vector.
- **Hidden Layers**: Two fully connected layers (e.g., 64 or 128 neurons each) with ReLU or Tanh activations.
- **Outputs**:
  - *Actor Head (Policy)*: Outputs logits for the 4 discrete actions.
  - *Critic Head (Value)*: Outputs a single scalar evaluating the current state.

## 6. Training Process
Training occurs iteratively in episodic rollouts. The step-by-step training loop is:
1. **Reset environment:** Start a new SUMO simulation episode.
2. **Collect trajectories:** The agent interacts with the environment for $N$ steps, storing states, actions, rewards, and log probabilities.
3. **Compute rewards & advantages:** Calculate expected returns and Advantages using GAE.
4. **Update policy using PPO:** Divide trajectories into mini-batches and optimize the PPO clipped objective using Gradient Descent over several `epochs`.
5. **Repeat:** Continue until target timesteps or convergence criteria are reached.

Key hyperparameters include `batch_size` (controls sample variance), `learning_rate` (Adam optimizer step size), and `epochs` (number of optimization passes per rollout).

## 7. Observation Pipeline
The `observation_builder.py` is responsible for translating raw SUMO data into RL-ready input representations. 
Raw metrics from `TraCI` (e.g., list of vehicle speeds, individual vehicle positions) cannot be fed efficiently into a fixed-size neural network. Feature engineering aggregates this unstructured data per lane or per intersection axis:
- Averaging vehicle speeds on an incoming edge.
- Summing halting vehicles over all incoming lanes to capture congestion.
- Appending the one-hot encoded `phase_elapsed` to explicitly give the agent a sense of time.
Finally, all engineered features are clipped and scaled (normalized) before being returned by the environment's `step()` method.

## 8. Reward Design
Reward shaping is critical. A naive reward function (e.g., just penalizing waiting time) can lead to the agent constantly switching lights, causing traffic to stall entirely due to continuous yellow phases.
The design balances multiple trade-offs:
- **Minimizing Waiting vs Maximizing Flow:** By giving positive rewards for throughput and negative for waiting time, the agent learns to prioritize busy lanes securely.
- **Penalizing Switches:** Adding a penalty for switching phases discourages the agent from flickering the lights, ensuring realistic, human-compatible traffic phases.

## 9. Key Components in Project
- `env/`: Contains the environment logic (`sumo_env.py`) and standardizes the Gym interface mapping between RL vectors and SUMO commands.
- `training/`: Holds the main training loops, trajectory rollout buffers, logging, and evaluation scripts.
- `agents/`: Contains the neural network models (actor and critic), PPO algorithm updates, and policy extraction logic.
- `configs/`: YAML or JSON files storing hyperparameters (learning rate, paths to SUMO networks, observation dimensions).
- `sumo/`: Includes SUMO-specific definitions, such as network topologies (`.net.xml`), routes (`.rou.xml`), and the configuration bundle (`.sumocfg`).

## 10. Training Workflow
To run training, the python environment must be configured and SUMO installed.
Example bash commands:

```bash
# 1. Setup bash variables and paths for TraCI
source setup_sumo.sh

# 2. Generate random traffic routes (if dynamic generation is enabled)
python sumo/tools/generate_routes.py --demand high

# 3. Launch the main training script
python scripts/train.py --config configs/ppo_config.yaml
```
The `setup_sumo.sh` script is essential for exporting `$SUMO_HOME` into the environment variable path, which the `traci` Python package depends on to locate the SUMO binary.

## 11. Challenges & Improvements
- **Exploration vs Exploitation:** Balancing when the agent should try new phase configurations vs. when it should act greedily on what it already knows to minimize real-time congestion.
- **Sparse Rewards:** In low traffic density, vehicles arriving infrequently can cause delayed reward signals, making action attribution difficult.
- **Scalability:** Expanding the state/action spaces to cover more complex intersections significantly increases training time and variance.
- **Multi-intersection Extension:** Moving from an isolated intersection to a grid network requires multi-agent coordination (MARL), where agents must communicate or infer neighboring traffic states to optimize city-wide throughput without moving bottlenecks.

## 12. Conclusion
NeuroTraffic-RL demonstrates the viability of modern Reinforcement Learning algorithms, specifically Proximal Policy Optimization, to solve complex traffic signal control problems. By cleanly decoupling the agent architecture from the SUMO simulation core via a Gym-like interface, the system remains maintainable, scalable, and modular. Future work will investigate integrating Graph Neural Networks (GNNs) for city-wide multi-intersection coordination.
