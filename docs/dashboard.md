# NeuroTraffic-RL Dashboard Overview

When you run `make dashboard`, a Streamlit web application launches and connects to the running training metrics file (`logs/metrics.json`). The dashboard automatically refreshes every 2 seconds, allowing you to monitor the training progress of the Reinforcement Learning agent in real time.

Here is a breakdown of everything you see on the dashboard:

## 📊 1. KPI Summary Cards (Top Row)
This section gives you immediate, high-level indicators of how the intersection is performing at the current moment:

* **⏱ Avg Waiting Time**: The average time (in seconds) vehicles are currently spending waiting at the intersection. A lower number means better traffic flow.
* **🚗 Throughput (last 60 steps)**: The total number of vehicles that have successfully passed through the intersection and arrived at their destination over the last ~5 minutes of simulation time. A higher number indicates the intersection is clearing efficiently.
* **🚦 Active Phase**: Shows which traffic light phase the agent currently has active (e.g., `NS 🟢` for North-South Green, or `EW 🟢` for East-West Green).
* **🏆 Episode Reward**: The cumulative numerical score the RL agent has achieved so far in the current episode. Because the reward function heavily penalizes waiting vehicles, this starts at `0` and typically becomes a large negative number over time. The agent's goal is to keep this negative number as close to 0 as possible.

## 📈 2. Real-Time Charts (Middle Rows)
These charts visualize the traffic flow dynamically as the simulation progresses:

* **🚦 Queue Length by Arm**: A bar chart (using Plotly) displaying the total waiting time accumulated on each of the four incoming arms (North, South, East, West). This helps you see if the agent is unfairly favoring one direction and letting queues build up in another.
* **Phase Indicator**: A large visual representation displaying the currently active green lights.
* **⏳ Waiting Time (last 60 steps)**: A line chart showing a rolling window of the total waiting time across all lanes over the last 60 simulation steps. You want to see this line trending downwards or staying flat rather than spiking.
* **📈 Episode Reward Curve**: A line chart tracking the total episode reward. Over many episodes, this line chart will demonstrate how well the agent is learning over time.

## 🔍 3. Reward Component Breakdown (Bottom Row)
The total reward is calculated by combining multiple different traffic factors. This bottom section breaks down the specific numbers the `RewardCalculator` emitted on the very last simulation step:

* **Waiting Penalty**: A negative penalty applied for every second a car spends waiting.
* **Throughput Bonus**: A positive reward given when vehicles successfully clear the intersection.
* **Phase Change Penalty**: A small negative penalty applied when the agent decides to switch from one green phase to another. This prevents the agent from flickering the lights too fast.
* **Max Queue Penalty**: A negative penalty applied if the queue on any single lane grows too long.

By monitoring these distinct factors, you can see exactly *why* the agent received its reward, which is incredibly useful for debugging the RL reward function.
