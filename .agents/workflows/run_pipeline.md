---
description: How to run the full NeuroTraffic-RL training and evaluation pipeline
---

This workflow walks you through the entire end-to-end process of NeuroTraffic-RL: from running tests, to training the PPO agent, monitoring its progress in the dashboard, and finally evaluating its performance.

### 1. Run Unit Tests
First, ensure the environment and all its logic (reward calculation, phase management) are working correctly.

// turbo
```bash
make test
```

### 2. Train the RL Agent
Start the PPO agent training process. This will run headless SUMO simulations in the background and train the model for the configured number of timesteps. The agent will repeatedly attempt to minimize traffic wait times by choosing optimal traffic light phases.

// turbo
```bash
make train
```

### 3. Monitor Training with the Dashboard
While the agent is training (or after it finishes), you can launch the Streamlit dashboard to monitor live metrics, such as the total waiting time, intersection throughput, and reward breakdowns. Because `make train` is already running, you should run this in a separate terminal.

```bash
make dashboard
```
*(This will launch a local server and provide a link like http://localhost:8501 to open in your browser)*

### 4. Evaluate the Trained Model
Once training is complete and the best model is saved to `models/best_model.zip`, you can evaluate the agent's performance against the default Fixed-Cycle traffic light baseline. This script will run both the AI agent and the baseline, and output a table comparing their metrics.

// turbo
```bash
make eval
```

### 5. (Optional) Train with GUI
If you want to visually observe the agent interacting with the SUMO environment during training, you can run the GUI version. This will pop up the SUMO interface so you can see cars moving and the traffic lights changing.

```bash
make train-gui
```
