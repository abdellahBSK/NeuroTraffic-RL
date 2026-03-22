# PPO Agent Migration Guide

This document provides all the necessary specifications, architecture details, and hyperparameters from the original `NeuroTraffic-RL` project to help **Engineer 2b** implement the PPO agent in the new `RL-Center` architecture.

## 1. Environment Interface Specifications

The new PPO agent must exactly match the inputs and outputs of the original SUMO environment (`SumoIntersectionEnv`).

*   **Observation Space (`state`):** A 1D `float32` array of size **26**. All values are normalized to `[0, 1]`.
*   **Action Space:** `Discrete(2)`. The agent predicts either `0` (NS_GREEN) or `1` (EW_GREEN). *(Note: Yellow phases are handled automatically by the environment logic).*

---

## 2. Infrastructure: `actor_critic_network.py`

The original implementation used `stable-baselines3.PPO` with the default `MlpPolicy`. This resolves to a standard fully connected network with separate Multi-Layer Perceptrons for the Actor and the Critic.

You MUST implement this architecture in `infrastructure/networks/actor_critic_network.py` (e.g., in PyTorch):

**Actor Network (Policy / $\pi$)**
*   **Input Layer:** 26 features
*   **Hidden Layer 1:** 64 units, **Tanh** activation
*   **Hidden Layer 2:** 64 units, **Tanh** activation
*   **Output Layer:** 2 units (Logits for the Categorical distribution)

**Critic Network (Value Function / $V$)**
*   **Input Layer:** 26 features
*   **Hidden Layer 1:** 64 units, **Tanh** activation
*   **Hidden Layer 2:** 64 units, **Tanh** activation
*   **Output Layer:** 1 unit (Linear, predicting State-Value)

> **Initialization Note:** Stable-Baselines3 normally uses orthogonal initialization (`gain=np.sqrt(2)` for hidden layers, `gain=0.01` for the policy output layer, and `gain=1.0` for the value output layer).

---

## 3. Adapters: `ppo_agent_adapter.py`

Your adapter lives in `adapters/agents/ppo/ppo_agent_adapter.py`. You are required to implement the `IAgentPort` interface.

Unlike DQN which trains on individual off-policy transitions, **PPO is an on-policy algorithm that trains in epochs over a collected rollout buffer**. You will need to maintain a Rollout Buffer internally within the adapter.

### Expected Behavior of Interface Methods:

*   `act(self, observation, training=True) -> Action`:
    *   Forward pass through the Actor network.
    *   If `training=True`, sample from the Categorical distribution and store the log-probability of the action (needed for the buffer).
    *   If `training=False`, take the `.argmax()` of the logits deterministically.
*   `store(self, observation, action, reward, next_observation, done) -> None`:
    *   Append the transition to your internal `RolloutBuffer`. **Wait for `n_steps` (2048) transitions before training.**
*   `train_step(self) -> float | None`:
    *   Check if the `RolloutBuffer` is full (size == 2048).
    *   If **not full**: return `None`.
    *   If **full**:
        1. Calculate Generalized Advantage Estimation (GAE).
        2. Run `n_epochs` (10) of optimization using mini-batches of size `batch_size` (64).
        3. Clear the `RolloutBuffer`.
        4. Return the calculated loss metric.
*   `save(self, path: str)` / `load(self, path: str)`:
    *   Serialize/Deserialize your PyTorch `nn.Module` (Actor, Critic) and the Optimizer state dictionary.

---

## 4. Required Hyperparameters

These exact hyperparameters were extracted from `configs/training.yaml` to ensure parity with the original performance:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `learning_rate` | `3.0e-4` | Adam optimizer learning rate |
| `n_steps` | `2048` | Number of steps to collect per rollout (Buffer Size) |
| `batch_size` | `64` | Mini-batch size during epoch training |
| `n_epochs` | `10` | Number of passes over the buffer during one `train_step()` |
| `gamma` | `0.99` | Reward discount factor |
| `gae_lambda` | `0.95` | Factor for bias vs variance tradeoff in GAE |
| `clip_range` | `0.2` | PPO surrogate objective clipping range |
| `ent_coef` | `0.01` | Entropy regularization coefficient (encourages exploration) |
| `vf_coef` | `0.5` | Value function loss coefficient in the total loss |
| `max_grad_norm` | `0.5` | Threshold for gradient clipping |

---

## 5. Development Checklist for Engineer 2b

1. [ ] **Branch Check:** Ensure you are on `feat/ppo`.
2. [ ] **Network:** Implement the dual-headed Actor-Critic in `infrastructure/networks/actor_critic_network.py` (PyTorch recommended).
3. [ ] **Rollout Buffer:** Implement a storage class/list capable of holding 2048 transitions, values, and log-probabilities.
4. [ ] **Adapter Logic:** Implement `PPOAgentAdapter` to satisfy `IAgentPort`, handling the deferred training logic inside `train_step()`.
5. [ ] **Testing:** Create and run `tests/integration/test_ppo_agent_adapter.py`. Check that `train_step()` returns `None` for the first 2047 steps, and returns a loss on step 2048.
6. [ ] **Run Eval:** `python main.py --agent ppo --log-level DEBUG`
