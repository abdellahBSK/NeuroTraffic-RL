# NeuroTraffic-RL Training Guide

## Prerequisites

- Python 3.10+
- SUMO ≥ 1.18 (install: `bash scripts/setup_sumo.sh`)
- `SUMO_HOME` set in environment

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt
cp .env.example .env   # edit SUMO_HOME if needed

# 2. Run tests (no SUMO required)
make test

# 3. Train
make train

# 4. Evaluate vs baseline
make eval

# 5. Launch dashboard
make dashboard
```

## Training Configuration

All hyperparameters live in `configs/training.yaml`. Key values:

| Parameter | Default | Effect |
|---|---|---|
| `total_timesteps` | 500 000 | Total env steps to train |
| `learning_rate` | 3e-4 | PPO learning rate |
| `n_steps` | 2048 | Rollout buffer length |
| `gamma` | 0.99 | Discount factor |
| `reward.alpha` | 1.0 | Waiting-time penalty weight |
| `reward.beta` | 0.5 | Throughput bonus weight |

## Training Command Reference

```bash
# Default training (configs/intersection_casablanca.yaml)
python training/train.py

# Custom config
python training/train.py \
    --intersection configs/intersection_casablanca.yaml \
    --config configs/training.yaml

# Resume from checkpoint
python training/train.py --resume models/best_model.zip

# With SUMO GUI (slower, useful for debugging)
python training/train.py --gui

# Override timesteps without editing YAML
python training/train.py --timesteps 1000000
```

## Expected Training Output

```
2026-01-01 12:00:00 [INFO    ] NeuroTraffic-RL — Training started
2026-01-01 12:00:01 [INFO    ] Starting SUMO: sumo (port=8813)
2026-01-01 12:00:02 [INFO    ] TraCI connection established on port 8813.
...
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 720      |
|    ep_rew_mean     | -1.23    |
| time/              |          |
|    fps             | 245      |
|    total_timesteps | 2048     |
---------------------------------
...
2026-01-01 12:45:00 [INFO    ] Final model saved to models/final_model.zip
```

Training 500k steps takes approximately **45–90 minutes** on a modern CPU (no GPU required).

## Monitoring with TensorBoard

```bash
tensorboard --logdir logs/tensorboard/
# Open http://localhost:6006
```

Key metrics to watch:
- `rollout/ep_rew_mean` — should increase and stabilise
- `train/loss` — should decrease
- `train/entropy_loss` — should stay near 0

## Evaluation

```bash
# Compare PPO vs FixedCycle baseline (5 episodes each)
python training/evaluate.py --model models/best_model.zip

# More episodes for statistical significance
python training/evaluate.py --model models/best_model.zip --episodes 20
```

Output includes a formatted comparison table and a JSON report in `logs/`.

## Hyperparameter Tuning Tips

- **Agent too aggressive (rapid phase switching):** Increase `reward.delta` (phase-change penalty).
- **Lanes starving (one direction always red):** Increase `reward.gamma` (starvation penalty) or reduce `max_duration` in intersection config.
- **Slow convergence:** Reduce `learning_rate` to 1e-4.
- **Unstable training:** Reduce `clip_range` to 0.1.

## Adding a New Intersection

1. Copy `configs/intersection_casablanca.yaml` → `configs/intersection_<name>.yaml`
2. Update `tl_id`, `incoming_lanes`, `observation_arms`, `phases`.
3. Create `sumo/networks/<name>.net.xml` and `.rou.xml`.
4. Run: `python training/train.py --intersection configs/intersection_<name>.yaml`