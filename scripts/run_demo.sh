#!/bin/bash
# run_demo.sh — One-command demo of NeuroTraffic-RL
# Runs a short training (10k steps) with SUMO GUI, then evaluates.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "======================================"
echo " NeuroTraffic-RL — Quick Demo"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Check Python environment
if ! command -v python &>/dev/null; then
    echo "❌ Python not found. Install Python 3.10+."
    exit 1
fi

# Check SUMO
if ! command -v sumo &>/dev/null; then
    echo "❌ SUMO not found. Run: bash scripts/setup_sumo.sh"
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing Python dependencies…"
pip install -r requirements.txt -q

# Short training run (headless to keep demo fast)
echo ""
echo "🚀 Starting 10,000-step training demo (headless SUMO)…"
python training/train.py \
    --intersection configs/intersection_casablanca.yaml \
    --config configs/training.yaml \
    --timesteps 10000

echo ""
echo "📊 Evaluating trained model vs FixedCycle baseline (3 episodes)…"
python training/evaluate.py \
    --model models/best_model.zip \
    --episodes 3

echo ""
echo "✅ Demo complete! Launch the dashboard with:"
echo "   streamlit run dashboard/app.py"
