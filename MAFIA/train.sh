#!/bin/bash
# MAFIA Training Script

set -euo pipefail

export PYTHONUNBUFFERED=1

cd /Users/nguyensiry/Documents/the_last_dance/agents/MAFIA

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "🔧 Activating virtual environment..."
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
    echo "   Python: $(which python)"
    echo "   Python version: $(python --version)"
    echo ""
fi

echo "🚀 Starting MAFIA Training..."
echo "⏰ Start: $(date)"
echo ""

# If RUN_MULTI_SEED=1 (default), run multi-seed experiment; otherwise run single entrance.py
RUN_MULTI_SEED="${RUN_MULTI_SEED:-1}"
NUM_SEEDS="${NUM_SEEDS:-10}"
BASE_SEED="${BASE_SEED:-2025}"
SEEDS_OVERRIDE="${SEEDS:-}"

if [ "$RUN_MULTI_SEED" = "1" ]; then
    echo "▶️ Running multi-seed experiment (NUM_SEEDS=${NUM_SEEDS}, BASE_SEED=${BASE_SEED}, SEEDS_OVERRIDE='${SEEDS_OVERRIDE}')"
    if [ -n "$SEEDS_OVERRIDE" ]; then
        python -u -W ignore::RuntimeWarning -W ignore::FutureWarning scripts/run_multi_seed_experiment.py --seeds "$SEEDS_OVERRIDE"
    else
        python -u -W ignore::RuntimeWarning -W ignore::FutureWarning scripts/run_multi_seed_experiment.py --num-seeds "$NUM_SEEDS" --base-seed "$BASE_SEED"
    fi
else
    echo "▶️ Running single training"
    python -u -W ignore::RuntimeWarning -W ignore::FutureWarning entrance.py
fi

echo ""
echo "✅ Training Completed!"
echo "⏰ End: $(date)"
