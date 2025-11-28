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

# Default: run walk-forward. Set WALK_FORWARD=0 to use legacy multi-seed/single run.
WALK_FORWARD="${WALK_FORWARD:-1}"
if [ "$WALK_FORWARD" = "1" ]; then
    WF_START_DATE="${WF_START_DATE:-2017-01-01}"
    WF_NUM_WINDOWS="${WF_NUM_WINDOWS:-0}"       # 0 => auto-compute windows until end-date in script
    WF_TRAIN_YEARS="${WF_TRAIN_YEARS:-3}"
    WF_VALID_YEARS="${WF_VALID_YEARS:-1}"
    WF_TEST_YEARS="${WF_TEST_YEARS:-1}"
    WF_STEP_YEARS="${WF_STEP_YEARS:-1}"
    WF_NUM_SEEDS="${WF_NUM_SEEDS:-10}"
    WF_BASE_SEED="${WF_BASE_SEED:-2025}"
    WF_RESUME_OVERLAP="${WF_RESUME_OVERLAP:-1}"
    RESUME_FLAG=""
    if [ "$WF_RESUME_OVERLAP" = "1" ]; then
        RESUME_FLAG="--resume-overlap"
    else
        RESUME_FLAG="--no-resume-overlap"
    fi

    echo "▶️ Running walk-forward training..."
    echo "   WF_START_DATE=${WF_START_DATE}"
    echo "   WF_NUM_WINDOWS=${WF_NUM_WINDOWS} (0 => auto)"
    python -u -W ignore::RuntimeWarning -W ignore::FutureWarning scripts/walk_forward.py \
        --start-date "$WF_START_DATE" \
        --num-windows "$WF_NUM_WINDOWS" \
        --train-years "$WF_TRAIN_YEARS" \
        --valid-years "$WF_VALID_YEARS" \
        --test-years "$WF_TEST_YEARS" \
        --step-years "$WF_STEP_YEARS" \
        --num-seeds "$WF_NUM_SEEDS" \
        --base-seed "$WF_BASE_SEED" \
        $RESUME_FLAG
    echo ""
    echo "✅ Training Completed!"
    echo "⏰ End: $(date)"
    exit 0
fi

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
