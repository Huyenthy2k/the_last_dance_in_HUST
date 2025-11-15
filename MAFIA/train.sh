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

python -u -W ignore::RuntimeWarning -W ignore::FutureWarning entrance.py

echo ""
echo "✅ Training Completed!"
echo "⏰ End: $(date)"