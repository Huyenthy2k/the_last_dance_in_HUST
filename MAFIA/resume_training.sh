#!/bin/bash
# MAFIA Resume Training Script

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

# Check if checkpoint path is provided as argument
if [ $# -eq 1 ]; then
    CHECKPOINT_PATH="$1"
    echo "📂 Using provided checkpoint: $CHECKPOINT_PATH"
    
    # Update config to use this checkpoint
    python3 << EOF
import sys
sys.path.insert(0, '.')
from config import Config
import json

# Read current config
config = Config()

# Update checkpoint path
config.resume_from_checkpoint = "$CHECKPOINT_PATH"
config.auto_resume_from_latest = False

# Save to temp config file (or just use environment variable)
print(f"Checkpoint path set to: {config.resume_from_checkpoint}")
EOF
    
    # Export as environment variable (will be read by entrance.py if needed)
    export MAFIA_RESUME_CHECKPOINT="$CHECKPOINT_PATH"
else
    echo "📂 Auto-detecting latest checkpoint..."
fi

echo "🚀 Resuming MAFIA Training..."
echo "⏰ Start: $(date)"
echo ""

# Run training (will auto-resume if no checkpoint specified)
python -u -W ignore::RuntimeWarning -W ignore::FutureWarning entrance.py

echo ""
echo "✅ Training Completed!"
echo "⏰ End: $(date)"

