#!/bin/bash
# Simple TensorBoard Launcher for MAFIA
# Usage: ./start_tensorboard_simple.sh

echo "🚀 Starting TensorBoard..."
echo "📂 Watching: observer_offline/ (bao gồm tb_logs + tensorboard)"
echo "🌐 URL: http://localhost:6006"
echo ""

cd "$(dirname "$0")"
tensorboard --logdir=observer_offline --reload_interval=5 --port=6006
