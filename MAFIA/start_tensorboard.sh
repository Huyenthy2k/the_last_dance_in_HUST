#!/bin/bash
# Start TensorBoard for MAFIA Observer Training
# Usage: ./start_tensorboard.sh

echo "================================================"
echo "Starting TensorBoard for MAFIA Observer Training"
echo "================================================"
echo ""

# Check if tensorboard is installed
if ! command -v tensorboard &> /dev/null
then
    echo "❌ TensorBoard not found!"
    echo "   Install with: pip install tensorboard"
    exit 1
fi

# Navigate to MAFIA directory
cd "$(dirname "$0")"

# Check if logs exist
LOGDIR="observer_offline/tb_logs"
if [ ! -d "$LOGDIR" ]; then
    # Try alternative path
    LOGDIR="observer_offline/tensorboard"
    if [ ! -d "$LOGDIR" ]; then
        echo "⚠️  Warning: No TensorBoard logs found yet"
        echo "   Logs will appear in: observer_offline/tb_logs/ or observer_offline/tensorboard/"
        echo "   Start your training and logs will be created automatically"
        echo ""
        LOGDIR="observer_offline/tb_logs"  # Default to tb_logs
    fi
fi

echo "📊 TensorBoard Configuration:"
echo "   Log Directory: $LOGDIR/"
echo "   Reload Interval: 5 seconds (real-time)"
echo "   Port: 6006"
echo ""
echo "🌐 Open in browser: http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo "================================================"
echo ""

# Start TensorBoard with auto-reload every 5 seconds
tensorboard --logdir "$LOGDIR" --reload_interval 5 --port 6006
