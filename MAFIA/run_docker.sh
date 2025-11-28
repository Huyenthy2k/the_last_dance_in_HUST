#!/bin/bash
# Bash script to run MAFIA training in Docker

set -e

# Default parameters
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO_ID=""
HPARAM_TRIALS=10
HPARAM_EPOCHS=10
TRAIN_EPOCHS=100
NUM_SEEDS=5
BUILD=false
UPLOAD_ALL=false

# Show help
show_help() {
    cat << EOF
MAFIA Docker Training Script
=============================

Usage:
  ./run_docker.sh [OPTIONS]

Options:
  --hf-token <token>        Hugging Face token (or use HF_TOKEN env variable)
  --hf-repo-id <repo-id>    Hugging Face repository ID (e.g., 'username/model-name')
  --hparam-trials <int>     Number of Optuna trials (default: 10)
  --hparam-epochs <int>     Epochs per hparam trial (default: 10)
  --train-epochs <int>      Epochs for full training (default: 100)
  --num-seeds <int>         Number of seeds to run (default: 5)
  --build                   Rebuild Docker image before running
  --upload-all              Upload all seed runs (not just official)
  --help                    Show this help message

Examples:
  # Basic run without upload
  ./run_docker.sh

  # Run with Hugging Face upload (official only)
  ./run_docker.sh --hf-repo-id "yourusername/mafia-model"

  # Run with custom parameters
  ./run_docker.sh --hparam-trials 5 --train-epochs 50 --num-seeds 3

  # Rebuild image and run
  ./run_docker.sh --build --hf-repo-id "yourusername/mafia-model"

  # Upload all seed runs (not just official)
  ./run_docker.sh --hf-repo-id "yourusername/mafia-model" --upload-all

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --hf-repo-id)
            HF_REPO_ID="$2"
            shift 2
            ;;
        --hparam-trials)
            HPARAM_TRIALS="$2"
            shift 2
            ;;
        --hparam-epochs)
            HPARAM_EPOCHS="$2"
            shift 2
            ;;
        --train-epochs)
            TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --num-seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --upload-all)
            UPLOAD_ALL=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

echo "====================================="
echo "MAFIA Docker Training Pipeline"
echo "====================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "✗ Docker is not installed"
    echo "Please install Docker from: https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo "✓ Docker found: $(docker --version)"

# Create necessary directories
for dir in data res log; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "✓ Created directory: $dir"
    fi
done

# Build Docker image if requested
if [ "$BUILD" = true ]; then
    echo ""
    echo "Building Docker image..."
    docker build -t mafia-rl-training:latest .
    echo "✓ Docker image built successfully"
fi

# Prepare command
CMD="python scripts/auto_pipeline.py"
CMD="$CMD --hparam-trials $HPARAM_TRIALS"
CMD="$CMD --hparam-epochs $HPARAM_EPOCHS"
CMD="$CMD --train-epochs $TRAIN_EPOCHS"
CMD="$CMD --num-seeds $NUM_SEEDS"

# Add Hugging Face parameters if provided
if [ -n "$HF_REPO_ID" ]; then
    CMD="$CMD --hf-repo-id \"$HF_REPO_ID\""
    if [ "$UPLOAD_ALL" = false ]; then
        CMD="$CMD --hf-upload-official-only"
    fi
    echo "✓ Hugging Face upload enabled: $HF_REPO_ID"
fi

# Prepare environment variables
ENV_VARS=""
if [ -n "$HF_TOKEN" ]; then
    ENV_VARS="-e HF_TOKEN=$HF_TOKEN"
    echo "✓ Hugging Face token configured"
elif [ -n "$HF_REPO_ID" ]; then
    echo "⚠ Warning: HF_REPO_ID set but no HF_TOKEN provided"
    echo "  Set HF_TOKEN environment variable or use --hf-token parameter"
fi

# Print configuration
echo ""
echo "Training Configuration:"
echo "  Hparam Trials: $HPARAM_TRIALS"
echo "  Hparam Epochs: $HPARAM_EPOCHS"
echo "  Train Epochs: $TRAIN_EPOCHS"
echo "  Num Seeds: $NUM_SEEDS"
if [ -n "$HF_REPO_ID" ]; then
    echo "  HF Repository: $HF_REPO_ID"
    if [ "$UPLOAD_ALL" = true ]; then
        echo "  Upload Mode: All runs"
    else
        echo "  Upload Mode: Official only"
    fi
fi

echo ""
echo "Starting Docker container..."
echo "Press Ctrl+C to stop training"
echo ""

# Run Docker container
docker run --rm -it \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/res:/app/res" \
    -v "$(pwd)/log:/app/log" \
    $ENV_VARS \
    mafia-rl-training:latest \
    bash -c "$CMD"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
    echo "Results saved in: ./res/"
    echo "Logs saved in: ./log/"
else
    echo ""
    echo "✗ Training failed"
    exit 1
fi
