#!/bin/bash
# Script to install all MASA dependencies

echo "============================================================"
echo "MASA Framework - Dependency Installation"
echo "============================================================"

# Detect Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo "Python path: $(which python3)"
echo ""

# Install core dependencies
echo "[1/4] Installing core dependencies..."
python3 -m pip install --upgrade pip --quiet
python3 -m pip install numpy pandas scipy matplotlib --quiet
echo "  ✓ Core dependencies installed"

# Install PyTorch (CPU version for compatibility)
echo "[2/4] Installing PyTorch..."
python3 -m pip install torch torchvision torchaudio --quiet
echo "  ✓ PyTorch installed"

# Install RL and optimization libraries
echo "[3/4] Installing RL and optimization libraries..."
python3 -m pip install gym stable-baselines3 cvxopt cvxpy --quiet
echo "  ✓ RL and optimization libraries installed"

# Install optional dependencies
echo "[4/4] Installing optional dependencies..."
python3 -m pip install tensorboard tensorboardX --quiet || echo "  ⚠ Tensorboard installation skipped (optional)"
echo "  ✓ Optional dependencies installed"

echo ""
echo "============================================================"
echo "Installation complete!"
echo "============================================================"
echo ""
echo "Verifying installation..."
python3 -c "
import numpy, pandas, torch, gym, stable_baselines3, cvxopt, cvxpy, scipy
print('✓ All critical packages imported successfully')
print(f'  NumPy: {numpy.__version__}')
print(f'  Pandas: {pandas.__version__}')
print(f'  PyTorch: {torch.__version__}')
print(f'  Gym: {gym.__version__}')
print(f'  Stable-Baselines3: {stable_baselines3.__version__}')
"

