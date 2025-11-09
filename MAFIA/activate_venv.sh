#!/bin/bash
# Script to activate virtual environment and verify setup

echo "============================================================"
echo "MASA Framework - Virtual Environment Setup"
echo "============================================================"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "✗ Virtual environment .venv not found!"
    echo "  Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check Python version
echo "Python version: $(python3 --version)"
echo "Python path: $(which python3)"

# Check if dependencies are installed
echo ""
echo "Checking dependencies..."
if python3 -c "import numpy" 2>/dev/null; then
    echo "  ✓ NumPy installed"
else
    echo "  ✗ NumPy not found - installing dependencies..."
    python3 install_dependencies.py
fi

echo ""
echo "============================================================"
echo "Virtual environment is ready!"
echo "============================================================"
echo ""
echo "To activate manually, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Or on Windows:"
echo "  .venv\\Scripts\\activate"
echo ""

