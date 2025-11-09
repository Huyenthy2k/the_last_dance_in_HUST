#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python script to install MASA dependencies
Works in any Python environment (conda, venv, etc.)
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"  ✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {description} failed: {e.stderr}")
        return False

def main():
    print("=" * 60)
    print("MASA Framework - Dependency Installation")
    print("=" * 60)
    
    python_version = sys.version
    python_path = sys.executable
    print(f"\nPython version: {python_version.split()[0]}")
    print(f"Python path: {python_path}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    if in_venv:
        print(f"✓ Running in virtual environment: {sys.prefix}")
    else:
        print("⚠ Not in a virtual environment (using system Python)")
    
    # Install dependencies
    print("\n" + "=" * 60)
    print("Installing dependencies...")
    print("=" * 60)
    
    # Check and install pip if missing
    print("\nChecking pip installation...")
    try:
        import pip
        print("  ✓ pip is available")
    except ImportError:
        print("  ⚠ pip not found, installing pip...")
        import urllib.request
        import subprocess
        
        try:
            # Try ensurepip first
            result = subprocess.run(
                [python_path, "-m", "ensurepip", "--upgrade"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                # Fallback: download get-pip.py
                print("  Downloading get-pip.py...")
                urllib.request.urlretrieve(
                    "https://bootstrap.pypa.io/get-pip.py",
                    "get-pip.py"
                )
                result = subprocess.run(
                    [python_path, "get-pip.py"],
                    capture_output=True,
                    text=True
                )
                # Clean up
                import os
                if os.path.exists("get-pip.py"):
                    os.remove("get-pip.py")
            
            if result.returncode == 0:
                print("  ✓ pip installed successfully")
            else:
                print(f"  ✗ Failed to install pip: {result.stderr}")
                sys.exit(1)
        except Exception as e:
            print(f"  ✗ Error installing pip: {e}")
            sys.exit(1)
    
    # Upgrade pip
    run_command(
        f"{python_path} -m pip install --upgrade pip --quiet",
        "Upgrading pip"
    )
    
    # Core dependencies
    run_command(
        f"{python_path} -m pip install numpy pandas scipy matplotlib --quiet",
        "Installing core dependencies (numpy, pandas, scipy, matplotlib)"
    )
    
    # PyTorch
    run_command(
        f"{python_path} -m pip install torch torchvision torchaudio --quiet",
        "Installing PyTorch"
    )
    
    # RL and optimization
    run_command(
        f"{python_path} -m pip install gym stable-baselines3 cvxopt cvxpy --quiet",
        "Installing RL and optimization libraries"
    )
    
    # Optional dependencies
    run_command(
        f"{python_path} -m pip install tensorboard tensorboardX --quiet",
        "Installing optional dependencies (tensorboard)"
    )
    
    # Verify installation
    print("\n" + "=" * 60)
    print("Verifying installation...")
    print("=" * 60)
    
    try:
        import numpy
        import pandas
        import torch
        import gym
        import stable_baselines3
        import cvxopt
        import cvxpy
        import scipy
        import matplotlib
        
        print("\n✓ All critical packages imported successfully!")
        print(f"  NumPy: {numpy.__version__}")
        print(f"  Pandas: {pandas.__version__}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  Gym: {gym.__version__}")
        print(f"  Stable-Baselines3: {stable_baselines3.__version__}")
        print(f"  CVXOPT: {cvxopt.__version__}")
        print(f"  CVXPY: {cvxpy.__version__}")
        print(f"  SciPy: {scipy.__version__}")
        print(f"  Matplotlib: {matplotlib.__version__}")
        
        print("\n" + "=" * 60)
        print("✅ Installation successful!")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python3 test_environment.py  # Test the environment")
        print("  python3 entrance.py          # Run MASA framework")
        
    except ImportError as e:
        print(f"\n✗ Verification failed: {e}")
        print("Please check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()

