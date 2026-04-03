#!/bin/bash
# Setup environment for InfiniteTalk

set -e

echo "Setting up InfiniteTalk environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (adjust for your CUDA version)
echo "Installing PyTorch..."
pip install torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
pip install -e .

echo "Setup complete!"
echo "Activate environment with: source venv/bin/activate"
