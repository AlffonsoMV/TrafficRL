#!/bin/bash
set -e

echo "Installing Traffic RL package..."

# Create a virtual environment (optional)
if [ "$1" == "--venv" ]; then
    echo "Creating a virtual environment..."
    python -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated."
fi

# Install the package
pip install -e .

echo "Installation complete. You can now run 'traffic_rl --help' to get started." 