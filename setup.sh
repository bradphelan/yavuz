#!/bin/bash
# Setup script for Yavuz project on Linux/Mac

set -e

echo "========================================"
echo "Yavuz Project Setup"
echo "========================================"
echo

# Check if uv is installed
echo "Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    pip install uv
fi

echo "uv is installed."
echo

# Create virtual environment
echo "Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
else
    uv venv
fi

echo
echo "Installing dependencies..."
source .venv/bin/activate
uv pip install -r requirements.txt

echo
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo
echo "To activate the virtual environment, run:"
echo "    source .venv/bin/activate"
echo
echo "To launch the demo selector, run:"
echo "    ./start.sh"
echo
echo "Or run the launcher directly:"
echo "    .venv/bin/python -m yavuz.launcher"
echo
