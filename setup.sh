#!/bin/bash
# Setup script for Yavuz project on Linux/Mac

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
}

print_step() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗ ERROR:${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ INFO:${NC} $1"
}

# Main
print_header "Yavuz Project Setup"

# Check Python 3
print_info "Checking for Python 3..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed!"
    echo "Please install Python 3 via your package manager."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_step "Python 3 found: $PYTHON_VERSION"
echo

# Install uv globally
print_info "Installing uv..."
python3 -m pip install -U uv > /dev/null 2>&1
print_step "uv installed"
echo

# Use uv to create venv and sync dependencies
print_info "Creating virtual environment and syncing dependencies..."
uv sync
print_step "Setup complete"
echo

# Success
print_header "Setup Complete!"

echo -e "${GREEN}Your project is ready to use!${NC}"
echo
echo "To launch the app:"
echo "  ./start.sh"
echo
