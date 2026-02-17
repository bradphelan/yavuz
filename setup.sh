#!/bin/bash
# Setup script for Yavuz project on Linux/Mac

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Detect OS
detect_platform() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ -f /etc/os-release ]]; then
        . /etc/os-release
        echo "$ID"
    elif [[ -f /etc/lsb-release ]]; then
        . /etc/lsb-release
        echo "$DISTRIB_ID" | tr '[:upper:]' '[:lower:]'
    else
        echo "linux"
    fi
}

get_platform_name() {
    case "$1" in
        macos) echo "macOS" ;;
        ubuntu) echo "Ubuntu" ;;
        debian) echo "Debian" ;;
        fedora) echo "Fedora" ;;
        rhel|centos) echo "RHEL/CentOS" ;;
        arch) echo "Arch Linux" ;;
        manjaro) echo "Manjaro" ;;
        opensuse*) echo "openSUSE" ;;
        *) echo "Linux" ;;
    esac
}

get_tkinter_install_cmd() {
    case "$1" in
        macos) echo "brew install python3" ;;
        ubuntu|debian) echo "sudo apt-get install python3-tk" ;;
        fedora|rhel|centos) echo "sudo dnf install python3-tkinter" ;;
        arch|manjaro) echo "sudo pacman -S tk" ;;
        opensuse*) echo "sudo zypper install python3-tk" ;;
        *) echo "Please install tkinter for your system" ;;
    esac
}

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

PLATFORM=$(detect_platform)
PLATFORM_NAME=$(get_platform_name "$PLATFORM")

print_info "Detected platform: $PLATFORM_NAME"
echo

# Check Python 3
print_info "Checking for Python 3..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed!"
    echo
    echo "On $PLATFORM_NAME, install with your package manager."
    echo
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_step "Python 3 found: $PYTHON_VERSION"
echo

# Check pip3
print_info "Checking for pip3..."
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed!"
    echo
    exit 1
fi

print_step "pip3 found"
echo

# Check tkinter
print_info "Checking for tkinter..."
if ! python3 -c "import tkinter" 2>/dev/null; then
    print_error "tkinter is not installed!"
    echo
    echo "On $PLATFORM_NAME, run:"
    echo
    TKINTER_CMD=$(get_tkinter_install_cmd "$PLATFORM")
    echo "  $TKINTER_CMD"
    echo
    exit 1
fi

print_step "tkinter found"
echo

# Check/install uv
print_info "Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip3 install uv
fi

print_step "uv is installed"
echo

# Run uv sync
print_info "Running uv sync..."
uv sync

print_step "Dependencies synchronized"
echo

# Success
print_header "Setup Complete!"

echo -e "${GREEN}Your project is ready to use!${NC}"
echo
echo "To launch the app:"
echo "  ./start.sh"
echo "  or"
echo "  python -m yavuz.launcher"
echo
