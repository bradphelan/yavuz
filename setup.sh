#!/bin/bash
# Setup script for Yavuz project on Linux/Mac

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect OS and distribution
detect_platform() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ -f /etc/os-release ]]; then
        # Source the os-release file to get ID
        . /etc/os-release
        echo "$ID"
    elif [[ -f /etc/lsb-release ]]; then
        # Fallback for older systems
        . /etc/lsb-release
        echo "$DISTRIB_ID" | tr '[:upper:]' '[:lower:]'
    else
        echo "linux"
    fi
}

get_platform_name() {
    local platform=$1
    case "$platform" in
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

get_python_install_cmd() {
    local platform=$1
    case "$platform" in
        macos)
            echo "brew install python3"
            ;;
        ubuntu|debian)
            echo "sudo apt-get update && sudo apt-get install python3 python3-venv python3-pip"
            ;;
        fedora|rhel|centos)
            echo "sudo dnf install python3 python3-pip"
            ;;
        arch|manjaro)
            echo "sudo pacman -S python"
            ;;
        opensuse*)
            echo "sudo zypper install python3 python3-pip"
            ;;
        *)
            echo "Please install Python 3.8+ for your system from https://www.python.org/downloads/"
            ;;
    esac
}

get_tkinter_install_cmd() {
    local platform=$1
    case "$platform" in
        macos)
            echo "tkinter is included with Python on macOS. Reinstall Python: brew install python3"
            ;;
        ubuntu|debian)
            echo "sudo apt-get install python3-tk"
            ;;
        fedora|rhel|centos)
            echo "sudo dnf install python3-tkinter"
            ;;
        arch|manjaro)
            echo "sudo pacman -S tk"
            ;;
        opensuse*)
            echo "sudo zypper install python3-tk"
            ;;
        *)
            echo "Please install tkinter for your $platform distribution"
            ;;
    esac
}

# Helper function to print colored output
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

print_warning() {
    echo -e "${YELLOW}⚠ WARNING:${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ INFO:${NC} $1"
}

# Main setup
print_header "Yavuz Project Setup"

# Detect platform
PLATFORM=$(detect_platform)
PLATFORM_NAME=$(get_platform_name "$PLATFORM")

print_info "Detected platform: $PLATFORM_NAME"
echo

# Check Python 3 exists
print_info "Checking for Python 3..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed!"
    echo
    echo "To install Python 3 on $PLATFORM_NAME, run:"
    echo
    INSTALL_CMD=$(get_python_install_cmd "$PLATFORM")
    echo "  $INSTALL_CMD"
    echo

    if [[ "$PLATFORM" == "macos" ]]; then
        echo "Don't have Homebrew? Visit https://brew.sh"
        echo
    fi

    echo "Then restart your terminal and run this script again."
    echo
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_step "Python 3 found: $PYTHON_VERSION"
echo

# Check pip3 exists
print_info "Checking for pip3..."
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed!"
    echo

    if [[ "$PLATFORM" == "macos" ]]; then
        echo "  On macOS, try:"
        echo "    python3 -m ensurepip --upgrade"
        echo "    or"
        echo "    brew install python3"
    elif [[ "$PLATFORM" == "ubuntu" ]] || [[ "$PLATFORM" == "debian" ]]; then
        echo "  On $PLATFORM_NAME, run:"
        echo "    sudo apt-get install python3-pip"
    elif [[ "$PLATFORM" == "fedora" ]] || [[ "$PLATFORM" == "rhel" ]] || [[ "$PLATFORM" == "centos" ]]; then
        echo "  On $PLATFORM_NAME, run:"
        echo "    sudo dnf install python3-pip"
    else
        echo "  Please install pip3 for your $PLATFORM_NAME distribution"
    fi
    echo
    exit 1
fi

print_step "pip3 found"
echo

# Check for tkinter (required for GUI)
print_info "Checking for tkinter (required for GUI)..."
if ! python3 -c "import tkinter" 2>/dev/null; then
    print_error "tkinter is not installed!"
    echo
    echo "Yavuz requires tkinter for the GUI. On $PLATFORM_NAME, run:"
    echo
    TKINTER_CMD=$(get_tkinter_install_cmd "$PLATFORM")
    echo "  $TKINTER_CMD"
    echo
    echo "Then run this script again."
    echo
    exit 1
fi

print_step "tkinter found"
echo

# Check for uv
print_info "Checking for uv..."
if ! command -v uv &> /dev/null; then
    print_warning "uv not found. Installing uv..."
    if ! pip3 install uv; then
        print_error "Failed to install uv"
        echo "Try installing manually: pip3 install uv"
        exit 1
    fi
    print_step "uv installed successfully"
else
    print_step "uv already installed"
fi
echo

# Create virtual environment
print_info "Setting up virtual environment..."
if [ -d ".venv" ]; then
    print_warning "Virtual environment already exists at .venv"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        print_step "Removed old virtual environment"
    fi
fi

if [ ! -d ".venv" ]; then
    if ! uv venv; then
        print_error "Failed to create virtual environment"
        exit 1
    fi
    print_step "Virtual environment created"
else
    print_step "Virtual environment ready"
fi
echo

# Activate virtual environment
print_info "Activating virtual environment..."
source .venv/bin/activate
if [ "$VIRTUAL_ENV" != "" ]; then
    print_step "Virtual environment activated"
else
    print_error "Failed to activate virtual environment"
    exit 1
fi
echo

# Install dependencies
print_info "Installing dependencies from requirements.txt..."
if ! uv pip install -r requirements.txt; then
    print_error "Failed to install dependencies"
    exit 1
fi
print_step "Dependencies installed"
echo

# Install the yavuz package
print_info "Installing yavuz package in development mode..."
if ! uv pip install -e .; then
    print_error "Failed to install yavuz package"
    exit 1
fi
print_step "Yavuz package installed"
echo

# Verify installation
print_info "Verifying installation..."
if python3 -c "import yavuz" 2>/dev/null; then
    print_step "yavuz module is importable"
else
    print_error "yavuz module could not be imported"
    echo "Try running: source .venv/bin/activate && pip install -e ."
    exit 1
fi
echo

# Success message
print_header "Setup Complete!"

echo -e "${GREEN}Your project is ready to use!${NC}"
echo
echo "Next steps:"
echo
echo "  1. Activate the virtual environment:"
echo "     ${BLUE}source .venv/bin/activate${NC}"
echo
echo "  2. Launch the demo selector:"
echo "     ${BLUE}./start.sh${NC}"
echo
echo "  3. Or run the launcher directly:"
echo "     ${BLUE}python -m yavuz.launcher${NC}"
echo
echo "To deactivate the virtual environment later, run:"
echo "     ${BLUE}deactivate${NC}"
echo
