#!/bin/bash

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Detect OS and distribution
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Validate system requirements
PLATFORM=$(detect_platform)
PLATFORM_NAME=$(get_platform_name "$PLATFORM")

if ! python3 -c "import tkinter" 2>/dev/null; then
    echo
    echo -e "${RED}[ERROR] tkinter is not installed!${NC}"
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

if [ ! -x ".venv/bin/python" ]; then
    echo
    echo -e "${BLUE}[*] Virtual environment not found. Running setup...${NC}"
    echo
    if ! ./setup.sh; then
        echo
        echo -e "${RED}[ERROR] Setup failed. Please fix the issues above and try again.${NC}"
        echo
        exit 1
    fi
fi

if [ ! -x ".venv/bin/python" ]; then
    echo
    echo -e "${RED}[ERROR] Virtual environment still missing after setup.${NC}"
    echo
    echo "Please run setup.sh manually:"
    echo "  ./setup.sh"
    echo
    exit 1
fi

echo
echo -e "${BLUE}[*] Launching Yavuz...${NC}"
echo

exec ".venv/bin/python" -m yavuz.launcher
