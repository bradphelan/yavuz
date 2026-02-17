#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -x ".venv/bin/python" ]; then
    echo "Virtual environment not found. Running setup..."
    ./setup.sh
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "Virtual environment still missing. Run ./setup.sh and try again."
    exit 1
fi

exec ".venv/bin/python" -m yavuz.launcher
