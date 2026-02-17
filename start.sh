#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run setup if needed
if [ ! -x ".venv/bin/python" ]; then
    ./setup.sh
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

echo "[*] Launching Yavuz..."
echo

# Use uv run which auto-syncs dependencies
uv run python -m yavuz.launcher
