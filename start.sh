#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run setup if needed (or always to validate requirements)
if [ ! -x ".venv/bin/python" ]; then
    ./setup.sh
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

# Launch the app
.venv/bin/python -m yavuz.launcher
