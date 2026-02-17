# Yavuz Sandbox Instructions

Yavuz is an AI-first algorithm sandbox for VS Code. The sandbox is the app: prompt an AI assistant to generate a new algorithm demo, then iterate and refine that demo with AI instead of hand-coding every step.

## Primary Start Commands

- Windows: run start.bat
- macOS/Linux: run ./start.sh or ./start.bash

The start scripts are robust: they check for a .venv and run setup if it is missing, then launch the app with:

- .venv/bin/python -m yavuz.launcher (macOS/Linux)
- .venv\Scripts\python.exe -m yavuz.launcher (Windows)

## VS Code Tasks

Use VS Code tasks for a frictionless workflow:

- Start app: runs the platform start script
- Setup env: runs the platform setup script

Tasks live in .vscode/tasks.json and are intended to be the default entry points.

## Where The Launcher Lives

The launcher entry point is a module:

- src/yavuz/launcher.py

Launch it via module invocation (python -m yavuz.launcher) so it works in any environment with the venv activated.

## Adding A New Demo

1. Create a new folder under demos/
2. Add your demo script inside that folder
3. Add the demo metadata in src/yavuz/launcher.py (demo_info)
4. Start the app and click Refresh

## Performance Guidance

Numpy provides fast vector math. Prefer vectorized operations and avoid manual Python loops when possible.
