# Yavuz Sandbox Instructions

Yavuz is an AI-first algorithm sandbox for VS Code. The sandbox is the app: prompt an AI assistant to generate a new algorithm demo, then iterate and refine that demo with AI instead of hand-coding every step.

## Quick Start

### First Time Setup
The fastest way to get started is to use VS Code tasks:

1. **Open this project in VS Code**
2. **Run the "Setup env" task** (Ctrl+Shift+B or Cmd+Shift+B on Mac, then select "Setup env")
3. **Run the "Start app" task** (Ctrl+Shift+B or Cmd+Shift+B on Mac, then select "Start app")

### Platform-Specific Quick Commands

**Windows:**
```batch
setup.bat
start.bat
```

**macOS/Linux:**
```bash
./setup.sh
./start.sh
```

## Prerequisites

### Windows
- **Python 3.8+** - Download from [python.org](https://www.python.org/downloads/) or install via:
  - Windows Store (search "Python")
  - Chocolatey: `choco install python`
  - Windows Package Manager: `winget install Python.Python.3.12`

### macOS
- **Python 3.8+** - Install via:
  - Homebrew (recommended): `brew install python3`
  - [Official installer](https://www.python.org/downloads/)
  - MacPorts: `sudo port install python311`

### Linux
- **Python 3.8+** - Install via your package manager:
  - **Ubuntu/Debian**: `sudo apt-get install python3 python3-venv python3-pip`
  - **Fedora/CentOS**: `sudo dnf install python3 python3-pip`
  - **Arch**: `sudo pacman -S python`

## How It Works

The start scripts are bulletproof: they check for a `.venv` and run setup if it is missing, then launch the app.

- **Windows**: `setup.bat` → `.venv\Scripts\python.exe -m yavuz.launcher`
- **macOS/Linux**: `setup.sh` → `.venv/bin/python -m yavuz.launcher`

## VS Code Tasks

For a frictionless workflow, use VS Code tasks:

- **Setup env**: runs the platform setup script
- **Start app**: runs the platform start script (auto-runs setup if needed)

Tasks are in `.venv/tasks.json` and are the recommended entry points.

## Where The Launcher Lives

The launcher entry point is:
- `src/yavuz/launcher.py`

It's launched via module invocation (`python -m yavuz.launcher`) so it works in any environment.

## Troubleshooting

### Python Not Found

**Error**: `python is not installed` or `pip is not installed`

**Solution**:
- Install Python from [python.org](https://www.python.org/downloads/)
- Restart your terminal after installation
- Run the setup script again

### Virtual Environment Issues

**Error**: `Virtual environment still missing after setup`

**Solution**:
1. Delete the `.venv` folder manually
2. Run the setup script again
3. If still failing, check that Python and pip work:
   ```bash
   python --version
   pip --version
   ```

### Module Not Found: yavuz

**Error**: `ModuleNotFoundError: No module named 'yavuz'`

**Solution**:
- Activate the virtual environment first:
  - Windows: `.venv\Scripts\activate.bat`
  - macOS/Linux: `source .venv/bin/activate`
- Then run: `python -m pip install -e .`

### Permission Denied (macOS/Linux)

**Error**: `Permission denied` when running scripts

**Solution**:
```bash
chmod +x setup.sh start.sh
./setup.sh
```

## Manual Setup (If Scripts Fail)

If the automated scripts aren't working, try this step-by-step:

```bash
# 1. Verify Python
python3 --version

# 2. Install uv
pip3 install uv

# 3. Create virtual environment
uv venv

# 4. Activate it
# Windows:
.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate

# 5. Install yavuz package (installs dependencies from pyproject.toml)
uv pip install -e .

# 6. Run the launcher
python -m yavuz.launcher
```

## Adding A New Demo

1. Create a new folder under `demos/`
2. Add your demo script inside that folder
3. Register the demo in `src/yavuz/launcher.py` (demo_info)
4. Start the app and click "Refresh"

## Performance Guidance

Numpy provides fast vector math. Prefer vectorized operations and avoid manual Python loops when possible.

## Agent Advice

If you use an AI coding assistant:
- Use the project virtual environment at `.venv/` for running and package inspection.
- Manage dependencies only in `pyproject.toml` (this project uses `uv`).

## Deactivating the Virtual Environment

When you're done working:

```bash
# Windows, macOS, and Linux:
deactivate
```
