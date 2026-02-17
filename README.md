# Yavuz

Yavuz is an AI-first algorithm sandbox for VS Code. The sandbox is the app: start with a prompt, let your AI assistant generate the first version, then keep evolving the algorithm through AI-guided iterations instead of writing everything by hand.

## Features

- 🤖 AI sandbox workflow: prompt, generate, run, refine
- 🧰 Lightweight framework to turn algorithm ideas into runnable demos fast
- ⚡ Fast vector math with numpy when algorithms avoid manual Python loops
- 🎨 Interactive 3D visualizations using PyVista
- 🎛️ GUI controls using PyVistaQt widgets
- 🚀 Easy-to-use demo launcher interface

## Installation

```bash
# Windows
.\start.bat

# Linux/Mac
./start.sh
```

This opens an interactive launcher where you can browse and run demos.

## Workflow

The core workflow is **AI-first development**:

1. **Start the launcher**: Run `start.bat` or `start.sh` to open the demo selector
2. **Describe your algorithm**: Tell your AI assistant what you want to build
3. **Generate code**: Let the AI create a new algorithm demo in `demos/my_algo/`
4. **Run immediately**: The launcher auto-detects the new demo—no restart needed
5. **Iterate**: Refine the algorithm through prompts without manual coding

The launcher watches for changes in the `demos/` folder, so new demos appear automatically as soon as they're created.

## Adding New Demos

Ask your AI assistant to create a new demo in the `demos/` folder. The launcher will auto-detect it on next refresh.

## Project Structure

```
yavuz/
├── start.bat           # Windows start script
├── start.sh            # Linux/Mac start script
├── start.bash          # Alternate bash entry point
├── demos/              # All demos, each in own subfolder
│   ├── surface_plot/
│   │   └── surface_plot_interactive.py
│   ├── algorithm_visualizer/
│   │   └── algorithm_visualizer.py
│   ├── parametric_3d/
│   ├── numerical_methods/
│   │   └── numerical_methods.py
│   └── douglas_peucker/
│       └── douglas_peucker
│       └── numerical_methods.py
├── src/
│   └── yavuz/          # Main package
│       └── launcher.py # Main demo launcher GUI
├── tests/              # Test suite
└── pyproject.toml      # Project configuration
```

## Development

Run tests:
```bash
pytest
```

Format code:
```bash
black src/ tests/ examples/
```

Type checking:
```bash
mypy src/
```

## Agent Advice

If you use an AI coding assistant:
- Use the project virtual environment at `.venv/` when running or inspecting Python packages.
- Manage dependencies only in `pyproject.toml` (this project uses `uv`).
- Choose renderers intentionally: default to PyVista for interactive visuals and widgets. Use PyVistaQt for UI controls, and make sure Qt is available on the target environment.

## License

MIT
