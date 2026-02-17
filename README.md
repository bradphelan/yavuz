# Yavuz

Yavuz is a structured sandbox app for algorithm development in VS Code. The sandbox is the app: use Copilot (or any AI assistant) to generate new algorithm demos, then run them from the launcher.

## Features

- 🎨 Interactive 3D visualizations using matplotlib
- 🔢 Numerical algorithms with numpy
- 🎛️ GUI controls (sliders, buttons, text boxes)
- 📊 Algorithm visualization tools
- 🧮 Numerical methods demonstrations
- 🚀 Easy-to-use demo launcher interface

## Installation

### Quick Setup (Recommended)

**Windows:**
```bash
.\setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

This will automatically install `uv`, create a virtual environment, and install all dependencies.

### Manual Setup

#### Using uv (Recommended)

Install `uv` if you haven't already:
```bash
pip install uv
```

Create a virtual environment and install dependencies:
```bash
uv venv
.venv\Scripts\activate  # On Windows
# or
source .venv/bin/activate  # On Linux/Mac

uv pip install -r requirements.txt
```

#### Using pip

Alternatively, use pip directly:
```bash
pip install -r requirements.txt
```

## Quick Start

Activate your virtual environment:
```bash
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

Launch the demo selector GUI:

```bash
python launcher.py
```

This opens an interactive interface where you can browse and launch any available demo with a single click.

The intended workflow is to ask your AI assistant to generate a new algorithm demo under `demos/`, then refresh the launcher to run it.

## Available Demos

The `demos/` directory contains interactive demonstration scripts, each in its own subfolder:

### 1. Interactive 3D Surface Plot
**Location:** `demos/surface_plot/`

Features:
- Real-time 3D surface visualization
- Adjustable frequency, amplitude, and phase parameters
- Interactive sliders for parameter control
- Mathematical function: Z = A·sin(f·√(X²+Y²) + φ)

### 2. Algorithm Visualizer
**Location:** `demos/algorithm_visualizer/`

Features:
- Visualize sorting algorithms (Bubble, Selection, Insertion)
- Adjustable array size (10-200 elements)
- Speed control for animation
- Color-coded comparisons
- Step-by-step visualization

### 3. Parametric 3D Curves
**Location:** `demos/parametric_3d/`

Features:
- Multiple curve types: Helix, Torus Knot, Lissajous, Spiral
- Three adjustable parameters per curve
- Real-time 3D rendering with color gradients
- Configurable point density
- Interactive 3D rotation

### 4. Numerical Methods
**Location:** `demos/numerical_methods/`

Features:
- Numerical integration (Trapezoidal & Simpson's rule)
- Numerical differentiation
- Side-by-side comparison with analytical solutions
- Adjustable sample points
- Interactive function parameters

### 5. Douglas-Peucker Line Simplification
**Location:** `demos/douglas_peucker/`

Features:
- Clean sine wave generation with adjustable frequency
- Interactive noise injection with amplitude control
- Douglas-Peucker algorithm visualization
- Real-time comparison between noisy and simplified signals
- Point reduction statistics and error metrics
- Noise regeneration for testing robustness

## Usage

### Using the Launcher (Recommended)
```bash
python launcher.py
```
Select a demo from the list and click "Run Selected Demo".

### Running Demos Directly
Each demo can also be run independently:

```bash
python demos/surface_plot/surface_plot_interactive.py
python demos/algorithm_visualizer/algorithm_visualizer.py
python demos/parametric_3d/parametric_3d.py
python demos/numerical_methods/numerical_methods.py
python demos/douglas_peucker/douglas_peucker.py
```

## Adding New Demos

To add a new demo to the project:

1. **Create a new subfolder** in the `demos/` directory:
   ```bash
   mkdir demos/my_new_demo
   ```

2. **Add your demo script** to the subfolder:
   ```bash
   # demos/my_new_demo/my_new_demo.py
   ```

3. **Update the launcher** by adding your demo info to the `demo_info` dictionary in `launcher.py`:
   ```python
   "my_new_demo": {
       "name": "My New Demo",
       "description": "Description of what this demo does...",
       "script": "my_new_demo.py"
   }
   ```

4. **Restart the launcher** and your demo will appear in the list!

## Project Structure

```
yavuz/
├── launcher.py         # Main demo launcher GUI
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
├── tests/              # Test suite
├── requirements.txt    # Production dependencies
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

## License

MIT
