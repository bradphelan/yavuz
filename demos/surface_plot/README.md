# Surface Plot Demo

Interactive 3D surface visualization with real-time parameter controls.

## Features

- 3D surface plotting using matplotlib
- Interactive sliders for frequency, amplitude, and phase
- Real-time surface updates
- Mathematical function: Z = A·sin(f·√(X²+Y²) + φ)

## Usage

Run directly:
```bash
python surface_plot_interactive.py
```

Or launch from the main launcher:
```bash
cd ../..
python launcher.py
```

## Controls

- **Frequency Slider**: Adjusts the wave frequency (0.1 - 5.0)
- **Amplitude Slider**: Adjusts the wave amplitude (0.1 - 3.0)
- **Phase Slider**: Adjusts the phase shift (0 - 2π)
- **Mouse**: Rotate and zoom the 3D view
