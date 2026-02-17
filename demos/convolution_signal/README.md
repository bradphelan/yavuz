# Convolution Signal Demo

Visualize discrete convolution with a sliding kernel in signal processing.

## Requirements

- Support multiple signal types (sine, square, triangle, white noise).
- Convolve the signal with an exponential decay kernel.
- Provide a three-panel layout: signal, kernel, and output.
- Show a step slider that advances the convolution window manually.
- Use valid convolution mode for the output.
- Keep charts flicker-free using persistent Chart2D plots.

## Features

- Signal and shifted-kernel overlay with selectable signal types
- Kernel view with normalized exponential decay
- Output curve with current step highlighted

## Usage

Run directly:
```bash
uv run python -m demos.convolution_signal.convolution_signal
```

## Controls

- **Signal frequency**: Change the input frequency for periodic signals
- **Signal type**: Select sine, square, triangle, or white noise
- **Noise seed**: Regenerate white noise deterministically
- **Kernel decay**: Adjust exponential decay rate
- **Convolution step**: Slide the kernel over the signal
