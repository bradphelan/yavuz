# Numerical Methods Demo

Demonstration of numerical algorithms with comparison to analytical solutions.

## Features

- Numerical integration methods
- Numerical differentiation
- Side-by-side comparison with exact solutions
- Interactive function parameters
- Adjustable sampling

## Usage

Run directly:
```bash
python numerical_methods.py
```

Or launch from the main launcher:
```bash
cd ../..
python launcher.py
```

## Controls

- **Frequency Slider**: Adjust test function frequency (0.1-5.0)
- **Amplitude Slider**: Adjust test function amplitude (0.1-3.0)
- **Points Slider**: Set number of sample points (10-500)
- **Method Radio Buttons**: Select numerical method

## Methods

- **Trapezoidal**: Numerical integration using trapezoidal rule
- **Simpson**: Numerical integration using Simpson's rule (more accurate)
- **Derivative**: Numerical differentiation using central differences

## Test Function

f(x) = A·sin(f·x) + 0.5·cos(2f·x)

The demo compares numerical results (red) with analytical solutions (blue).
