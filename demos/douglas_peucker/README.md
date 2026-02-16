# Douglas-Peucker Algorithm Demo

Interactive visualization of the Ramer-Douglas-Peucker line simplification algorithm with noise reduction.

## Overview

The Douglas-Peucker (or Ramer-Douglas-Peucker) algorithm is a classic algorithm for reducing the number of points in a curve while preserving its overall shape. This demo shows how the algorithm can be used to simplify noisy signals.

## Features

- Clean sine wave generation
- Adjustable noise amplitude
- Interactive Douglas-Peucker simplification
- Real-time comparison between noisy and simplified signals
- Statistics showing point reduction and error metrics
- Configurable signal frequency

## Usage

Run directly:
```bash
python douglas_peucker.py
```

Or launch from the main launcher:
```bash
cd ../..
python launcher.py
```

## Controls

- **Noise Amplitude Slider**: Add random noise to the signal (0.0 - 0.5)
- **DP Tolerance Slider**: Set simplification threshold (0.001 - 0.3)
  - Lower tolerance = more points kept (closer to original)
  - Higher tolerance = fewer points (more simplified)
- **Frequency Slider**: Adjust sine wave frequency (0.5 - 5.0)
- **Regenerate Noise Button**: Create new random noise with same amplitude

## Algorithm Details

The Douglas-Peucker algorithm works by:

1. **Starting** with a curve defined by a series of points
2. **Finding** the point that is farthest from the line connecting the first and last points
3. **Checking** if this maximum distance exceeds the tolerance threshold
4. **If yes**: Keep that point and recursively simplify the two resulting segments
5. **If no**: Discard all intermediate points and connect first and last points directly

### Time Complexity
- Worst case: O(n²) where n is the number of points
- Average case: O(n log n)

### Applications
- GPS track simplification
- Cartographic line generalization
- Data compression for time series
- Computer graphics and vector simplification
- Noise reduction in sensor data

## How to Interpret the Results

**Top Plot**: Shows the original clean signal (green dashed line) overlaid with the noisy signal (blue line with points).

**Bottom Plot**: Compares three curves:
- **Green dashed**: Clean reference signal
- **Blue thin**: Noisy signal (faded)
- **Red with markers**: Simplified signal after Douglas-Peucker

**Statistics Box**: Shows:
- Point reduction percentage
- Mean Squared Error (MSE) compared to clean signal

## Tips

1. **Start with low noise** (0.05-0.1) to see the algorithm in action
2. **Adjust tolerance** to find the sweet spot between simplification and accuracy
3. **Higher tolerance** removes more noise but may lose signal features
4. **Lower tolerance** preserves more detail but keeps more noise
5. **Use "Regenerate Noise"** to test robustness across different noise patterns

## Example Settings

**Light denoising:**
- Noise Amplitude: 0.1
- Tolerance: 0.05

**Aggressive simplification:**
- Noise Amplitude: 0.2
- Tolerance: 0.15

**Fine detail preservation:**
- Noise Amplitude: 0.15
- Tolerance: 0.02
