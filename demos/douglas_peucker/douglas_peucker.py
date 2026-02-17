"""
Douglas-Peucker Algorithm Visualization
Demonstrates line simplification with noise reduction using the Ramer-Douglas-Peucker algorithm.
"""

from pathlib import Path
from urllib.parse import quote
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class DouglasPeuckerDemo:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))
        self.fig.subplots_adjust(bottom=0.25, hspace=0.3)

        # Parameters
        self.noise_amplitude = 0.1
        self.tolerance = 0.05
        self.frequency = 2.0
        self.n_points = 200

        # Generate initial data
        self.x_clean = None
        self.y_clean = None
        self.y_noisy = None
        self.generate_data()

        # Setup controls and initial plot
        self.setup_controls()
        self.update_plot()

    def generate_data(self):
        """Generate clean sine wave and add noise."""
        self.x_clean = np.linspace(0, 4 * np.pi, self.n_points)
        self.y_clean = np.sin(self.frequency * self.x_clean)

        # Add noise
        noise = np.random.normal(0, self.noise_amplitude, self.n_points)
        self.y_noisy = self.y_clean + noise

    def douglas_peucker(self, points, tolerance):
        """
        Ramer-Douglas-Peucker algorithm for line simplification.

        Args:
            points: Array of shape (n, 2) containing x, y coordinates
            tolerance: Maximum distance threshold

        Returns:
            Simplified array of points
        """
        if len(points) < 3:
            return points

        # Find the point with maximum distance from line between first and last
        start = points[0]
        end = points[-1]

        max_distance = 0
        max_index = 0

        # Calculate perpendicular distance for each point
        for i in range(1, len(points) - 1):
            distance = self.perpendicular_distance(points[i], start, end)
            if distance > max_distance:
                max_distance = distance
                max_index = i

        # If max distance is greater than tolerance, recursively simplify
        if max_distance > tolerance:
            # Recursive call
            left_points = self.douglas_peucker(points[:max_index + 1], tolerance)
            right_points = self.douglas_peucker(points[max_index:], tolerance)

            # Combine results (remove duplicate point at max_index)
            result = np.vstack([left_points[:-1], right_points])
        else:
            # All points between start and end can be removed
            result = np.array([start, end])

        return result

    def perpendicular_distance(self, point, line_start, line_end):
        """
        Calculate perpendicular distance from point to line.

        Args:
            point: Point coordinates (x, y)
            line_start: Start of line segment (x, y)
            line_end: End of line segment (x, y)

        Returns:
            Perpendicular distance
        """
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Handle case where line segment is actually a point
        if x1 == x2 and y1 == y2:
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

        # Formula for perpendicular distance from point to line
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

        return numerator / denominator

    def setup_controls(self):
        """Setup GUI controls."""
        # Noise amplitude slider
        ax_noise = plt.axes([0.2, 0.15, 0.6, 0.03])
        self.slider_noise = Slider(
            ax_noise, 'Noise Amplitude', 0.0, 0.5,
            valinit=self.noise_amplitude, valstep=0.01
        )
        self.slider_noise.on_changed(self.update_noise)

        # Tolerance slider
        ax_tolerance = plt.axes([0.2, 0.10, 0.6, 0.03])
        self.slider_tolerance = Slider(
            ax_tolerance, 'DP Tolerance', 0.001, 0.3,
            valinit=self.tolerance, valstep=0.001
        )
        self.slider_tolerance.on_changed(self.update_tolerance)

        # Frequency slider
        ax_freq = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider_freq = Slider(
            ax_freq, 'Frequency', 0.5, 5.0,
            valinit=self.frequency, valstep=0.1
        )
        self.slider_freq.on_changed(self.update_frequency)

        # Regenerate button
        ax_regen = plt.axes([0.4, 0.01, 0.2, 0.03])
        self.btn_regen = Button(ax_regen, 'Regenerate Noise')
        self.btn_regen.on_clicked(self.regenerate_noise)

    def update_noise(self, val):
        """Update noise amplitude."""
        self.noise_amplitude = val
        self.generate_data()
        self.update_plot()

    def update_tolerance(self, val):
        """Update Douglas-Peucker tolerance."""
        self.tolerance = val
        self.update_plot()

    def update_frequency(self, val):
        """Update sine wave frequency."""
        self.frequency = val
        self.generate_data()
        self.update_plot()

    def regenerate_noise(self, event):
        """Regenerate noise with same parameters."""
        self.generate_data()
        self.update_plot()

    def update_plot(self):
        """Redraw plots with current parameters."""
        # Clear both axes
        self.ax1.clear()
        self.ax2.clear()

        # Top plot: Original data with noise
        self.ax1.plot(self.x_clean, self.y_clean, 'g--', linewidth=2,
                     label='Clean Signal', alpha=0.7)
        self.ax1.plot(self.x_clean, self.y_noisy, 'b-', linewidth=1,
                     label=f'Noisy Signal (noise={self.noise_amplitude:.2f})', alpha=0.8)
        self.ax1.scatter(self.x_clean, self.y_noisy, c='blue', s=10, alpha=0.3)

        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.ax1.set_title('Original Signal with Noise')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylim(-2, 2)

        # Apply Douglas-Peucker algorithm
        points = np.column_stack([self.x_clean, self.y_noisy])
        simplified = self.douglas_peucker(points, self.tolerance)

        # Bottom plot: Comparison
        self.ax2.plot(self.x_clean, self.y_clean, 'g--', linewidth=2,
                     label='Clean Signal', alpha=0.7)
        self.ax2.plot(self.x_clean, self.y_noisy, 'b-', linewidth=1,
                     label='Noisy Signal', alpha=0.3)
        self.ax2.plot(simplified[:, 0], simplified[:, 1], 'r-', linewidth=2,
                     label=f'Douglas-Peucker (tol={self.tolerance:.3f})', marker='o',
                     markersize=4)

        # Calculate and display statistics
        original_points = len(self.x_clean)
        simplified_points = len(simplified)
        reduction = (1 - simplified_points / original_points) * 100

        # Calculate error metrics
        # Interpolate simplified curve to compare with original
        simplified_y_interp = np.interp(self.x_clean, simplified[:, 0], simplified[:, 1])
        mse = np.mean((self.y_clean - simplified_y_interp)**2)

        stats_text = f'Points: {original_points} → {simplified_points} ({reduction:.1f}% reduction)\n'
        stats_text += f'MSE vs Clean: {mse:.4f}'

        self.ax2.text(0.02, 0.98, stats_text, transform=self.ax2.transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round',
                     facecolor='wheat', alpha=0.8), fontsize=9, family='monospace')

        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        self.ax2.set_title('Douglas-Peucker Line Simplification')
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(-2, 2)

        self.fig.canvas.draw_idle()

    def show(self):
        """Display the demo."""
        plt.show()


def main():
    """Run the Douglas-Peucker demo."""
    demo = DouglasPeuckerDemo()
    demo.show()


def _build_vscode_url(path):
    posix_path = Path(path).resolve().as_posix()
    if not posix_path.startswith("/"):
        posix_path = f"/{posix_path}"
    return f"vscode://file{quote(posix_path, safe='/:')}"


def get_manifest():
    return {
        "title": "Douglas-Peucker Line Simplification",
        "description": "Visualize noise reduction using the Ramer-Douglas-Peucker algorithm.\n\n"
                       "Features:\n"
                       "- Clean sine wave generation\n"
                       "- Adjustable noise amplitude\n"
                       "- Interactive tolerance control for simplification\n"
                       "- Real-time comparison of noisy vs simplified signals\n"
                       "- Point reduction statistics and error metrics\n"
                       "- Configurable signal frequency",
        "source_url": _build_vscode_url(__file__),
    }


if __name__ == "__main__":
    main()
