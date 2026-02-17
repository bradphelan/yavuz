"""
Numerical methods demonstration with interactive controls.
Demonstrates various numerical algorithms like integration, differentiation, and root finding.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.parse import quote
from matplotlib.widgets import Slider, RadioButtons, Button, TextBox


class NumericalMethodsDemo:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.subplots_adjust(bottom=0.35)

        # Parameters
        self.n_points = 100
        self.n_iterations = 10
        self.method = 'Trapezoidal'

        # Function parameters
        self.freq = 1.0
        self.amp = 1.0

        self.setup_controls()
        self.update_plot()

    def test_function(self, x):
        """Test function to integrate/differentiate."""
        return self.amp * np.sin(self.freq * x) + 0.5 * np.cos(2 * self.freq * x)

    def test_function_derivative(self, x):
        """Analytical derivative for comparison."""
        return self.amp * self.freq * np.cos(self.freq * x) - \
               self.freq * np.sin(2 * self.freq * x)

    def test_function_integral(self, x):
        """Analytical integral for comparison."""
        return -self.amp / self.freq * np.cos(self.freq * x) + \
               1 / (4 * self.freq) * np.sin(2 * self.freq * x)

    def trapezoidal_integration(self, x, y):
        """Trapezoidal rule for numerical integration."""
        dx = x[1] - x[0]
        integral = np.cumsum(y) * dx
        # Correct using trapezoidal rule
        integral[1:] = integral[1:] - dx * y[1:] / 2
        integral[0] = 0
        return integral

    def simpson_integration(self, x, y):
        """Simpson's rule for numerical integration."""
        n = len(x) - 1
        if n % 2 == 1:
            n -= 1

        dx = x[1] - x[0]
        integral = np.zeros_like(y)

        for i in range(1, n+1):
            if i % 2 == 0:
                # Even index - use Simpson's rule
                integral[i] = integral[i-2] + dx/3 * (y[i-2] + 4*y[i-1] + y[i])
            else:
                # Odd index - interpolate
                integral[i] = integral[i-1] + dx * (y[i-1] + y[i]) / 2

        return integral

    def numerical_derivative(self, x, y):
        """Numerical derivative using central differences."""
        dy = np.zeros_like(y)
        dx = x[1] - x[0]

        # Central difference for interior points
        dy[1:-1] = (y[2:] - y[:-2]) / (2 * dx)

        # Forward difference for first point
        dy[0] = (y[1] - y[0]) / dx

        # Backward difference for last point
        dy[-1] = (y[-1] - y[-2]) / dx

        return dy

    def setup_controls(self):
        """Setup GUI controls."""
        # Frequency slider
        ax_freq = plt.axes([0.2, 0.25, 0.6, 0.03])
        self.slider_freq = Slider(ax_freq, 'Frequency', 0.1, 5.0, valinit=self.freq, valstep=0.1)
        self.slider_freq.on_changed(self.update_freq)

        # Amplitude slider
        ax_amp = plt.axes([0.2, 0.20, 0.6, 0.03])
        self.slider_amp = Slider(ax_amp, 'Amplitude', 0.1, 3.0, valinit=self.amp, valstep=0.1)
        self.slider_amp.on_changed(self.update_amp)

        # Points slider
        ax_points = plt.axes([0.2, 0.15, 0.6, 0.03])
        self.slider_points = Slider(
            ax_points, 'Points', 10, 500,
            valinit=self.n_points, valstep=10
        )
        self.slider_points.on_changed(self.update_points)

        # Method selector
        ax_method = plt.axes([0.025, 0.4, 0.15, 0.15])
        self.radio = RadioButtons(
            ax_method,
            ('Trapezoidal', 'Simpson', 'Derivative')
        )
        self.radio.on_clicked(self.update_method)

    def update_freq(self, val):
        """Update frequency."""
        self.freq = val
        self.update_plot()

    def update_amp(self, val):
        """Update amplitude."""
        self.amp = val
        self.update_plot()

    def update_points(self, val):
        """Update number of points."""
        self.n_points = int(val)
        self.update_plot()

    def update_method(self, label):
        """Update numerical method."""
        self.method = label
        self.update_plot()

    def update_plot(self):
        """Redraw plots with current parameters."""
        # Generate data
        x = np.linspace(0, 4*np.pi, self.n_points)
        y = self.test_function(x)

        # Clear axes
        self.ax1.clear()
        self.ax2.clear()

        # Plot original function
        x_fine = np.linspace(0, 4*np.pi, 500)
        y_fine = self.test_function(x_fine)
        self.ax1.plot(x_fine, y_fine, 'b-', label='Function', linewidth=2)
        self.ax1.plot(x, y, 'ro', markersize=3, label='Sample Points')
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('f(x)')
        self.ax1.set_title('Original Function')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

        # Compute and plot numerical result
        if self.method == 'Trapezoidal':
            numerical = self.trapezoidal_integration(x, y)
            analytical = self.test_function_integral(x)
            analytical -= analytical[0]  # Normalize to start at 0
            self.ax2.plot(x, numerical, 'r-', label='Numerical (Trapezoidal)', linewidth=2)
            self.ax2.plot(x_fine, self.test_function_integral(x_fine) - self.test_function_integral(0),
                         'b--', label='Analytical', linewidth=2, alpha=0.7)
            self.ax2.set_title('Numerical Integration')
            self.ax2.set_ylabel('∫f(x)dx')

        elif self.method == 'Simpson':
            numerical = self.simpson_integration(x, y)
            analytical = self.test_function_integral(x)
            analytical -= analytical[0]
            self.ax2.plot(x, numerical, 'r-', label='Numerical (Simpson)', linewidth=2)
            self.ax2.plot(x_fine, self.test_function_integral(x_fine) - self.test_function_integral(0),
                         'b--', label='Analytical', linewidth=2, alpha=0.7)
            self.ax2.set_title('Numerical Integration')
            self.ax2.set_ylabel('∫f(x)dx')

        elif self.method == 'Derivative':
            numerical = self.numerical_derivative(x, y)
            analytical = self.test_function_derivative(x)
            self.ax2.plot(x, numerical, 'r-', label='Numerical', linewidth=2)
            self.ax2.plot(x_fine, self.test_function_derivative(x_fine),
                         'b--', label='Analytical', linewidth=2, alpha=0.7)
            self.ax2.set_title('Numerical Differentiation')
            self.ax2.set_ylabel("f'(x)")

        self.ax2.set_xlabel('x')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)

        self.fig.canvas.draw_idle()

    def show(self):
        """Display the demo."""
        plt.show()


def main():
    """Run the numerical methods demo."""
    demo = NumericalMethodsDemo()
    demo.show()


def _build_vscode_url(path):
    posix_path = Path(path).resolve().as_posix()
    if not posix_path.startswith("/"):
        posix_path = f"/{posix_path}"
    return f"vscode://file{quote(posix_path, safe='/:')}"


def get_manifest():
    return {
        "title": "Numerical Methods Demonstration",
        "description": "Compare numerical algorithms with analytical solutions.\n\n"
                       "Features:\n"
                       "- Numerical integration (Trapezoidal and Simpson)\n"
                       "- Numerical differentiation\n"
                       "- Side-by-side comparison with exact solutions\n"
                       "- Adjustable sample points\n"
                       "- Interactive function parameters",
        "source_url": _build_vscode_url(__file__),
    }


if __name__ == "__main__":
    main()
