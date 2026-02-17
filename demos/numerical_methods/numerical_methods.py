"""
Numerical methods demonstration with interactive controls.
Demonstrates integration and differentiation visuals.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter


class NumericalMethodsDemo:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1400, 700),
            title="Numerical Methods Demonstration",
            shape=(1, 2),
        )
        self.plotter.set_background("white")

        self.n_points = 100
        self.method_options = ["Trapezoidal", "Simpson", "Derivative"]
        self.method_index = 0

        self.freq = 1.0
        self.amp = 1.0

        self._setup_controls()
        self.update_plot()

    def test_function(self, x):
        return self.amp * np.sin(self.freq * x) + 0.5 * np.cos(2 * self.freq * x)

    def test_function_derivative(self, x):
        return self.amp * self.freq * np.cos(self.freq * x) - self.freq * np.sin(2 * self.freq * x)

    def test_function_integral(self, x):
        return -self.amp / self.freq * np.cos(self.freq * x) + 1 / (4 * self.freq) * np.sin(2 * self.freq * x)

    def trapezoidal_integration(self, x, y):
        dx = x[1] - x[0]
        integral = np.cumsum(y) * dx
        integral[1:] = integral[1:] - dx * y[1:] / 2
        integral[0] = 0
        return integral

    def simpson_integration(self, x, y):
        n = len(x) - 1
        if n % 2 == 1:
            n -= 1

        dx = x[1] - x[0]
        integral = np.zeros_like(y)

        for i in range(1, n + 1):
            if i % 2 == 0:
                integral[i] = integral[i - 2] + dx / 3 * (y[i - 2] + 4 * y[i - 1] + y[i])
            else:
                integral[i] = integral[i - 1] + dx * (y[i - 1] + y[i]) / 2

        return integral

    def numerical_derivative(self, x, y):
        dy = np.zeros_like(y)
        dx = x[1] - x[0]
        dy[1:-1] = (y[2:] - y[:-2]) / (2 * dx)
        dy[0] = (y[1] - y[0]) / dx
        dy[-1] = (y[-1] - y[-2]) / dx
        return dy

    def _setup_controls(self):
        self.plotter.add_slider_widget(
            self._on_freq_change,
            [0.1, 5.0],
            value=self.freq,
            title="Frequency",
            pointa=(0.02, 0.06),
            pointb=(0.32, 0.06),
        )
        self.plotter.add_slider_widget(
            self._on_amp_change,
            [0.1, 3.0],
            value=self.amp,
            title="Amplitude",
            pointa=(0.35, 0.06),
            pointb=(0.65, 0.06),
        )
        self.plotter.add_slider_widget(
            self._on_points_change,
            [10, 500],
            value=self.n_points,
            title="Points",
            pointa=(0.68, 0.06),
            pointb=(0.98, 0.06),
        )
        self.plotter.add_slider_widget(
            self._on_method_change,
            [0, len(self.method_options) - 1],
            value=self.method_index,
            title="Method",
            pointa=(0.02, 0.1),
            pointb=(0.32, 0.1),
        )

    def _on_freq_change(self, val):
        self.freq = float(val)
        self.update_plot()

    def _on_amp_change(self, val):
        self.amp = float(val)
        self.update_plot()

    def _on_points_change(self, val):
        self.n_points = int(round(val))
        self.update_plot()

    def _on_method_change(self, val):
        self.method_index = int(round(val))
        self.update_plot()

    def _clear_subplot(self, row, col):
        self.plotter.subplot(row, col)
        self.plotter.clear()
        self.plotter.set_background("white")
        self.plotter.show_grid()

    def update_plot(self):
        self._clear_subplot(0, 0)
        self._clear_subplot(0, 1)

        x = np.linspace(0, 4 * np.pi, self.n_points)
        y = self.test_function(x)

        x_fine = np.linspace(0, 4 * np.pi, 500)
        y_fine = self.test_function(x_fine)

        points_fine = np.column_stack([x_fine, y_fine, np.zeros_like(x_fine)])
        points_samples = np.column_stack([x, y, np.zeros_like(x)])
        line_fine = pv.lines_from_points(points_fine)
        points_poly = pv.PolyData(points_samples)

        self.plotter.subplot(0, 0)
        self.plotter.add_text("Original Function", position="upper_left", font_size=12)
        self.plotter.add_mesh(line_fine, color="blue", line_width=3)
        self.plotter.add_mesh(points_poly, color="red", point_size=6, render_points_as_spheres=True)

        method = self.method_options[self.method_index]
        self.plotter.subplot(0, 1)

        if method == "Trapezoidal":
            numerical = self.trapezoidal_integration(x, y)
            analytical = self.test_function_integral(x)
            analytical -= analytical[0]
            title = "Numerical Integration (Trapezoidal)"
        elif method == "Simpson":
            numerical = self.simpson_integration(x, y)
            analytical = self.test_function_integral(x)
            analytical -= analytical[0]
            title = "Numerical Integration (Simpson)"
        else:
            numerical = self.numerical_derivative(x, y)
            analytical = self.test_function_derivative(x)
            title = "Numerical Differentiation"

        numerical_points = np.column_stack([x, numerical, np.zeros_like(x)])
        analytical_points = np.column_stack([x_fine, analytical, np.zeros_like(x_fine)])
        numerical_line = pv.lines_from_points(numerical_points)
        analytical_line = pv.lines_from_points(analytical_points)

        self.plotter.add_text(title, position="upper_left", font_size=12)
        self.plotter.add_mesh(numerical_line, color="red", line_width=3)
        self.plotter.add_mesh(analytical_line, color="blue", line_width=2, opacity=0.6)

        self.plotter.reset_camera()

    def show(self):
        self.plotter.show()
        self.plotter.app.exec_()


def main():
    demo = NumericalMethodsDemo()
    demo.show()


def _build_vscode_url(path):
    posix_path = Path(path).resolve().as_posix()
    if not posix_path.startswith("/"):
        posix_path = f"/{posix_path}"
    return f"vscode://file{quote(posix_path, safe='/:')}"


def get_manifest():
    manifest = dict(DEMO_MANIFEST)
    manifest["source_url"] = _build_vscode_url(__file__)
    return manifest


DEMO_MANIFEST = {
    "title": "Numerical Methods Demonstration",
    "description": "Compare numerical algorithms with analytical solutions.\n\n"
    "Features:\n"
    "- Numerical integration (Trapezoidal and Simpson)\n"
    "- Numerical differentiation\n"
    "- Side-by-side comparison with exact solutions\n"
    "- Adjustable sample points\n"
    "- Interactive function parameters",
}


if __name__ == "__main__":
    main()
