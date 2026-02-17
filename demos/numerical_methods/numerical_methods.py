"""
Numerical methods demonstration with interactive controls.
Demonstrates integration and differentiation visuals.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
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

        # Initialize charts
        self.chart_left = None
        self.chart_right = None
        self.line_fine_plot = None
        self.points_plot = None
        self.numerical_plot = None
        self.analytical_plot = None

        self._setup_controls()
        self._build_scene()
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
        dock = QtWidgets.QDockWidget("Controls", self.plotter)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Frequency slider (0.1-5.0, scaled as 1-50)
        layout.addWidget(QtWidgets.QLabel("Frequency"))
        self.freq_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.freq_slider.setMinimum(1)
        self.freq_slider.setMaximum(50)
        self.freq_slider.setValue(int(self.freq * 10))
        self.freq_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.freq_slider.setTickInterval(5)
        self.freq_slider.valueChanged.connect(self._on_freq_change)
        layout.addWidget(self.freq_slider)

        # Amplitude slider (0.1-3.0, scaled as 1-30)
        layout.addWidget(QtWidgets.QLabel("Amplitude"))
        self.amp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.amp_slider.setMinimum(1)
        self.amp_slider.setMaximum(30)
        self.amp_slider.setValue(int(self.amp * 10))
        self.amp_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.amp_slider.setTickInterval(3)
        self.amp_slider.valueChanged.connect(self._on_amp_change)
        layout.addWidget(self.amp_slider)

        # Points slider (10-500)
        layout.addWidget(QtWidgets.QLabel("Points"))
        self.points_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.points_slider.setMinimum(10)
        self.points_slider.setMaximum(500)
        self.points_slider.setValue(self.n_points)
        self.points_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.points_slider.setTickInterval(25)
        self.points_slider.valueChanged.connect(self._on_points_change)
        layout.addWidget(self.points_slider)

        # Method slider (0-2)
        layout.addWidget(QtWidgets.QLabel("Method"))
        self.method_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.method_slider.setMinimum(0)
        self.method_slider.setMaximum(len(self.method_options) - 1)
        self.method_slider.setValue(self.method_index)
        self.method_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.method_slider.valueChanged.connect(self._on_method_change)
        layout.addWidget(self.method_slider)

        # Stretch to push controls to top
        layout.addStretch(1)

        dock.setWidget(panel)
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _on_freq_change(self, value):
        self.freq = value / 10.0
        self.update_plot()

    def _on_amp_change(self, value):
        self.amp = value / 10.0
        self.update_plot()

    def _on_points_change(self, value):
        self.n_points = int(value)
        self.update_plot()

    def _on_method_change(self, value):
        self.method_index = int(value)
        self.update_plot()

    def _build_scene(self):
        """Build initial chart structure."""
        # Left chart: Original Function
        self.plotter.subplot(0, 0)
        self.chart_left = pv.Chart2D()
        self.chart_left.title = "Original Function"
        self.chart_left.x_label = "x"
        self.chart_left.y_label = "f(x)"
        self.chart_left.grid = False
        self.plotter.add_chart(self.chart_left)

        # Right chart: Numerical Method Result
        self.plotter.subplot(0, 1)
        self.chart_right = pv.Chart2D()
        self.chart_right.x_label = "x"
        self.chart_right.grid = False
        self.plotter.add_chart(self.chart_right)

    def update_plot(self):
        """Update both charts with current parameters."""
        x = np.linspace(0, 4 * np.pi, self.n_points)
        y = self.test_function(x)

        x_fine = np.linspace(0, 4 * np.pi, 500)
        y_fine = self.test_function(x_fine)

        # Update left chart: Original function
        if self.line_fine_plot is None:
            self.line_fine_plot = self.chart_left.line(x_fine, y_fine, color="blue", width=2)
            self.points_plot = self.chart_left.scatter(x, y, color="red", size=6)
            self.chart_left.x_axis.range = [x_fine.min(), x_fine.max()]
            y_range = max(abs(y_fine.min()), abs(y_fine.max()))
            self.chart_left.y_axis.range = [-y_range * 1.1, y_range * 1.1]
        else:
            self.line_fine_plot.update(x_fine, y_fine)
            self.points_plot.update(x, y)
            y_range = max(abs(y_fine.min()), abs(y_fine.max()))
            self.chart_left.y_axis.range = [-y_range * 1.1, y_range * 1.1]

        # Update right chart: Numerical method
        method = self.method_options[self.method_index]

        if method == "Trapezoidal":
            numerical = self.trapezoidal_integration(x, y)
            analytical = self.test_function_integral(x_fine)
            analytical -= analytical[0]
            title = "Numerical Integration (Trapezoidal)"
            y_label = "∫f(x)dx"
        elif method == "Simpson":
            numerical = self.simpson_integration(x, y)
            analytical = self.test_function_integral(x_fine)
            analytical -= analytical[0]
            title = "Numerical Integration (Simpson)"
            y_label = "∫f(x)dx"
        else:
            numerical = self.numerical_derivative(x, y)
            analytical = self.test_function_derivative(x_fine)
            title = "Numerical Differentiation"
            y_label = "df/dx"

        self.chart_right.title = title
        self.chart_right.y_label = y_label

        if self.numerical_plot is None:
            self.numerical_plot = self.chart_right.line(x, numerical, color="red", width=3, label="Numerical")
            self.analytical_plot = self.chart_right.line(x_fine, analytical, color="blue", width=2, label="Analytical")
            self.chart_right.x_axis.range = [x_fine.min(), x_fine.max()]
            y_min = min(numerical.min(), analytical.min())
            y_max = max(numerical.max(), analytical.max())
            y_range = y_max - y_min
            self.chart_right.y_axis.range = [y_min - y_range * 0.1, y_max + y_range * 0.1]
        else:
            self.numerical_plot.update(x, numerical)
            self.analytical_plot.update(x_fine, analytical)
            y_min = min(numerical.min(), analytical.min())
            y_max = max(numerical.max(), analytical.max())
            y_range = y_max - y_min
            self.chart_right.y_axis.range = [y_min - y_range * 0.1, y_max + y_range * 0.1]

        self.plotter.render()

    def show(self):
        self.plotter.show()
        self.plotter.app.exec()


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
