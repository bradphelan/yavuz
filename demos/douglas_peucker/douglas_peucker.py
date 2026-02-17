"""
Douglas-Peucker Algorithm Visualization
Demonstrates line simplification with noise reduction using the RDP algorithm.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import BackgroundPlotter


class DouglasPeuckerDemo:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1200, 800),
            title="Douglas-Peucker Line Simplification",
            shape=(2, 1),
        )
        self.plotter.set_background("white")
        self.plotter.show_grid()

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

        # UI controls
        self._setup_controls()

        self.clean_line = None
        self.noisy_line = None
        self.noisy_points = None
        self.simplified_line = None
        self._camera_initialized = False

        self._build_scene()
        self.update_plot()

    def generate_data(self):
        """Generate clean sine wave and add noise."""
        self.x_clean = np.linspace(0, 4 * np.pi, self.n_points)
        self.y_clean = np.sin(self.frequency * self.x_clean)
        noise = np.random.normal(0, self.noise_amplitude, self.n_points)
        self.y_noisy = self.y_clean + noise

    def douglas_peucker(self, points, tolerance):
        if len(points) < 3:
            return points

        start = points[0]
        end = points[-1]

        max_distance = 0
        max_index = 0

        for i in range(1, len(points) - 1):
            distance = self.perpendicular_distance(points[i], start, end)
            if distance > max_distance:
                max_distance = distance
                max_index = i

        if max_distance > tolerance:
            left_points = self.douglas_peucker(points[: max_index + 1], tolerance)
            right_points = self.douglas_peucker(points[max_index:], tolerance)
            result = np.vstack([left_points[:-1], right_points])
        else:
            result = np.array([start, end])

        return result

    def perpendicular_distance(self, point, line_start, line_end):
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        if x1 == x2 and y1 == y2:
            return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return numerator / denominator

    def _setup_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self.plotter)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        self.noise_slider = self._make_slider(0, 50, int(self.noise_amplitude * 100))
        self.tolerance_slider = self._make_slider(1, 300, int(self.tolerance * 1000))
        self.frequency_slider = self._make_slider(5, 50, int(self.frequency * 10))

        layout.addWidget(QtWidgets.QLabel("Noise Amplitude"))
        layout.addWidget(self.noise_slider)
        layout.addWidget(QtWidgets.QLabel("DP Tolerance"))
        layout.addWidget(self.tolerance_slider)
        layout.addWidget(QtWidgets.QLabel("Frequency"))
        layout.addWidget(self.frequency_slider)

        regen_button = QtWidgets.QPushButton("Regenerate Noise")
        regen_button.clicked.connect(self.regenerate_noise)
        layout.addWidget(regen_button)
        layout.addStretch(1)

        self.noise_slider.valueChanged.connect(self._on_noise_slider)
        self.tolerance_slider.valueChanged.connect(self._on_tolerance_slider)
        self.frequency_slider.valueChanged.connect(self._on_frequency_slider)

        dock.setWidget(panel)
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    @staticmethod
    def _make_slider(min_val, max_val, value):
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(value)
        return slider

    def _on_noise_slider(self, value):
        self.update_noise(value / 100.0)

    def _on_tolerance_slider(self, value):
        self.update_tolerance(value / 1000.0)

    def _on_frequency_slider(self, value):
        self.update_frequency(value / 10.0)

    def update_noise(self, val):
        self.noise_amplitude = float(val)
        self.generate_data()
        self.update_plot()

    def update_tolerance(self, val):
        self.tolerance = float(val)
        self.update_plot()

    def update_frequency(self, val):
        self.frequency = float(val)
        self.generate_data()
        self.update_plot()

    def regenerate_noise(self):
        self.generate_data()
        self.update_plot()

    @staticmethod
    def _line_cells(points):
        n_points = len(points)
        if n_points < 2:
            return np.array([], dtype=np.int64)
        indices = np.arange(n_points - 1)
        lines = np.column_stack((
            np.full(n_points - 1, 2, dtype=np.int64),
            indices,
            indices + 1,
        ))
        return lines.ravel()

    def _build_scene(self):
        points_clean = np.column_stack(
            [self.x_clean, self.y_clean, np.zeros_like(self.x_clean)]
        )
        points_noisy = np.column_stack(
            [self.x_clean, self.y_noisy, np.zeros_like(self.x_clean)]
        )

        self.clean_line = pv.PolyData(points_clean, self._line_cells(points_clean))
        self.noisy_line = pv.PolyData(points_noisy, self._line_cells(points_noisy))
        self.noisy_points = pv.PolyData(points_noisy)

        points = np.column_stack([self.x_clean, self.y_noisy])
        simplified = self.douglas_peucker(points, self.tolerance)
        simplified_points = np.column_stack(
            [simplified[:, 0], simplified[:, 1], np.zeros(len(simplified))]
        )
        self.simplified_line = pv.PolyData(
            simplified_points, self._line_cells(simplified_points)
        )

        self.plotter.subplot(0, 0)
        self.plotter.add_text(
            "Original Signal with Noise",
            position="upper_left",
            font_size=12,
            name="title_top",
        )
        self.plotter.add_mesh(self.clean_line, color="green", line_width=3)
        self.plotter.add_mesh(self.noisy_line, color="blue", line_width=2, opacity=0.7)
        self.plotter.add_mesh(
            self.noisy_points,
            color="blue",
            point_size=6,
            render_points_as_spheres=True,
        )

        self.plotter.subplot(1, 0)
        self.plotter.add_text(
            "Douglas-Peucker Line Simplification",
            position="upper_left",
            font_size=12,
            name="title_bottom",
        )
        self.plotter.add_mesh(self.clean_line, color="green", line_width=3, opacity=0.6)
        self.plotter.add_mesh(self.noisy_line, color="blue", line_width=2, opacity=0.3)
        self.plotter.add_mesh(self.simplified_line, color="red", line_width=3)

    def update_plot(self):
        points_clean = np.column_stack(
            [self.x_clean, self.y_clean, np.zeros_like(self.x_clean)]
        )
        points_noisy = np.column_stack(
            [self.x_clean, self.y_noisy, np.zeros_like(self.x_clean)]
        )

        self.clean_line.points = points_clean
        self.clean_line.lines = self._line_cells(points_clean)
        self.clean_line.Modified()

        self.noisy_line.points = points_noisy
        self.noisy_line.lines = self._line_cells(points_noisy)
        self.noisy_line.Modified()

        self.noisy_points.points = points_noisy
        self.noisy_points.Modified()

        points = np.column_stack([self.x_clean, self.y_noisy])
        simplified = self.douglas_peucker(points, self.tolerance)
        simplified_points = np.column_stack(
            [simplified[:, 0], simplified[:, 1], np.zeros(len(simplified))]
        )

        self.simplified_line.points = simplified_points
        self.simplified_line.lines = self._line_cells(simplified_points)
        self.simplified_line.Modified()

        original_points = len(self.x_clean)
        simplified_points_count = len(simplified)
        reduction = (1 - simplified_points_count / original_points) * 100
        simplified_y_interp = np.interp(self.x_clean, simplified[:, 0], simplified[:, 1])
        mse = np.mean((self.y_clean - simplified_y_interp) ** 2)

        stats_text = (
            f"Points: {original_points} -> {simplified_points_count} ({reduction:.1f}% reduction)\n"
            f"MSE vs Clean: {mse:.4f}"
        )
        self.plotter.subplot(1, 0)
        self.plotter.add_text(stats_text, position=(10, 10), font_size=10, name="stats")
        self._ensure_camera()
        self.plotter.render()

    def show(self):
        self.plotter.show()
        self.plotter.app.exec_()

    def _ensure_camera(self):
        if self._camera_initialized:
            return
        for row in range(2):
            self.plotter.subplot(row, 0)
            self.plotter.reset_camera()
            self.plotter.enable_parallel_projection()
            self.plotter.view_xy()
        self._camera_initialized = True


def main():
    demo = DouglasPeuckerDemo()
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
    "title": "Douglas-Peucker Line Simplification",
    "description": "Visualize noise reduction using the Ramer-Douglas-Peucker algorithm.\n\n"
    "Features:\n"
    "- Clean sine wave generation\n"
    "- Adjustable noise amplitude\n"
    "- Interactive tolerance control for simplification\n"
    "- Real-time comparison of noisy vs simplified signals\n"
    "- Point reduction statistics and error metrics\n"
    "- Configurable signal frequency",
}


if __name__ == "__main__":
    main()
