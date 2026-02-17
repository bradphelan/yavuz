"""
Interactive 3D surface plot with parameter controls.
Demonstrates numpy computations and PyVista 3D plotting with Qt widgets.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import BackgroundPlotter


class InteractiveSurface:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1200, 800),
            title="Interactive 3D Surface",
        )
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.plotter.show_grid()

        # Initial parameters
        self.frequency = 1.0
        self.amplitude = 1.0
        self.phase = 0.0

        self.grid = None
        self.mesh_actor = None
        self._camera_initialized = False

        # Create controls
        self.setup_controls()
        self._build_scene()
        self.update_plot()

    def compute_surface(self):
        """Compute the surface based on current parameters."""
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Z = self.amplitude * np.sin(self.frequency * R + self.phase)
        return X, Y, Z

    def setup_controls(self):
        """Setup GUI controls (dock panel)."""
        dock = QtWidgets.QDockWidget("Controls", self.plotter.app_window)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        self.frequency_slider = self._make_slider(1, 50, int(self.frequency * 10))
        self.amplitude_slider = self._make_slider(1, 30, int(self.amplitude * 10))
        self.phase_slider = self._make_slider(0, 628, int(self.phase * 100))

        layout.addWidget(QtWidgets.QLabel("Frequency"))
        layout.addWidget(self.frequency_slider)
        layout.addWidget(QtWidgets.QLabel("Amplitude"))
        layout.addWidget(self.amplitude_slider)
        layout.addWidget(QtWidgets.QLabel("Phase"))
        layout.addWidget(self.phase_slider)
        layout.addStretch(1)

        self.frequency_slider.valueChanged.connect(self._on_frequency_slider)
        self.amplitude_slider.valueChanged.connect(self._on_amplitude_slider)
        self.phase_slider.valueChanged.connect(self._on_phase_slider)

        dock.setWidget(panel)
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    @staticmethod
    def _make_slider(min_val, max_val, value):
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(value)
        return slider

    def _on_frequency_slider(self, value):
        self.update_frequency(value / 10.0)

    def _on_amplitude_slider(self, value):
        self.update_amplitude(value / 10.0)

    def _on_phase_slider(self, value):
        self.update_phase(value / 100.0)

    def update_frequency(self, val):
        """Update frequency parameter."""
        self.frequency = float(val)
        self.update_plot()

    def update_amplitude(self, val):
        """Update amplitude parameter."""
        self.amplitude = float(val)
        self.update_plot()

    def update_phase(self, val):
        """Update phase parameter."""
        self.phase = float(val)
        self.update_plot()

    def _build_scene(self):
        x, y, z = self.compute_surface()
        self.grid = pv.StructuredGrid(x, y, z)
        self.grid["height"] = z.ravel(order="F")
        self.mesh_actor = self.plotter.add_mesh(
            self.grid,
            scalars="height",
            cmap="viridis",
            smooth_shading=True,
        )

    def update_plot(self):
        """Redraw the surface with updated parameters."""
        x, y, z = self.compute_surface()
        points = np.column_stack([x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")])
        self.grid.points = points
        self.grid.point_data["height"] = z.ravel(order="F")
        self.grid.Modified()
        if self.mesh_actor is not None:
            self.mesh_actor.mapper.SetInputData(self.grid)
            self.mesh_actor.mapper.Update()

        if not self._camera_initialized:
            self.plotter.reset_camera()
            self._camera_initialized = True
        self.plotter.render()

    def show(self):
        """Display the interactive plot."""
        self.plotter.show()
        self.plotter.app.exec()


def main():
    """Run the interactive surface plot."""
    app = InteractiveSurface()
    app.show()


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
    "title": "Interactive 3D Surface Plot",
    "description": "Real-time 3D surface visualization with adjustable parameters.\n\n"
    "Features:\n"
    "- Interactive 3D surface rendering\n"
    "- Frequency, amplitude, and phase controls\n"
    "- Slider-based parameter adjustment\n"
    "- Mathematical function: Z = A*sin(f*sqrt(X^2+Y^2) + phi)",
}


if __name__ == "__main__":
    main()
