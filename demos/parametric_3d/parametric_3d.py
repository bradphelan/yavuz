"""
Parametric 3D curves with interactive parameter controls.
Demonstrates various 3D mathematical curves with adjustable parameters.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import BackgroundPlotter


class Parametric3DVisualizer:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1200, 800),
            title="Parametric 3D Curves",
        )
        self.plotter.set_background("white")
        self.plotter.add_axes()

        # Parameters
        self.param_a = 1.0
        self.param_b = 1.0
        self.param_c = 1.0
        self.n_points = 1000
        self.curve_types = ["Helix", "Torus Knot", "Lissajous", "Spiral"]
        self.curve_type = self.curve_types[0]

        # Initial plot
        self.mesh = None
        self.mesh_actor = None
        self._build_scene()

        # Setup controls
        self.setup_controls()

    def compute_helix(self):
        """Compute helix curve."""
        t = np.linspace(0, 4*np.pi, self.n_points)
        x = self.param_a * np.cos(t)
        y = self.param_a * np.sin(t)
        z = self.param_b * t
        return x, y, z

    def compute_torus_knot(self):
        """Compute torus knot curve."""
        t = np.linspace(0, 2*np.pi, self.n_points)
        p = int(self.param_a)
        q = int(self.param_b)
        r = self.param_c
        R = 3

        x = (R + r * np.cos(q * t)) * np.cos(p * t)
        y = (R + r * np.cos(q * t)) * np.sin(p * t)
        z = r * np.sin(q * t)
        return x, y, z

    def compute_lissajous(self):
        """Compute 3D Lissajous curve."""
        t = np.linspace(0, 2*np.pi, self.n_points)
        x = np.sin(self.param_a * t)
        y = np.sin(self.param_b * t)
        z = np.sin(self.param_c * t)
        return x, y, z

    def compute_spiral(self):
        """Compute spherical spiral."""
        t = np.linspace(0, 4*np.pi, self.n_points)
        x = self.param_a * np.cos(t) * np.cos(self.param_b * t)
        y = self.param_a * np.sin(t) * np.cos(self.param_b * t)
        z = self.param_a * np.sin(self.param_b * t)
        return x, y, z

    def get_curve_data(self):
        """Get curve data based on selected type."""
        if self.curve_type == 'Helix':
            return self.compute_helix()
        elif self.curve_type == 'Torus Knot':
            return self.compute_torus_knot()
        elif self.curve_type == 'Lissajous':
            return self.compute_lissajous()
        elif self.curve_type == 'Spiral':
            return self.compute_spiral()

    def _build_scene(self):
        """Build the initial scene with mesh."""
        x, y, z = self.get_curve_data()
        points = np.column_stack((x, y, z))
        self.mesh = pv.lines_from_points(points)
        self.mesh["t"] = np.linspace(0, 1, len(points))

        self.mesh_actor = self.plotter.add_mesh(
            self.mesh,
            scalars="t",
            cmap="viridis",
            line_width=3,
        )

        # Get bounds and add padding for initial grid
        bounds = self.mesh.bounds
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]
        max_range = max(x_range, y_range, z_range)
        padding = max_range * 0.2

        # Create grid that encompasses the curve
        grid_bounds = [
            bounds[0] - padding, bounds[1] + padding,
            bounds[2] - padding, bounds[3] + padding,
            bounds[4] - padding, bounds[5] + padding
        ]
        self.plotter.show_bounds(
            bounds=grid_bounds,
            grid='back',
            location='outer',
            ticks='both'
        )

        self.plotter.reset_camera()

    def setup_controls(self):
        """Setup GUI controls using Qt dock widget."""
        dock = QtWidgets.QDockWidget("Controls", self.plotter)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Param A slider (0.1-5.0, scaled as 1-50)
        layout.addWidget(QtWidgets.QLabel("Param A"))
        self.slider_a = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_a.setMinimum(1)
        self.slider_a.setMaximum(50)
        self.slider_a.setValue(int(self.param_a * 10))
        self.slider_a.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_a.setTickInterval(5)
        self.slider_a.valueChanged.connect(self._on_param_a)
        layout.addWidget(self.slider_a)

        # Param B slider (0.1-5.0, scaled as 1-50)
        layout.addWidget(QtWidgets.QLabel("Param B"))
        self.slider_b = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b.setMinimum(1)
        self.slider_b.setMaximum(50)
        self.slider_b.setValue(int(self.param_b * 10))
        self.slider_b.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b.setTickInterval(5)
        self.slider_b.valueChanged.connect(self._on_param_b)
        layout.addWidget(self.slider_b)

        # Param C slider (0.1-5.0, scaled as 1-50)
        layout.addWidget(QtWidgets.QLabel("Param C"))
        self.slider_c = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_c.setMinimum(1)
        self.slider_c.setMaximum(50)
        self.slider_c.setValue(int(self.param_c * 10))
        self.slider_c.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_c.setTickInterval(5)
        self.slider_c.valueChanged.connect(self._on_param_c)
        layout.addWidget(self.slider_c)

        # Curve Type slider (0-3)
        layout.addWidget(QtWidgets.QLabel("Curve Type"))
        self.slider_curve = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_curve.setMinimum(0)
        self.slider_curve.setMaximum(len(self.curve_types) - 1)
        self.slider_curve.setValue(0)
        self.slider_curve.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_curve.valueChanged.connect(self._on_curve_type)
        layout.addWidget(self.slider_curve)

        # Points slider (200-3000, scaled as 20-300)
        layout.addWidget(QtWidgets.QLabel("Points"))
        self.slider_points = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_points.setMinimum(20)
        self.slider_points.setMaximum(300)
        self.slider_points.setValue(int(self.n_points / 10))
        self.slider_points.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_points.setTickInterval(10)
        self.slider_points.valueChanged.connect(self._on_points)
        layout.addWidget(self.slider_points)

        # Stretch to push controls to top
        layout.addStretch(1)

        dock.setWidget(panel)
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _on_param_a(self, value):
        """Update parameter A."""
        self.param_a = value / 10.0
        self.update_plot()

    def _on_param_b(self, value):
        """Update parameter B."""
        self.param_b = value / 10.0
        self.update_plot()

    def _on_param_c(self, value):
        """Update parameter C."""
        self.param_c = value / 10.0
        self.update_plot()

    def _on_curve_type(self, value):
        """Update curve type."""
        idx = int(value)
        if 0 <= idx < len(self.curve_types):
            self.curve_type = self.curve_types[idx]
            self.update_plot()

    def _on_points(self, value):
        """Update number of points."""
        self.n_points = max(50, value * 10)
        self.update_plot()

    def update_plot(self):
        """Redraw the curve with updated parameters."""
        x, y, z = self.get_curve_data()
        points = np.column_stack((x, y, z))

        # Reuse mesh, update in-place
        self.mesh.points = points
        self.mesh["t"] = np.linspace(0, 1, len(points))
        self.mesh.Modified()

        if self.mesh_actor is not None:
            self.mesh_actor.mapper.SetInputData(self.mesh)
            self.mesh_actor.mapper.Update()

        # Reset camera to fully encompass the curve
        self.plotter.reset_camera()
        self.plotter.render()

    def show(self):
        """Display the visualizer."""
        self.plotter.show()
        self.plotter.app.exec()


def main():
    """Run the parametric 3D visualizer."""
    visualizer = Parametric3DVisualizer()
    visualizer.show()


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
    "title": "Parametric 3D Curves",
    "description": "Explore 3D mathematical curves with interactive parameters.\n\n"
    "Features:\n"
    "- Curve types: Helix, Torus Knot, Lissajous, Spiral\n"
    "- Three adjustable parameters per curve\n"
    "- Color gradients based on position\n"
    "- Configurable point density\n"
    "- Interactive 3D rotation",
}


if __name__ == "__main__":
    main()
