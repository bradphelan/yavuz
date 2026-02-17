"""
Parametric 3D curves with interactive parameter controls.
Demonstrates various 3D mathematical curves with adjustable parameters.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter


class Parametric3DVisualizer:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1200, 800),
            title="Parametric 3D Curves",
        )
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.plotter.show_grid()

        # Parameters
        self.param_a = 1.0
        self.param_b = 1.0
        self.param_c = 1.0
        self.n_points = 1000
        self.curve_types = ["Helix", "Torus Knot", "Lissajous", "Spiral"]
        self.curve_type = self.curve_types[0]

        # Initial plot
        self.actor = None
        self.update_plot()

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

    def setup_controls(self):
        """Setup GUI controls."""
        self.plotter.add_slider_widget(
            self.update_param_a,
            [0.1, 5.0],
            value=self.param_a,
            title="Param A",
            pointa=(0.02, 0.1),
            pointb=(0.3, 0.1),
        )
        self.plotter.add_slider_widget(
            self.update_param_b,
            [0.1, 5.0],
            value=self.param_b,
            title="Param B",
            pointa=(0.35, 0.1),
            pointb=(0.63, 0.1),
        )
        self.plotter.add_slider_widget(
            self.update_param_c,
            [0.1, 5.0],
            value=self.param_c,
            title="Param C",
            pointa=(0.68, 0.1),
            pointb=(0.96, 0.1),
        )
        self.plotter.add_slider_widget(
            self.update_curve_type,
            [0, len(self.curve_types) - 1],
            value=0,
            title="Curve Type",
            pointa=(0.02, 0.04),
            pointb=(0.48, 0.04),
        )
        self.plotter.add_slider_widget(
            self.update_points,
            [200, 3000],
            value=self.n_points,
            title="Points",
            pointa=(0.55, 0.04),
            pointb=(0.96, 0.04),
        )

    def update_param_a(self, val):
        """Update parameter A."""
        self.param_a = val
        self.update_plot()

    def update_param_b(self, val):
        """Update parameter B."""
        self.param_b = val
        self.update_plot()

    def update_param_c(self, val):
        """Update parameter C."""
        self.param_c = val
        self.update_plot()

    def update_curve_type(self, label):
        """Update curve type."""
        idx = int(round(label))
        idx = max(0, min(idx, len(self.curve_types) - 1))
        self.curve_type = self.curve_types[idx]
        self.update_plot()

    def update_points(self, text):
        """Update number of points."""
        self.n_points = max(50, int(round(text)))
        self.update_plot()

    def update_plot(self):
        """Redraw the curve with updated parameters."""
        if self.actor:
            self.plotter.remove_actor(self.actor)

        x, y, z = self.get_curve_data()
        points = np.column_stack((x, y, z))
        poly = pv.lines_from_points(points)
        poly["t"] = np.linspace(0, 1, len(points))

        self.actor = self.plotter.add_mesh(
            poly,
            scalars="t",
            cmap="viridis",
            line_width=3,
        )

        self.plotter.reset_camera()

    def show(self):
        """Display the visualizer."""
        self.plotter.show()
        self.plotter.app.exec_()


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
