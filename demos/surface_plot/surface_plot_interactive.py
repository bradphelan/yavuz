"""
Interactive 3D surface plot with parameter controls.
Demonstrates numpy computations and PyVista 3D plotting with Qt widgets.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
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

        # Create initial surface
        self.mesh_actor = None
        self.update_plot()

        # Create sliders
        self.setup_controls()

    def compute_surface(self):
        """Compute the surface based on current parameters."""
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Z = self.amplitude * np.sin(self.frequency * R + self.phase)
        return X, Y, Z

    def setup_controls(self):
        """Setup GUI controls (sliders and text boxes)."""
        self.plotter.add_slider_widget(
            self.update_frequency,
            [0.1, 5.0],
            value=self.frequency,
            title="Frequency",
            pointa=(0.02, 0.1),
            pointb=(0.34, 0.1),
        )
        self.plotter.add_slider_widget(
            self.update_amplitude,
            [0.1, 3.0],
            value=self.amplitude,
            title="Amplitude",
            pointa=(0.36, 0.1),
            pointb=(0.68, 0.1),
        )
        self.plotter.add_slider_widget(
            self.update_phase,
            [0.0, 2 * np.pi],
            value=self.phase,
            title="Phase",
            pointa=(0.7, 0.1),
            pointb=(0.98, 0.1),
        )

    def update_frequency(self, val):
        """Update frequency parameter."""
        self.frequency = val
        self.update_plot()

    def update_amplitude(self, val):
        """Update amplitude parameter."""
        self.amplitude = val
        self.update_plot()

    def update_phase(self, val):
        """Update phase parameter."""
        self.phase = val
        self.update_plot()

    def update_plot(self):
        """Redraw the surface with updated parameters."""
        if self.mesh_actor:
            self.plotter.remove_actor(self.mesh_actor)

        x, y, z = self.compute_surface()
        grid = pv.StructuredGrid(x, y, z)
        grid["height"] = z.ravel(order="F")

        self.mesh_actor = self.plotter.add_mesh(
            grid,
            scalars="height",
            cmap="viridis",
            smooth_shading=True,
        )
        self.plotter.reset_camera()

    def show(self):
        """Display the interactive plot."""
        self.plotter.show()
        self.plotter.app.exec_()


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
