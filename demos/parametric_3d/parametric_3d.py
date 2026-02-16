"""
Parametric 3D curves with interactive parameter controls.
Demonstrates various 3D mathematical curves with adjustable parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, TextBox
from mpl_toolkits.mplot3d import Axes3D


class Parametric3DVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(left=0.25, bottom=0.25)

        # Parameters
        self.param_a = 1.0
        self.param_b = 1.0
        self.param_c = 1.0
        self.n_points = 1000
        self.curve_type = 'Helix'

        # Initial plot
        self.line = None
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
        # Parameter A slider
        ax_a = plt.axes([0.25, 0.15, 0.5, 0.03])
        self.slider_a = Slider(ax_a, 'Param A', 0.1, 5.0, valinit=self.param_a, valstep=0.1)
        self.slider_a.on_changed(self.update_param_a)

        # Parameter B slider
        ax_b = plt.axes([0.25, 0.10, 0.5, 0.03])
        self.slider_b = Slider(ax_b, 'Param B', 0.1, 5.0, valinit=self.param_b, valstep=0.1)
        self.slider_b.on_changed(self.update_param_b)

        # Parameter C slider
        ax_c = plt.axes([0.25, 0.05, 0.5, 0.03])
        self.slider_c = Slider(ax_c, 'Param C', 0.1, 5.0, valinit=self.param_c, valstep=0.1)
        self.slider_c.on_changed(self.update_param_c)

        # Curve type selector
        ax_type = plt.axes([0.025, 0.3, 0.15, 0.2])
        self.radio = RadioButtons(
            ax_type,
            ('Helix', 'Torus Knot', 'Lissajous', 'Spiral')
        )
        self.radio.on_clicked(self.update_curve_type)

        # Points input box
        ax_points = plt.axes([0.1, 0.02, 0.1, 0.04])
        self.textbox = TextBox(ax_points, 'Points:', initial=str(self.n_points))
        self.textbox.on_submit(self.update_points)

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
        self.curve_type = label
        self.update_plot()

    def update_points(self, text):
        """Update number of points."""
        try:
            self.n_points = int(text)
            self.update_plot()
        except ValueError:
            pass

    def update_plot(self):
        """Redraw the curve with updated parameters."""
        self.ax.clear()

        x, y, z = self.get_curve_data()

        # Create color gradient based on parameter
        colors = np.linspace(0, 1, len(x))

        self.line = self.ax.scatter(x, y, z, c=colors, cmap='viridis', s=1)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'3D Parametric Curve: {self.curve_type}')

        # Set equal aspect ratio
        max_range = max(np.ptp(x), np.ptp(y), np.ptp(z)) / 2
        mid_x = (np.max(x) + np.min(x)) / 2
        mid_y = (np.max(y) + np.min(y)) / 2
        mid_z = (np.max(z) + np.min(z)) / 2
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        self.fig.canvas.draw_idle()

    def show(self):
        """Display the visualizer."""
        plt.show()


def main():
    """Run the parametric 3D visualizer."""
    visualizer = Parametric3DVisualizer()
    visualizer.show()


if __name__ == "__main__":
    main()
