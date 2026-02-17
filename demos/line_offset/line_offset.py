"""
Line Offset Visualizer
Visualize parallel line offsets with interactive control.
"""

from pathlib import Path
from urllib.parse import quote
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np


class LineOffsetDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Line Offset Visualizer")
        self.root.geometry("1000x700")

        # Parameters
        self.start_x = 1.0
        self.start_y = 1.0
        self.end_x = 8.0
        self.end_y = 6.0
        self.num_offsets = 5
        self.stepover = 0.5
        self.dragging_point = None

        # Create main layout
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Controls panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        # Number of offsets slider
        ttk.Label(control_frame, text="Number of Offsets:").pack(anchor=tk.W, pady=(0, 5))
        self.num_offsets_var = tk.IntVar(value=self.num_offsets)
        num_offsets_scale = ttk.Scale(
            control_frame,
            from_=1,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self.num_offsets_var,
            command=self._on_param_change,
        )
        num_offsets_scale.pack(fill=tk.X, pady=(0, 10))
        self.num_offsets_label = ttk.Label(control_frame, text=f"{self.num_offsets}")
        self.num_offsets_label.pack(anchor=tk.W, pady=(0, 15))

        # Stepover slider
        ttk.Label(control_frame, text="Stepover Distance:").pack(anchor=tk.W, pady=(0, 5))
        self.stepover_var = tk.DoubleVar(value=self.stepover)
        stepover_scale = ttk.Scale(
            control_frame,
            from_=0.1,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.stepover_var,
            command=self._on_param_change,
        )
        stepover_scale.pack(fill=tk.X, pady=(0, 10))
        self.stepover_label = ttk.Label(control_frame, text=f"{self.stepover:.2f}")
        self.stepover_label.pack(anchor=tk.W, pady=(0, 15))

        # Point coordinates display
        ttk.Label(control_frame, text="Start Point:", font=("Arial", 10, "bold")).pack(
            anchor=tk.W, pady=(10, 5)
        )
        self.start_label = ttk.Label(
            control_frame, text=f"({self.start_x:.2f}, {self.start_y:.2f})"
        )
        self.start_label.pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(control_frame, text="End Point:", font=("Arial", 10, "bold")).pack(
            anchor=tk.W, pady=(5, 5)
        )
        self.end_label = ttk.Label(
            control_frame, text=f"({self.end_x:.2f}, {self.end_y:.2f})"
        )
        self.end_label.pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(control_frame, text="(Drag points on plot)", font=("Arial", 9, "italic")).pack(
            anchor=tk.W
        )

        # Canvas for plot
        canvas_frame = ttk.LabelFrame(main_frame, text="Visualization", padding=5)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bind mouse events for dragging
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

        # Initial draw
        self._update_plot()

    def _on_param_change(self, value):
        """Update parameters from sliders."""
        self.num_offsets = int(self.num_offsets_var.get())
        self.stepover = float(self.stepover_var.get())
        self.num_offsets_label.config(text=f"{self.num_offsets}")
        self.stepover_label.config(text=f"{self.stepover:.2f}")
        self._update_plot()

    def _on_press(self, event):
        """Handle mouse press for point dragging."""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Check proximity to start point
        dist_start = np.sqrt((x - self.start_x) ** 2 + (y - self.start_y) ** 2)
        # Check proximity to end point
        dist_end = np.sqrt((x - self.end_x) ** 2 + (y - self.end_y) ** 2)

        threshold = 0.3
        if dist_start < threshold:
            self.dragging_point = "start"
        elif dist_end < threshold:
            self.dragging_point = "end"

    def _on_release(self, event):
        """Handle mouse release."""
        self.dragging_point = None

    def _on_motion(self, event):
        """Handle mouse motion for dragging."""
        if event.inaxes != self.ax or self.dragging_point is None:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        if self.dragging_point == "start":
            self.start_x = x
            self.start_y = y
            self.start_label.config(text=f"({self.start_x:.2f}, {self.start_y:.2f})")
        elif self.dragging_point == "end":
            self.end_x = x
            self.end_y = y
            self.end_label.config(text=f"({self.end_x:.2f}, {self.end_y:.2f})")

        self._update_plot()

    def _get_offset_line(self, p1, p2, distance):
        """Calculate offset line parallel to original line."""
        # Direction vector
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx**2 + dy**2)

        if length == 0:
            return p1, p2

        # Perpendicular vector (rotated 90 degrees)
        px = -dy / length
        py = dx / length

        # Offset points
        offset_p1 = (p1[0] + px * distance, p1[1] + py * distance)
        offset_p2 = (p2[0] + px * distance, p2[1] + py * distance)

        return offset_p1, offset_p2

    def _get_end_cap(self, center, angle_start, angle_end, radius, steps=60):
        """Generate arc points for a rounded end cap."""
        if angle_end < angle_start:
            angle_end += 2 * np.pi
        angles = np.linspace(angle_start, angle_end, steps)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        return x, y

    def _update_plot(self):
        """Redraw the plot with current parameters."""
        self.ax.clear()

        p1 = (self.start_x, self.start_y)
        p2 = (self.end_x, self.end_y)

        # Draw original line
        self.ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], "b-", linewidth=2.5, label="Original", zorder=10
        )
        self.ax.plot([p1[0]], [p1[1]], "bo", markersize=10, zorder=11)
        self.ax.plot([p2[0]], [p2[1]], "bs", markersize=10, zorder=11)

        # Draw offset lines on both sides with rounded end caps
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, self.num_offsets))
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        theta = np.arctan2(dy, dx)

        for i in range(self.num_offsets):
            distance = (i + 1) * self.stepover
            offset_p1_pos, offset_p2_pos = self._get_offset_line(p1, p2, distance)
            offset_p1_neg, offset_p2_neg = self._get_offset_line(p1, p2, -distance)

            self.ax.plot(
                [offset_p1_pos[0], offset_p2_pos[0]],
                [offset_p1_pos[1], offset_p2_pos[1]],
                color=colors[i],
                linewidth=1.5,
                alpha=0.7,
                label=f"Offset +{i + 1} ({distance:.2f})",
            )
            self.ax.plot(
                [offset_p1_neg[0], offset_p2_neg[0]],
                [offset_p1_neg[1], offset_p2_neg[1]],
                color=colors[i],
                linewidth=1.5,
                alpha=0.7,
                label=f"Offset -{i + 1} ({distance:.2f})",
            )

            radius = abs(distance)
            cap_start_x, cap_start_y = self._get_end_cap(
                p1, theta + np.pi / 2, theta + 3 * np.pi / 2, radius
            )
            cap_end_x, cap_end_y = self._get_end_cap(
                p2, theta - np.pi / 2, theta + np.pi / 2, radius
            )
            self.ax.plot(
                cap_start_x,
                cap_start_y,
                color=colors[i],
                linewidth=1.0,
                alpha=0.7,
                label="_nolegend_",
            )
            self.ax.plot(
                cap_end_x,
                cap_end_y,
                color=colors[i],
                linewidth=1.0,
                alpha=0.7,
                label="_nolegend_",
            )

        # Setup plot
        self.ax.set_xlabel("X", fontsize=10)
        self.ax.set_ylabel("Y", fontsize=10)
        self.ax.set_title("Line Offset Visualization", fontsize=12, fontweight="bold")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect("equal")
        self.ax.legend(loc="upper right", fontsize=8)

        # Set reasonable limits
        all_x = [p1[0], p2[0]]
        all_y = [p1[1], p2[1]]
        for i in range(self.num_offsets):
            distance = (i + 1) * self.stepover
            offset_p1_pos, offset_p2_pos = self._get_offset_line(p1, p2, distance)
            offset_p1_neg, offset_p2_neg = self._get_offset_line(p1, p2, -distance)
            all_x.extend([offset_p1_pos[0], offset_p2_pos[0], offset_p1_neg[0], offset_p2_neg[0]])
            all_y.extend([offset_p1_pos[1], offset_p2_pos[1], offset_p1_neg[1], offset_p2_neg[1]])

        margin = 1.0
        self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        self.canvas.draw()


def main():
    """Launch the demo."""
    root = tk.Tk()
    app = LineOffsetDemo(root)
    root.mainloop()


def _build_vscode_url(path):
    posix_path = Path(path).resolve().as_posix()
    if not posix_path.startswith("/"):
        posix_path = f"/{posix_path}"
    return f"vscode://file{quote(posix_path, safe='/:')}"


def get_manifest():
    return {
        "title": "Line Offset Visualizer",
        "description": "Create and visualize parallel line offsets.\n\n"
                       "Features:\n"
                       "- Drag start and end points interactively\n"
                       "- Control number of offset lines (1-20)\n"
                       "- Adjust stepover distance between offsets\n"
                       "- Real-time visualization with color gradients\n"
                       "- Useful for CAM toolpath planning",
        "source_url": _build_vscode_url(__file__),
    }


if __name__ == "__main__":
    main()
