"""
Line Offset Visualizer
Visualize parallel line offsets with interactive control.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter


class LineOffsetDemo:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1200, 800),
            title="Line Offset Visualizer",
        )
        self.plotter.set_background("white")
        self.plotter.show_grid()

        self.start_x = 1.0
        self.start_y = 1.0
        self.end_x = 8.0
        self.end_y = 6.0
        self.num_offsets = 5
        self.stepover = 0.5

        self.actors = []

        self._setup_controls()
        self._setup_widgets()
        self._update_plot()

    def _setup_controls(self):
        self.plotter.add_slider_widget(
            self._on_num_offsets,
            [1, 20],
            value=self.num_offsets,
            title="Offsets",
            pointa=(0.02, 0.1),
            pointb=(0.32, 0.1),
        )
        self.plotter.add_slider_widget(
            self._on_stepover,
            [0.1, 2.0],
            value=self.stepover,
            title="Stepover",
            pointa=(0.35, 0.1),
            pointb=(0.65, 0.1),
        )
        self.plotter.add_text("Drag the spheres to move the line", position="upper_left", font_size=10)
        self._update_labels()

    def _setup_widgets(self):
        self.plotter.add_sphere_widget(
            self._on_start_move,
            center=(self.start_x, self.start_y, 0.0),
            radius=0.15,
            color="blue",
        )
        self.plotter.add_sphere_widget(
            self._on_end_move,
            center=(self.end_x, self.end_y, 0.0),
            radius=0.15,
            color="blue",
        )

    def _on_num_offsets(self, value):
        self.num_offsets = int(round(value))
        self._update_plot()

    def _on_stepover(self, value):
        self.stepover = float(value)
        self._update_plot()

    def _on_start_move(self, center):
        self.start_x, self.start_y = center[0], center[1]
        self._update_plot()

    def _on_end_move(self, center):
        self.end_x, self.end_y = center[0], center[1]
        self._update_plot()

    def _get_offset_line(self, p1, p2, distance):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx ** 2 + dy ** 2)

        if length == 0:
            return p1, p2

        px = -dy / length
        py = dx / length

        offset_p1 = (p1[0] + px * distance, p1[1] + py * distance)
        offset_p2 = (p2[0] + px * distance, p2[1] + py * distance)

        return offset_p1, offset_p2

    def _get_end_cap(self, center, angle_start, angle_end, radius, steps=60):
        if angle_end < angle_start:
            angle_end += 2 * np.pi
        angles = np.linspace(angle_start, angle_end, steps)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        return x, y

    def _line_actor(self, points, color, width=2):
        poly = pv.lines_from_points(points)
        return self.plotter.add_mesh(poly, color=color, line_width=width)

    def _update_labels(self):
        self.plotter.add_text(
            f"Start: ({self.start_x:.2f}, {self.start_y:.2f})",
            position=(10, 60),
            font_size=10,
            name="start_label",
        )
        self.plotter.add_text(
            f"End: ({self.end_x:.2f}, {self.end_y:.2f})",
            position=(10, 40),
            font_size=10,
            name="end_label",
        )

    def _clear_actors(self):
        for actor in self.actors:
            self.plotter.remove_actor(actor)
        self.actors = []

    def _update_plot(self):
        self._clear_actors()

        p1 = (self.start_x, self.start_y)
        p2 = (self.end_x, self.end_y)

        base_points = np.array([[p1[0], p1[1], 0.0], [p2[0], p2[1], 0.0]])
        self.actors.append(self._line_actor(base_points, color="blue", width=4))

        colors = self._gradient_colors(self.num_offsets)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        theta = np.arctan2(dy, dx)

        all_x = [p1[0], p2[0]]
        all_y = [p1[1], p2[1]]

        for i in range(self.num_offsets):
            distance = (i + 1) * self.stepover
            offset_p1_pos, offset_p2_pos = self._get_offset_line(p1, p2, distance)
            offset_p1_neg, offset_p2_neg = self._get_offset_line(p1, p2, -distance)

            pos_points = np.array([
                [offset_p1_pos[0], offset_p1_pos[1], 0.0],
                [offset_p2_pos[0], offset_p2_pos[1], 0.0],
            ])
            neg_points = np.array([
                [offset_p1_neg[0], offset_p1_neg[1], 0.0],
                [offset_p2_neg[0], offset_p2_neg[1], 0.0],
            ])

            self.actors.append(self._line_actor(pos_points, color=colors[i], width=2))
            self.actors.append(self._line_actor(neg_points, color=colors[i], width=2))

            radius = abs(distance)
            cap_start_x, cap_start_y = self._get_end_cap(
                p1, theta + np.pi / 2, theta + 3 * np.pi / 2, radius
            )
            cap_end_x, cap_end_y = self._get_end_cap(
                p2, theta - np.pi / 2, theta + np.pi / 2, radius
            )

            cap_start = np.column_stack([cap_start_x, cap_start_y, np.zeros_like(cap_start_x)])
            cap_end = np.column_stack([cap_end_x, cap_end_y, np.zeros_like(cap_end_x)])

            self.actors.append(self._line_actor(cap_start, color=colors[i], width=1))
            self.actors.append(self._line_actor(cap_end, color=colors[i], width=1))

            all_x.extend([offset_p1_pos[0], offset_p2_pos[0], offset_p1_neg[0], offset_p2_neg[0]])
            all_y.extend([offset_p1_pos[1], offset_p2_pos[1], offset_p1_neg[1], offset_p2_neg[1]])

        self._update_labels()

        margin = 1.0
        bounds = (
            min(all_x) - margin,
            max(all_x) + margin,
            min(all_y) - margin,
            max(all_y) + margin,
            -1.0,
            1.0,
        )
        self.plotter.reset_camera(bounds=bounds)

    def _gradient_colors(self, count):
        if count <= 1:
            return [(0.2, 0.8, 0.2)]
        colors = []
        for i in range(count):
            t = i / (count - 1)
            r = 0.2 + 0.6 * t
            g = 0.8 - 0.5 * t
            b = 0.2
            colors.append((r, g, b))
        return colors

    def show(self):
        self.plotter.show()
        self.plotter.app.exec_()


def main():
    demo = LineOffsetDemo()
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
    "title": "Line Offset Visualizer",
    "description": "Create and visualize parallel line offsets.\n\n"
    "Features:\n"
    "- Drag start and end points interactively\n"
    "- Control number of offset lines (1-20)\n"
    "- Adjust stepover distance between offsets\n"
    "- Real-time visualization with color gradients\n"
    "- Useful for CAM toolpath planning",
}


if __name__ == "__main__":
    main()
