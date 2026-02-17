"""
Line Offset Visualizer
Visualize parallel line offsets with interactive control.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
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

        # Store meshes and actors for reuse (no flickering)
        self.base_line_mesh = None
        self.base_line_actor = None
        self.offset_meshes = []  # List of (pos_line, neg_line, start_cap, end_cap) tuples
        self.offset_actors = []  # Corresponding actors
        self.start_widget = None
        self.end_widget = None

        self._setup_controls()
        self._build_scene()
        self._update_plot()
        self._setup_widgets()  # Add widgets last so they're on top
        self.plotter.view_xy()  # Set to top-down view

    def _setup_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self.plotter)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Num Offsets slider (1-20)
        layout.addWidget(QtWidgets.QLabel("Number of Offsets"))
        self.offsets_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.offsets_slider.setMinimum(1)
        self.offsets_slider.setMaximum(20)
        self.offsets_slider.setValue(self.num_offsets)
        self.offsets_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.offsets_slider.setTickInterval(1)
        self.offsets_slider.valueChanged.connect(self._on_num_offsets)
        layout.addWidget(self.offsets_slider)

        # Stepover slider (0.1-2.0, scaled as 1-20)
        layout.addWidget(QtWidgets.QLabel("Stepover"))
        self.stepover_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.stepover_slider.setMinimum(1)
        self.stepover_slider.setMaximum(20)
        self.stepover_slider.setValue(int(self.stepover * 10))
        self.stepover_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.stepover_slider.setTickInterval(1)
        self.stepover_slider.valueChanged.connect(self._on_stepover)
        layout.addWidget(self.stepover_slider)

        # Info label
        layout.addWidget(QtWidgets.QLabel("Drag the blue spheres to move the line"))

        # Stretch to push controls to top
        layout.addStretch(1)

        dock.setWidget(panel)
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        self._update_labels()

    def _setup_widgets(self):
        self.start_widget = self.plotter.add_sphere_widget(
            self._on_start_move,
            center=(self.start_x, self.start_y, 0.0),
            radius=0.15,
            color="blue",
            pass_widget=True,
            interaction_event='always',
        )
        self.end_widget = self.plotter.add_sphere_widget(
            self._on_end_move,
            center=(self.end_x, self.end_y, 0.0),
            radius=0.15,
            color="blue",
            pass_widget=True,
            interaction_event='always',
        )

    def _on_num_offsets(self, value):
        self.num_offsets = int(value)
        self._update_plot()

    def _on_stepover(self, value):
        self.stepover = value / 10.0
        self._update_plot()

    def _on_start_move(self, center, widget):
        # Constrain to 2D plane (x-y only)
        self.start_x, self.start_y = center[0], center[1]
        # Force z coordinate back to 0
        widget.SetCenter(self.start_x, self.start_y, 0.0)
        self._update_plot()

    def _on_end_move(self, center, widget):
        # Constrain to 2D plane (x-y only)
        self.end_x, self.end_y = center[0], center[1]
        # Force z coordinate back to 0
        widget.SetCenter(self.end_x, self.end_y, 0.0)
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

    def _build_scene(self):
        """Initialize all meshes and actors once."""
        # Create base line using lines_from_points for better dynamic updates
        base_points = np.array([
            [self.start_x, self.start_y, 0.0],
            [self.end_x, self.end_y, 0.0]
        ])
        self.base_line_mesh = pv.lines_from_points(base_points)
        self.base_line_actor = self.plotter.add_mesh(
            self.base_line_mesh, color="blue", line_width=4
        )

        # Create max number of offset lines (20)
        max_offsets = 20
        for i in range(max_offsets):
            # Create straight lines for pos/neg offsets
            pos_mesh = pv.Line((0, 0, 0), (1, 0, 0))
            neg_mesh = pv.Line((0, 0, 0), (1, 0, 0))

            # Create cap arcs (need multiple points)
            angles = np.linspace(0, np.pi, 60)
            cap_x = np.cos(angles)
            cap_y = np.sin(angles)
            cap_points = np.column_stack([cap_x, cap_y, np.zeros_like(cap_x)])
            start_cap_mesh = pv.lines_from_points(cap_points)
            end_cap_mesh = pv.lines_from_points(cap_points)

            color = (0.5, 0.5, 0.5)  # Will be updated

            pos_actor = self.plotter.add_mesh(pos_mesh, color=color, line_width=2)
            neg_actor = self.plotter.add_mesh(neg_mesh, color=color, line_width=2)
            start_cap_actor = self.plotter.add_mesh(start_cap_mesh, color=color, line_width=1)
            end_cap_actor = self.plotter.add_mesh(end_cap_mesh, color=color, line_width=1)

            # Hide initially
            pos_actor.SetVisibility(False)
            neg_actor.SetVisibility(False)
            start_cap_actor.SetVisibility(False)
            end_cap_actor.SetVisibility(False)

            self.offset_meshes.append((pos_mesh, neg_mesh, start_cap_mesh, end_cap_mesh))
            self.offset_actors.append((pos_actor, neg_actor, start_cap_actor, end_cap_actor))

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

    def _update_plot(self):
        """Update all meshes in place without removing/re-adding."""
        # Skip if scene not built yet
        if self.base_line_mesh is None:
            return

        p1 = (self.start_x, self.start_y)
        p2 = (self.end_x, self.end_y)

        # Update base line points directly
        self.base_line_mesh.points = np.array([
            [p1[0], p1[1], 0.0],
            [p2[0], p2[1], 0.0]
        ])
        self.base_line_mesh.Modified()
        self.base_line_actor.mapper.SetInputData(self.base_line_mesh)
        self.base_line_actor.mapper.Update()

        colors = self._gradient_colors(self.num_offsets)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        theta = np.arctan2(dy, dx)

        all_x = [p1[0], p2[0]]
        all_y = [p1[1], p2[1]]

        # Update visible offset lines
        for i in range(len(self.offset_actors)):
            if i < self.num_offsets:
                distance = (i + 1) * self.stepover
                offset_p1_pos, offset_p2_pos = self._get_offset_line(p1, p2, distance)
                offset_p1_neg, offset_p2_neg = self._get_offset_line(p1, p2, -distance)

                # Update meshes
                pos_mesh, neg_mesh, start_cap_mesh, end_cap_mesh = self.offset_meshes[i]
                pos_actor, neg_actor, start_cap_actor, end_cap_actor = self.offset_actors[i]

                # Update straight line points directly
                pos_mesh.points = np.array([
                    [offset_p1_pos[0], offset_p1_pos[1], 0.0],
                    [offset_p2_pos[0], offset_p2_pos[1], 0.0]
                ])
                pos_mesh.Modified()
                pos_actor.mapper.SetInputData(pos_mesh)
                pos_actor.mapper.Update()
                pos_actor.GetProperty().SetColor(colors[i])

                neg_mesh.points = np.array([
                    [offset_p1_neg[0], offset_p1_neg[1], 0.0],
                    [offset_p2_neg[0], offset_p2_neg[1], 0.0]
                ])
                neg_mesh.Modified()
                neg_actor.mapper.SetInputData(neg_mesh)
                neg_actor.mapper.Update()
                neg_actor.GetProperty().SetColor(colors[i])

                # Update caps
                radius = abs(distance)
                cap_start_x, cap_start_y = self._get_end_cap(
                    p1, theta + np.pi / 2, theta + 3 * np.pi / 2, radius
                )
                cap_end_x, cap_end_y = self._get_end_cap(
                    p2, theta - np.pi / 2, theta + np.pi / 2, radius
                )

                start_cap_mesh.points = np.column_stack([
                    cap_start_x, cap_start_y, np.zeros_like(cap_start_x)
                ])
                start_cap_mesh.Modified()
                start_cap_actor.mapper.SetInputData(start_cap_mesh)
                start_cap_actor.mapper.Update()
                start_cap_actor.GetProperty().SetColor(colors[i])

                end_cap_mesh.points = np.column_stack([
                    cap_end_x, cap_end_y, np.zeros_like(cap_end_x)
                ])
                end_cap_mesh.Modified()
                end_cap_actor.mapper.SetInputData(end_cap_mesh)
                end_cap_actor.mapper.Update()
                end_cap_actor.GetProperty().SetColor(colors[i])

                # Show these actors
                pos_actor.SetVisibility(True)
                neg_actor.SetVisibility(True)
                start_cap_actor.SetVisibility(True)
                end_cap_actor.SetVisibility(True)

                all_x.extend([offset_p1_pos[0], offset_p2_pos[0], offset_p1_neg[0], offset_p2_neg[0]])
                all_y.extend([offset_p1_pos[1], offset_p2_pos[1], offset_p1_neg[1], offset_p2_neg[1]])
            else:
                # Hide unused actors
                pos_actor, neg_actor, start_cap_actor, end_cap_actor = self.offset_actors[i]
                pos_actor.SetVisibility(False)
                neg_actor.SetVisibility(False)
                start_cap_actor.SetVisibility(False)
                end_cap_actor.SetVisibility(False)

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
        self.plotter.render()

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
        self.plotter.app.exec()


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
