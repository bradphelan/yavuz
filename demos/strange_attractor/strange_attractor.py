"""
Strange Attractor Demo
Animate classic strange attractors with interactive controls.
"""

from collections import deque
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import BackgroundPlotter


class StrangeAttractorDemo:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1200, 800),
            title="Strange Attractor Explorer",
        )
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.plotter.show_grid()

        self.systems = ["Lorenz", "Rossler"]
        self.system_index = 0

        self.dt = 0.01
        self.steps_per_frame = 10
        self.trail_length = 5000
        self.is_paused = False

        self.defaults = {
            "Lorenz": {"a": 10.0, "b": 28.0, "c": 8.0 / 3.0},
            "Rossler": {"a": 0.2, "b": 0.2, "c": 5.7},
        }
        self.params = {
            "Lorenz": dict(self.defaults["Lorenz"]),
            "Rossler": dict(self.defaults["Rossler"]),
        }

        self.state = np.array([0.1, 0.0, 0.0])
        self.trail = deque(maxlen=self.trail_length)
        self._seed_trail()

        self.line_mesh = None
        self.line_actor = None
        self.head_mesh = None
        self.head_actor = None
        self._camera_initialized = False

        self._setup_controls()
        self._update_hud()
        self._update_scene()

        self.callback_id = self.plotter.add_callback(self._animate, interval=30)

    def _seed_trail(self):
        self.trail.clear()
        current = self.state.copy()
        for _ in range(self.trail.maxlen // 4):
            current = self._step(current)
            if current is None:
                break
            self.trail.append(current.copy())
        if len(self.trail) > 0:
            self.state = self.trail[-1].copy()

    def _step(self, state):
        system = self.systems[self.system_index]
        if system == "Lorenz":
            a = self.params["Lorenz"]["a"]
            b = self.params["Lorenz"]["b"]
            c = self.params["Lorenz"]["c"]
            x, y, z = state
            dx = a * (y - x)
            dy = x * (b - z) - y
            dz = x * y - c * z
        else:
            a = self.params["Rossler"]["a"]
            b = self.params["Rossler"]["b"]
            c = self.params["Rossler"]["c"]
            x, y, z = state
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)

        next_state = state + self.dt * np.array([dx, dy, dz])
        if not np.all(np.isfinite(next_state)):
            return None
        return next_state

    def _setup_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self.plotter)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # System selector (0-1)
        layout.addWidget(QtWidgets.QLabel("System"))
        self.slider_system = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_system.setMinimum(0)
        self.slider_system.setMaximum(len(self.systems) - 1)
        self.slider_system.setValue(self.system_index)
        self.slider_system.valueChanged.connect(self._on_system_change)
        layout.addWidget(self.slider_system)

        # Param A slider (0.1-30.0, scaled as 1-300)
        layout.addWidget(QtWidgets.QLabel("Param A"))
        self.slider_a = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_a.setMinimum(1)
        self.slider_a.setMaximum(300)
        self.slider_a.setValue(int(self.defaults["Lorenz"]["a"] * 10))
        self.slider_a.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_a.setTickInterval(20)
        self.slider_a.valueChanged.connect(self._on_param_a)
        layout.addWidget(self.slider_a)

        # Param B slider (0.1-35.0, scaled as 1-350)
        layout.addWidget(QtWidgets.QLabel("Param B"))
        self.slider_b = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_b.setMinimum(1)
        self.slider_b.setMaximum(350)
        self.slider_b.setValue(int(self.defaults["Lorenz"]["b"] * 10))
        self.slider_b.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_b.setTickInterval(20)
        self.slider_b.valueChanged.connect(self._on_param_b)
        layout.addWidget(self.slider_b)

        # Param C slider (0.1-10.0, scaled as 1-100)
        layout.addWidget(QtWidgets.QLabel("Param C"))
        self.slider_c = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_c.setMinimum(1)
        self.slider_c.setMaximum(100)
        self.slider_c.setValue(int(self.defaults["Lorenz"]["c"] * 10))
        self.slider_c.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_c.setTickInterval(5)
        self.slider_c.valueChanged.connect(self._on_param_c)
        layout.addWidget(self.slider_c)

        # Time Step slider (0.001-0.03, scaled as 1-30)
        layout.addWidget(QtWidgets.QLabel("Time Step"))
        self.slider_dt = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_dt.setMinimum(1)
        self.slider_dt.setMaximum(30)
        self.slider_dt.setValue(int(self.dt * 1000))
        self.slider_dt.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_dt.setTickInterval(2)
        self.slider_dt.valueChanged.connect(self._on_dt)
        layout.addWidget(self.slider_dt)

        # Steps/Frame slider (1-60)
        layout.addWidget(QtWidgets.QLabel("Steps/Frame"))
        self.slider_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_speed.setMinimum(1)
        self.slider_speed.setMaximum(60)
        self.slider_speed.setValue(self.steps_per_frame)
        self.slider_speed.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_speed.setTickInterval(5)
        self.slider_speed.valueChanged.connect(self._on_speed)
        layout.addWidget(self.slider_speed)

        # Trail Length slider (500-12000, scaled as 50-1200)
        layout.addWidget(QtWidgets.QLabel("Trail Length"))
        self.slider_trail = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_trail.setMinimum(50)
        self.slider_trail.setMaximum(1200)
        self.slider_trail.setValue(int(self.trail_length / 10))
        self.slider_trail.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_trail.setTickInterval(50)
        self.slider_trail.valueChanged.connect(self._on_trail)
        layout.addWidget(self.slider_trail)

        # Pause checkbox
        self.pause_checkbox = QtWidgets.QCheckBox("Pause")
        self.pause_checkbox.setChecked(self.is_paused)
        self.pause_checkbox.stateChanged.connect(self._toggle_pause)
        layout.addWidget(self.pause_checkbox)

        # Stretch to push controls to top
        layout.addStretch(1)

        dock.setWidget(panel)
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        # Add key event for reset
        self.plotter.add_text("Reset: R", position=(10, 14), font_size=10)
        self.plotter.add_key_event("r", self._reset)

    def _update_scene(self):
        data = np.array(self.trail)
        if len(data) < 2:
            return

        if self.line_mesh is None:
            # First time: create the mesh and actor
            self.line_mesh = pv.lines_from_points(data)
            self.line_actor = self.plotter.add_mesh(self.line_mesh, color="#1f77b4", line_width=2)
        else:
            # Update line mesh: rebuild connectivity for new points
            new_line_mesh = pv.lines_from_points(data)
            self.line_mesh.copy_from(new_line_mesh)
            self.line_mesh.Modified()

            if self.line_actor is not None:
                self.line_actor.mapper.SetInputData(self.line_mesh)
                self.line_actor.mapper.Update()

        if self.head_mesh is None:
            # First time: create head mesh and actor
            head_point = data[-1:]
            self.head_mesh = pv.PolyData(head_point)
            self.head_actor = self.plotter.add_mesh(
                self.head_mesh,
                color="#ff7f0e",
                point_size=10,
                render_points_as_spheres=True,
            )
        else:
            # Subsequent frames: update head position
            head_point = data[-1:]
            self.head_mesh.points = head_point
            self.head_mesh.Modified()

            if self.head_actor is not None:
                self.head_actor.mapper.SetInputData(self.head_mesh)
                self.head_actor.mapper.Update()

        if not self._camera_initialized:
            self.plotter.reset_camera()
            self._camera_initialized = True

    def _update_hud(self):
        system = self.systems[self.system_index]
        text = (
            "Strange Attractor Explorer\n"
            f"System: {system}\n"
            f"Time Step: {self.dt:.3f}  Steps/Frame: {self.steps_per_frame}\n"
            f"Trail: {self.trail_length}"
        )
        self.plotter.add_text(text, position="upper_left", font_size=10, name="hud")

    def _on_system_change(self, value):
        self.system_index = int(value)
        self._apply_defaults()

    def _on_param_a(self, value):
        system = self.systems[self.system_index]
        self.params[system]["a"] = value / 10.0

    def _on_param_b(self, value):
        system = self.systems[self.system_index]
        self.params[system]["b"] = value / 10.0

    def _on_param_c(self, value):
        system = self.systems[self.system_index]
        self.params[system]["c"] = value / 10.0

    def _on_dt(self, value):
        self.dt = value / 1000.0
        self._update_hud()

    def _on_speed(self, value):
        self.steps_per_frame = int(value)
        self._update_hud()

    def _on_trail(self, value):
        self.trail_length = value * 10
        self.trail = deque(self.trail, maxlen=self.trail_length)
        self._update_hud()

    def _toggle_pause(self, state):
        self.is_paused = bool(state)

    def _apply_defaults(self):
        system = self.systems[self.system_index]
        defaults = self.defaults[system]
        self.params[system] = dict(defaults)
        self.slider_a.setValue(int(defaults["a"] * 10))
        self.slider_b.setValue(int(defaults["b"] * 10))
        self.slider_c.setValue(int(defaults["c"] * 10))
        self._reset_state()

    def _reset_state(self):
        self.state = np.array([0.1, 0.0, 0.0])
        self.dt = 0.01
        self.steps_per_frame = 10
        self.trail_length = 5000
        self.slider_dt.setValue(int(self.dt * 1000))
        self.slider_speed.setValue(self.steps_per_frame)
        self.slider_trail.setValue(int(self.trail_length / 10))
        self._seed_trail()
        self._update_hud()
        self._update_scene()

    def _reset(self):
        self._apply_defaults()

    def _animate(self):
        if self.is_paused:
            return
        for _ in range(self.steps_per_frame):
            next_state = self._step(self.state)
            if next_state is None:
                self._reset_state()
                break
            self.state = next_state
            self.trail.append(self.state.copy())
        self._update_scene()

    def show(self):
        self.plotter.show()
        self.plotter.app.exec()


def main():
    demo = StrangeAttractorDemo()
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
    "title": "Strange Attractor Explorer",
    "description": "Animate classic strange attractors with interactive control.\n\n"
    "Features:\n"
    "- Lorenz and Rossler attractors\n"
    "- Adjustable parameters and time step\n"
    "- Trail length and speed controls\n"
    "- Pause and reset controls",
}


if __name__ == "__main__":
    main()
