"""
Strange Attractor Demo
Animate classic strange attractors with interactive controls.
"""

from collections import deque
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
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

        self.line_actor = None
        self.head_actor = None

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
        self.slider_system = self.plotter.add_slider_widget(
            self._on_system_change,
            [0, len(self.systems) - 1],
            value=self.system_index,
            title="System",
            pointa=(0.02, 0.1),
            pointb=(0.28, 0.1),
        )
        self.slider_a = self.plotter.add_slider_widget(
            self._on_param_a,
            [0.1, 30.0],
            value=self.defaults["Lorenz"]["a"],
            title="Param A",
            pointa=(0.32, 0.1),
            pointb=(0.58, 0.1),
        )
        self.slider_b = self.plotter.add_slider_widget(
            self._on_param_b,
            [0.1, 35.0],
            value=self.defaults["Lorenz"]["b"],
            title="Param B",
            pointa=(0.62, 0.1),
            pointb=(0.88, 0.1),
        )
        self.slider_c = self.plotter.add_slider_widget(
            self._on_param_c,
            [0.1, 10.0],
            value=self.defaults["Lorenz"]["c"],
            title="Param C",
            pointa=(0.02, 0.06),
            pointb=(0.28, 0.06),
        )
        self.slider_dt = self.plotter.add_slider_widget(
            self._on_dt,
            [0.001, 0.03],
            value=self.dt,
            title="Time Step",
            pointa=(0.32, 0.06),
            pointb=(0.58, 0.06),
        )
        self.slider_speed = self.plotter.add_slider_widget(
            self._on_speed,
            [1, 60],
            value=self.steps_per_frame,
            title="Steps/Frame",
            pointa=(0.62, 0.06),
            pointb=(0.88, 0.06),
        )
        self.slider_trail = self.plotter.add_slider_widget(
            self._on_trail,
            [500, 12000],
            value=self.trail_length,
            title="Trail",
            pointa=(0.02, 0.02),
            pointb=(0.28, 0.02),
        )
        self.plotter.add_checkbox_button_widget(
            self._toggle_pause,
            value=self.is_paused,
            position=(10, 10),
            size=30,
        )
        self.plotter.add_text("Pause", position=(45, 14), font_size=10)
        self.plotter.add_text("Reset: R", position=(95, 14), font_size=10)
        self.plotter.add_key_event("r", self._reset)

    def _update_scene(self):
        data = np.array(self.trail)
        if len(data) < 2:
            return

        if self.line_actor:
            self.plotter.remove_actor(self.line_actor)
            self.line_actor = None
        if self.head_actor:
            self.plotter.remove_actor(self.head_actor)
            self.head_actor = None

        line = pv.lines_from_points(data)
        self.line_actor = self.plotter.add_mesh(line, color="#1f77b4", line_width=2)

        head = pv.PolyData(data[-1:])
        self.head_actor = self.plotter.add_mesh(
            head,
            color="#ff7f0e",
            point_size=10,
            render_points_as_spheres=True,
        )

        self.plotter.reset_camera()

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
        self.system_index = int(round(value))
        self._apply_defaults()

    def _on_param_a(self, value):
        system = self.systems[self.system_index]
        self.params[system]["a"] = float(value)

    def _on_param_b(self, value):
        system = self.systems[self.system_index]
        self.params[system]["b"] = float(value)

    def _on_param_c(self, value):
        system = self.systems[self.system_index]
        self.params[system]["c"] = float(value)

    def _on_dt(self, value):
        self.dt = float(value)
        self._update_hud()

    def _on_speed(self, value):
        self.steps_per_frame = int(round(value))
        self._update_hud()

    def _on_trail(self, value):
        self.trail_length = int(round(value))
        self.trail = deque(self.trail, maxlen=self.trail_length)
        self._update_hud()

    def _toggle_pause(self, state):
        self.is_paused = bool(state)

    def _apply_defaults(self):
        system = self.systems[self.system_index]
        defaults = self.defaults[system]
        self.params[system] = dict(defaults)
        self._set_slider_value(self.slider_a, defaults["a"])
        self._set_slider_value(self.slider_b, defaults["b"])
        self._set_slider_value(self.slider_c, defaults["c"])
        self._reset_state()

    def _reset_state(self):
        self.state = np.array([0.1, 0.0, 0.0])
        self.dt = 0.01
        self.steps_per_frame = 10
        self.trail_length = 5000
        self._set_slider_value(self.slider_dt, self.dt)
        self._set_slider_value(self.slider_speed, self.steps_per_frame)
        self._set_slider_value(self.slider_trail, self.trail_length)
        self._seed_trail()
        self._update_hud()
        self._update_scene()

    @staticmethod
    def _set_slider_value(slider, value):
        slider.GetRepresentation().SetValue(value)

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
        self.plotter.app.exec_()


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
