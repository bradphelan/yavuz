"""
Strange Attractor Demo
Animate classic strange attractors with interactive controls and mouse dragging.
"""

from collections import deque
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, RadioButtons, Button
import numpy as np


class StrangeAttractorDemo:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.fig.subplots_adjust(left=0.25, bottom=0.3)

        # Simulation state
        self.system = "Lorenz"
        self.dt = 0.01
        self.steps_per_frame = 10
        self.trail_length = 5000
        self.offset = np.array([0.0, 0.0])
        self.dragging = False
        self.last_mouse = None
        self.is_paused = False

        # Parameters
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

        # Plot elements
        self.line, = self.ax.plot([], [], color="#1f77b4", linewidth=0.8)
        self.head, = self.ax.plot([], [], "o", color="#ff7f0e", markersize=4)
        self.origin, = self.ax.plot([], [], "+", color="#111111", markersize=8)

        self.ax.set_title("Strange Attractor: Lorenz")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        self._setup_controls()
        self._update_param_labels()
        self._connect_mouse_events()

        self._draw_from_trail()

        self.anim = FuncAnimation(
            self.fig,
            self._animate,
            interval=30,
            blit=False,
            cache_frame_data=False,
        )

    def _seed_trail(self):
        self.trail.clear()
        current = self.state.copy()
        for _ in range(self.trail.maxlen // 4):
            current = self._step(current)
            self.trail.append(current.copy())
        self.state = current

    def _step(self, state):
        if self.system == "Lorenz":
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
        ax_system = plt.axes([0.025, 0.65, 0.18, 0.2])
        self.radio = RadioButtons(ax_system, ("Lorenz", "Rossler"))
        self.radio.on_clicked(self._on_system_change)

        ax_a = plt.axes([0.25, 0.22, 0.6, 0.03])
        ax_b = plt.axes([0.25, 0.17, 0.6, 0.03])
        ax_c = plt.axes([0.25, 0.12, 0.6, 0.03])

        self.slider_a = Slider(ax_a, "Param A", 0.1, 30.0, valinit=10.0, valstep=0.1)
        self.slider_b = Slider(ax_b, "Param B", 0.1, 35.0, valinit=28.0, valstep=0.1)
        self.slider_c = Slider(ax_c, "Param C", 0.1, 10.0, valinit=8.0 / 3.0, valstep=0.05)

        self.slider_a.on_changed(self._on_param_change)
        self.slider_b.on_changed(self._on_param_change)
        self.slider_c.on_changed(self._on_param_change)

        ax_dt = plt.axes([0.25, 0.07, 0.6, 0.03])
        self.slider_dt = Slider(ax_dt, "Time Step", 0.001, 0.03, valinit=self.dt, valstep=0.001)
        self.slider_dt.on_changed(self._on_dt_change)

        ax_speed = plt.axes([0.25, 0.02, 0.6, 0.03])
        self.slider_speed = Slider(
            ax_speed, "Steps/Frame", 1, 60, valinit=self.steps_per_frame, valstep=1
        )
        self.slider_speed.on_changed(self._on_speed_change)

        ax_trail = plt.axes([0.025, 0.55, 0.18, 0.03])
        self.slider_trail = Slider(ax_trail, "Trail", 500, 12000, valinit=self.trail_length, valstep=100)
        self.slider_trail.on_changed(self._on_trail_change)

        ax_pause = plt.axes([0.025, 0.45, 0.18, 0.05])
        self.btn_pause = Button(ax_pause, "Pause")
        self.btn_pause.on_clicked(self._toggle_pause)

        ax_reset = plt.axes([0.025, 0.38, 0.18, 0.05])
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_reset.on_clicked(self._reset)

    def _connect_mouse_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def _on_system_change(self, label):
        self.system = label
        defaults = self.defaults[label]
        self.slider_a.set_val(defaults["a"])
        self.slider_b.set_val(defaults["b"])
        self.slider_c.set_val(defaults["c"])
        self._update_param_labels()
        self._reset(None)

    def _on_param_change(self, _):
        self.params[self.system]["a"] = float(self.slider_a.val)
        self.params[self.system]["b"] = float(self.slider_b.val)
        self.params[self.system]["c"] = float(self.slider_c.val)

    def _on_dt_change(self, val):
        self.dt = float(val)

    def _on_speed_change(self, val):
        self.steps_per_frame = int(val)

    def _on_trail_change(self, val):
        self.trail_length = int(val)
        new_trail = deque(self.trail, maxlen=self.trail_length)
        self.trail = new_trail

    def _toggle_pause(self, _):
        if self.anim.event_source is None:
            return
        if self.is_paused:
            self.anim.event_source.start()
            self.btn_pause.label.set_text("Pause")
        else:
            self.anim.event_source.stop()
            self.btn_pause.label.set_text("Resume")
        self.is_paused = not self.is_paused

    def _reset(self, _):
        defaults = self.defaults[self.system]
        self.slider_a.set_val(defaults["a"])
        self.slider_b.set_val(defaults["b"])
        self.slider_c.set_val(defaults["c"])
        self.slider_dt.set_val(0.01)
        self.slider_speed.set_val(10)
        self.slider_trail.set_val(5000)

        self.offset = np.array([0.0, 0.0])
        self.state = np.array([0.1, 0.0, 0.0])
        self._seed_trail()
        self._draw_from_trail()

    def _update_param_labels(self):
        if self.system == "Lorenz":
            self.slider_a.label.set_text("Sigma")
            self.slider_b.label.set_text("Rho")
            self.slider_c.label.set_text("Beta")
        else:
            self.slider_a.label.set_text("a")
            self.slider_b.label.set_text("b")
            self.slider_c.label.set_text("c")
        self.ax.set_title(f"Strange Attractor: {self.system}")

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.dragging = True
        self.last_mouse = np.array([event.xdata, event.ydata])

    def _on_mouse_release(self, _event):
        self.dragging = False
        self.last_mouse = None

    def _on_mouse_move(self, event):
        if not self.dragging:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        current = np.array([event.xdata, event.ydata])
        delta = current - self.last_mouse
        self.offset += delta
        self.last_mouse = current

    def _animate(self, _frame):
        for _ in range(self.steps_per_frame):
            next_state = self._step(self.state)
            if next_state is None:
                self._reset(None)
                break
            self.state = next_state
            self.trail.append(self.state.copy())
        return self._draw_from_trail()

    def _draw_from_trail(self):
        if len(self.trail) == 0:
            return self.line, self.head, self.origin

        data = np.array(self.trail)
        if not np.all(np.isfinite(data)):
            self._reset(None)
            return self.line, self.head, self.origin
        x = data[:, 0] + self.offset[0]
        y = data[:, 1] + self.offset[1]

        self.line.set_data(x, y)
        self.head.set_data([x[-1]], [y[-1]])
        self.origin.set_data([self.offset[0]], [self.offset[1]])

        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        span_x = x_max - x_min
        span_y = y_max - y_min
        pad_x = max(span_x * 0.1, 1.0)
        pad_y = max(span_y * 0.1, 1.0)
        self.ax.set_xlim(x_min - pad_x, x_max + pad_x)
        self.ax.set_ylim(y_min - pad_y, y_max + pad_y)

        return self.line, self.head, self.origin

    def show(self):
        """Display the demo."""
        plt.show()


def main():
    demo = StrangeAttractorDemo()
    demo.show()


def _build_vscode_url(path):
    posix_path = Path(path).resolve().as_posix()
    if not posix_path.startswith("/"):
        posix_path = f"/{posix_path}"
    return f"vscode://file{quote(posix_path, safe='/:')}"


def get_manifest():
    return {
        "title": "Strange Attractor Explorer",
        "description": "Animate classic strange attractors with interactive control.\n\n"
                       "Features:\n"
                       "- Lorenz and Rossler attractors\n"
                       "- Mouse drag to move the attractor locus\n"
                       "- Adjustable parameters and time step\n"
                       "- Trail length and speed controls\n"
                       "- Pause and reset controls",
        "source_url": _build_vscode_url(__file__),
    }


if __name__ == "__main__":
    main()
