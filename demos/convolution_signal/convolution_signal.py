"""
Convolution Signal Demo
Visualizes discrete convolution with a sliding kernel.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import BackgroundPlotter


class ConvolutionSignalDemo:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1500, 600),
            title="Convolution in Signal Processing",
            shape=(1, 3),
        )
        self.plotter.set_background("white")

        self.n_samples = 240
        self.kernel_len = 40
        self.frequency = 2.0
        self.decay = 4.0
        self.noise_seed = 0
        self.signal_type = "Sine"
        self.signal_options = ["Sine", "Square", "Triangle", "White noise"]
        self.step_index = 0

        self.t = np.array([])
        self.signal = np.array([])
        self.kernel = np.array([])
        self.output = np.array([])
        self.output_x = np.array([])

        self.chart_signal = None
        self.chart_kernel = None
        self.chart_output = None

        self.line_signal = None
        self.line_kernel_shifted = None
        self.line_kernel = None
        self.line_output_full = None
        self.line_output_partial = None
        self.point_output_current = None

        self._setup_controls()
        self._build_scene()
        self._recompute()
        self._update_plot()

    def _setup_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self.plotter)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        layout.addWidget(QtWidgets.QLabel("Signal frequency"))
        self.freq_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.freq_slider.setMinimum(1)
        self.freq_slider.setMaximum(50)
        self.freq_slider.setValue(int(self.frequency * 10))
        self.freq_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.freq_slider.setTickInterval(5)
        self.freq_slider.valueChanged.connect(self._on_frequency_change)
        layout.addWidget(self.freq_slider)

        layout.addWidget(QtWidgets.QLabel("Signal type"))
        self.signal_combo = QtWidgets.QComboBox()
        self.signal_combo.addItems(self.signal_options)
        self.signal_combo.setCurrentText(self.signal_type)
        self.signal_combo.currentTextChanged.connect(self._on_signal_type_change)
        layout.addWidget(self.signal_combo)

        layout.addWidget(QtWidgets.QLabel("Noise seed"))
        self.noise_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.noise_slider.setMinimum(0)
        self.noise_slider.setMaximum(999)
        self.noise_slider.setValue(self.noise_seed)
        self.noise_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.noise_slider.setTickInterval(100)
        self.noise_slider.valueChanged.connect(self._on_noise_seed_change)
        layout.addWidget(self.noise_slider)

        layout.addWidget(QtWidgets.QLabel("Kernel decay"))
        self.decay_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.decay_slider.setMinimum(5)
        self.decay_slider.setMaximum(120)
        self.decay_slider.setValue(int(self.decay * 10))
        self.decay_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.decay_slider.setTickInterval(10)
        self.decay_slider.valueChanged.connect(self._on_decay_change)
        layout.addWidget(self.decay_slider)

        layout.addWidget(QtWidgets.QLabel("Convolution step"))
        self.step_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.step_slider.setMinimum(0)
        self.step_slider.setMaximum(0)
        self.step_slider.setValue(self.step_index)
        self.step_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.step_slider.setTickInterval(10)
        self.step_slider.valueChanged.connect(self._on_step_change)
        layout.addWidget(self.step_slider)

        self.status_label = QtWidgets.QLabel("Step: 0 | Dot product: 0.0")
        layout.addWidget(self.status_label)

        layout.addStretch(1)

        dock.setWidget(panel)
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _build_scene(self):
        self.plotter.subplot(0, 0)
        self.chart_signal = pv.Chart2D()
        self.chart_signal.title = "Signal and Shifted Kernel"
        self.chart_signal.x_label = "Time"
        self.chart_signal.y_label = "Amplitude"
        self.chart_signal.grid = False
        self.plotter.add_chart(self.chart_signal)

        self.plotter.subplot(0, 1)
        self.chart_kernel = pv.Chart2D()
        self.chart_kernel.title = "Kernel (Exponential Decay)"
        self.chart_kernel.x_label = "Kernel time"
        self.chart_kernel.y_label = "Amplitude"
        self.chart_kernel.grid = False
        self.plotter.add_chart(self.chart_kernel)

        self.plotter.subplot(0, 2)
        self.chart_output = pv.Chart2D()
        self.chart_output.title = "Valid Convolution Output"
        self.chart_output.x_label = "Time"
        self.chart_output.y_label = "Output"
        self.chart_output.grid = False
        self.plotter.add_chart(self.chart_output)

    def _recompute(self):
        self.t = np.linspace(0.0, 1.0, self.n_samples)
        dt = self.t[1] - self.t[0]
        self.signal = self._compute_signal(self.t)

        tau = np.arange(self.kernel_len) * dt
        self.kernel = np.exp(-self.decay * tau)
        self.kernel /= max(self.kernel.sum(), 1e-12)

        self.output = np.convolve(self.signal, self.kernel, mode="valid") * dt
        self.output_x = self.t[: self.output.size]

        max_step = max(self.output.size - 1, 0)
        self.step_slider.setMaximum(max_step)
        self.step_index = min(self.step_index, max_step)
        self.step_slider.setValue(self.step_index)

    def _compute_signal(self, t):
        omega = 2.0 * np.pi * self.frequency * t
        if self.signal_type == "Sine":
            return np.sin(omega)
        if self.signal_type == "Square":
            return np.sign(np.sin(omega))
        if self.signal_type == "Triangle":
            return (2.0 / np.pi) * np.arcsin(np.sin(omega))
        rng = np.random.default_rng(self.noise_seed)
        return rng.normal(0.0, 0.6, size=t.size)

    def _update_plot(self):
        if self.signal.size == 0:
            return

        dt = self.t[1] - self.t[0]
        kernel_scale = 0.6 * max(np.max(np.abs(self.signal)), 1e-6) / max(
            np.max(self.kernel), 1e-6
        )
        shifted = np.zeros_like(self.signal)
        start = self.step_index
        end = start + self.kernel_len
        if end <= self.signal.size:
            shifted[start:end] = self.kernel * kernel_scale

        if self.line_signal is None:
            self.line_signal = self.chart_signal.line(self.t, self.signal, color="navy", width=2)
            self.line_kernel_shifted = self.chart_signal.line(
                self.t, shifted, color="orange", width=2
            )
            self.chart_signal.x_axis.range = [self.t.min(), self.t.max()]
        else:
            self.line_signal.update(self.t, self.signal)
            self.line_kernel_shifted.update(self.t, shifted)

        self.chart_signal.title = f"Signal and Shifted Kernel ({self.signal_type})"

        y_min = min(self.signal.min(), shifted.min())
        y_max = max(self.signal.max(), shifted.max())
        y_range = max(y_max - y_min, 1e-6)
        self.chart_signal.y_axis.range = [y_min - 0.1 * y_range, y_max + 0.1 * y_range]

        tau = np.arange(self.kernel_len) * dt
        if self.line_kernel is None:
            self.line_kernel = self.chart_kernel.line(tau, self.kernel, color="teal", width=3)
            self.chart_kernel.x_axis.range = [tau.min(), tau.max()]
        else:
            self.line_kernel.update(tau, self.kernel)

        k_min = float(self.kernel.min())
        k_max = float(self.kernel.max())
        k_range = max(k_max - k_min, 1e-6)
        self.chart_kernel.y_axis.range = [k_min - 0.1 * k_range, k_max + 0.1 * k_range]

        if self.line_output_full is None:
            self.line_output_full = self.chart_output.line(
                self.output_x, self.output, color="lightgray", width=2
            )
            self.line_output_partial = self.chart_output.line(
                self.output_x[:1], self.output[:1], color="purple", width=3
            )
            self.point_output_current = self.chart_output.scatter(
                self.output_x[:1], self.output[:1], color="purple", size=8
            )
            self.chart_output.x_axis.range = [self.output_x.min(), self.output_x.max()]
        else:
            self.line_output_full.update(self.output_x, self.output)
            idx = min(self.step_index + 1, self.output.size)
            self.line_output_partial.update(self.output_x[:idx], self.output[:idx])
            self.point_output_current.update(
                self.output_x[idx - 1 : idx], self.output[idx - 1 : idx]
            )

        out_min = float(self.output.min())
        out_max = float(self.output.max())
        out_range = max(out_max - out_min, 1e-6)
        self.chart_output.y_axis.range = [out_min - 0.1 * out_range, out_max + 0.1 * out_range]

        self._update_status()
        self.plotter.render()

    def _update_status(self):
        if self.output.size == 0:
            self.status_label.setText("Step: 0 | Dot product: 0.0")
            return
        dot_value = self.output[self.step_index]
        self.status_label.setText(
            f"Step: {self.step_index} / {self.output.size - 1} | Dot product: {dot_value:.4e}"
        )

    def _on_frequency_change(self, value):
        self.frequency = value / 10.0
        self._recompute()
        self._update_plot()

    def _on_decay_change(self, value):
        self.decay = value / 10.0
        self._recompute()
        self._update_plot()

    def _on_signal_type_change(self, value):
        self.signal_type = value
        self._recompute()
        self._update_plot()

    def _on_noise_seed_change(self, value):
        self.noise_seed = int(value)
        if self.signal_type == "White noise":
            self._recompute()
            self._update_plot()

    def _on_step_change(self, value):
        self.step_index = int(value)
        self._update_plot()

    def show(self):
        self.plotter.show()
        self.plotter.app.exec()


def main():
    demo = ConvolutionSignalDemo()
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
    "title": "Convolution in Signal Processing",
    "description": "Step through a discrete convolution with a sliding kernel.\n\n"
    "Features:\n"
    "- Sine signal convolved with exponential decay kernel\n"
    "- Shifted kernel overlay for each convolution step\n"
    "- Valid convolution output visualization\n"
    "- Interactive frequency, decay, and step controls",
}


if __name__ == "__main__":
    main()
