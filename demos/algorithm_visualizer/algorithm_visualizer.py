"""
Algorithm visualization with interactive controls.
Demonstrates sorting algorithms with adjustable speed and array size.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import BackgroundPlotter


class AlgorithmVisualizer:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1200, 800),
            title="Sorting Algorithm Visualizer",
        )

        self.array_size = 50
        self.delay = 0.02
        self.algorithms = ["Bubble Sort", "Selection Sort", "Insertion Sort"]
        self.algorithm_index = 0
        self.is_sorting = False
        self.callback_id = None
        self.sort_steps = None

        self.array = None
        self.chart = None
        self.bars_plot = None
        self.highlight_bars = None

        self.generate_array()
        self._setup_controls()
        self._build_chart()
        self._update_bars()

    def generate_array(self):
        """Generate a random array."""
        self.array = np.random.rand(self.array_size)

    def _build_chart(self):
        """Create the 2D chart for bar visualization."""
        self.chart = pv.Chart2D()
        self.chart.x_label = "Index"
        self.chart.y_label = "Value"
        self.chart.grid = False
        self.plotter.add_chart(self.chart)

    def _setup_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self.plotter)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Array Size slider (10-200)
        layout.addWidget(QtWidgets.QLabel("Array Size"))
        self.size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.size_slider.setMinimum(10)
        self.size_slider.setMaximum(200)
        self.size_slider.setValue(self.array_size)
        self.size_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.size_slider.setTickInterval(10)
        self.size_slider.valueChanged.connect(self._on_size_change)
        layout.addWidget(self.size_slider)

        # Speed slider (0.002-0.1 seconds, scaled as 2-100)
        layout.addWidget(QtWidgets.QLabel("Speed (delay in seconds)"))
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setMinimum(2)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(int(self.delay * 1000))
        self.speed_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.speed_slider.setTickInterval(10)
        self.speed_slider.valueChanged.connect(self._on_speed_change)
        layout.addWidget(self.speed_slider)

        # Algorithm selector (0-2)
        layout.addWidget(QtWidgets.QLabel("Algorithm"))
        self.algorithm_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.algorithm_slider.setMinimum(0)
        self.algorithm_slider.setMaximum(len(self.algorithms) - 1)
        self.algorithm_slider.setValue(self.algorithm_index)
        self.algorithm_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.algorithm_slider.valueChanged.connect(self._on_algorithm_change)
        layout.addWidget(self.algorithm_slider)

        self.algorithm_label = QtWidgets.QLabel(self.algorithms[self.algorithm_index])
        layout.addWidget(self.algorithm_label)

        # Start button
        self.start_button = QtWidgets.QPushButton("Start Sort")
        self.start_button.clicked.connect(self._start_sort)
        layout.addWidget(self.start_button)

        # Stop button
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_sort)
        layout.addWidget(self.stop_button)

        # Reset button
        self.reset_button = QtWidgets.QPushButton("Reset Array")
        self.reset_button.clicked.connect(self._reset_array)
        layout.addWidget(self.reset_button)

        # Stretch to push controls to top
        layout.addStretch(1)

        dock.setWidget(panel)
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _update_bars(self, highlight=None):
        """Update the bar chart visualization."""
        highlight = highlight or []
        x = np.arange(len(self.array))

        # Create or update main bars
        if self.bars_plot is None:
            self.bars_plot = self.chart.bar(x, self.array, color="skyblue")
            self.chart.x_axis.range = [-1, len(self.array)]
            self.chart.y_axis.range = [0, 1.1]
        else:
            self.bars_plot.update(x, self.array)

        # Create or update highlight bars if needed
        if highlight:
            h_y = np.zeros(len(self.array))
            h_y[highlight] = self.array[highlight]
            if self.highlight_bars is None:
                self.highlight_bars = self.chart.bar(x, h_y, color="red")
            else:
                self.highlight_bars.update(x, h_y)
        else:
            if self.highlight_bars is not None:
                # Hide highlights by setting to zero
                self.highlight_bars.update(x, np.zeros(len(self.array)))

    def _on_size_change(self, value):
        if self.is_sorting:
            return
        self.array_size = int(value)
        # Clear plots when array size changes
        if self.bars_plot is not None:
            self.chart.remove_plot(self.bars_plot)
            self.bars_plot = None
        if self.highlight_bars is not None:
            self.chart.remove_plot(self.highlight_bars)
            self.highlight_bars = None
        self.generate_array()
        self._update_bars()

    def _on_speed_change(self, value):
        self.delay = value / 1000.0
        if self.is_sorting:
            self._restart_callback()

    def _on_algorithm_change(self, value):
        if self.is_sorting:
            return
        self.algorithm_index = int(value)
        self.algorithm_label.setText(self.algorithms[self.algorithm_index])

    def _reset_array(self):
        # Stop sorting if active
        if self.is_sorting:
            self._stop_sort()
        # Clear plots for fresh start
        if self.bars_plot is not None:
            self.chart.remove_plot(self.bars_plot)
            self.bars_plot = None
        if self.highlight_bars is not None:
            self.chart.remove_plot(self.highlight_bars)
            self.highlight_bars = None
        self.generate_array()
        self._update_bars()

    def _start_sort(self):
        if self.is_sorting:
            return
        self.is_sorting = True
        self.sort_steps = self._sort_generator()
        self._start_callback()

    def _start_callback(self):
        interval = max(1, int(self.delay * 1000))
        self.callback_id = self.plotter.add_callback(self._step_sort, interval=interval)

    def _restart_callback(self):
        if self.callback_id is not None:
            self.plotter.remove_callback(self.callback_id)
            self.callback_id = None
        if self.is_sorting:
            self._start_callback()

    def _stop_sort(self):
        if self.callback_id is not None:
            self.plotter.remove_callback(self.callback_id)
            self.callback_id = None
        self.is_sorting = False
        self.sort_steps = None
        self._update_bars()

    def _step_sort(self):
        if not self.sort_steps:
            self._stop_sort()
            return
        try:
            highlight = next(self.sort_steps)
            self._update_bars(highlight=highlight)
        except StopIteration:
            self._stop_sort()

    def _sort_generator(self):
        if self.algorithms[self.algorithm_index] == "Bubble Sort":
            yield from self._bubble_sort_steps()
        elif self.algorithms[self.algorithm_index] == "Selection Sort":
            yield from self._selection_sort_steps()
        else:
            yield from self._insertion_sort_steps()

    def _bubble_sort_steps(self):
        n = len(self.array)
        for i in range(n):
            for j in range(0, n - i - 1):
                if self.array[j] > self.array[j + 1]:
                    self.array[j], self.array[j + 1] = self.array[j + 1], self.array[j]
                yield [j, j + 1]
        yield []

    def _selection_sort_steps(self):
        n = len(self.array)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if self.array[j] < self.array[min_idx]:
                    min_idx = j
                yield [i, min_idx]
            self.array[i], self.array[min_idx] = self.array[min_idx], self.array[i]
            yield [i, min_idx]
        yield []

    def _insertion_sort_steps(self):
        for i in range(1, len(self.array)):
            key = self.array[i]
            j = i - 1
            while j >= 0 and self.array[j] > key:
                self.array[j + 1] = self.array[j]
                j -= 1
                yield [j + 1, i]
            self.array[j + 1] = key
            yield [j + 1]
        yield []

    def show(self):
        self.plotter.show()
        self.plotter.app.exec()


def main():
    visualizer = AlgorithmVisualizer()
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
    "title": "Sorting Algorithm Visualizer",
    "description": "Visualize classic sorting algorithms in real-time.\n\n"
    "Features:\n"
    "- Bubble Sort, Selection Sort, Insertion Sort\n"
    "- Adjustable array size (10-200 elements)\n"
    "- Speed control for animation\n"
    "- Color-coded comparisons\n"
    "- Step-by-step visualization",
}


if __name__ == "__main__":
    main()
