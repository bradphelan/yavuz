"""
Algorithm visualization with interactive controls.
Demonstrates sorting algorithms with adjustable speed and array size.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter


class AlgorithmVisualizer:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1200, 800),
            title="Sorting Algorithm Visualizer",
        )
        self.plotter.set_background("white")
        self.plotter.show_grid()

        self.array_size = 50
        self.delay = 0.02
        self.algorithms = ["Bubble Sort", "Selection Sort", "Insertion Sort"]
        self.algorithm_index = 0
        self.is_sorting = False
        self.callback_id = None
        self.sort_steps = None

        self.array = None
        self.bars_poly = None
        self.bars_actor = None
        self.highlight_poly = pv.PolyData()
        self.highlight_actor = None

        self.generate_array()
        self._setup_controls()
        self._update_bars()
        self._update_title()

    def generate_array(self):
        """Generate a random array."""
        self.array = np.random.rand(self.array_size)

    def _setup_controls(self):
        self.plotter.add_slider_widget(
            self._on_size_change,
            [10, 200],
            value=self.array_size,
            title="Array Size",
            pointa=(0.02, 0.1),
            pointb=(0.35, 0.1),
        )
        self.plotter.add_slider_widget(
            self._on_speed_change,
            [0.002, 0.1],
            value=self.delay,
            title="Speed (sec)",
            pointa=(0.4, 0.1),
            pointb=(0.72, 0.1),
        )
        self.plotter.add_slider_widget(
            self._on_algorithm_change,
            [0, len(self.algorithms) - 1],
            value=self.algorithm_index,
            title="Algorithm",
            pointa=(0.76, 0.1),
            pointb=(0.98, 0.1),
        )
        self.plotter.add_text(
            "Keys: S start, R reset",
            position=(10, 10),
            font_size=10,
            name="controls_hint",
        )
        self.plotter.add_key_event("s", self._start_sort)
        self.plotter.add_key_event("r", self._reset_array)

    def _update_title(self):
        algorithm = self.algorithms[self.algorithm_index]
        self.plotter.add_text(
            f"{algorithm} Visualization",
            position="upper_left",
            font_size=12,
            name="title",
        )

    def _build_lines(self, heights):
        n = len(heights)
        points = np.zeros((2 * n, 3), dtype=float)
        points[0::2, 0] = np.arange(n)
        points[1::2, 0] = np.arange(n)
        points[1::2, 1] = heights

        lines = np.zeros((n, 3), dtype=np.int64)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(0, 2 * n, 2)
        lines[:, 2] = np.arange(1, 2 * n, 2)
        return points, lines.ravel()

    def _update_bars(self, highlight=None):
        highlight = highlight or []
        points, lines = self._build_lines(self.array)

        if self.bars_poly is None:
            self.bars_poly = pv.PolyData(points, lines)
            self.bars_actor = self.plotter.add_mesh(
                self.bars_poly,
                color="skyblue",
                line_width=4,
            )
        else:
            self.bars_poly.points = points
            self.bars_poly.lines = lines
            self.bars_poly.Modified()

        if highlight:
            h_points, h_lines = self._build_lines(self.array[highlight])
            h_points[:, 0] = np.repeat(highlight, 2)
            self.highlight_poly.points = h_points
            self.highlight_poly.lines = h_lines
        else:
            self.highlight_poly.points = np.empty((0, 3))
            self.highlight_poly.lines = np.empty((0,), dtype=np.int64)

        if self.highlight_actor is None:
            self.highlight_actor = self.plotter.add_mesh(
                self.highlight_poly,
                color="red",
                line_width=6,
            )
        else:
            self.highlight_poly.Modified()

        self.plotter.reset_camera()

    def _on_size_change(self, value):
        if self.is_sorting:
            return
        self.array_size = int(round(value))
        self.generate_array()
        self._update_bars()

    def _on_speed_change(self, value):
        self.delay = float(value)
        if self.is_sorting:
            self._restart_callback()

    def _on_algorithm_change(self, value):
        if self.is_sorting:
            return
        self.algorithm_index = int(round(value))
        self._update_title()

    def _reset_array(self):
        if self.is_sorting:
            return
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
        self.plotter.app.exec_()


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
