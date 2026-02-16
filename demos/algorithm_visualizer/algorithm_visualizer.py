"""
Algorithm visualization with interactive controls.
Demonstrates sorting algorithms with adjustable speed and array size.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import time


class AlgorithmVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.subplots_adjust(bottom=0.30)

        # Initial parameters
        self.array_size = 50
        self.delay = 0.01
        self.algorithm = 'Bubble Sort'
        self.is_sorting = False

        # Generate initial array
        self.array = None
        self.generate_array()

        # Initial plot
        self.bars = self.ax.bar(range(len(self.array)), self.array, color='skyblue')
        self.ax.set_xlim(0, self.array_size)
        self.ax.set_ylim(0, 1.1)
        self.ax.set_title(f'{self.algorithm} Visualization')
        self.ax.set_xlabel('Index')
        self.ax.set_ylabel('Value')

        # Setup controls
        self.setup_controls()

    def generate_array(self):
        """Generate a random array."""
        self.array = np.random.rand(self.array_size)

    def setup_controls(self):
        """Setup GUI controls."""
        # Array size slider
        ax_size = plt.axes([0.2, 0.20, 0.6, 0.03])
        self.slider_size = Slider(
            ax_size, 'Array Size', 10, 200,
            valinit=self.array_size, valstep=10
        )
        self.slider_size.on_changed(self.update_size)

        # Speed slider
        ax_speed = plt.axes([0.2, 0.15, 0.6, 0.03])
        self.slider_speed = Slider(
            ax_speed, 'Speed', 0.001, 0.1,
            valinit=self.delay, valstep=0.001
        )
        self.slider_speed.on_changed(self.update_speed)

        # Algorithm selector
        ax_algo = plt.axes([0.025, 0.4, 0.15, 0.15])
        self.radio = RadioButtons(
            ax_algo,
            ('Bubble Sort', 'Selection Sort', 'Insertion Sort')
        )
        self.radio.on_clicked(self.update_algorithm)

        # Start button
        ax_start = plt.axes([0.3, 0.05, 0.15, 0.05])
        self.btn_start = Button(ax_start, 'Start')
        self.btn_start.on_clicked(self.start_sorting)

        # Reset button
        ax_reset = plt.axes([0.5, 0.05, 0.15, 0.05])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_array)

    def update_size(self, val):
        """Update array size."""
        self.array_size = int(val)
        self.generate_array()
        self.redraw()

    def update_speed(self, val):
        """Update sorting speed."""
        self.delay = val

    def update_algorithm(self, label):
        """Update selected algorithm."""
        self.algorithm = label
        self.ax.set_title(f'{self.algorithm} Visualization')
        self.fig.canvas.draw_idle()

    def reset_array(self, event):
        """Reset to a new random array."""
        self.generate_array()
        self.redraw()

    def redraw(self):
        """Redraw the bars."""
        self.ax.clear()
        self.bars = self.ax.bar(range(len(self.array)), self.array, color='skyblue')
        self.ax.set_xlim(0, self.array_size)
        self.ax.set_ylim(0, 1.1)
        self.ax.set_title(f'{self.algorithm} Visualization')
        self.ax.set_xlabel('Index')
        self.ax.set_ylabel('Value')
        self.fig.canvas.draw_idle()

    def update_bars(self, highlight=None):
        """Update bar heights and colors."""
        for i, (bar, val) in enumerate(zip(self.bars, self.array)):
            bar.set_height(val)
            if highlight and i in highlight:
                bar.set_color('red')
            else:
                bar.set_color('skyblue')
        self.fig.canvas.draw_idle()
        plt.pause(self.delay)

    def bubble_sort(self):
        """Bubble sort algorithm with visualization."""
        n = len(self.array)
        for i in range(n):
            for j in range(0, n-i-1):
                if self.array[j] > self.array[j+1]:
                    self.array[j], self.array[j+1] = self.array[j+1], self.array[j]
                self.update_bars(highlight=[j, j+1])
        self.update_bars()

    def selection_sort(self):
        """Selection sort algorithm with visualization."""
        n = len(self.array)
        for i in range(n):
            min_idx = i
            for j in range(i+1, n):
                if self.array[j] < self.array[min_idx]:
                    min_idx = j
                self.update_bars(highlight=[i, min_idx])
            self.array[i], self.array[min_idx] = self.array[min_idx], self.array[i]
            self.update_bars(highlight=[i, min_idx])
        self.update_bars()

    def insertion_sort(self):
        """Insertion sort algorithm with visualization."""
        for i in range(1, len(self.array)):
            key = self.array[i]
            j = i-1
            while j >= 0 and self.array[j] > key:
                self.array[j+1] = self.array[j]
                j -= 1
                self.update_bars(highlight=[j+1, i])
            self.array[j+1] = key
            self.update_bars(highlight=[j+1])
        self.update_bars()

    def start_sorting(self, event):
        """Start the sorting algorithm."""
        if self.is_sorting:
            return

        self.is_sorting = True

        if self.algorithm == 'Bubble Sort':
            self.bubble_sort()
        elif self.algorithm == 'Selection Sort':
            self.selection_sort()
        elif self.algorithm == 'Insertion Sort':
            self.insertion_sort()

        self.is_sorting = False

    def show(self):
        """Display the visualizer."""
        plt.show()


def main():
    """Run the algorithm visualizer."""
    visualizer = AlgorithmVisualizer()
    visualizer.show()


if __name__ == "__main__":
    main()
