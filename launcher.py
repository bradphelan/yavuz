"""
Yavuz Demo Launcher
Interactive GUI to select and run different algorithm demonstrations.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import sys
from pathlib import Path


class DemoLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Yavuz Demo Launcher")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # Configure style
        style = ttk.Style()
        style.theme_use('clam')

        # Main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="🚀 Yavuz Algorithm Demos",
            font=('Arial', 20, 'bold')
        )
        title_label.grid(row=0, column=0, pady=(0, 20))

        # Demo list frame
        list_frame = ttk.LabelFrame(main_frame, text="Available Demos", padding="10")
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        # Listbox with scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.demo_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=('Arial', 11),
            height=10,
            selectmode=tk.SINGLE
        )
        self.demo_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.config(command=self.demo_listbox.yview)

        # Bind selection event
        self.demo_listbox.bind('<<ListboxSelect>>', self.on_demo_select)
        self.demo_listbox.bind('<Double-Button-1>', lambda e: self.run_selected_demo())

        # Description frame
        desc_frame = ttk.LabelFrame(main_frame, text="Demo Description", padding="10")
        desc_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        desc_frame.columnconfigure(0, weight=1)
        desc_frame.rowconfigure(0, weight=1)

        self.description_text = scrolledtext.ScrolledText(
            desc_frame,
            wrap=tk.WORD,
            width=60,
            height=8,
            font=('Arial', 10),
            state=tk.DISABLED
        )
        self.description_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=(10, 0))

        self.run_button = ttk.Button(
            button_frame,
            text="▶ Run Selected Demo",
            command=self.run_selected_demo,
            state=tk.DISABLED
        )
        self.run_button.grid(row=0, column=0, padx=5)

        ttk.Button(
            button_frame,
            text="🔄 Refresh List",
            command=self.refresh_demos
        ).grid(row=0, column=1, padx=5)

        ttk.Button(
            button_frame,
            text="❌ Exit",
            command=root.quit
        ).grid(row=0, column=2, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Select a demo to see details.")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        # Demo definitions
        self.demos = self.discover_demos()
        self.populate_demo_list()

    def discover_demos(self):
        """Discover all available demos in the demos directory."""
        demos = []
        demos_dir = Path(__file__).parent / "demos"

        if not demos_dir.exists():
            return []

        # Demo metadata
        demo_info = {
            "surface_plot": {
                "name": "Interactive 3D Surface Plot",
                "description": "Real-time 3D surface visualization with adjustable parameters.\n\n"
                             "Features:\n"
                             "• Interactive 3D surface rendering\n"
                             "• Frequency, amplitude, and phase controls\n"
                             "• Slider-based parameter adjustment\n"
                             "• Mathematical function: Z = A*sin(f*√(X²+Y²) + φ)",
                "script": "surface_plot_interactive.py"
            },
            "algorithm_visualizer": {
                "name": "Sorting Algorithm Visualizer",
                "description": "Visualize classic sorting algorithms in real-time.\n\n"
                             "Features:\n"
                             "• Bubble Sort, Selection Sort, Insertion Sort\n"
                             "• Adjustable array size (10-200 elements)\n"
                             "• Speed control for animation\n"
                             "• Color-coded comparisons\n"
                             "• Step-by-step visualization",
                "script": "algorithm_visualizer.py"
            },
            "parametric_3d": {
                "name": "Parametric 3D Curves",
                "description": "Explore beautiful 3D mathematical curves.\n\n"
                             "Features:\n"
                             "• Multiple curve types: Helix, Torus Knot, Lissajous, Spiral\n"
                             "• Three adjustable parameters per curve\n"
                             "• Color gradients based on position\n"
                             "• Configurable point density\n"
                             "• Interactive 3D rotation",
                "script": "parametric_3d.py"
            },
            "numerical_methods": {
                "name": "Numerical Methods Demonstration",
                "description": "Compare numerical algorithms with analytical solutions.\n\n"
                             "Features:\n"
                             "• Numerical integration (Trapezoidal & Simpson's rule)\n"
                             "• Numerical differentiation\n"
                             "• Side-by-side comparison with exact solutions\n"
                             "• Adjustable sample points\n"
                             "• Interactive function parameters",
                "script": "numerical_methods.py"
            },
            "douglas_peucker": {
                "name": "Douglas-Peucker Line Simplification",
                "description": "Visualize noise reduction using the Ramer-Douglas-Peucker algorithm.\n\n"
                             "Features:\n"
                             "• Clean sine wave generation\n"
                             "• Adjustable noise amplitude\n"
                             "• Interactive tolerance control for simplification\n"
                             "• Real-time comparison of noisy vs simplified signals\n"
                             "• Point reduction statistics and error metrics\n"
                             "• Configurable signal frequency",
                "script": "douglas_peucker.py"
            }
        }

        # Scan demos directory
        for demo_folder in sorted(demos_dir.iterdir()):
            if demo_folder.is_dir() and demo_folder.name in demo_info:
                info = demo_info[demo_folder.name]
                script_path = demo_folder / info["script"]

                if script_path.exists():
                    demos.append({
                        "id": demo_folder.name,
                        "name": info["name"],
                        "description": info["description"],
                        "path": script_path
                    })

        return demos

    def populate_demo_list(self):
        """Populate the listbox with available demos."""
        self.demo_listbox.delete(0, tk.END)

        if not self.demos:
            self.demo_listbox.insert(tk.END, "No demos found. Run 'refresh' to scan again.")
            self.status_var.set("No demos found in demos/ directory")
            return

        for demo in self.demos:
            self.demo_listbox.insert(tk.END, demo["name"])

        self.status_var.set(f"Found {len(self.demos)} demo(s). Select one to view details.")

    def on_demo_select(self, event):
        """Handle demo selection."""
        selection = self.demo_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        if idx < len(self.demos):
            demo = self.demos[idx]

            # Update description
            self.description_text.config(state=tk.NORMAL)
            self.description_text.delete(1.0, tk.END)
            self.description_text.insert(1.0, demo["description"])
            self.description_text.config(state=tk.DISABLED)

            # Enable run button
            self.run_button.config(state=tk.NORMAL)

            self.status_var.set(f"Selected: {demo['name']}")

    def run_selected_demo(self):
        """Run the selected demo."""
        selection = self.demo_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        if idx < len(self.demos):
            demo = self.demos[idx]
            self.status_var.set(f"Launching: {demo['name']}...")

            try:
                # Run the demo in a new process
                subprocess.Popen([sys.executable, str(demo["path"])])
                self.status_var.set(f"✓ Launched: {demo['name']}")
            except Exception as e:
                self.status_var.set(f"✗ Error launching demo: {str(e)}")

    def refresh_demos(self):
        """Refresh the demo list."""
        self.demos = self.discover_demos()
        self.populate_demo_list()
        self.description_text.config(state=tk.NORMAL)
        self.description_text.delete(1.0, tk.END)
        self.description_text.config(state=tk.DISABLED)
        self.run_button.config(state=tk.DISABLED)


def main():
    """Launch the demo selector GUI."""
    root = tk.Tk()
    app = DemoLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
