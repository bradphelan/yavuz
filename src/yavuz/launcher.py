"""
Yavuz Demo Launcher
Interactive GUI to select and run different algorithm demonstrations.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import sys
import threading
import webbrowser
from importlib import util as importlib_util
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class DemoFolderWatcher(FileSystemEventHandler):
    """Watch the demos folder for changes and auto-reload when detected."""

    def __init__(self, launcher):
        self.launcher = launcher
        self.debounce_timer = None

    def on_any_event(self, event):
        """Handle file system events with debouncing."""
        # Ignore hidden files and cache directories
        if event.src_path.endswith(('.pyc', '.pyo', '__pycache__', '.git')):
            return

        # Debounce - wait a bit for multiple events to settle
        if self.debounce_timer:
            self.debounce_timer.cancel()

        self.debounce_timer = threading.Timer(0.5, self.trigger_reload)
        self.debounce_timer.daemon = True
        self.debounce_timer.start()

    def trigger_reload(self):
        """Trigger a reload of the demo list."""
        self.launcher.refresh_demos()


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

        self.open_source_button = ttk.Button(
            button_frame,
            text="Open Source",
            command=self.open_selected_source,
            state=tk.DISABLED
        )
        self.open_source_button.grid(row=0, column=1, padx=5)

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

        # Setup file watcher for demos folder
        self.observer = None
        self.start_demos_watcher()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def discover_demos(self):
        """Discover all available demos in the demos directory."""
        demos = []
        demos_dir = Path(__file__).resolve().parents[2] / "demos"

        if not demos_dir.exists():
            return []

        for script_path in sorted(demos_dir.rglob("*.py")):
            if script_path.name == "__init__.py":
                continue
            if "__pycache__" in script_path.parts:
                continue

            manifest = self.load_manifest(script_path)
            if not manifest:
                continue

            demos.append({
                "id": f"{script_path.parent.name}/{script_path.stem}",
                "name": manifest["title"],
                "description": manifest["description"],
                "source_url": manifest["source_url"],
                "path": script_path
            })

        demos.sort(key=lambda demo: demo["name"].lower())
        return demos

    def load_manifest(self, script_path):
        """Load a demo manifest from a script file."""
        module_name = f"yavuz_demo_{script_path.parent.name}_{script_path.stem}"
        spec = importlib_util.spec_from_file_location(module_name, script_path)
        if not spec or not spec.loader:
            return None

        module = importlib_util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            return None

        get_manifest = getattr(module, "get_manifest", None)
        if not callable(get_manifest):
            return None

        try:
            manifest = get_manifest()
        except Exception:
            return None

        if not isinstance(manifest, dict):
            return None

        required_keys = {"title", "description", "source_url"}
        if not required_keys.issubset(manifest.keys()):
            return None

        return manifest

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
            description = f"{demo['description']}\n\nSource: {demo['source_url']}"
            self.description_text.insert(1.0, description)
            self.description_text.config(state=tk.DISABLED)

            # Enable run button
            self.run_button.config(state=tk.NORMAL)
            self.open_source_button.config(state=tk.NORMAL)

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

    def open_selected_source(self):
        """Open the selected demo source in VS Code."""
        selection = self.demo_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        if idx < len(self.demos):
            demo = self.demos[idx]
            try:
                webbrowser.open(demo["source_url"])
                self.status_var.set(f"Opened source for: {demo['name']}")
            except Exception as e:
                self.status_var.set(f"✗ Error opening source: {str(e)}")

    def refresh_demos(self):
        """Refresh the demo list."""
        self.demos = self.discover_demos()
        self.populate_demo_list()
        self.description_text.config(state=tk.NORMAL)
        self.description_text.delete(1.0, tk.END)
        self.description_text.config(state=tk.DISABLED)
        self.run_button.config(state=tk.DISABLED)
        self.open_source_button.config(state=tk.DISABLED)
        self.status_var.set(f"Demo list updated. Found {len(self.demos)} demo(s).")

    def start_demos_watcher(self):
        """Start watching the demos folder for changes."""
        try:
            demos_dir = Path(__file__).resolve().parents[2] / "demos"
            if demos_dir.exists():
                self.observer = Observer()
                self.observer.schedule(
                    DemoFolderWatcher(self),
                    str(demos_dir),
                    recursive=True
                )
                self.observer.start()
        except Exception as e:
            print(f"Warning: Could not start file watcher: {e}")

    def on_close(self):
        """Clean up and close the application."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        self.root.destroy()


def main():
    """Launch the demo selector GUI."""
    root = tk.Tk()
    app = DemoLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
