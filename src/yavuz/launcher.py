"""
Yavuz Demo Launcher
Interactive GUI to select and run different algorithm demonstrations.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import sys
import threading
import traceback
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
                self._launch_demo_process(demo)
            except Exception as e:
                self.status_var.set(f"✗ Error launching demo: {str(e)}")

    def _launch_demo_process(self, demo):
        """Run the demo in a subprocess and capture crashes."""
        def _run():
            try:
                process = subprocess.Popen(
                    [sys.executable, str(demo["path"])],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate()
                self.root.after(
                    0,
                    self._handle_demo_result,
                    demo,
                    process.returncode,
                    stdout,
                    stderr
                )
            except Exception:
                error_text = traceback.format_exc()
                self.root.after(
                    0,
                    self._handle_demo_result,
                    demo,
                    1,
                    "",
                    error_text
                )

        threading.Thread(target=_run, daemon=True).start()
        self.status_var.set(f"✓ Launched: {demo['name']}")

    def _handle_demo_result(self, demo, return_code, stdout, stderr):
        """Show crash modal when a demo exits with errors."""
        error_hint = "Traceback" in (stderr or "")
        if return_code == 0 and not error_hint:
            return

        report = self._build_crash_report(demo, return_code, stdout, stderr)
        self._show_crash_modal(demo["name"], report)

    def _build_crash_report(self, demo, return_code, stdout, stderr):
        """Build a structured report for AI assistance."""
        return (
            "Demo Crash Report\n"
            f"Demo Name: {demo['name']}\n"
            f"Demo Id: {demo['id']}\n"
            f"Demo Path: {demo['path']}\n"
            f"Python: {sys.version.replace(chr(10), ' ')}\n"
            f"Platform: {sys.platform}\n"
            f"Exit Code: {return_code}\n"
            "\n"
            "--- STDERR ---\n"
            f"{stderr.strip()}\n"
            "\n"
            "--- STDOUT ---\n"
            f"{stdout.strip()}\n"
        )

    def _show_crash_modal(self, demo_name, report):
        """Show a modal dialog with crash details and copy helper."""
        modal = tk.Toplevel(self.root)
        modal.title("Demo crashed")
        modal.geometry("700x500")
        modal.transient(self.root)
        modal.grab_set()

        header = ttk.Label(
            modal,
            text=(
                f"{demo_name} stopped unexpectedly. "
                "Copy the report and paste it into the chat to get AI help."
            ),
            wraplength=660,
            justify=tk.LEFT
        )
        header.pack(padx=12, pady=(12, 8), anchor=tk.W)

        report_box = scrolledtext.ScrolledText(
            modal,
            wrap=tk.WORD,
            height=18,
            font=("Consolas", 10)
        )
        report_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        report_box.insert(tk.END, report)
        report_box.config(state=tk.DISABLED)

        button_frame = ttk.Frame(modal)
        button_frame.pack(pady=(0, 12))

        ttk.Button(
            button_frame,
            text="Copy Report",
            command=lambda: self._copy_to_clipboard(report)
        ).grid(row=0, column=0, padx=6)

        ttk.Button(
            button_frame,
            text="Close",
            command=modal.destroy
        ).grid(row=0, column=1, padx=6)

    def _copy_to_clipboard(self, text):
        """Copy crash report to clipboard with demo metadata."""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()

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
