"""
Yavuz Demo Launcher
Interactive GUI to select and run different algorithm demonstrations.
"""

import ast
import subprocess
import sys
import threading
import traceback
import webbrowser
from importlib import util as importlib_util
from pathlib import Path
from urllib.parse import quote

from PySide6 import QtCore, QtWidgets
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class DemoFolderWatcher(FileSystemEventHandler):
    """Watch the demos folder for changes and auto-reload when detected."""

    def __init__(self, refresh_callback):
        self.refresh_callback = refresh_callback
        self.debounce_timer = None

    def on_any_event(self, event):
        if event.src_path.endswith((".pyc", ".pyo", "__pycache__", ".git")):
            return

        if self.debounce_timer:
            self.debounce_timer.cancel()

        self.debounce_timer = threading.Timer(0.5, self.trigger_reload)
        self.debounce_timer.daemon = True
        self.debounce_timer.start()

    def trigger_reload(self):
        QtCore.QTimer.singleShot(0, self.refresh_callback)


class DemoLauncher(QtWidgets.QWidget):
    demo_result = QtCore.Signal(object, object, str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Yavuz Demo Launcher")
        self.resize(900, 650)

        self.demos = []
        self.observer = None

        self._build_ui()
        self.demo_result.connect(self._handle_demo_result)
        self.refresh_demos()
        self.start_demos_watcher()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Yavuz Algorithm Demos")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title_font = title.font()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_demo_select)
        self.list_widget.itemDoubleClicked.connect(lambda _: self.run_selected_demo())
        layout.addWidget(self.list_widget, stretch=2)

        self.description = QtWidgets.QTextEdit()
        self.description.setReadOnly(True)
        self.description.setMinimumHeight(160)
        layout.addWidget(self.description)

        button_layout = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run Selected Demo")
        self.run_button.clicked.connect(self.run_selected_demo)
        self.run_button.setEnabled(False)
        button_layout.addWidget(self.run_button)

        self.open_source_button = QtWidgets.QPushButton("Open Source")
        self.open_source_button.clicked.connect(self.open_selected_source)
        self.open_source_button.setEnabled(False)
        button_layout.addWidget(self.open_source_button)

        exit_button = QtWidgets.QPushButton("Exit")
        exit_button.clicked.connect(self.close)
        button_layout.addWidget(exit_button)

        layout.addLayout(button_layout)

        self.status = QtWidgets.QLabel("Ready. Select a demo to see details.")
        layout.addWidget(self.status)

    def discover_demos(self):
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

            demos.append(
                {
                    "id": f"{script_path.parent.name}/{script_path.stem}",
                    "name": manifest["title"],
                    "description": manifest["description"],
                    "source_url": manifest["source_url"],
                    "path": script_path,
                }
            )

        demos.sort(key=lambda demo: demo["name"].lower())
        return demos

    def load_manifest(self, script_path):
        manifest = self._load_manifest_from_source(script_path)
        if manifest:
            return manifest

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

        manifest.setdefault("source_url", self._build_vscode_url(script_path))
        required_keys = {"title", "description", "source_url"}
        if not required_keys.issubset(manifest.keys()):
            return None

        return manifest

    def _load_manifest_from_source(self, script_path):
        try:
            source = script_path.read_text(encoding="utf-8")
        except Exception:
            return None

        try:
            module_ast = ast.parse(source)
        except SyntaxError:
            return None

        docstring = ast.get_docstring(module_ast) or ""
        manifest = None

        for node in module_ast.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "DEMO_MANIFEST":
                        try:
                            manifest = ast.literal_eval(node.value)
                        except Exception:
                            manifest = None
                        break

        if manifest is None:
            manifest = {}

        if not isinstance(manifest, dict):
            return None

        if not manifest.get("title"):
            first_line = docstring.splitlines()[0] if docstring else script_path.stem
            manifest["title"] = first_line

        if not manifest.get("description"):
            manifest["description"] = docstring or "No description available."

        manifest["source_url"] = self._build_vscode_url(script_path)
        return manifest

    @staticmethod
    def _build_vscode_url(path):
        posix_path = Path(path).resolve().as_posix()
        if not posix_path.startswith("/"):
            posix_path = f"/{posix_path}"
        return f"vscode://file{quote(posix_path, safe='/:')}"

    def refresh_demos(self):
        self.demos = self.discover_demos()
        self.list_widget.clear()

        if not self.demos:
            self.list_widget.addItem("No demos found. Refresh to scan again.")
            self.status.setText("No demos found in demos/ directory")
            return

        for demo in self.demos:
            self.list_widget.addItem(demo["name"])

        self.status.setText(f"Found {len(self.demos)} demo(s). Select one to view details.")
        self.description.clear()
        self.run_button.setEnabled(False)
        self.open_source_button.setEnabled(False)

    def on_demo_select(self, idx):
        if idx < 0 or idx >= len(self.demos):
            return

        demo = self.demos[idx]
        description = f"{demo['description']}\n\nSource: {demo['source_url']}"
        self.description.setPlainText(description)
        self.run_button.setEnabled(True)
        self.open_source_button.setEnabled(True)
        self.status.setText(f"Selected: {demo['name']}")

    def run_selected_demo(self):
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.demos):
            return

        demo = self.demos[idx]
        self.status.setText(f"Launching: {demo['name']}...")
        try:
            self._launch_demo_process(demo)
        except Exception as exc:
            self.status.setText(f"Error launching demo: {exc}")

    def _launch_demo_process(self, demo):
        def _run():
            try:
                process = subprocess.Popen(
                    [sys.executable, str(demo["path"])],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, stderr = process.communicate()
                self.demo_result.emit(demo, process.returncode, stdout, stderr)
            except Exception:
                error_text = traceback.format_exc()
                self.demo_result.emit(demo, 1, "", error_text)

        threading.Thread(target=_run, daemon=True).start()
        self.status.setText(f"Launched: {demo['name']}")

    @QtCore.Slot(object, object, str, str)
    def _handle_demo_result(self, demo, return_code, stdout, stderr):
        stdout = stdout or ""
        stderr = stderr or ""
        has_output = bool(stdout.strip() or stderr.strip())
        error_hint = "Traceback" in stderr
        if return_code == 0 and not error_hint and not has_output:
            return

        report = self._build_crash_report(demo, return_code, stdout, stderr)
        self._show_crash_modal(demo["name"], report)

    def _build_crash_report(self, demo, return_code, stdout, stderr):
        return (
            "Demo Crash Report\n"
            f"Demo Name: {demo['name']}\n"
            f"Demo Id: {demo['id']}\n"
            f"Demo Path: {demo['path']}\n"
            f"Python: {sys.version.replace(chr(10), ' ')}\n"
            f"Platform: {sys.platform}\n"
            f"Exit Code: {return_code}\n\n"
            "--- STDERR ---\n"
            f"{stderr.strip()}\n\n"
            "--- STDOUT ---\n"
            f"{stdout.strip()}\n"
        )

    def _show_crash_modal(self, demo_name, report):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Demo crashed")
        dialog.resize(720, 520)

        layout = QtWidgets.QVBoxLayout(dialog)
        header = QtWidgets.QLabel(
            f"{demo_name} stopped unexpectedly. Copy the report to share in chat."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        report_box = QtWidgets.QTextEdit()
        report_box.setReadOnly(True)
        report_box.setPlainText(report)
        layout.addWidget(report_box)

        button_row = QtWidgets.QHBoxLayout()
        copy_button = QtWidgets.QPushButton("Copy Report")
        copy_button.clicked.connect(lambda: self._copy_to_clipboard(report))
        button_row.addWidget(copy_button)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        button_row.addWidget(close_button)

        layout.addLayout(button_row)
        dialog.exec()


    def _copy_to_clipboard(self, text):
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(text)

    def open_selected_source(self):
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self.demos):
            return

        demo = self.demos[idx]
        try:
            webbrowser.open(demo["source_url"])
            self.status.setText(f"Opened source for: {demo['name']}")
        except Exception as exc:
            self.status.setText(f"Error opening source: {exc}")

    def start_demos_watcher(self):
        try:
            demos_dir = Path(__file__).resolve().parents[2] / "demos"
            if demos_dir.exists():
                self.observer = Observer()
                self.observer.schedule(
                    DemoFolderWatcher(self.refresh_demos),
                    str(demos_dir),
                    recursive=True,
                )
                self.observer.start()
        except Exception as exc:
            print(f"Warning: Could not start file watcher: {exc}")

    def closeEvent(self, event):
        if self.observer:
            self.observer.stop()
            self.observer.join()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    launcher = DemoLauncher()
    launcher.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
