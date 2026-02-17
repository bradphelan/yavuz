# Demo Template

This template shows how to create a new demo for the Yavuz project following established patterns.

## Steps to Create a New Demo

1. **Create a new folder** in the `demos/` directory:
   ```bash
   mkdir demos/my_new_demo
   ```

2. **Create your main script** (e.g., `my_new_demo.py`):
   ```python
   """
   My New Demo
   Brief description of what this demo does.
   """

   from pathlib import Path
   from urllib.parse import quote

   import numpy as np
   import pyvista as pv
   from PySide6 import QtCore, QtWidgets
   from pyvistaqt import BackgroundPlotter


   class MyNewDemo:
       def __init__(self):
           self.plotter = BackgroundPlotter(
               window_size=(1200, 800),
               title="My New Demo",
           )
           self.plotter.set_background("white")
           self.plotter.show_grid()

           # Initialize state variables
           self.param_a = 1.0
           self.param_b = 2.0

           # Store meshes and actors for reuse (no flickering)
           self.mesh = None
           self.mesh_actor = None

           # Setup in order: controls, scene, widgets, camera
           self._setup_controls()
           self._build_scene()
           self._update_plot()
           self._setup_widgets()  # Add widgets last so they're on top
           self.plotter.view_xy()  # Set camera view if needed

       def _setup_controls(self):
           """Setup Qt dock widget with controls."""
           dock = QtWidgets.QDockWidget("Controls", self.plotter)
           dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
           panel = QtWidgets.QWidget()
           layout = QtWidgets.QVBoxLayout(panel)

           # Parameter A slider
           layout.addWidget(QtWidgets.QLabel("Parameter A"))
           self.slider_a = QtWidgets.QSlider(QtCore.Qt.Horizontal)
           self.slider_a.setMinimum(1)
           self.slider_a.setMaximum(100)
           self.slider_a.setValue(int(self.param_a * 10))
           self.slider_a.valueChanged.connect(self._on_param_a_changed)
           layout.addWidget(self.slider_a)

           # Parameter B slider
           layout.addWidget(QtWidgets.QLabel("Parameter B"))
           self.slider_b = QtWidgets.QSlider(QtCore.Qt.Horizontal)
           self.slider_b.setMinimum(1)
           self.slider_b.setMaximum(100)
           self.slider_b.setValue(int(self.param_b * 10))
           self.slider_b.valueChanged.connect(self._on_param_b_changed)
           layout.addWidget(self.slider_b)

           # Push controls to top
           layout.addStretch(1)

           dock.setWidget(panel)
           self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

       def _build_scene(self):
           """Initialize all meshes and actors once."""
           # Create your mesh here
           self.mesh = pv.Sphere(radius=1.0)
           self.mesh_actor = self.plotter.add_mesh(
               self.mesh, color="skyblue", smooth_shading=True
           )

       def _update_plot(self):
           """Update visualization in place (no flicker)."""
           if self.mesh is None:
               return

           # Update mesh geometry/data
           # self.mesh.points = new_points
           # self.mesh.Modified()
           # self.mesh_actor.mapper.SetInputData(self.mesh)
           # self.mesh_actor.mapper.Update()

           self.plotter.render()

       def _setup_widgets(self):
           """Setup interactive 3D widgets (if needed)."""
           # Example: add_sphere_widget with 2D constraint
           # self.widget = self.plotter.add_sphere_widget(
           #     self._on_widget_move,
           #     center=(0, 0, 0),
           #     radius=0.15,
           #     color="blue",
           #     pass_widget=True,
           #     interaction_event='always',  # Update during drag
           # )
           pass

       def _on_param_a_changed(self, value):
           """Callback for parameter A slider."""
           self.param_a = value / 10.0
           self._update_plot()

       def _on_param_b_changed(self, value):
           """Callback for parameter B slider."""
           self.param_b = value / 10.0
           self._update_plot()

       def show(self):
           """Display the demo."""
           self.plotter.show()
           self.plotter.app.exec()


   def main():
       """Run the demo."""
       demo = MyNewDemo()
       demo.show()


   def _build_vscode_url(path):
       """Build VS Code file URL."""
       posix_path = Path(path).resolve().as_posix()
       if not posix_path.startswith("/"):
           posix_path = f"/{posix_path}"
       return f"vscode://file{quote(posix_path, safe='/:')}"


   def get_manifest():
       """Get demo manifest with source URL."""
       manifest = dict(DEMO_MANIFEST)
       manifest["source_url"] = _build_vscode_url(__file__)
       return manifest


   DEMO_MANIFEST = {
       "title": "My New Demo",
       "description": "Brief description of what this demo does.\n\n"
       "Features:\n"
       "- Feature 1\n"
       "- Feature 2\n"
       "- Feature 3",
   }


   if __name__ == "__main__":
       main()
   ```

3. **Create a README.md** in your demo folder:
   ```markdown
   # My New Demo

   Brief description of the demo.

   ## Features
   - Feature 1
   - Feature 2
   - Feature 3

   ## Usage
   Run from the launcher or directly:
   ```bash
   uv run python -m demos.my_new_demo.my_new_demo
   ```

   ## Controls
   - **Parameter A**: Adjust with slider
   - **Parameter B**: Adjust with slider
   ```

4. **Register your demo** automatically:
   The launcher discovers demos automatically from `DEMO_MANIFEST` in each demo module.
   Just create your `DEMO_MANIFEST` dict with `title` and `description` keys (see template above).

5. **Test your demo**:
   ```bash
   # Test directly
   uv run python -m demos.my_new_demo.my_new_demo

   # Test from launcher
   uv run python src/yavuz/launcher.py
   ```

## Key Patterns to Follow

### 1. Class Structure
- Single demo class that encapsulates everything
- Initialize BackgroundPlotter with window_size and title
- Use `self.plotter.set_background("white")`
- Store state variables before scene setup

### 2. Initialization Order
1. `_setup_controls()` - Set up Qt dock widget with sliders/buttons
2. `_build_scene()` - Create meshes/actors once, stored as instance variables
3. `_update_plot()` - Called on init to render first frame
4. `_setup_widgets()` - Add interactive widgets (optional)
5. `plotter.view_xy()` or other camera setup - Set view last

### 3. Controls with PySide6
**NEVER use PyVista widgets** - they're not supported on Windows.
Always use Qt dock widgets:
```python
def _setup_controls(self):
    dock = QtWidgets.QDockWidget("Controls", self.plotter)
    dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)

    # Add sliders
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.valueChanged.connect(self._on_value_changed)
    layout.addWidget(slider)

    layout.addStretch(1)  # Push controls to top
    dock.setWidget(panel)
    self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
```

### 4. Mesh Updates (No Flicker)
For dynamic updates, reuse actors - never remove/re-add:
```python
# Init: create once
self.mesh = pv.Grid(...)
self.actor = self.plotter.add_mesh(self.mesh, ...)

# Update: modify in place
self.mesh.points = new_points
self.mesh.Modified()
self.actor.mapper.SetInputData(self.mesh)
self.actor.mapper.Update()
self.plotter.render()
```

Use `pv.lines_from_points()` for dynamic lines, NOT `pv.Line()`.

### 5. Interactive Widgets (3D)
For 2D-constrained dragging (like in line_offset demo):
```python
self.widget = self.plotter.add_sphere_widget(
    self._on_widget_move,
    center=(x, y, 0),
    radius=0.15,
    color="blue",
    pass_widget=True,           # Get widget object in callback
    interaction_event='always',  # Update during drag, not just drop
)

def _on_widget_move(self, center, widget):
    # Constrain to 2D plane
    x, y = center[0], center[1]
    widget.SetCenter(x, y, 0)   # Force z to 0
    self._update_plot()
```

### 6. DEMO_MANIFEST
Required for launcher discovery:
```python
DEMO_MANIFEST = {
    "title": "Demo Title",
    "description": "Multi-line description with features:\n\n"
    "Features:\n"
    "- Feature 1\n"
    "- Feature 2",
}

def get_manifest():
    """Launcher calls this to get manifest with source_url."""
    manifest = dict(DEMO_MANIFEST)
    manifest["source_url"] = _build_vscode_url(__file__)
    return manifest
```

## Best Practices

1. **Refer to PyVista documentation** - Always check API docs before using methods
2. **Use descriptive names** for functions and classes
3. **Add docstrings** to explain functionality
4. **Store state variables** as instance attributes
5. **Reuse meshes/actors** - Never add/remove, just update in place
6. **Call `.Modified()` then `mapper.Update()`** after mesh changes
7. **Use button callbacks** instead of key bindings (more discoverable)
8. **Test edge cases** before adding to launcher
9. **Include meaningful labels** in your controls
10. **Provide visual feedback** for user actions

## PyVista Documentation URLs

**ALWAYS refer to these before using an API:**
- Main API docs: https://docs.pyvista.org/api/
- Sphere widget: https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_sphere_widget.html
- Chart2D: https://docs.pyvista.org/api/plotting/charts/
- Common patterns: Check `AGENTS.md` in root for tested patterns

## Tips for Algorithm Visualization

1. **Start simple** - Get basic visualization working first
2. **Add interactivity gradually** - One control at a time
3. **Use colors effectively** - Highlight important elements with color gradients
4. **Provide feedback** - Show what the algorithm is doing in real-time
5. **Control performance** - Update only what changed, reuse meshes/actors
6. **Test on Windows** - Avoid PyVista widgets (not supported)
7. **Add descriptions** - Explain controls and what's happening
8. **Use top-down views** for 2D demos: `self.plotter.view_xy()`

