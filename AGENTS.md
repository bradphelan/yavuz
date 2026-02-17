
## Python environment
We use ``uv`` so don't create requirements.txt. pyproject.toml is all you need

## Initializing the sandbox
Run setup.bat or setup.sh

## Starting the app
Run start.bat or start.sh and it will sort out the venv

## Running individual demos
uv run .\demos\douglas_peucker\douglas_peucker.py

## Agent
You are a helpful agent designed to create algorithmic demos on demand and integrate them in the
demos folder. You are using python, numpy, pyvista qt. Always look for documentation or look
at the style of other demos for idea. Don't be afraid to ask the user to clarify. Questions
are expected.

https://docs.pyvista.org/api/plotting/charts/
https://docs.pyvista.org/api/plotting/charts/_autosummary/pyvista.chart2d
https://docs.pyvista.org/examples/02-plot/chart_basics#chart-basics-example

## UI/UX Guide lines for Yavuz Algorithm Demos

### PyVista + PySide6 Integration

#### 1. Control Panel
- Use Qt dock widget: `dock.setWidget(panel)` then `plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)`
- Never use PyVista widgets (not supported on Windows)
- Add `layout.addStretch(1)` to push controls to top
- Use buttons in control panel, not key bindings
- For animations: separate Start, Stop, and Reset buttons

#### 2. 2D Charts (Bar Charts, Line Charts)

**Use Chart2D API for proper 2D visualizations:**
```python
# Init: create chart once
self.chart = pv.Chart2D()
self.chart.x_label = "Index"
self.chart.y_label = "Value"
self.chart.grid = False
self.plotter.add_chart(self.chart)

# Create bar plot on first call
if self.bars_plot is None:
    x = np.arange(len(data))
    self.bars_plot = self.chart.bar(x, data, color="skyblue")
    self.chart.x_axis.range = [x.min()-1, x.max()+1]
    self.chart.y_axis.range = [0, data.max()*1.1]
else:
    # Update existing bars - maintains bar width
    self.bars_plot.update(x, data)
```

#### 3. Mesh Updates (No Flicker)
**Fixed geometry (StructuredGrid):**
```python
# Init: create once
self.grid = pv.StructuredGrid(x, y, z)
self.mesh_actor = self.plotter.add_mesh(self.grid, ...)

# Update: modify in place
self.grid.points = new_points
self.grid.point_data["scalars"] = new_scalars
self.grid.Modified()
self.mesh_actor.mapper.SetInputData(self.grid)
self.mesh_actor.mapper.Update()
```

**Changing connectivity (PolyData lines):**
```python
# Init: create once
self.line_mesh = pv.lines_from_points(points)
self.line_actor = self.plotter.add_mesh(self.line_mesh, ...)

# Update: use copy_from() - NEVER assign .cells directly
new_mesh = pv.lines_from_points(new_points)
self.line_mesh.copy_from(new_mesh)
self.line_mesh.Modified()
self.line_actor.mapper.SetInputData(self.line_mesh)
self.line_actor.mapper.Update()
```

**Optional/conditional meshes:**
```python
# Init: lazy initialization
self.highlight_mesh = None
self.highlight_actor = None

# Update: create on first use, hide when empty
if has_data:
    if self.highlight_actor is None:
        self.highlight_mesh = pv.PolyData(points, lines)
        self.highlight_actor = self.plotter.add_mesh(self.highlight_mesh, ...)
    else:
        self.highlight_mesh.copy_from(new_mesh)  # or .points = ... if same connectivity
        self.highlight_mesh.Modified()
        self.highlight_actor.mapper.SetInputData(self.highlight_mesh)
        self.highlight_actor.mapper.Update()
        self.highlight_actor.SetVisibility(True)
else:
    if self.highlight_actor:
        self.highlight_actor.SetVisibility(False)
```

#### 4. Multi-Panel Layouts
```python
self.plotter = BackgroundPlotter(shape=(rows, cols))
self.plotter.subplot(row, col)  # Switch active subplot
```

#### 5. Lighting
**Add lights for better 3D visuals:**
```python
light1 = pv.Light(position=(3, 3, 3), light_type='cameralight')
light2 = pv.Light(position=(-3, -3, 3), light_type='cameralight', intensity=0.5)
light3 = pv.Light(position=(0, 3, -3), light_type='cameralight', intensity=0.3)
plotter.add_light(light1)
plotter.add_light(light2)
plotter.add_light(light3)
```

#### 6. Common Patterns
- Slider scaling: `value / 100.0` for int→float conversion
- Camera for 3D: always call `plotter.reset_camera()` after updating mesh if bounds change
- Camera for 2D/fixed bounds: reset once on init, use `plotter.render()` for updates
- Rendering: call `plotter.render()` at end of update functions to refresh display
- Don't use `show_grid()` for 3D visualizations with dynamic bounds - it constrains the view
- For 3D with changing bounds: use `show_bounds()` ONCE in init (not on every update to avoid flickering), then `reset_camera()` adjusts view
- App entry: `plotter.app.exec()` (PySide6 style)
- Manifest: define `DEMO_MANIFEST` dict for launcher
- Lighting: Use light_type='cameralight' (not 'camera')

#### 7. PyVista API Reference
**ALWAYS consult PyVista documentation before using widget or method - DO NOT GUESS.**

Key URLs:
- Plotter widgets: https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_sphere_widget.html
- For any method/class, use: https://docs.pyvista.org/api/ (search for exact method name)

#### Demo template
Use [demos/DEMO_TEMPLATE.md] when creating a new demo


### Checklist
- [ ] **ALWAYS CONSULT PYVISTA DOCS** before using methods - never guess API signatures
- [ ] Qt dock widget attached to `app_window`
- [ ] Use Chart2D API for bar charts and 2D plots (not PolyData hacks)
- [ ] Meshes reused, never removed/re-added
- [ ] Call `.Modified()` after mesh updates
- [ ] Call `mapper.SetInputData()` + `mapper.Update()` after mesh changes
- [ ] Use `mesh.copy_from()` for connectivity changes (never assign `.cells`)
- [ ] Lazy init optional meshes (None), use `SetVisibility()` to show/hide
- [ ] Never add empty meshes (zero points)
- [ ] Use buttons in control panel, not key bindings
- [ ] For 3D: always reset camera after updates if data bounds change
- [ ] Add lights for 3D scenes (use light_type='cameralight')
- [ ] Call `plotter.render()` at end of update functions
- [ ] Use `pv.lines_from_points()` for dynamic lines, not `pv.Line()`
- [ ] For widgets with continuous updates: set `interaction_event='always'`
- [ ] For 2D constrained widgets: use `pass_widget=True` and `widget.SetCenter()` to force constraints

