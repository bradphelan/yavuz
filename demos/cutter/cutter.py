"""
Cutter Tool Sweep Visualizer
Visualize a spherical cutting tool sweeping through a triangulated 3D geometry using manifold
geometry processing.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import BackgroundPlotter

try:
    import manifold3d as manifold
    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False


class CutterDemo:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1200, 800),
            title="Cutter Tool Sweep",
        )
        self.plotter.set_background("white")

        # State variables
        self.tool_position = np.array([0.0, 0.0, 0.0])
        self.tool_radius = 0.5
        self.sweep_progress = 0.0
        self.show_swept_volume = True
        self.show_tool = True
        self.sweep_step = 1.0
        self.tess_tolerance = 0.05

        # Store meshes and actors for reuse
        self.cutter_mesh = None
        self.cutter_actor = None
        self.bunny_base_mesh = None
        self.bunny_manifold = None
        self.tool_mesh = None
        self.tool_actor = None
        self.swept_volume_mesh = None
        self.swept_volume_actor = None
        self.tool_widget = None

        # Sweep path (tool trajectory)
        self.sweep_path = self._create_sweep_path()

        # Setup in order: controls, scene, widgets, camera
        self._setup_controls()
        self._build_scene()
        self._update_plot()
        self._setup_widgets()
        self.plotter.reset_camera()

    def _create_sweep_path(self):
        """Create a spiral sweep path in the XY plane."""
        turns = 3.0
        t = np.linspace(0.0, 2.0 * np.pi * turns, 160)
        radius = np.linspace(0.5, 3.0, t.size)
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = np.zeros_like(x)
        return np.column_stack([x, y, z])

    def _create_cutter_geometry(self):
        """Create a cube mesh for the tool to cut through."""
        cube = pv.Cube(center=(0.0, 0.0, 0.0), x_length=6.0, y_length=4.0, z_length=4.0)
        return cube.triangulate()

    def _create_tool_mesh(self):
        """Create a lollipop tool composed of a sphere and a handle."""
        sphere = pv.Sphere(
            radius=self.tool_radius,
            center=self.tool_position,
            theta_resolution=32,
            phi_resolution=32,
        )
        handle_length = self.tool_radius * 3.5
        handle_radius = self.tool_radius * 0.35
        handle_center = self.tool_position + np.array(
            [0.0, 0.0, self.tool_radius + handle_length / 2.0]
        )
        handle = pv.Cylinder(
            center=handle_center,
            direction=(0.0, 0.0, 1.0),
            radius=handle_radius,
            height=handle_length,
            resolution=48,
        )
        return sphere.merge(handle)

    def _create_tool_manifold(self, center):
        """Create a lollipop tool as a manifold centered at the given position."""
        sphere = manifold.Manifold.sphere(self.tool_radius).translate(center)

        handle_length = self.tool_radius * 3.5
        handle_radius = self.tool_radius * 0.35
        handle_offset = self.tool_radius + handle_length / 2.0
        handle_center = np.asarray(center) + np.array([0.0, 0.0, handle_offset])

        handle = (
            manifold.Manifold.cylinder(
                height=handle_length,
                radius_low=handle_radius,
                radius_high=handle_radius,
                center=True,
            )
            .translate(handle_center)
        )

        return sphere + handle

    def _mesh_to_manifold(self, mesh):
        """Convert a triangulated PyVista mesh to a manifold."""
        vertices = mesh.points.astype(np.float32)
        faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.uint32)
        m = manifold.Mesh(vertices, faces)
        m.merge()
        return manifold.Manifold(m)

    def _manifold_to_mesh(self, solid):
        """Convert a manifold to PyVista PolyData."""
        mesh = solid.to_mesh()
        vertices = np.asarray(mesh.vert_properties)[:, :3]
        triangles = np.asarray(mesh.tri_verts)
        cell_array = np.column_stack([
            np.full(len(triangles), 3, dtype=np.int64),
            triangles.astype(np.int64)
        ]).flatten()
        return pv.PolyData(vertices, cell_array)

    def _setup_controls(self):
        """Setup Qt dock widget with controls."""
        dock = QtWidgets.QDockWidget("Controls", self.plotter)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Sweep Progress slider
        layout.addWidget(QtWidgets.QLabel("Sweep Progress"))
        self.slider_progress = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_progress.setMinimum(0)
        self.slider_progress.setMaximum(100)
        self.slider_progress.setValue(int(self.sweep_progress * 100))
        self.slider_progress.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_progress.setTickInterval(10)
        self.slider_progress.valueChanged.connect(self._on_progress_changed)
        layout.addWidget(self.slider_progress)

        # Tool Radius slider
        layout.addWidget(QtWidgets.QLabel("Tool Radius"))
        self.slider_radius = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_radius.setMinimum(10)
        self.slider_radius.setMaximum(100)
        self.slider_radius.setValue(int(self.tool_radius * 100))
        self.slider_radius.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_radius.setTickInterval(10)
        self.slider_radius.valueChanged.connect(self._on_radius_changed)
        layout.addWidget(self.slider_radius)

        # Sweep Step Size slider
        layout.addWidget(QtWidgets.QLabel("Sweep Step Size"))
        self.slider_step = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_step.setMinimum(1)
        self.slider_step.setMaximum(100)
        self.slider_step.setValue(int(self.sweep_step * 10))
        self.slider_step.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_step.setTickInterval(10)
        self.slider_step.valueChanged.connect(self._on_step_changed)
        layout.addWidget(self.slider_step)

        # Tessellation Tolerance slider
        layout.addWidget(QtWidgets.QLabel("Tessellation Tolerance"))
        self.slider_tess = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_tess.setMinimum(1)
        self.slider_tess.setMaximum(100)
        self.slider_tess.setValue(int(self.tess_tolerance * 1000))
        self.slider_tess.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_tess.setTickInterval(10)
        self.slider_tess.valueChanged.connect(self._on_tess_changed)
        layout.addWidget(self.slider_tess)

        # Show/Hide Swept Volume checkbox
        self.check_swept = QtWidgets.QCheckBox("Show Swept Volume")
        self.check_swept.setChecked(self.show_swept_volume)
        self.check_swept.stateChanged.connect(self._on_swept_toggled)
        layout.addWidget(self.check_swept)

        # Show/Hide Tool checkbox
        self.check_tool = QtWidgets.QCheckBox("Show Tool")
        self.check_tool.setChecked(self.show_tool)
        self.check_tool.stateChanged.connect(self._on_tool_toggled)
        layout.addWidget(self.check_tool)

        # Push controls to top
        layout.addStretch(1)

        dock.setWidget(panel)
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _build_scene(self):
        """Initialize all meshes and actors once."""
        # Create cutter geometry
        self.bunny_base_mesh = self._create_cutter_geometry()
        if HAS_MANIFOLD:
            self.bunny_manifold = self._mesh_to_manifold(self.bunny_base_mesh)
            if self.bunny_manifold.is_empty() or self.bunny_manifold.status() != manifold.Error.NoError:
                print(
                    "Base manifold invalid; status="
                    f"{self.bunny_manifold.status().name}. Cutting disabled."
                )
                self.bunny_manifold = None
        self.cutter_mesh = self.bunny_base_mesh.copy()
        self.cutter_actor = self.plotter.add_mesh(
            self.cutter_mesh,
            color="gainsboro",
            edge_color="slategray",
            show_edges=False,
            opacity=1.0,
        )
        self.plotter.show_bounds()

        # Create tool sphere (will be updated)
        self.tool_mesh = self._create_tool_mesh()
        self.tool_actor = self.plotter.add_mesh(
            self.tool_mesh, color="red", opacity=0.6
        )

        # Create swept volume mesh with initial sphere
        self.swept_volume_mesh = pv.Sphere(radius=self.tool_radius, center=self.tool_position)
        self.swept_volume_actor = self.plotter.add_mesh(
            self.swept_volume_mesh, color="yellow", opacity=0.3
        )

    def _update_plot(self):
        """Update visualization in place."""
        if self.cutter_mesh is None:
            return

        # Update tool position based on sweep progress
        idx = int(self.sweep_progress * (len(self.sweep_path) - 1))
        idx = min(idx, len(self.sweep_path) - 1)
        self.tool_position = self.sweep_path[idx].copy()

        # Update tool sphere
        self.tool_mesh = self._create_tool_mesh()
        self.tool_actor.mapper.SetInputData(self.tool_mesh)
        self.tool_actor.mapper.Update()

        # Update swept volume using manifold for proper geometry
        swept_manifold = None
        if HAS_MANIFOLD:
            try:
                manifold.set_min_circular_edge_length(self.tess_tolerance)
                # Create swept volume by unioning spheres along path using manifold

                centers = self._sample_sweep_centers(idx)
                tool_manifolds = [self._create_tool_manifold(center) for center in centers]
                if tool_manifolds:
                    swept_manifold = manifold.Manifold.batch_boolean(
                        tool_manifolds,
                        manifold.OpType.Add,
                    )
            except Exception as e:
                # Fallback to simple merge if manifold fails
                print(f"Manifold operation failed: {e}, using fallback merge")
                self._update_swept_volume_fallback(idx)
        if self.show_swept_volume and swept_manifold is not None and not swept_manifold.is_empty():
            new_mesh = self._manifold_to_mesh(swept_manifold)
            self.swept_volume_mesh.copy_from(new_mesh)
            self.swept_volume_mesh.Modified()
        elif self.show_swept_volume and not HAS_MANIFOLD:
            # Fallback if manifold not available
            self._update_swept_volume_fallback(idx)

        if HAS_MANIFOLD and self.bunny_manifold is not None and swept_manifold is not None:
            try:
                cut_manifold = self.bunny_manifold - swept_manifold
                if not cut_manifold.is_empty():
                    cut_mesh = self._manifold_to_mesh(cut_manifold)
                    self.cutter_mesh.copy_from(cut_mesh)
                    self.cutter_mesh.Modified()
                    self.cutter_actor.SetVisibility(True)
                else:
                    self.cutter_mesh.copy_from(self.bunny_base_mesh)
                    self.cutter_mesh.Modified()
                    self.cutter_actor.SetVisibility(True)
            except Exception as e:
                print(f"Manifold cut failed: {e}, using original bunny")
                self.cutter_mesh.copy_from(self.bunny_base_mesh)
                self.cutter_mesh.Modified()
                self.cutter_actor.SetVisibility(True)
        else:
            self.cutter_mesh.copy_from(self.bunny_base_mesh)
            self.cutter_mesh.Modified()
            self.cutter_actor.SetVisibility(True)

        self.swept_volume_actor.mapper.SetInputData(self.swept_volume_mesh)
        self.swept_volume_actor.mapper.Update()

        self.cutter_actor.mapper.SetInputData(self.cutter_mesh)
        self.cutter_actor.mapper.Update()

        # Update visibility
        self.tool_actor.SetVisibility(self.show_tool)
        self.swept_volume_actor.SetVisibility(self.show_swept_volume)

        self.plotter.render()

    def _update_swept_volume_fallback(self, idx):
        """Fallback method to compute swept volume without manifold."""
        swept_points = []
        centers = self._sample_sweep_centers(idx)

        for center in centers:
            sphere = pv.Sphere(radius=self.tool_radius, center=center, theta_resolution=12, phi_resolution=12)
            swept_points.append(sphere)

        if swept_points:
            merged = swept_points[0]
            for sphere in swept_points[1:]:
                merged = merged.merge(sphere)

            self.swept_volume_mesh.copy_from(merged)
            self.swept_volume_mesh.Modified()

    def _setup_widgets(self):
        """Setup interactive 3D widgets."""
        self.tool_widget = self.plotter.add_sphere_widget(
            self._on_tool_drag,
            center=tuple(self.tool_position),
            radius=0.2,
            color="darkred",
            pass_widget=True,
            interaction_event="always",
        )

    def _on_tool_drag(self, center, widget):
        """Handle tool widget dragging."""
        self.tool_position = np.array(center)
        # Update slider to match tool position along sweep path
        distances = np.linalg.norm(
            self.sweep_path - self.tool_position, axis=1
        )
        closest_idx = np.argmin(distances)
        self.sweep_progress = closest_idx / (len(self.sweep_path) - 1)
        self.slider_progress.setValue(int(self.sweep_progress * 100))
        widget.SetCenter(*self.tool_position)

    def _on_progress_changed(self, value):
        """Handle sweep progress slider change."""
        self.sweep_progress = value / 100.0
        self._update_plot()

    def _on_radius_changed(self, value):
        """Handle tool radius slider change."""
        self.tool_radius = value / 100.0

        # Recreate tool sphere with new radius
        self.tool_mesh = self._create_tool_mesh()
        self.tool_actor.mapper.SetInputData(self.tool_mesh)
        self.tool_actor.mapper.Update()

        # Update swept volume
        self._update_plot()

    def _on_step_changed(self, value):
        """Handle sweep step size change."""
        self.sweep_step = max(0.1, value / 10.0)
        self._update_plot()

    def _on_tess_changed(self, value):
        """Handle tessellation tolerance change."""
        self.tess_tolerance = max(0.001, value / 1000.0)
        self._update_plot()

    def _sample_sweep_centers(self, idx):
        """Sample centers along the sweep path with fractional step support."""
        step = max(0.1, float(self.sweep_step))
        max_idx = len(self.sweep_path) - 1
        last_t = min(float(idx), float(max_idx))
        samples = np.arange(0.0, last_t + 1e-6, step)
        if samples.size == 0 or samples[-1] < last_t:
            samples = np.append(samples, last_t)

        centers = []
        for t in samples:
            i0 = int(np.floor(t))
            i1 = min(i0 + 1, max_idx)
            alpha = t - i0
            center = (1.0 - alpha) * self.sweep_path[i0] + alpha * self.sweep_path[i1]
            centers.append(center)

        return centers

    def _on_swept_toggled(self, state):
        """Handle swept volume visibility toggle."""
        self.show_swept_volume = bool(state)
        self._update_plot()

    def _on_tool_toggled(self, state):
        """Handle tool visibility toggle."""
        self.show_tool = bool(state)
        self._update_plot()

    def show(self):
        """Display the demo."""
        self.plotter.show()
        self.plotter.app.exec()


def main():
    """Run the demo."""
    demo = CutterDemo()
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
    "title": "Cutter Tool Sweep",
    "description": "Visualize a spherical cutting tool sweeping through 3D geometry.\n\n"
    "Features:\n"
    "- Interactive tool path with sweep progress slider\n"
    "- Adjustable tool radius\n"
    "- Real-time swept volume visualization\n"
    "- Drag tool widget along sweep path\n"
    "- Toggle swept volume and tool visibility\n"
    "- Useful for CAM simulation and verification",
}


if __name__ == "__main__":
    main()
