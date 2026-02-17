"""
Mesh Boolean Playground (PyVista)
Demonstrate manifold3d mesh booleans with PyVista controls.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import manifold3d as m
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import BackgroundPlotter


class MeshBooleanDemo:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1200, 800),
            title="Mesh Boolean Playground",
        )
        self.plotter.set_background("white")
        self.plotter.show_grid()

        # Add lighting for better visuals
        light1 = pv.Light(position=(3, 3, 3), light_type='cameralight')
        light2 = pv.Light(position=(-3, -3, 3), light_type='cameralight', intensity=0.5)
        light3 = pv.Light(position=(0, 3, -3), light_type='cameralight', intensity=0.3)
        self.plotter.add_light(light1)
        self.plotter.add_light(light2)
        self.plotter.add_light(light3)

        self.shape_options = ["Cube", "Sphere", "Cylinder", "Tetrahedron"]
        self.op_options = ["Union", "Intersect", "Subtract (A - B)"]

        self.shape_a = 0
        self.shape_b = 1
        self.operation = 0
        self.distance = 1.8
        self.shear = 0.0
        self.wireframe = False

        self.colors = {
            "A": (0.12, 0.47, 0.71),
            "B": (1.0, 0.5, 0.1),
            "Result": (0.17, 0.63, 0.17),
        }

        self.actors = {
            "A_clipped": None,
            "B_clipped": None,
            "Result": None
        }
        self.meshes = {
            "A_clipped": None,
            "B_clipped": None,
            "Result": None
        }

        self._setup_controls()
        self._build_scene()
        self._update_scene()

    def _setup_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self.plotter)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Shape A combo box
        layout.addWidget(QtWidgets.QLabel("Shape A"))
        self.shape_a_combo = QtWidgets.QComboBox()
        self.shape_a_combo.addItems(self.shape_options)
        self.shape_a_combo.setCurrentIndex(self.shape_a)
        self.shape_a_combo.currentIndexChanged.connect(self._on_shape_a)
        layout.addWidget(self.shape_a_combo)

        # Shape B combo box
        layout.addWidget(QtWidgets.QLabel("Shape B"))
        self.shape_b_combo = QtWidgets.QComboBox()
        self.shape_b_combo.addItems(self.shape_options)
        self.shape_b_combo.setCurrentIndex(self.shape_b)
        self.shape_b_combo.currentIndexChanged.connect(self._on_shape_b)
        layout.addWidget(self.shape_b_combo)

        # Operation combo box
        layout.addWidget(QtWidgets.QLabel("Operation"))
        self.op_combo = QtWidgets.QComboBox()
        self.op_combo.addItems(self.op_options)
        self.op_combo.setCurrentIndex(self.operation)
        self.op_combo.currentIndexChanged.connect(self._on_operation)
        layout.addWidget(self.op_combo)

        # Distance slider (0.2-3.5, scaled as 2-35)
        layout.addWidget(QtWidgets.QLabel("Distance"))
        self.dist_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dist_slider.setMinimum(2)
        self.dist_slider.setMaximum(35)
        self.dist_slider.setValue(int(self.distance * 10))
        self.dist_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.dist_slider.setTickInterval(2)
        self.dist_slider.valueChanged.connect(self._on_distance)
        layout.addWidget(self.dist_slider)

        # Shear slider (-2.0-2.0, scaled as -20-20)
        layout.addWidget(QtWidgets.QLabel("Shear"))
        self.shear_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.shear_slider.setMinimum(-20)
        self.shear_slider.setMaximum(20)
        self.shear_slider.setValue(int(self.shear * 10))
        self.shear_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.shear_slider.setTickInterval(5)
        self.shear_slider.valueChanged.connect(self._on_shear)
        layout.addWidget(self.shear_slider)

        # Wireframe checkbox
        self.wire_checkbox = QtWidgets.QCheckBox("Wireframe")
        self.wire_checkbox.setChecked(self.wireframe)
        self.wire_checkbox.stateChanged.connect(self._on_wireframe)
        layout.addWidget(self.wire_checkbox)

        # Stretch to push controls to top
        layout.addStretch(1)

        dock.setWidget(panel)
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        self.plotter.add_key_event("r", self._reset)
        self.plotter.add_key_event("n", self._randomize)

    def _make_shape(self, name):
        if name == "Cube":
            return m.Manifold.cube((1.0, 1.0, 1.0), center=True)
        if name == "Sphere":
            return m.Manifold.sphere(0.65, 32)
        if name == "Cylinder":
            return m.Manifold.cylinder(1.2, 0.5, 0.5, 40, True)
        if name == "Tetrahedron":
            return m.Manifold.tetrahedron().scale((0.9, 0.9, 0.9))
        return m.Manifold.cube((1.0, 1.0, 1.0), center=True)

    def _apply_boolean(self, a, b):
        if self.operation == 0:
            return m.Manifold.batch_boolean([a, b], m.OpType.Add)
        if self.operation == 1:
            return m.Manifold.batch_boolean([a, b], m.OpType.Intersect)
        return m.Manifold.batch_boolean([a, b], m.OpType.Subtract)

    def _polydata_from_manifold(self, manifold):
        if manifold is None or manifold.is_empty():
            return None
        mesh = manifold.to_mesh()
        vertices = np.asarray(mesh.vert_properties, dtype=np.float32)
        faces = np.asarray(mesh.tri_verts, dtype=np.int64)
        face_sizes = np.full((faces.shape[0], 1), 3, dtype=np.int64)
        faces = np.hstack((face_sizes, faces)).ravel()
        return pv.PolyData(vertices, faces)

    def _build_scene(self):
        """Initialize mesh actors once."""
        # Create placeholder meshes and actors
        for key in ["A_clipped", "B_clipped", "Result"]:
            # Start with a simple sphere as placeholder
            mesh = pv.Sphere(radius=0.1)
            if self.wireframe:
                actor = self.plotter.add_mesh(
                    mesh,
                    color=self.colors.get(key.split("_")[0], self.colors["Result"]),
                    opacity=1.0,
                    style="wireframe",
                    line_width=1,
                )
            else:
                opacity = 0.05 if "clipped" in key else 1.0
                actor = self.plotter.add_mesh(
                    mesh,
                    color=self.colors.get(key.split("_")[0], self.colors["Result"]),
                    opacity=opacity,
                    smooth_shading=True,
                )
            self.meshes[key] = mesh
            self.actors[key] = actor
            # Hide initially
            actor.SetVisibility(False)

    def _update_mesh(self, key, new_mesh, opacity):
        """Update mesh in-place without removing actor."""
        if new_mesh is None:
            # Hide if no mesh
            if self.actors[key]:
                self.actors[key].SetVisibility(False)
            return

        # Update mesh data
        self.meshes[key].copy_from(new_mesh)
        self.meshes[key].Modified()

        # Update mapper
        self.actors[key].mapper.SetInputData(self.meshes[key])
        self.actors[key].mapper.Update()

        # Update properties
        if not self.wireframe:
            self.actors[key].GetProperty().SetOpacity(opacity)

        # Show the actor
        self.actors[key].SetVisibility(True)

    def _update_scene(self):
        pos_a = np.array([-0.6, 0.0, 0.0], dtype=float)
        pos_b = pos_a + np.array([self.distance, self.shear, 0.0])

        shape_a = self._make_shape(self.shape_options[self.shape_a]).translate(tuple(pos_a))
        shape_b = self._make_shape(self.shape_options[self.shape_b]).translate(tuple(pos_b))
        result = self._apply_boolean(shape_a, shape_b)

        # Compute clipped parts (parts that are removed)
        clipped_a = None
        clipped_b = None

        if not result.is_empty():
            # Clipped A = A - Result (parts of A that are cut away)
            clipped_a = m.Manifold.batch_boolean([shape_a, result], m.OpType.Subtract)
            # Clipped B = B - Result (parts of B that are cut away)
            clipped_b = m.Manifold.batch_boolean([shape_b, result], m.OpType.Subtract)
        else:
            # When result is empty (no intersection), show original shapes at full opacity
            clipped_a = shape_a
            clipped_b = shape_b

        mesh_a_clipped = self._polydata_from_manifold(clipped_a)
        mesh_b_clipped = self._polydata_from_manifold(clipped_b)
        mesh_result = self._polydata_from_manifold(result)

        # Update meshes in place (no remove/add to avoid flickering)
        # When result is empty, show originals at 1.0 opacity, otherwise show clipped at 0.05
        opacity_clipped = 1.0 if result.is_empty() else 0.05
        self._update_mesh("A_clipped", mesh_a_clipped, opacity_clipped)
        self._update_mesh("B_clipped", mesh_b_clipped, opacity_clipped)
        self._update_mesh("Result", mesh_result, 1.0)

        bounds = self._combine_bounds([mesh_a_clipped, mesh_b_clipped, mesh_result])
        self.plotter.reset_camera(bounds=bounds)
        self.plotter.render()

    def _combine_bounds(self, meshes):
        valid = [mesh.bounds for mesh in meshes if mesh is not None]
        if not valid:
            return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        mins = np.min(np.array([b[::2] for b in valid]), axis=0)
        maxs = np.max(np.array([b[1::2] for b in valid]), axis=0)
        return (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])

    def _on_shape_a(self, value):
        self.shape_a = int(value)
        self._update_scene()

    def _on_shape_b(self, value):
        self.shape_b = int(value)
        self._update_scene()

    def _on_operation(self, value):
        self.operation = int(value)
        self._update_scene()

    def _on_distance(self, value):
        self.distance = value / 10.0
        self._update_scene()

    def _on_shear(self, value):
        self.shear = value / 10.0
        self._update_scene()

    def _on_wireframe(self, state):
        self.wireframe = bool(state)
        # Rebuild scene to change rendering style
        for key in self.actors:
            if self.actors[key]:
                self.plotter.remove_actor(self.actors[key])
        self._build_scene()
        self._update_scene()

    def _reset(self):
        self.shape_a = 0
        self.shape_b = 1
        self.operation = 0
        self.distance = 1.8
        self.shear = 0.0
        self.wireframe = False
        self.shape_a_combo.setCurrentIndex(self.shape_a)
        self.shape_b_combo.setCurrentIndex(self.shape_b)
        self.op_combo.setCurrentIndex(self.operation)
        self.dist_slider.setValue(int(self.distance * 10))
        self.shear_slider.setValue(int(self.shear * 10))
        self.wire_checkbox.setChecked(self.wireframe)
        self._update_scene()

    def _randomize(self):
        rng = np.random.default_rng()
        self.shape_a = int(rng.integers(0, len(self.shape_options)))
        self.shape_b = int(rng.integers(0, len(self.shape_options)))
        self.operation = int(rng.integers(0, len(self.op_options)))
        self.distance = float(rng.uniform(0.4, 3.2))
        self.shear = float(rng.uniform(-1.6, 1.6))
        self.shape_a_combo.setCurrentIndex(self.shape_a)
        self.shape_b_combo.setCurrentIndex(self.shape_b)
        self.op_combo.setCurrentIndex(self.operation)
        self.dist_slider.setValue(int(self.distance * 10))
        self.shear_slider.setValue(int(self.shear * 10))
        self._update_scene()

    def show(self):
        self.plotter.show()
        self.plotter.app.exec()


def main():
    demo = MeshBooleanDemo()
    demo.show()


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
    "title": "Mesh Boolean Playground",
    "description": "Real-time mesh booleans with PyVista rendering.\n\n"
    "Features:\n"
    "- Cube, sphere, cylinder, and tetrahedron operands\n"
    "- Union, intersection, and subtraction booleans\n"
    "- Dropdowns for shape and operation selection\n"
    "- Sliders for distance and shear\n"
    "- Visual distinction: result (1.0 opacity) vs clipped parts (0.05 opacity)\n"
    "- Wireframe/solid toggle",
}


if __name__ == "__main__":
    main()
