"""
Mesh Boolean Playground (PyVista)
Demonstrate manifold3d mesh booleans with PyVista controls.
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import manifold3d as m
import pyvista as pv
from pyvistaqt import BackgroundPlotter


class MeshBooleanDemo:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1200, 800),
            title="Mesh Boolean Playground",
        )
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.plotter.show_grid()

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

        self.actors = {"A": None, "B": None, "Result": None}
        self.hud = self.plotter.add_text("", position="upper_left", font_size=10)

        self._setup_controls()
        self._update_scene()
        self._update_hud()

    def _setup_controls(self):
        self.plotter.add_slider_widget(
            self._on_shape_a,
            [0, len(self.shape_options) - 1],
            value=self.shape_a,
            title="Shape A",
            pointa=(0.02, 0.1),
            pointb=(0.32, 0.1),
        )
        self.plotter.add_slider_widget(
            self._on_shape_b,
            [0, len(self.shape_options) - 1],
            value=self.shape_b,
            title="Shape B",
            pointa=(0.35, 0.1),
            pointb=(0.65, 0.1),
        )
        self.plotter.add_slider_widget(
            self._on_operation,
            [0, len(self.op_options) - 1],
            value=self.operation,
            title="Operation",
            pointa=(0.68, 0.1),
            pointb=(0.98, 0.1),
        )
        self.plotter.add_slider_widget(
            self._on_distance,
            [0.2, 3.5],
            value=self.distance,
            title="Distance",
            pointa=(0.02, 0.04),
            pointb=(0.48, 0.04),
        )
        self.plotter.add_slider_widget(
            self._on_shear,
            [-2.0, 2.0],
            value=self.shear,
            title="Shear",
            pointa=(0.52, 0.04),
            pointb=(0.98, 0.04),
        )
        self.plotter.add_checkbox_button_widget(
            self._on_wireframe,
            value=self.wireframe,
            position=(10, 10),
            size=30,
        )
        self.plotter.add_text(
            "Keys: R reset, N random",
            position=(60, 14),
            font_size=10,
            name="controls_hint",
        )
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

    def _add_mesh(self, key, mesh, color, opacity):
        if self.actors[key]:
            self.plotter.remove_actor(self.actors[key])
            self.actors[key] = None
        if mesh is None:
            return
        if self.wireframe:
            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                opacity=1.0,
                style="wireframe",
                line_width=1,
            )
        else:
            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                smooth_shading=True,
            )
        self.actors[key] = actor

    def _update_scene(self):
        pos_a = np.array([-0.6, 0.0, 0.0], dtype=float)
        pos_b = pos_a + np.array([self.distance, self.shear, 0.0])

        shape_a = self._make_shape(self.shape_options[self.shape_a]).translate(tuple(pos_a))
        shape_b = self._make_shape(self.shape_options[self.shape_b]).translate(tuple(pos_b))
        result = self._apply_boolean(shape_a, shape_b)

        mesh_a = self._polydata_from_manifold(shape_a)
        mesh_b = self._polydata_from_manifold(shape_b)
        mesh_r = self._polydata_from_manifold(result)

        self._add_mesh("A", mesh_a, self.colors["A"], 0.35)
        self._add_mesh("B", mesh_b, self.colors["B"], 0.35)
        self._add_mesh("Result", mesh_r, self.colors["Result"], 0.85)

        bounds = self._combine_bounds([mesh_a, mesh_b, mesh_r])
        self.plotter.reset_camera(bounds=bounds)

    def _combine_bounds(self, meshes):
        valid = [mesh.bounds for mesh in meshes if mesh is not None]
        if not valid:
            return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        mins = np.min(np.array([b[::2] for b in valid]), axis=0)
        maxs = np.max(np.array([b[1::2] for b in valid]), axis=0)
        return (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])

    def _update_hud(self):
        op_label = self.op_options[self.operation]
        mode = "Wireframe" if self.wireframe else "Solid"
        text = (
            "Mesh Booleans (PyVista)\n"
            f"Shape A: {self.shape_options[self.shape_a]}\n"
            f"Shape B: {self.shape_options[self.shape_b]}\n"
            f"Op: {op_label}\n"
            f"Distance: {self.distance:.2f}  Shear: {self.shear:.2f}\n"
            f"Mode: {mode}\n\n"
            "Controls:\n"
            "- Sliders for shapes, operation, distance, shear\n"
            "- Checkbox toggles wireframe\n"
            "- Buttons reset or randomize"
        )
        self.hud.SetInput(text)

    def _on_shape_a(self, value):
        self.shape_a = int(round(value))
        self._update_scene()
        self._update_hud()

    def _on_shape_b(self, value):
        self.shape_b = int(round(value))
        self._update_scene()
        self._update_hud()

    def _on_operation(self, value):
        self.operation = int(round(value))
        self._update_scene()
        self._update_hud()

    def _on_distance(self, value):
        self.distance = float(value)
        self._update_scene()
        self._update_hud()

    def _on_shear(self, value):
        self.shear = float(value)
        self._update_scene()
        self._update_hud()

    def _on_wireframe(self, state):
        self.wireframe = bool(state)
        self._update_scene()
        self._update_hud()

    def _reset(self):
        self.shape_a = 0
        self.shape_b = 1
        self.operation = 0
        self.distance = 1.8
        self.shear = 0.0
        self.wireframe = False
        self._update_scene()
        self._update_hud()

    def _randomize(self):
        rng = np.random.default_rng()
        self.shape_a = int(rng.integers(0, len(self.shape_options)))
        self.shape_b = int(rng.integers(0, len(self.shape_options)))
        self.operation = int(rng.integers(0, len(self.op_options)))
        self.distance = float(rng.uniform(0.4, 3.2))
        self.shear = float(rng.uniform(-1.6, 1.6))
        self._update_scene()
        self._update_hud()

    def show(self):
        self.plotter.show()
        self.plotter.app.exec_()


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
    "- Sliders for distance and shear\n"
    "- Wireframe/solid toggle",
}


if __name__ == "__main__":
    main()
