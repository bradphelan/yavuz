"""
Tri-Dexel CNC Machining Simulation
Demonstrates the tri-dexel volumetric representation used in CNC simulation.
A sphere tool is swept along predefined zigzag toolpaths, subtracting material
from a workpiece represented as three orthogonal sets of dexel ray segments.
"""

from pathlib import Path
from urllib.parse import quote
import time

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import BackgroundPlotter


# ---------------------------------------------------------------------------
# Tri-dexel data structure
# ---------------------------------------------------------------------------

class TriDexelGrid:
    """Three orthogonal grids of dexel (depth-element) ray segments.

    Each axis stores a 2D array of segment lists.  A segment is a pair
    [lo, hi] along the ray direction.  The workpiece starts as a single
    segment per ray spanning the full block extent.
    """

    def __init__(self, bounds: tuple, resolution: int):
        """
        Parameters
        ----------
        bounds : (xmin, xmax, ymin, ymax, zmin, zmax)
        resolution : number of rays per axis per dimension
        """
        self.bounds = bounds
        self.resolution = resolution
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        # X-axis dexels: rays parallel to X, grid over (Y, Z)
        self.y_ticks_x = np.linspace(ymin, ymax, resolution)
        self.z_ticks_x = np.linspace(zmin, zmax, resolution)
        self.x_segs = [
            [[xmin, xmax]] for _ in range(resolution * resolution)
        ]

        # Y-axis dexels: rays parallel to Y, grid over (X, Z)
        self.x_ticks_y = np.linspace(xmin, xmax, resolution)
        self.z_ticks_y = np.linspace(zmin, zmax, resolution)
        self.y_segs = [
            [[ymin, ymax]] for _ in range(resolution * resolution)
        ]

        # Z-axis dexels: rays parallel to Z, grid over (X, Y)
        self.x_ticks_z = np.linspace(xmin, xmax, resolution)
        self.y_ticks_z = np.linspace(ymin, ymax, resolution)
        self.z_segs = [
            [[zmin, zmax]] for _ in range(resolution * resolution)
        ]

    def copy(self) -> "TriDexelGrid":
        """Return a deep copy for undo support."""
        import copy
        return copy.deepcopy(self)

    # ----- sphere subtraction -----

    def subtract_sphere(self, cx: float, cy: float, cz: float, r: float):
        """Subtract a sphere centred at (cx, cy, cz) with radius r."""
        r2 = r * r
        self._subtract_sphere_axis(
            self.x_segs, self.y_ticks_x, self.z_ticks_x,
            cx, cy, cz, r, r2, axis="x",
        )
        self._subtract_sphere_axis(
            self.y_segs, self.x_ticks_y, self.z_ticks_y,
            cx, cy, cz, r, r2, axis="y",
        )
        self._subtract_sphere_axis(
            self.z_segs, self.x_ticks_z, self.y_ticks_z,
            cx, cy, cz, r, r2, axis="z",
        )

    @staticmethod
    def _subtract_sphere_axis(
        segs_list, ticks_u, ticks_v,
        cx, cy, cz, r, r2, axis,
    ):
        """Subtract sphere from one axis of dexels."""
        nu = len(ticks_u)
        nv = len(ticks_v)
        for iv in range(nv):
            v = ticks_v[iv]
            for iu in range(nu):
                u = ticks_u[iu]
                # Compute distance² from ray to sphere centre in the
                # two transverse coordinates
                if axis == "x":
                    d2 = (u - cy) ** 2 + (v - cz) ** 2
                    c_along = cx
                elif axis == "y":
                    d2 = (u - cx) ** 2 + (v - cz) ** 2
                    c_along = cy
                else:
                    d2 = (u - cx) ** 2 + (v - cy) ** 2
                    c_along = cz

                if d2 >= r2:
                    continue  # ray misses sphere

                half = np.sqrt(r2 - d2)
                s_lo = c_along - half
                s_hi = c_along + half

                idx = iv * nu + iu
                segs_list[idx] = _subtract_interval(segs_list[idx], s_lo, s_hi)

    # ----- query helpers -----

    def collect_dexel_lines(self, axis: str):
        """Return (N, 2, 3) array of line-segment endpoints for one axis."""
        pts = []
        if axis == "x":
            segs, ticks_u, ticks_v = self.x_segs, self.y_ticks_x, self.z_ticks_x
        elif axis == "y":
            segs, ticks_u, ticks_v = self.y_segs, self.x_ticks_y, self.z_ticks_y
        else:
            segs, ticks_u, ticks_v = self.z_segs, self.x_ticks_z, self.y_ticks_z

        nu = len(ticks_u)
        for iv, v in enumerate(ticks_v):
            for iu, u in enumerate(ticks_u):
                for seg in segs[iv * nu + iu]:
                    if axis == "x":
                        pts.append([[seg[0], u, v], [seg[1], u, v]])
                    elif axis == "y":
                        pts.append([[u, seg[0], v], [u, seg[1], v]])
                    else:
                        pts.append([[u, v, seg[0]], [u, v, seg[1]]])
        if not pts:
            return np.empty((0, 2, 3))
        return np.array(pts)

    def to_scalar_field(self, grid_res: int = 60):
        """Convert dexels to a scalar field for marching-cubes surface extraction.

        Returns (ImageData, scalar_name).  Uses a uniform grid (ImageData)
        so that VTK marching cubes can operate on it directly.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        pad = 0.01

        # Build uniform grid
        img = pv.ImageData(
            dimensions=(grid_res, grid_res, grid_res),
            spacing=(
                (xmax - xmin + 2 * pad) / (grid_res - 1),
                (ymax - ymin + 2 * pad) / (grid_res - 1),
                (zmax - zmin + 2 * pad) / (grid_res - 1),
            ),
            origin=(xmin - pad, ymin - pad, zmin - pad),
        )

        xs = np.linspace(xmin - pad, xmax + pad, grid_res)
        ys = np.linspace(ymin - pad, ymax + pad, grid_res)
        zs = np.linspace(zmin - pad, zmax + pad, grid_res)

        # Build an occupancy field: +1 inside, -1 outside
        # Check the Z-axis dexels to determine occupancy.
        field = -np.ones((grid_res, grid_res, grid_res), dtype=np.float32)

        x_ticks = self.x_ticks_z
        y_ticks = self.y_ticks_z
        nu = len(x_ticks)

        for iz, zv in enumerate(zs):
            for iy, yv in enumerate(ys):
                iy_near = int(np.clip(
                    np.searchsorted(y_ticks, yv) - 1, 0, len(y_ticks) - 2
                ))
                for ix, xv in enumerate(xs):
                    ix_near = int(np.clip(
                        np.searchsorted(x_ticks, xv) - 1, 0, len(x_ticks) - 2
                    ))
                    seg_list = self.z_segs[iy_near * nu + ix_near]
                    for seg in seg_list:
                        if seg[0] <= zv <= seg[1]:
                            field[ix, iy, iz] = 1.0
                            break

        # ImageData expects Fortran ordering matching (x, y, z) dimensions
        img["occupancy"] = field.ravel(order="F")
        return img, "occupancy"


# ---------------------------------------------------------------------------
# Interval arithmetic helper
# ---------------------------------------------------------------------------

def _subtract_interval(segments: list, lo: float, hi: float) -> list:
    """Subtract interval [lo, hi] from a sorted list of segments."""
    result = []
    for seg in segments:
        s0, s1 = seg[0], seg[1]
        if s1 <= lo or s0 >= hi:
            # no overlap
            result.append([s0, s1])
        else:
            if s0 < lo:
                result.append([s0, lo])
            if s1 > hi:
                result.append([hi, s1])
    return result


# ---------------------------------------------------------------------------
# Toolpath generators
# ---------------------------------------------------------------------------

def zigzag_path(
    bounds: tuple, z_depth: float, step_over: float, n_points_per_pass: int = 40,
) -> np.ndarray:
    """Generate a zigzag raster path in the XY plane at constant Z."""
    xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
    margin = step_over * 0.5
    y_positions = np.arange(ymin + margin, ymax - margin + 1e-9, step_over)
    pts = []
    for i, y in enumerate(y_positions):
        if i % 2 == 0:
            xs = np.linspace(xmin + margin, xmax - margin, n_points_per_pass)
        else:
            xs = np.linspace(xmax - margin, xmin + margin, n_points_per_pass)
        for x in xs:
            pts.append([x, y, z_depth])
    return np.array(pts) if pts else np.empty((0, 3))


def spiral_path(
    bounds: tuple, z_depth: float, n_turns: int = 5, n_points: int = 300,
) -> np.ndarray:
    """Generate a spiral path from centre outward at constant Z."""
    xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    rx = (xmax - xmin) / 2 * 0.85
    ry = (ymax - ymin) / 2 * 0.85
    t = np.linspace(0, 2 * np.pi * n_turns, n_points)
    r_frac = np.linspace(0.05, 1.0, n_points)
    x = cx + rx * r_frac * np.cos(t)
    y = cy + ry * r_frac * np.sin(t)
    z = np.full(n_points, z_depth)
    return np.column_stack([x, y, z])


def contour_path(
    bounds: tuple, z_depth: float, n_offsets: int = 6, n_points_per_loop: int = 60,
) -> np.ndarray:
    """Generate concentric rectangular contour offsets at constant Z."""
    xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    hw = (xmax - xmin) / 2
    hh = (ymax - ymin) / 2
    pts = []
    for i in range(n_offsets):
        frac = 1.0 - i / max(n_offsets, 1)
        w = hw * frac * 0.85
        h = hh * frac * 0.85
        if w < 0.05 or h < 0.05:
            break
        corners = [
            [cx - w, cy - h], [cx + w, cy - h],
            [cx + w, cy + h], [cx - w, cy + h],
            [cx - w, cy - h],  # close
        ]
        for j in range(len(corners) - 1):
            n = n_points_per_loop // 4
            for t in np.linspace(0, 1, n, endpoint=False):
                x = corners[j][0] + t * (corners[j + 1][0] - corners[j][0])
                y = corners[j][1] + t * (corners[j + 1][1] - corners[j][1])
                pts.append([x, y, z_depth])
    return np.array(pts) if pts else np.empty((0, 3))


def cross_hatch_path(
    bounds: tuple, z_depth: float, step_over: float, n_points_per_pass: int = 40,
) -> np.ndarray:
    """Zigzag in X then zigzag in Y (cross-hatch)."""
    p1 = zigzag_path(bounds, z_depth, step_over, n_points_per_pass)
    # Rotate 90°: swap X/Y sweep direction
    xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
    x_positions = np.arange(xmin + step_over * 0.5, xmax - step_over * 0.5 + 1e-9, step_over)
    pts = []
    for i, x in enumerate(x_positions):
        if i % 2 == 0:
            ys = np.linspace(ymin + step_over * 0.5, ymax - step_over * 0.5, n_points_per_pass)
        else:
            ys = np.linspace(ymax - step_over * 0.5, ymin + step_over * 0.5, n_points_per_pass)
        for y in ys:
            pts.append([x, y, z_depth])
    p2 = np.array(pts) if pts else np.empty((0, 3))
    if p1.size and p2.size:
        return np.vstack([p1, p2])
    return p1 if p1.size else p2


# ---------------------------------------------------------------------------
# Demo class
# ---------------------------------------------------------------------------

WORKPIECE_BOUNDS = (-5.0, 5.0, -5.0, 5.0, -3.0, 3.0)
DEFAULT_RESOLUTION = 30
DEFAULT_TOOL_RADIUS = 1.0
DEFAULT_Z_DEPTH = 1.5  # Z from top surface going down


class TriDexelDemo:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1400, 900),
            title="Tri-Dexel CNC Simulation",
        )
        self.plotter.set_background("white")

        # Lighting
        light1 = pv.Light(position=(5, 5, 8), light_type="cameralight")
        light2 = pv.Light(position=(-5, -5, 8), light_type="cameralight", intensity=0.5)
        light3 = pv.Light(position=(0, 5, -5), light_type="cameralight", intensity=0.3)
        self.plotter.add_light(light1)
        self.plotter.add_light(light2)
        self.plotter.add_light(light3)

        # State
        self.resolution = DEFAULT_RESOLUTION
        self.tool_radius = DEFAULT_TOOL_RADIUS
        self.z_depth = DEFAULT_Z_DEPTH
        self.step_over = 2.0
        self.sweep_progress = 0.0
        self.show_dexels_x = False
        self.show_dexels_y = False
        self.show_dexels_z = False
        self.show_tool = True
        self.show_path = True

        # Toolpath
        self.path_names = ["Zigzag", "Spiral", "Contour", "Cross-hatch"]
        self.current_path_idx = 0
        self.toolpath = np.empty((0, 3))

        # Tri-dexel grid
        self.grid = TriDexelGrid(WORKPIECE_BOUNDS, self.resolution)
        self.undo_stack: list = []

        # Mesh/actor storage (created once, updated in place)
        self.workpiece_mesh = None
        self.workpiece_actor = None
        self.tool_mesh = None
        self.tool_actor = None
        self.path_mesh = None
        self.path_actor = None
        self.dexel_meshes = {"x": None, "y": None, "z": None}
        self.dexel_actors = {"x": None, "y": None, "z": None}

        # Timing label
        self.last_cut_ms = 0.0

        # Build
        self._generate_toolpath()
        self._setup_controls()
        self._build_scene()
        self._update_workpiece_mesh()
        self._update_tool_position()
        self._update_path_mesh()
        self.plotter.reset_camera()

    # ---- toolpath generation ----

    def _generate_toolpath(self):
        b = WORKPIECE_BOUNDS
        name = self.path_names[self.current_path_idx]
        z = b[5] - self.z_depth  # from top surface down
        if name == "Zigzag":
            self.toolpath = zigzag_path(b, z, self.step_over)
        elif name == "Spiral":
            self.toolpath = spiral_path(b, z)
        elif name == "Contour":
            self.toolpath = contour_path(b, z)
        elif name == "Cross-hatch":
            self.toolpath = cross_hatch_path(b, z, self.step_over)

    # ---- controls ----

    def _setup_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self.plotter)
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Toolpath selector
        layout.addWidget(QtWidgets.QLabel("Toolpath"))
        self.path_combo = QtWidgets.QComboBox()
        self.path_combo.addItems(self.path_names)
        self.path_combo.setCurrentIndex(self.current_path_idx)
        self.path_combo.currentIndexChanged.connect(self._on_path_changed)
        layout.addWidget(self.path_combo)

        # Sweep progress
        layout.addWidget(QtWidgets.QLabel("Sweep Progress"))
        self.progress_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(1000)
        self.progress_slider.setValue(0)
        self.progress_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.progress_slider.setTickInterval(100)
        self.progress_slider.valueChanged.connect(self._on_progress_changed)
        layout.addWidget(self.progress_slider)

        # Tool radius
        layout.addWidget(QtWidgets.QLabel("Tool Radius"))
        self.radius_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.radius_slider.setMinimum(2)
        self.radius_slider.setMaximum(30)
        self.radius_slider.setValue(int(self.tool_radius * 10))
        self.radius_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.radius_slider.setTickInterval(5)
        self.radius_slider.valueChanged.connect(self._on_radius_changed)
        layout.addWidget(self.radius_slider)

        # Depth of cut
        layout.addWidget(QtWidgets.QLabel("Depth of Cut"))
        self.depth_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.depth_slider.setMinimum(1)
        self.depth_slider.setMaximum(50)
        self.depth_slider.setValue(int(self.z_depth * 10))
        self.depth_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.depth_slider.setTickInterval(5)
        self.depth_slider.valueChanged.connect(self._on_depth_changed)
        layout.addWidget(self.depth_slider)

        # Step-over
        layout.addWidget(QtWidgets.QLabel("Step Over"))
        self.stepover_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.stepover_slider.setMinimum(5)
        self.stepover_slider.setMaximum(40)
        self.stepover_slider.setValue(int(self.step_over * 10))
        self.stepover_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.stepover_slider.setTickInterval(5)
        self.stepover_slider.valueChanged.connect(self._on_stepover_changed)
        layout.addWidget(self.stepover_slider)

        # Separator
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        layout.addWidget(sep)

        # Cut button
        self.cut_btn = QtWidgets.QPushButton("Run Cut")
        self.cut_btn.clicked.connect(self._on_cut)
        layout.addWidget(self.cut_btn)

        # Reset button
        self.reset_btn = QtWidgets.QPushButton("Reset Workpiece")
        self.reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(self.reset_btn)

        # Separator
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        layout.addWidget(sep2)

        # Dexel visualisation toggles
        layout.addWidget(QtWidgets.QLabel("Show Dexels"))
        self.check_dx = QtWidgets.QCheckBox("X-axis (red)")
        self.check_dx.stateChanged.connect(lambda s: self._on_dexel_toggle("x", s))
        layout.addWidget(self.check_dx)

        self.check_dy = QtWidgets.QCheckBox("Y-axis (green)")
        self.check_dy.stateChanged.connect(lambda s: self._on_dexel_toggle("y", s))
        layout.addWidget(self.check_dy)

        self.check_dz = QtWidgets.QCheckBox("Z-axis (blue)")
        self.check_dz.stateChanged.connect(lambda s: self._on_dexel_toggle("z", s))
        layout.addWidget(self.check_dz)

        # Show tool / path
        self.check_tool = QtWidgets.QCheckBox("Show Tool")
        self.check_tool.setChecked(True)
        self.check_tool.stateChanged.connect(self._on_tool_toggle)
        layout.addWidget(self.check_tool)

        self.check_path = QtWidgets.QCheckBox("Show Toolpath")
        self.check_path.setChecked(True)
        self.check_path.stateChanged.connect(self._on_path_toggle)
        layout.addWidget(self.check_path)

        # Status
        self.status_label = QtWidgets.QLabel("Ready")
        layout.addWidget(self.status_label)

        layout.addStretch(1)

        dock.setWidget(panel)
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    # ---- scene ----

    def _build_scene(self):
        # Workpiece — start with a placeholder box
        box = pv.Box(bounds=WORKPIECE_BOUNDS)
        self.workpiece_mesh = box.extract_surface(algorithm=None)
        self.workpiece_actor = self.plotter.add_mesh(
            self.workpiece_mesh,
            color="#b8d4e3",  # pastel blue
            smooth_shading=True,
            opacity=1.0,
        )

        # Tool sphere
        pos = self.toolpath[0] if len(self.toolpath) > 0 else [0, 0, 0]
        self.tool_mesh = pv.Sphere(
            radius=self.tool_radius, center=pos,
            theta_resolution=24, phi_resolution=24,
        )
        self.tool_actor = self.plotter.add_mesh(
            self.tool_mesh, color="#e88d8d", opacity=0.6,
            smooth_shading=True,
        )

        # Toolpath line
        if len(self.toolpath) >= 2:
            self.path_mesh = pv.lines_from_points(self.toolpath)
        else:
            self.path_mesh = pv.PolyData()
        self.path_actor = self.plotter.add_mesh(
            self.path_mesh, color="#666666", line_width=1, opacity=0.5,
        )

    # ---- mesh updates ----

    def _update_workpiece_mesh(self):
        """Reconstruct surface from the dexel grid and update the actor."""
        scalar_grid, name = self.grid.to_scalar_field(grid_res=60)
        contour = scalar_grid.contour(
            isosurfaces=[0.0], scalars=name, method="marching_cubes",
        )
        if contour.n_points == 0:
            self.workpiece_actor.SetVisibility(False)
            return
        self.workpiece_mesh.copy_from(contour)
        self.workpiece_mesh.Modified()
        self.workpiece_actor.mapper.SetInputData(self.workpiece_mesh)
        self.workpiece_actor.mapper.Update()
        self.workpiece_actor.SetVisibility(True)

    def _update_tool_position(self):
        """Move the tool sphere to the current sweep position."""
        if len(self.toolpath) == 0:
            return
        idx = int(self.sweep_progress * (len(self.toolpath) - 1))
        idx = min(idx, len(self.toolpath) - 1)
        pos = self.toolpath[idx]
        new_sphere = pv.Sphere(
            radius=self.tool_radius, center=pos,
            theta_resolution=24, phi_resolution=24,
        )
        self.tool_mesh.copy_from(new_sphere)
        self.tool_mesh.Modified()
        self.tool_actor.mapper.SetInputData(self.tool_mesh)
        self.tool_actor.mapper.Update()
        self.tool_actor.SetVisibility(self.show_tool)
        self.plotter.render()

    def _update_path_mesh(self):
        """Update the toolpath polyline."""
        if len(self.toolpath) >= 2:
            new_line = pv.lines_from_points(self.toolpath)
            self.path_mesh.copy_from(new_line)
            self.path_mesh.Modified()
            self.path_actor.mapper.SetInputData(self.path_mesh)
            self.path_actor.mapper.Update()
            self.path_actor.SetVisibility(self.show_path)
        else:
            self.path_actor.SetVisibility(False)
        self.plotter.render()

    def _update_dexel_lines(self, axis: str):
        """Update dexel line visualisation for one axis."""
        show = {
            "x": self.show_dexels_x,
            "y": self.show_dexels_y,
            "z": self.show_dexels_z,
        }[axis]
        colors = {"x": "#cc4444", "y": "#44aa44", "z": "#4444cc"}

        if not show:
            if self.dexel_actors[axis] is not None:
                self.dexel_actors[axis].SetVisibility(False)
            return

        segs = self.grid.collect_dexel_lines(axis)
        if segs.shape[0] == 0:
            if self.dexel_actors[axis] is not None:
                self.dexel_actors[axis].SetVisibility(False)
            return

        # Build line mesh from segment pairs
        n = segs.shape[0]
        points = segs.reshape(-1, 3)  # (2N, 3)
        lines = np.column_stack([
            np.full(n, 2, dtype=np.int64),
            np.arange(0, 2 * n, 2, dtype=np.int64),
            np.arange(1, 2 * n, 2, dtype=np.int64),
        ]).ravel()
        new_mesh = pv.PolyData(points, lines=lines)

        if self.dexel_actors[axis] is None:
            self.dexel_meshes[axis] = new_mesh
            self.dexel_actors[axis] = self.plotter.add_mesh(
                self.dexel_meshes[axis],
                color=colors[axis], line_width=1, opacity=0.4,
            )
        else:
            self.dexel_meshes[axis].copy_from(new_mesh)
            self.dexel_meshes[axis].Modified()
            self.dexel_actors[axis].mapper.SetInputData(self.dexel_meshes[axis])
            self.dexel_actors[axis].mapper.Update()
            self.dexel_actors[axis].SetVisibility(True)

    # ---- cut operation ----

    def _perform_cut(self):
        """Sweep the sphere along the toolpath up to the current progress and subtract."""
        if len(self.toolpath) == 0:
            return

        t0 = time.perf_counter()

        max_idx = int(self.sweep_progress * (len(self.toolpath) - 1))
        max_idx = min(max_idx, len(self.toolpath) - 1)

        # Sample positions along the path (every few points for performance)
        step = max(1, len(self.toolpath) // 200)
        indices = list(range(0, max_idx + 1, step))
        if max_idx not in indices:
            indices.append(max_idx)

        for i in indices:
            pos = self.toolpath[i]
            self.grid.subtract_sphere(pos[0], pos[1], pos[2], self.tool_radius)

        elapsed = (time.perf_counter() - t0) * 1000
        self.last_cut_ms = elapsed
        self.status_label.setText(
            f"Cut complete — {len(indices)} positions, {elapsed:.1f} ms"
        )

        self._update_workpiece_mesh()
        self._refresh_visible_dexels()
        self.plotter.render()

    def _refresh_visible_dexels(self):
        for axis in ("x", "y", "z"):
            show = {"x": self.show_dexels_x, "y": self.show_dexels_y, "z": self.show_dexels_z}[axis]
            if show:
                self._update_dexel_lines(axis)

    # ---- callbacks ----

    def _on_path_changed(self, idx: int):
        self.current_path_idx = idx
        self._generate_toolpath()
        self._update_path_mesh()
        self._update_tool_position()

    def _on_progress_changed(self, value: int):
        self.sweep_progress = value / 1000.0
        self._update_tool_position()

    def _on_radius_changed(self, value: int):
        self.tool_radius = value / 10.0
        self._update_tool_position()

    def _on_depth_changed(self, value: int):
        self.z_depth = value / 10.0
        self._generate_toolpath()
        self._update_path_mesh()
        self._update_tool_position()

    def _on_stepover_changed(self, value: int):
        self.step_over = value / 10.0
        self._generate_toolpath()
        self._update_path_mesh()
        self._update_tool_position()

    def _on_cut(self):
        self._perform_cut()

    def _on_reset(self):
        self.grid = TriDexelGrid(WORKPIECE_BOUNDS, self.resolution)
        self._update_workpiece_mesh()
        self._refresh_visible_dexels()
        self.status_label.setText("Workpiece reset")
        self.plotter.render()

    def _on_dexel_toggle(self, axis: str, state: int):
        val = bool(state)
        if axis == "x":
            self.show_dexels_x = val
        elif axis == "y":
            self.show_dexels_y = val
        else:
            self.show_dexels_z = val
        self._update_dexel_lines(axis)
        self.plotter.render()

    def _on_tool_toggle(self, state: int):
        self.show_tool = bool(state)
        self.tool_actor.SetVisibility(self.show_tool)
        self.plotter.render()

    def _on_path_toggle(self, state: int):
        self.show_path = bool(state)
        self.path_actor.SetVisibility(self.show_path)
        self.plotter.render()

    def show(self):
        self.plotter.show()
        self.plotter.app.exec()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main():
    demo = TriDexelDemo()
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
    "title": "Tri-Dexel CNC Simulation",
    "description": "Interactive tri-dexel volumetric CNC machining simulation.\n\n"
    "Features:\n"
    "- Tri-dexel workpiece representation (X/Y/Z ray segments)\n"
    "- Sphere tool with Boolean subtraction\n"
    "- Zigzag, spiral, contour, and cross-hatch toolpaths\n"
    "- Dexel ray visualisation per axis\n"
    "- Surface reconstruction via marching cubes\n"
    "- Real-time performance readout",
}


if __name__ == "__main__":
    main()
