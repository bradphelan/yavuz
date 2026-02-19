"""
Tri-Dexel CNC Machining Simulation
Demonstrates the tri-dexel volumetric representation used in CNC simulation.
A sphere tool is swept along predefined toolpaths, subtracting material from a
workpiece represented as three orthogonal sets of dexel ray segments.

Surface reconstruction uses pyvista contour_labels (vtkSurfaceNets3D) so that
ALL faces — top, walls, and cut surfaces — are generated correctly with two
distinct colours: uncut stock and freshly machined surface.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import BackgroundPlotter

pv.global_theme.allow_empty_mesh = True


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
        self.bounds = bounds
        self.resolution = resolution
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        # X-axis dexels: rays parallel to X, grid over (Y, Z)
        self.y_ticks_x = np.linspace(ymin, ymax, resolution)
        self.z_ticks_x = np.linspace(zmin, zmax, resolution)
        self.x_segs = [[[xmin, xmax]] for _ in range(resolution * resolution)]

        # Y-axis dexels: rays parallel to Y, grid over (X, Z)
        self.x_ticks_y = np.linspace(xmin, xmax, resolution)
        self.z_ticks_y = np.linspace(zmin, zmax, resolution)
        self.y_segs = [[[ymin, ymax]] for _ in range(resolution * resolution)]

        # Z-axis dexels: rays parallel to Z, grid over (X, Y)
        self.x_ticks_z = np.linspace(xmin, xmax, resolution)
        self.y_ticks_z = np.linspace(ymin, ymax, resolution)
        self.z_segs = [[[zmin, zmax]] for _ in range(resolution * resolution)]

    def copy(self) -> TriDexelGrid:
        return copy.deepcopy(self)

    # ----- sphere subtraction -----

    def subtract_sphere(self, cx: float, cy: float, cz: float, r: float):
        r2 = r * r
        self._subtract_sphere_axis(
            self.x_segs, self.y_ticks_x, self.z_ticks_x, cx, cy, cz, r, r2, "x",
        )
        self._subtract_sphere_axis(
            self.y_segs, self.x_ticks_y, self.z_ticks_y, cx, cy, cz, r, r2, "y",
        )
        self._subtract_sphere_axis(
            self.z_segs, self.x_ticks_z, self.y_ticks_z, cx, cy, cz, r, r2, "z",
        )

    @staticmethod
    def _subtract_sphere_axis(segs_list, ticks_u, ticks_v, cx, cy, cz, r, r2, axis):
        nu = len(ticks_u)
        # Vectorised distance² computation for all (u, v) pairs
        if axis == "x":
            du = ticks_u - cy           # shape (nu,)
            dv = ticks_v - cz           # shape (nv,)
            c_along = cx
        elif axis == "y":
            du = ticks_u - cx
            dv = ticks_v - cz
            c_along = cy
        else:
            du = ticks_u - cx
            dv = ticks_v - cy
            c_along = cz

        # d2[iv, iu] = du[iu]² + dv[iv]²
        du2 = du * du                   # (nu,)
        dv2 = dv * dv                   # (nv,)
        d2 = dv2[:, np.newaxis] + du2[np.newaxis, :]   # (nv, nu)

        # Find rays that intersect the sphere
        hit_iv, hit_iu = np.where(d2 < r2)
        if len(hit_iv) == 0:
            return

        half = np.sqrt(r2 - d2[hit_iv, hit_iu])
        lo_arr = c_along - half
        hi_arr = c_along + half

        for k in range(len(hit_iv)):
            idx = int(hit_iv[k]) * nu + int(hit_iu[k])
            segs_list[idx] = _subtract_interval(
                segs_list[idx], lo_arr[k], hi_arr[k],
            )

    # ----- dexel line query -----

    def collect_dexel_lines(self, axis: str):
        if axis == "x":
            segs, ticks_u, ticks_v = self.x_segs, self.y_ticks_x, self.z_ticks_x
        elif axis == "y":
            segs, ticks_u, ticks_v = self.y_segs, self.x_ticks_y, self.z_ticks_y
        else:
            segs, ticks_u, ticks_v = self.z_segs, self.x_ticks_z, self.y_ticks_z
        pts = []
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
        return np.array(pts) if pts else np.empty((0, 2, 3))

    # ----- label field for contour_labels -----

    def to_label_field(self, grid_res: int = 60) -> tuple[pv.ImageData, str]:
        """Build an integer label ImageData for use with contour_labels.

        Labels:
          0 = air / background
          1 = uncut solid stock
          2 = cut / machined surface (exposed by tool)

        We use Z-dexels to determine which voxels are solid.
        A voxel is labeled 2 (cut-exposed) if it is solid AND any of its
        six face-neighbours is air — i.e. it is on the machined surface.
        Internal solid voxels that were never touched get label 1.

        contour_labels will then generate:
          boundary_labels [1, 0] → uncut outer face  → colour A
          boundary_labels [2, 0] → machined face      → colour B
          boundary_labels [1, 2] → interior interface (ignored / hidden)
        """
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        pad = 0.02

        xs = np.linspace(xmin - pad, xmax + pad, grid_res)
        ys = np.linspace(ymin - pad, ymax + pad, grid_res)
        zs = np.linspace(zmin - pad, zmax + pad, grid_res)

        img = pv.ImageData(
            dimensions=(grid_res, grid_res, grid_res),
            spacing=(xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]),
            origin=(xs[0], ys[0], zs[0]),
        )

        # --- occupancy pass (Z-dexels) ---
        x_ticks = self.x_ticks_z
        y_ticks = self.y_ticks_z
        nu = len(x_ticks)
        nv = len(y_ticks)

        # Map grid positions to nearest dexel index
        xi_map = np.clip(
            np.searchsorted(x_ticks, xs) - 1, 0, nu - 2
        )
        yi_map = np.clip(
            np.searchsorted(y_ticks, ys) - 1, 0, nv - 2
        )

        # Build reverse maps: dexel index → array of grid indices
        gx_for_dexel: dict[int, np.ndarray] = {}
        for dx in range(nu):
            mask = xi_map == dx
            if mask.any():
                gx_for_dexel[dx] = np.where(mask)[0]
        gy_for_dexel: dict[int, np.ndarray] = {}
        for dy in range(nv):
            mask = yi_map == dy
            if mask.any():
                gy_for_dexel[dy] = np.where(mask)[0]

        # Iterate dexel columns (resolution²), build z-mask, broadcast to grid
        solid = np.zeros((grid_res, grid_res, grid_res), dtype=bool)
        for iy_near, gy_arr in gy_for_dexel.items():
            for ix_near, gx_arr in gx_for_dexel.items():
                seg_list = self.z_segs[iy_near * nu + ix_near]
                if not seg_list:
                    continue
                # Batch searchsorted for all segments in this column
                seg_arr = np.asarray(seg_list)          # (n_segs, 2)
                lo_iz = np.searchsorted(zs, seg_arr[:, 0])
                hi_iz = np.searchsorted(zs, seg_arr[:, 1])
                # Build z-mask for this column
                z_mask = np.zeros(grid_res, dtype=bool)
                for s in range(len(seg_arr)):
                    z_mask[lo_iz[s]:hi_iz[s]] = True
                # Broadcast to all grid cells that map to this dexel column
                solid[np.ix_(gx_arr, gy_arr)] |= z_mask

        # --- label assignment ---
        # Start: all solid = label 1 (uncut)
        labels = np.where(solid, np.int32(1), np.int32(0))

        # Mark surface voxels that are adjacent to air as label 2 (cut/exposed)
        # A solid voxel is "on the machined surface" only if it was NOT solid
        # in the original block — but we don't store original here.
        # Instead we detect the machined surface as any solid voxel where
        # neighbouring voxels in any direction are air (label 0).
        # The original block faces (min/max extents) stay label 1 (uncut).
        s = solid
        # interior neighbour check — exclude the outer shell of the bounding box
        exposed = np.zeros_like(solid)
        exposed[1:-1, 1:-1, 1:-1] = (
            s[1:-1, 1:-1, 1:-1] & (
                ~s[0:-2, 1:-1, 1:-1] |
                ~s[2:,   1:-1, 1:-1] |
                ~s[1:-1, 0:-2, 1:-1] |
                ~s[1:-1, 2:,   1:-1] |
                ~s[1:-1, 1:-1, 0:-2] |
                ~s[1:-1, 1:-1, 2:  ]
            )
        )

        # Voxels on the outer shell of the original bounding box
        # remain label 1 (they are uncut stock faces).
        # We detect "original outer shell" as voxels outside the dexel range
        # that are solid due to the pad, plus the first/last rows at the extents.
        labels[exposed] = 2

        # Restore outer bounding faces to label 1 (uncut stock walls/top/bottom)
        # These are voxels at grid boundary — the pad region ensures the block
        # edges are always solid and reachable.
        labels[0,  :, :] = np.where(solid[0,  :, :], 1, 0)
        labels[-1, :, :] = np.where(solid[-1, :, :], 1, 0)
        labels[:,  0, :] = np.where(solid[:,  0, :], 1, 0)
        labels[:, -1, :] = np.where(solid[:, -1, :], 1, 0)
        labels[:, :,  0] = np.where(solid[:, :,  0], 1, 0)
        labels[:, :, -1] = np.where(solid[:, :, -1], 1, 0)

        img["labels"] = labels.ravel(order="F").astype(np.int32)
        return img, "labels"


# ---------------------------------------------------------------------------
# Interval helper
# ---------------------------------------------------------------------------

def _subtract_interval(segments: list, lo: float, hi: float) -> list:
    result = []
    for seg in segments:
        s0, s1 = seg[0], seg[1]
        if s1 <= lo or s0 >= hi:
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

def zigzag_path(bounds, z_depth, step_over, n_points_per_pass=40):
    xmin, xmax, ymin, ymax = bounds[:4]
    margin = step_over * 0.5
    y_positions = np.arange(ymin + margin, ymax - margin + 1e-9, step_over)
    pts = []
    for i, y in enumerate(y_positions):
        xs = np.linspace(xmin + margin, xmax - margin, n_points_per_pass)
        if i % 2:
            xs = xs[::-1]
        for x in xs:
            pts.append([x, y, z_depth])
    return np.array(pts) if pts else np.empty((0, 3))


def spiral_path(bounds, z_depth, n_turns=5, n_points=300):
    xmin, xmax, ymin, ymax = bounds[:4]
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    rx, ry = (xmax - xmin) / 2 * 0.85, (ymax - ymin) / 2 * 0.85
    t = np.linspace(0, 2 * np.pi * n_turns, n_points)
    r = np.linspace(0.05, 1.0, n_points)
    return np.column_stack([cx + rx * r * np.cos(t), cy + ry * r * np.sin(t),
                             np.full(n_points, z_depth)])


def contour_path(bounds, z_depth, n_offsets=6, n_per_loop=60):
    xmin, xmax, ymin, ymax = bounds[:4]
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    hw, hh = (xmax - xmin) / 2, (ymax - ymin) / 2
    pts = []
    for i in range(n_offsets):
        frac = 1.0 - i / max(n_offsets, 1)
        w, h = hw * frac * 0.85, hh * frac * 0.85
        if w < 0.05 or h < 0.05:
            break
        corners = [
            [cx - w, cy - h], [cx + w, cy - h],
            [cx + w, cy + h], [cx - w, cy + h], [cx - w, cy - h],
        ]
        for j in range(4):
            for t in np.linspace(0, 1, n_per_loop // 4, endpoint=False):
                x = corners[j][0] + t * (corners[j+1][0] - corners[j][0])
                y = corners[j][1] + t * (corners[j+1][1] - corners[j][1])
                pts.append([x, y, z_depth])
    return np.array(pts) if pts else np.empty((0, 3))


def cross_hatch_path(bounds, z_depth, step_over, n_per_pass=40):
    p1 = zigzag_path(bounds, z_depth, step_over, n_per_pass)
    xmin, xmax, ymin, ymax = bounds[:4]
    margin = step_over * 0.5
    x_positions = np.arange(xmin + margin, xmax - margin + 1e-9, step_over)
    pts = []
    for i, x in enumerate(x_positions):
        ys = np.linspace(ymin + margin, ymax - margin, n_per_pass)
        if i % 2:
            ys = ys[::-1]
        for y in ys:
            pts.append([x, y, z_depth])
    p2 = np.array(pts) if pts else np.empty((0, 3))
    if p1.size and p2.size:
        return np.vstack([p1, p2])
    return p1 if p1.size else p2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKPIECE_BOUNDS = (-5.0, 5.0, -5.0, 5.0, -3.0, 3.0)
DEFAULT_RESOLUTION = 30
DEFAULT_TOOL_RADIUS = 1.0
DEFAULT_Z_DEPTH = 1.5
GRID_RES_FACTOR = 2    # voxel grid = dexel resolution * this factor
CACHE_STEPS = 80       # number of snapshots stored when pre-computing

# Colours
COL_UNCUT  = "#c8a882"   # warm beige — uncut stock
COL_CUT    = "#5b9bbf"   # steel blue — machined surface
COL_TOOL   = "#e88d8d"
COL_PATH   = "#888888"


# ---------------------------------------------------------------------------
# Demo class
# ---------------------------------------------------------------------------

class TriDexelDemo:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1400, 900),
            title="Tri-Dexel CNC Simulation",
        )
        self.plotter.set_background("white")

        # ---- lighting: 6 lights from all axis directions, balanced ----
        # Each at intensity 0.30 so all six sum to ~1.8 — bright but not blown out.
        # Using scene lights (not camera lights) so they stay fixed in world space
        # and illuminate inside cavities regardless of view angle.
        self.plotter.remove_all_lights()
        D = 12.0   # distance from origin
        I = 0.32   # per-light intensity
        light_positions = [
            ( D,  0,  0), (-D,  0,  0),
            ( 0,  D,  0), ( 0, -D,  0),
            ( 0,  0,  D), ( 0,  0, -D),
        ]
        for pos in light_positions:
            lt = pv.Light(
                position=pos,
                focal_point=(0, 0, 0),
                color="white",
                intensity=I,
            )
            lt.positional = True
            self.plotter.add_light(lt)

        # ---- state ----
        self.resolution  = DEFAULT_RESOLUTION
        self.tool_radius = DEFAULT_TOOL_RADIUS
        self.z_depth     = DEFAULT_Z_DEPTH
        self.step_over   = 2.0
        self.sweep_progress = 0.0
        self.show_dexels_x = False
        self.show_dexels_y = False
        self.show_dexels_z = False
        self.show_mesh  = True
        self.show_tool  = True
        self.show_path  = True

        self.path_names = ["Zigzag", "Spiral", "Contour", "Cross-hatch"]
        self.current_path_idx = 1   # default: spiral (looks nice)
        self.toolpath: np.ndarray = np.empty((0, 3))

        # Live dexel grid
        self.grid = TriDexelGrid(WORKPIECE_BOUNDS, self.resolution)

        # Pre-computed cache: list of (TriDexelGrid snapshots) at CACHE_STEPS
        # evenly-spaced positions along the toolpath.
        self._cache: list[TriDexelGrid] | None = None
        self._cache_toolpath_hash: int | None = None   # detect stale cache

        # Actors
        self.uncut_mesh:  pv.PolyData | None = None
        self.uncut_actor = None
        self.cut_mesh:    pv.PolyData | None = None
        self.cut_actor   = None
        self.tool_mesh:   pv.PolyData | None = None
        self.tool_actor  = None
        self.path_mesh:   pv.PolyData | None = None
        self.path_actor  = None
        self.dexel_meshes = {"x": None, "y": None, "z": None}
        self.dexel_actors = {"x": None, "y": None, "z": None}

        self._generate_toolpath()
        self._setup_controls()
        self._build_scene()
        self._rebuild_workpiece()
        self._update_tool_position()
        self._update_path_mesh()
        self.plotter.reset_camera()

    # -----------------------------------------------------------------------
    # Toolpath
    # -----------------------------------------------------------------------

    def _generate_toolpath(self):
        b = WORKPIECE_BOUNDS
        z = b[5] - self.z_depth
        name = self.path_names[self.current_path_idx]
        if name == "Zigzag":
            self.toolpath = zigzag_path(b, z, self.step_over)
        elif name == "Spiral":
            self.toolpath = spiral_path(b, z)
        elif name == "Contour":
            self.toolpath = contour_path(b, z)
        elif name == "Cross-hatch":
            self.toolpath = cross_hatch_path(b, z, self.step_over)
        self._cache = None   # invalidate cache on new toolpath

    # -----------------------------------------------------------------------
    # Controls
    # -----------------------------------------------------------------------

    def _setup_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self.plotter)
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Toolpath
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

        # Dexel density (1x–5x base resolution)
        self.density_label = QtWidgets.QLabel(
            f"Dexel Density  (1x = {DEFAULT_RESOLUTION})"
        )
        layout.addWidget(self.density_label)
        self.density_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.density_slider.setMinimum(10)   # 1.0x
        self.density_slider.setMaximum(50)   # 5.0x
        self.density_slider.setValue(10)
        self.density_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.density_slider.setTickInterval(10)
        self.density_slider.valueChanged.connect(self._on_density_changed)
        layout.addWidget(self.density_slider)

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

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        layout.addWidget(sep)

        # Run cut (cut up to current progress on live grid)
        self.cut_btn = QtWidgets.QPushButton("Run Cut")
        self.cut_btn.setToolTip("Cut to current sweep progress on the live grid")
        self.cut_btn.clicked.connect(self._on_cut)
        layout.addWidget(self.cut_btn)

        # Pre-compute all intermediate states for instant slider scrubbing
        self.precalc_btn = QtWidgets.QPushButton("Pre-compute All States")
        self.precalc_btn.setToolTip(
            "Pre-calculate and cache all intermediate cut states so the\n"
            "Sweep Progress slider plays back instantly."
        )
        self.precalc_btn.clicked.connect(self._on_precalculate)
        layout.addWidget(self.precalc_btn)

        # Reset
        self.reset_btn = QtWidgets.QPushButton("Reset Workpiece")
        self.reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(self.reset_btn)

        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        layout.addWidget(sep2)

        # Mesh toggle
        self.check_mesh = QtWidgets.QCheckBox("Show Mesh")
        self.check_mesh.setChecked(True)
        self.check_mesh.stateChanged.connect(self._on_mesh_toggle)
        layout.addWidget(self.check_mesh)

        # Dexel toggles
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

        self.check_tool = QtWidgets.QCheckBox("Show Tool")
        self.check_tool.setChecked(True)
        self.check_tool.stateChanged.connect(self._on_tool_toggle)
        layout.addWidget(self.check_tool)

        self.check_path = QtWidgets.QCheckBox("Show Toolpath")
        self.check_path.setChecked(True)
        self.check_path.stateChanged.connect(self._on_path_toggle)
        layout.addWidget(self.check_path)

        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch(1)
        dock.setWidget(panel)
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    # -----------------------------------------------------------------------
    # Scene setup
    # -----------------------------------------------------------------------

    def _build_scene(self):
        # Two workpiece actors: uncut faces and cut/machined faces
        placeholder = pv.PolyData()
        self.uncut_mesh = placeholder.copy()
        self.uncut_actor = self.plotter.add_mesh(
            self.uncut_mesh,
            color=COL_UNCUT,
            smooth_shading=True,
            opacity=1.0,
            specular=0.3,
            specular_power=20,
        )
        self.cut_mesh = placeholder.copy()
        self.cut_actor = self.plotter.add_mesh(
            self.cut_mesh,
            color=COL_CUT,
            smooth_shading=True,
            opacity=1.0,
            specular=0.6,
            specular_power=60,
        )

        # Tool
        pos = self.toolpath[0] if len(self.toolpath) else [0, 0, 0]
        self.tool_mesh = pv.Sphere(radius=self.tool_radius, center=pos,
                                   theta_resolution=24, phi_resolution=24)
        self.tool_actor = self.plotter.add_mesh(
            self.tool_mesh, color=COL_TOOL, opacity=0.65,
            smooth_shading=True, specular=0.5, specular_power=40,
        )

        # Toolpath line
        self.path_mesh = (pv.lines_from_points(self.toolpath)
                          if len(self.toolpath) >= 2 else pv.PolyData())
        self.path_actor = self.plotter.add_mesh(
            self.path_mesh, color=COL_PATH, line_width=1, opacity=0.5,
        )

    # -----------------------------------------------------------------------
    # Workpiece surface reconstruction
    # -----------------------------------------------------------------------

    def _rebuild_workpiece(self, grid: TriDexelGrid | None = None):
        """Reconstruct the surface from *grid* (defaults to self.grid).

        Uses contour_labels / vtkSurfaceNets3D so walls and flat surfaces are
        fully closed.  Splits output into uncut (label 1) and cut (label 2)
        actors via the boundary_labels cell array.
        """
        if not self.show_mesh:
            self.uncut_actor.SetVisibility(False)
            self.cut_actor.SetVisibility(False)
            return
        g = grid if grid is not None else self.grid
        grid_res = self.resolution * GRID_RES_FACTOR
        img, name = g.to_label_field(grid_res=grid_res)

        surface = img.contour_labels(
            scalars=name,
            boundary_style="all",
            smoothing=True,
            smoothing_iterations=12,
            smoothing_relaxation=0.4,
        )

        if surface.n_points == 0:
            self.uncut_actor.SetVisibility(False)
            self.cut_actor.SetVisibility(False)
            return

        # boundary_labels is a 2-component array: [inside_label, outside_label]
        bl = surface.cell_data["boundary_labels"]  # shape (N, 2)

        # Uncut faces: boundary between label 1 and label 0 (air)
        mask_uncut = ((bl[:, 0] == 1) & (bl[:, 1] == 0)) | \
                     ((bl[:, 0] == 0) & (bl[:, 1] == 1))
        # Cut/machined faces: boundary between label 2 and label 0
        mask_cut   = ((bl[:, 0] == 2) & (bl[:, 1] == 0)) | \
                     ((bl[:, 0] == 0) & (bl[:, 1] == 2))

        def extract(mask):
            if not mask.any():
                return pv.PolyData()
            cell_ids = np.where(mask)[0]
            return surface.extract_cells(cell_ids).extract_surface()

        uncut_surf = extract(mask_uncut)
        cut_surf   = extract(mask_cut)

        def _push(mesh, actor, surf):
            if surf.n_points == 0:
                actor.SetVisibility(False)
                return
            mesh.copy_from(surf)
            mesh.Modified()
            actor.mapper.SetInputData(mesh)
            actor.mapper.Update()
            actor.SetVisibility(True)

        _push(self.uncut_mesh, self.uncut_actor, uncut_surf)
        _push(self.cut_mesh,   self.cut_actor,   cut_surf)

        self.plotter.render()

    # -----------------------------------------------------------------------
    # Tool / path updates
    # -----------------------------------------------------------------------

    def _update_tool_position(self):
        if not len(self.toolpath):
            return
        idx = int(self.sweep_progress * (len(self.toolpath) - 1))
        idx = min(idx, len(self.toolpath) - 1)
        pos = self.toolpath[idx]
        new_sphere = pv.Sphere(radius=self.tool_radius, center=pos,
                               theta_resolution=24, phi_resolution=24)
        self.tool_mesh.copy_from(new_sphere)
        self.tool_mesh.Modified()
        self.tool_actor.mapper.SetInputData(self.tool_mesh)
        self.tool_actor.mapper.Update()
        self.tool_actor.SetVisibility(self.show_tool)
        self.plotter.render()

    def _update_path_mesh(self):
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

    def _update_dexel_lines(self, axis: str, grid: TriDexelGrid | None = None):
        show = {"x": self.show_dexels_x,
                "y": self.show_dexels_y,
                "z": self.show_dexels_z}[axis]
        colors = {"x": "#cc4444", "y": "#44aa44", "z": "#4444cc"}

        if not show:
            if self.dexel_actors[axis] is not None:
                self.dexel_actors[axis].SetVisibility(False)
            return

        g = grid if grid is not None else self.grid
        segs = g.collect_dexel_lines(axis)
        if segs.shape[0] == 0:
            if self.dexel_actors[axis] is not None:
                self.dexel_actors[axis].SetVisibility(False)
            return

        n = segs.shape[0]
        points = segs.reshape(-1, 3)
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

    def _refresh_visible_dexels(self, grid: TriDexelGrid | None = None):
        for axis in ("x", "y", "z"):
            if {"x": self.show_dexels_x,
                "y": self.show_dexels_y,
                "z": self.show_dexels_z}[axis]:
                self._update_dexel_lines(axis, grid)

    # -----------------------------------------------------------------------
    # Cut operation — live grid
    # -----------------------------------------------------------------------

    def _perform_cut(self):
        if not len(self.toolpath):
            return
        t0 = time.perf_counter()
        max_idx = int(self.sweep_progress * (len(self.toolpath) - 1))
        max_idx = min(max_idx, len(self.toolpath) - 1)
        step = max(1, len(self.toolpath) // 200)
        indices = list(range(0, max_idx + 1, step))
        if max_idx not in indices:
            indices.append(max_idx)
        for i in indices:
            pos = self.toolpath[i]
            self.grid.subtract_sphere(pos[0], pos[1], pos[2], self.tool_radius)
        elapsed = (time.perf_counter() - t0) * 1000
        self.status_label.setText(
            f"Cut complete — {len(indices)} positions, {elapsed:.1f} ms"
        )
        self._rebuild_workpiece()
        self._refresh_visible_dexels()

    # -----------------------------------------------------------------------
    # Pre-computation of all intermediate states
    # -----------------------------------------------------------------------

    def _on_precalculate(self):
        """Pre-compute CACHE_STEPS evenly-spaced dexel grid snapshots.

        After this, dragging the Sweep Progress slider shows the cached
        surface instantly without re-running the cut simulation.
        """
        if not len(self.toolpath):
            return

        self.status_label.setText("Pre-computing… please wait")
        QtWidgets.QApplication.processEvents()

        t0 = time.perf_counter()
        n = len(self.toolpath)

        # Snap indices evenly across the toolpath
        cache_indices = np.linspace(0, n - 1, CACHE_STEPS, dtype=int)

        # Fresh grid; accumulate cuts progressively
        g = TriDexelGrid(WORKPIECE_BOUNDS, self.resolution)
        self._cache = []
        prev_i = 0
        for snap_i in cache_indices:
            for i in range(prev_i, snap_i + 1):
                pos = self.toolpath[i]
                g.subtract_sphere(pos[0], pos[1], pos[2], self.tool_radius)
            self._cache.append(g.copy())
            prev_i = snap_i + 1

        elapsed = (time.perf_counter() - t0) * 1000
        self.status_label.setText(
            f"Pre-computed {CACHE_STEPS} states in {elapsed:.0f} ms\n"
            "Slider now plays back instantly."
        )

        # Show the final state
        self._rebuild_workpiece(self._cache[-1])

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _on_path_changed(self, idx: int):
        self.current_path_idx = idx
        self._generate_toolpath()
        self._update_path_mesh()
        self._update_tool_position()

    def _on_progress_changed(self, value: int):
        self.sweep_progress = value / 1000.0
        self._update_tool_position()

        # If cache is available, look up and display the nearest snapshot
        if self._cache is not None:
            snap_idx = int(round(self.sweep_progress * (len(self._cache) - 1)))
            snap_idx = max(0, min(snap_idx, len(self._cache) - 1))
            cached_grid = self._cache[snap_idx]
            self._rebuild_workpiece(cached_grid)
            self._refresh_visible_dexels(cached_grid)

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

    def _on_density_changed(self, value: int):
        multiplier = value / 10.0
        new_res = int(round(DEFAULT_RESOLUTION * multiplier))
        if new_res == self.resolution:
            return
        self.resolution = new_res
        self.density_label.setText(
            f"Dexel Density  ({multiplier:.1f}x = {new_res})"
        )
        # Rebuild grid at new resolution, invalidate cache
        self.grid = TriDexelGrid(WORKPIECE_BOUNDS, self.resolution)
        self._cache = None
        self._rebuild_workpiece()
        self._refresh_visible_dexels()
        self.status_label.setText(
            f"Resolution changed to {new_res} ({multiplier:.1f}x)"
        )

    def _on_cut(self):
        self._perform_cut()

    def _on_reset(self):
        self.grid = TriDexelGrid(WORKPIECE_BOUNDS, self.resolution)
        self._cache = None
        self._rebuild_workpiece()
        self._refresh_visible_dexels()
        self.status_label.setText("Workpiece reset")

    def _on_dexel_toggle(self, axis: str, state: int):
        val = bool(state)
        if axis == "x":
            self.show_dexels_x = val
        elif axis == "y":
            self.show_dexels_y = val
        else:
            self.show_dexels_z = val
        # Use cached grid during playback so dexels match the mesh
        grid = None
        if self._cache is not None:
            snap_idx = int(round(self.sweep_progress * (len(self._cache) - 1)))
            snap_idx = max(0, min(snap_idx, len(self._cache) - 1))
            grid = self._cache[snap_idx]
        self._update_dexel_lines(axis, grid)
        self.plotter.render()

    def _on_mesh_toggle(self, state: int):
        self.show_mesh = bool(state)
        if self.show_mesh:
            # Re-render the current state to bring mesh back
            if self._cache is not None:
                snap_idx = int(round(self.sweep_progress * (len(self._cache) - 1)))
                snap_idx = max(0, min(snap_idx, len(self._cache) - 1))
                self._rebuild_workpiece(self._cache[snap_idx])
            else:
                self._rebuild_workpiece()
        else:
            self.uncut_actor.SetVisibility(False)
            self.cut_actor.SetVisibility(False)
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
    "description": (
        "Interactive tri-dexel volumetric CNC machining simulation.\n\n"
        "Features:\n"
        "- Tri-dexel workpiece representation (X/Y/Z ray segments)\n"
        "- Sphere tool with Boolean subtraction\n"
        "- Zigzag, spiral, contour, and cross-hatch toolpaths\n"
        "- Full closed-surface reconstruction via contour_labels (vtkSurfaceNets3D)\n"
        "- Two-colour surface: uncut stock vs machined faces\n"
        "- Shadows and balanced three-point lighting\n"
        "- Pre-compute all states for instant slider scrubbing"
    ),
}


if __name__ == "__main__":
    main()
