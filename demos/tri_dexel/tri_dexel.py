"""
Tri-Dexel CNC Machining Simulation
Demonstrates the tri-dexel volumetric representation used in CNC simulation.
A sphere tool is swept along predefined toolpaths, subtracting material from a
workpiece represented as three orthogonal sets of dexel ray segments.

Surface reconstruction uses dual contouring: the grid cells ARE the dexel
cells, edge crossings are detected from the sign field, and QEF (Quadratic
Error Function) minimisation places one vertex per active cell.  Surface
normals for QEF are computed from the gradient of the combined signed
distance field (central finite differences), giving normals consistent with
the actual isosurface geometry.  Sharp features (workpiece edges/corners) are
preserved, and flat machined surfaces stay flat.
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
        # Segment format: [lo, hi, lo_nx, lo_ny, lo_nz, hi_nx, hi_ny, hi_nz]
        # Normals point outward from material into empty space.
        self.y_ticks_x = np.linspace(ymin, ymax, resolution)
        self.z_ticks_x = np.linspace(zmin, zmax, resolution)
        self.x_segs = [[[xmin, xmax, -1, 0, 0, 1, 0, 0]] for _ in range(resolution * resolution)]

        # Y-axis dexels: rays parallel to Y, grid over (X, Z)
        self.x_ticks_y = np.linspace(xmin, xmax, resolution)
        self.z_ticks_y = np.linspace(zmin, zmax, resolution)
        self.y_segs = [[[ymin, ymax, 0, -1, 0, 0, 1, 0]] for _ in range(resolution * resolution)]

        # Z-axis dexels: rays parallel to Z, grid over (X, Y)
        self.x_ticks_z = np.linspace(xmin, xmax, resolution)
        self.y_ticks_z = np.linspace(ymin, ymax, resolution)
        self.z_segs = [[[zmin, zmax, 0, 0, -1, 0, 0, 1]] for _ in range(resolution * resolution)]

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

        # Pre-compute outward normals at cut points (vectorised).
        # Normal = (sphere_centre - cut_point) / r  (points into void).
        du_hits = du[hit_iu]          # (n_hits,)
        dv_hits = dv[hit_iv]          # (n_hits,)
        if axis == "z":
            lo_nx = -du_hits / r;  lo_ny = -dv_hits / r;  lo_nz =  half / r
            hi_nx = -du_hits / r;  hi_ny = -dv_hits / r;  hi_nz = -half / r
        elif axis == "x":
            lo_nx =  half / r;     lo_ny = -du_hits / r;  lo_nz = -dv_hits / r
            hi_nx = -half / r;     hi_ny = -du_hits / r;  hi_nz = -dv_hits / r
        else:   # y
            lo_nx = -du_hits / r;  lo_ny =  half / r;     lo_nz = -dv_hits / r
            hi_nx = -du_hits / r;  hi_ny = -half / r;     hi_nz = -dv_hits / r

        for k in range(len(hit_iv)):
            idx = int(hit_iv[k]) * nu + int(hit_iu[k])
            lo_n = (lo_nx[k], lo_ny[k], lo_nz[k])
            hi_n = (hi_nx[k], hi_ny[k], hi_nz[k])
            segs_list[idx] = _subtract_interval(
                segs_list[idx], lo_arr[k], hi_arr[k], lo_n, hi_n,
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

    # ----- dual contouring surface reconstruction -----

    def to_dual_contour_mesh(self) -> pv.PolyData:
        """Surface reconstruction via dual contouring.

        The grid cells ARE the dexel cells (no interpolation).  Edge
        crossings are detected from the sign field, and QEF minimisation
        places one vertex per active cell.  Surface normals are derived
        from the gradient of the combined signed distance field (central
        finite differences), giving normals consistent with the actual
        isosurface rather than individual sphere intersection normals.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        res = self.resolution

        # Grid: dexel ticks + 1 padding cell on each side
        step_x = (xmax - xmin) / (res - 1) if res > 1 else 1.0
        step_y = (ymax - ymin) / (res - 1) if res > 1 else 1.0
        step_z = (zmax - zmin) / (res - 1) if res > 1 else 1.0

        grid_x = np.concatenate([[xmin - step_x], self.x_ticks_z, [xmax + step_x]])
        grid_y = np.concatenate([[ymin - step_y], self.y_ticks_z, [ymax + step_y]])
        grid_z = np.concatenate([[zmin - step_z], self.z_ticks_x, [zmax + step_z]])

        nx, ny, nz = len(grid_x), len(grid_y), len(grid_z)

        # -- Per-axis signed distance (normals from gradient, not stored) --
        # Z-axis: z_segs[iv * res + iu], iu=x index, iv=y index
        sd_z = np.full((nx, ny, nz), -1e6)
        for iv in range(res):
            for iu in range(res):
                sd_z[iu + 1, iv + 1, 1:-1] = _column_signed_distance(
                    self.z_segs[iv * res + iu], grid_z[1:-1])

        # X-axis: x_segs[iv * res + iu], iu=y index, iv=z index
        sd_x = np.full((nx, ny, nz), -1e6)
        for iv in range(res):
            for iu in range(res):
                sd_x[1:-1, iu + 1, iv + 1] = _column_signed_distance(
                    self.x_segs[iv * res + iu], grid_x[1:-1])

        # Y-axis: y_segs[iv * res + iu], iu=x index, iv=z index
        sd_y = np.full((nx, ny, nz), -1e6)
        for iv in range(res):
            for iu in range(res):
                sd_y[iu + 1, 1:-1, iv + 1] = _column_signed_distance(
                    self.y_segs[iv * res + iu], grid_y[1:-1])

        # Combined sign field (tri-dexel intersection = min)
        sd_combined = np.minimum(np.minimum(sd_z, sd_x), sd_y)
        signs = sd_combined >= 0

        # Surface normals from gradient of combined SD field.
        # Central finite differences give normals consistent with the
        # actual isosurface, not individual sphere normals.
        grad = np.zeros((nx, ny, nz, 3))
        grad[1:-1, :, :, 0] = (sd_combined[2:, :, :] -
                                sd_combined[:-2, :, :]) / (2 * step_x)
        grad[:, 1:-1, :, 1] = (sd_combined[:, 2:, :] -
                                sd_combined[:, :-2, :]) / (2 * step_y)
        grad[:, :, 1:-1, 2] = (sd_combined[:, :, 2:] -
                                sd_combined[:, :, :-2]) / (2 * step_z)
        grad_len = np.linalg.norm(grad, axis=-1, keepdims=True)
        grad_len = np.where(grad_len < 1e-10, 1.0, grad_len)
        grad_normals = grad / grad_len

        # -- Edge crossing detection --
        def _crossings(s0, s1, sgn0, sgn1):
            cross = sgn0 != sgn1
            denom = s0 - s1
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            return cross, s0 / denom

        # Helper: interpolate gradient normal at crossing position
        def _interp_normal(n0, n1, tv):
            n = (1 - tv[:, None]) * n0 + tv[:, None] * n1
            nlen = np.linalg.norm(n, axis=1, keepdims=True)
            nlen = np.where(nlen < 1e-10, 1.0, nlen)
            return n / nlen

        # Z-edges
        z_cross, z_t = _crossings(
            sd_combined[:, :, :-1], sd_combined[:, :, 1:],
            signs[:, :, :-1], signs[:, :, 1:])
        z_ei, z_ej, z_ek = np.where(z_cross)
        if len(z_ei):
            z_tv = z_t[z_ei, z_ej, z_ek]
            cp_z = np.column_stack([
                grid_x[z_ei], grid_y[z_ej],
                grid_z[z_ek] + z_tv * (grid_z[z_ek + 1] - grid_z[z_ek])])
            cn_z = _interp_normal(
                grad_normals[z_ei, z_ej, z_ek],
                grad_normals[z_ei, z_ej, z_ek + 1], z_tv)
        else:
            cp_z = np.empty((0, 3)); cn_z = np.empty((0, 3))

        # X-edges
        x_cross, x_t = _crossings(
            sd_combined[:-1, :, :], sd_combined[1:, :, :],
            signs[:-1, :, :], signs[1:, :, :])
        x_ei, x_ej, x_ek = np.where(x_cross)
        if len(x_ei):
            x_tv = x_t[x_ei, x_ej, x_ek]
            cp_x = np.column_stack([
                grid_x[x_ei] + x_tv * (grid_x[x_ei + 1] - grid_x[x_ei]),
                grid_y[x_ej], grid_z[x_ek]])
            cn_x = _interp_normal(
                grad_normals[x_ei, x_ej, x_ek],
                grad_normals[x_ei + 1, x_ej, x_ek], x_tv)
        else:
            cp_x = np.empty((0, 3)); cn_x = np.empty((0, 3))

        # Y-edges
        y_cross, y_t = _crossings(
            sd_combined[:, :-1, :], sd_combined[:, 1:, :],
            signs[:, :-1, :], signs[:, 1:, :])
        y_ei, y_ej, y_ek = np.where(y_cross)
        if len(y_ei):
            y_tv = y_t[y_ei, y_ej, y_ek]
            cp_y = np.column_stack([
                grid_x[y_ei],
                grid_y[y_ej] + y_tv * (grid_y[y_ej + 1] - grid_y[y_ej]),
                grid_z[y_ek]])
            cn_y = _interp_normal(
                grad_normals[y_ei, y_ej, y_ek],
                grad_normals[y_ei, y_ej + 1, y_ek], y_tv)
        else:
            cp_y = np.empty((0, 3)); cn_y = np.empty((0, 3))

        total_crossings = len(z_ei) + len(x_ei) + len(y_ei)
        if total_crossings == 0:
            return pv.PolyData()

        # -- QEF accumulation --
        ncx, ncy, ncz = nx - 1, ny - 1, nz - 1
        n_cells = ncx * ncy * ncz

        ATA = np.zeros((n_cells, 3, 3))
        ATb = np.zeros((n_cells, 3))
        cross_count = np.zeros(n_cells, dtype=np.int32)
        cross_pos_sum = np.zeros((n_cells, 3))

        def _cell_flat(cx, cy, cz):
            return cx * ncy * ncz + cy * ncz + cz

        def _scatter(eix, eij, eik, cpos, cnorm, offsets):
            if len(eix) == 0:
                return
            nn = cnorm[:, :, None] * cnorm[:, None, :]
            ndp = np.sum(cnorm * cpos, axis=1)
            atb = ndp[:, None] * cnorm
            for di, dj, dk in offsets:
                cx = eix + di
                cy = eij + dj
                cz = eik + dk
                ok = ((cx >= 0) & (cx < ncx) &
                      (cy >= 0) & (cy < ncy) &
                      (cz >= 0) & (cz < ncz))
                if not ok.any():
                    continue
                ci = _cell_flat(cx[ok], cy[ok], cz[ok])
                np.add.at(ATA, ci, nn[ok])
                np.add.at(ATb, ci, atb[ok])
                np.add.at(cross_count, ci, 1)
                np.add.at(cross_pos_sum, ci, cpos[ok])

        # Z-edge at (i,j,k) → 4 cells in XY plane
        _scatter(z_ei, z_ej, z_ek, cp_z, cn_z,
                 [(-1, -1, 0), (0, -1, 0), (0, 0, 0), (-1, 0, 0)])
        # X-edge at (i,j,k) → 4 cells in YZ plane
        _scatter(x_ei, x_ej, x_ek, cp_x, cn_x,
                 [(0, -1, -1), (0, 0, -1), (0, 0, 0), (0, -1, 0)])
        # Y-edge at (i,j,k) → 4 cells in XZ plane
        _scatter(y_ei, y_ej, y_ek, cp_y, cn_y,
                 [(-1, 0, -1), (-1, 0, 0), (0, 0, 0), (0, 0, -1)])

        # -- QEF solve --
        active_idx = np.where(cross_count > 0)[0]
        if len(active_idx) == 0:
            return pv.PolyData()

        mass_point = (cross_pos_sum[active_idx] /
                      np.maximum(cross_count[active_idx, None], 1))
        bias = 0.1
        A = ATA[active_idx] + (bias ** 2) * np.eye(3)
        b = ATb[active_idx] + (bias ** 2) * mass_point

        try:
            vertices = np.linalg.solve(A, b[..., np.newaxis])[..., 0]
        except np.linalg.LinAlgError:
            vertices = mass_point.copy()

        # Clamp vertices to cell bounds
        acx = active_idx // (ncy * ncz)
        acy = (active_idx % (ncy * ncz)) // ncz
        acz = active_idx % ncz
        cell_lo = np.column_stack([
            grid_x[acx], grid_y[acy], grid_z[acz]])
        cell_hi = np.column_stack([
            grid_x[acx + 1], grid_y[acy + 1], grid_z[acz + 1]])
        vertices = np.clip(vertices, cell_lo, cell_hi)

        # Vertex index map: cell flat index → vertex index
        vert_map = np.full(n_cells, -1, dtype=np.int64)
        vert_map[active_idx] = np.arange(len(active_idx))

        # -- Quad emission --
        faces_list = []

        def _emit(eix, eij, eik, sign_first, offsets):
            if len(eix) == 0:
                return
            ne = len(eix)
            vidx = np.full((ne, 4), -1, dtype=np.int64)
            ok = np.ones(ne, dtype=bool)
            for col, (di, dj, dk) in enumerate(offsets):
                cx = eix + di; cy = eij + dj; cz = eik + dk
                inb = ((cx >= 0) & (cx < ncx) &
                       (cy >= 0) & (cy < ncy) &
                       (cz >= 0) & (cz < ncz))
                ok &= inb
                cf = np.where(inb, _cell_flat(cx, cy, cz), 0)
                vi = np.where(inb, vert_map[cf], -1)
                ok &= (vi >= 0)
                vidx[:, col] = vi
            if not ok.any():
                return
            vidx = vidx[ok]
            sf = sign_first[ok]
            vidx[~sf] = vidx[~sf][:, ::-1]
            faces_list.append(vidx[:, [0, 1, 2]])
            faces_list.append(vidx[:, [0, 2, 3]])

        # Z-edges → quads in XY, +Z normal when first node is inside
        _emit(z_ei, z_ej, z_ek, signs[z_ei, z_ej, z_ek],
              [(-1, -1, 0), (0, -1, 0), (0, 0, 0), (-1, 0, 0)])
        # X-edges → quads in YZ, +X normal when first node is inside
        _emit(x_ei, x_ej, x_ek, signs[x_ei, x_ej, x_ek],
              [(0, -1, -1), (0, 0, -1), (0, 0, 0), (0, -1, 0)])
        # Y-edges → quads in XZ, +Y normal when first node is inside
        _emit(y_ei, y_ej, y_ek, signs[y_ei, y_ej, y_ek],
              [(-1, 0, -1), (-1, 0, 0), (0, 0, 0), (0, 0, -1)])

        if not faces_list:
            return pv.PolyData()

        all_tris = np.concatenate(faces_list, axis=0)
        n_tri = len(all_tris)
        faces = np.column_stack([
            np.full(n_tri, 3, dtype=np.int64), all_tris]).ravel()
        return pv.PolyData(vertices, faces=faces)


# ---------------------------------------------------------------------------
# Signed distance helper
# ---------------------------------------------------------------------------

def _column_signed_distance(segments: list, ray_positions: np.ndarray) -> np.ndarray:
    """Signed distance from ray_positions to the union of segments along a ray.

    Convention: positive = inside material, negative = outside.
    """
    n = len(ray_positions)
    if not segments:
        return np.full(n, -1e6)

    seg_arr = np.asarray(segments, dtype=np.float64)  # (n_segs, 8)
    lo = seg_arr[:, 0:1]   # (n_segs, 1)
    hi = seg_arr[:, 1:2]   # (n_segs, 1)
    r = ray_positions[np.newaxis, :]  # (1, n)

    # Per-segment inside check: (n_segs, n)
    inside = (r >= lo) & (r <= hi)

    # Distance to nearest boundary when inside a segment
    dist_lo = r - lo    # (n_segs, n)
    dist_hi = hi - r    # (n_segs, n)
    internal = np.minimum(dist_lo, dist_hi)

    any_inside = inside.any(axis=0)        # (n,)
    inside_sd = np.where(inside, internal, -np.inf).max(axis=0)  # (n,)

    # Distance to nearest boundary when outside all segments
    boundaries = np.concatenate([seg_arr[:, 0], seg_arr[:, 1]])  # (2*n_segs,)
    dists = np.abs(ray_positions[:, np.newaxis] - boundaries[np.newaxis, :])
    nearest = dists.min(axis=1)            # (n,)

    return np.where(any_inside, inside_sd, -nearest)


def _column_sd_and_normals(segments: list,
                           ray_positions: np.ndarray):
    """Signed distance AND 3-D outward normal of the nearest boundary.

    Returns (sd, normals) where sd has shape (n,) and normals (n, 3).
    Positive sd = inside material.
    """
    n = len(ray_positions)
    if not segments:
        return np.full(n, -1e6), np.zeros((n, 3))

    seg_arr = np.asarray(segments, dtype=np.float64)   # (n_segs, 8)
    lo = seg_arr[:, 0:1]   # (n_segs, 1)
    hi = seg_arr[:, 1:2]
    r = ray_positions[np.newaxis, :]  # (1, n)

    inside = (r >= lo) & (r <= hi)
    dist_lo = r - lo
    dist_hi = hi - r
    internal = np.minimum(dist_lo, dist_hi)

    any_inside = inside.any(axis=0)
    inside_sd = np.where(inside, internal, -np.inf).max(axis=0)

    # All boundaries and their stored normals
    all_pos = np.concatenate([seg_arr[:, 0], seg_arr[:, 1]])
    all_norm = np.concatenate([seg_arr[:, 2:5], seg_arr[:, 5:8]], axis=0)

    dists = np.abs(ray_positions[:, np.newaxis] - all_pos[np.newaxis, :])
    nearest_idx = np.argmin(dists, axis=1)
    nearest_dist = dists[np.arange(n), nearest_idx]
    normals = all_norm[nearest_idx]

    sd = np.where(any_inside, inside_sd, -nearest_dist)
    return sd, normals


# ---------------------------------------------------------------------------
# Interval helper
# ---------------------------------------------------------------------------

def _subtract_interval(segments: list, lo: float, hi: float,
                       lo_normal=None, hi_normal=None) -> list:
    """Remove [lo, hi] from segments, propagating boundary normals.

    lo_normal / hi_normal are (nx, ny, nz) outward normals at the cut
    boundaries.  Existing segment normals are preserved on unchanged ends.
    """
    result = []
    for seg in segments:
        s0, s1 = seg[0], seg[1]
        if s1 <= lo or s0 >= hi:
            result.append(seg)            # no overlap — keep as-is
        else:
            if s0 < lo:
                # Left remnant: keep s0's normal, new hi gets lo_normal
                result.append([s0, lo,
                               seg[2], seg[3], seg[4],
                               lo_normal[0], lo_normal[1], lo_normal[2]])
            if s1 > hi:
                # Right remnant: new lo gets hi_normal, keep s1's normal
                result.append([hi, s1,
                               hi_normal[0], hi_normal[1], hi_normal[2],
                               seg[5], seg[6], seg[7]])
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
DEFAULT_TOOL_RADIUS = 1.5
DEFAULT_Z_DEPTH = 1.0
CACHE_STEPS = 80       # number of snapshots stored when pre-computing

# Colours / materials
COL_UNCUT  = "#b0b0b0"   # brushed aluminium grey — uncut stock
COL_CUT    = "#7ec8e3"   # light machined-steel blue — freshly cut
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

        # ---- lighting for PBR metallic surfaces ----
        # Metallic PBR reflects light colour, so varied-colour directional
        # lights create the tinted reflections that read as brushed metal.
        # Non-positional (directional) lights avoid bright spot circles.
        self.plotter.remove_all_lights()
        light_cfgs = [
            # (position, color, intensity) — warm key, cool fill, neutral rim
            (( 8,  5, 10), "#fffaf0", 0.9),   # warm white key (top-right-front)
            ((-6, -4,  8), "#c0d8f0", 0.5),   # cool blue fill (top-left-back)
            (( 0,  8, -3), "#f0e8d0", 0.4),   # warm amber side
            ((-8,  0,  2), "#d0e0f0", 0.35),  # cool left
            (( 3, -6, -5), "#e8e0f0", 0.25),  # lavender under-fill
            (( 0,  0, 12), "#ffffff", 0.3),    # neutral top for cavity fill
        ]
        for pos, color, intensity in light_cfgs:
            lt = pv.Light(
                position=pos,
                focal_point=(0, 0, 0),
                color=color,
                intensity=intensity,
            )
            lt.positional = False  # directional — no spot circles
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
        self._cache_indices: np.ndarray | None = None  # toolpath index per cache slot
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
        self._cache = None; self._cache_indices = None   # invalidate cache on new toolpath

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
            pbr=True,
            metallic=0.8,
            roughness=0.45,
            smooth_shading=True,
            opacity=1.0,
        )
        self.cut_mesh = placeholder.copy()
        self.cut_actor = self.plotter.add_mesh(
            self.cut_mesh,
            color=COL_CUT,
            pbr=True,
            metallic=0.2,
            roughness=0.7,
            smooth_shading=True,
            opacity=1.0,
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
        """Reconstruct surface via dual contouring from all three dexel axes."""
        if not self.show_mesh:
            self.uncut_actor.SetVisibility(False)
            self.cut_actor.SetVisibility(False)
            return
        g = grid if grid is not None else self.grid
        surface = g.to_dual_contour_mesh()

        if surface.n_points == 0:
            self.uncut_actor.SetVisibility(False)
            self.cut_actor.SetVisibility(False)
            return

        # Compute cell normals first so we can use them for classification
        surface = surface.compute_normals(
            cell_normals=True, point_normals=False,
            auto_orient_normals=True,
        )

        # Classify faces as uncut stock vs machined.
        # A face is "uncut" only if its centroid is near a bounding-box face
        # AND its normal is approximately aligned with that face's outward
        # direction.  This prevents edge triangles (tilted normals) from
        # being misclassified as stock.
        xmin, xmax, ymin, ymax, zmin, zmax = WORKPIECE_BOUNDS
        cc = surface.cell_centers().points
        cn = surface.cell_data["Normals"]
        extent = max(xmax - xmin, ymax - ymin, zmax - zmin)
        tol = extent / self.resolution * 0.6
        cos_thresh = 0.7   # ~45° alignment required

        on_boundary = (
            ((np.abs(cc[:, 0] - xmin) < tol) & (-cn[:, 0] > cos_thresh)) |
            ((np.abs(cc[:, 0] - xmax) < tol) & ( cn[:, 0] > cos_thresh)) |
            ((np.abs(cc[:, 1] - ymin) < tol) & (-cn[:, 1] > cos_thresh)) |
            ((np.abs(cc[:, 1] - ymax) < tol) & ( cn[:, 1] > cos_thresh)) |
            ((np.abs(cc[:, 2] - zmin) < tol) & (-cn[:, 2] > cos_thresh)) |
            ((np.abs(cc[:, 2] - zmax) < tol) & ( cn[:, 2] > cos_thresh))
        )

        def _extract(mask):
            if not mask.any():
                return pv.PolyData()
            return surface.extract_cells(np.where(mask)[0]).extract_surface()

        uncut_surf = _extract(on_boundary)
        cut_surf = _extract(~on_boundary)

        # Recompute smooth point normals with feature angle for rendering
        feature_angle = 30.0
        if uncut_surf.n_points > 0:
            uncut_surf = uncut_surf.compute_normals(
                cell_normals=False, point_normals=True,
                feature_angle=feature_angle, split_vertices=True,
            )
        if cut_surf.n_points > 0:
            cut_surf = cut_surf.compute_normals(
                cell_normals=False, point_normals=True,
                feature_angle=feature_angle, split_vertices=True,
            )

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
        self._cache_indices = cache_indices
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

        # If cache is available, look up and display the nearest snapshot
        # and position the tool at the exact toolpath index for that snapshot
        if self._cache is not None:
            snap_idx = int(round(self.sweep_progress * (len(self._cache) - 1)))
            snap_idx = max(0, min(snap_idx, len(self._cache) - 1))
            # Position tool at the toolpath index this cache slot was built at
            tp_idx = int(self._cache_indices[snap_idx])
            if len(self.toolpath):
                tp_idx = min(tp_idx, len(self.toolpath) - 1)
                pos = self.toolpath[tp_idx]
                new_sphere = pv.Sphere(
                    radius=self.tool_radius, center=pos,
                    theta_resolution=24, phi_resolution=24)
                self.tool_mesh.copy_from(new_sphere)
                self.tool_mesh.Modified()
                self.tool_actor.mapper.SetInputData(self.tool_mesh)
                self.tool_actor.mapper.Update()
                self.tool_actor.SetVisibility(self.show_tool)
            cached_grid = self._cache[snap_idx]
            self._rebuild_workpiece(cached_grid)
            self._refresh_visible_dexels(cached_grid)
        else:
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
        self._cache = None; self._cache_indices = None
        self._rebuild_workpiece()
        self._refresh_visible_dexels()
        self.status_label.setText(
            f"Resolution changed to {new_res} ({multiplier:.1f}x)"
        )

    def _on_cut(self):
        self._perform_cut()

    def _on_reset(self):
        self.grid = TriDexelGrid(WORKPIECE_BOUNDS, self.resolution)
        self._cache = None; self._cache_indices = None
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
        "- Dual contouring with QEF vertex placement → sharp edges & smooth curves\n"
        "- Two-colour surface: uncut stock vs machined faces\n"
        "- Shadows and balanced three-point lighting\n"
        "- Pre-compute all states for instant slider scrubbing"
    ),
}


if __name__ == "__main__":
    main()
