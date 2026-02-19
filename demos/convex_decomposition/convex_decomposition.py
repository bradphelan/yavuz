"""
Convex Decomposition Demo
Generates a random concave polygon and decomposes it into convex sub-polygons
using ear-clipping triangulation followed by Hertel-Mehlhorn merging.
"""

from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import BackgroundPlotter


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _cross2d(o, a, b):
    """Z-component of (a-o) × (b-o). Positive = left turn (CCW)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _signed_area(pts):
    n = len(pts)
    return sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    ) / 2.0


def _segments_cross(p1, p2, p3, p4):
    """Return True if segment p1-p2 properly crosses segment p3-p4."""
    d1 = _cross2d(p3, p4, p1)
    d2 = _cross2d(p3, p4, p2)
    d3 = _cross2d(p1, p2, p3)
    d4 = _cross2d(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _is_simple(pts):
    """Return True if polygon has no self-intersecting edges."""
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # Adjacent wrap-around edges share a vertex
            c, d = pts[j], pts[(j + 1) % n]
            if _segments_cross(a, b, c, d):
                return False
    return True


def _has_concavity(pts):
    """Return True if the CCW polygon has at least one reflex vertex."""
    n = len(pts)
    return any(
        _cross2d(pts[(i - 1) % n], pts[i], pts[(i + 1) % n]) < -1e-9
        for i in range(n)
    )


# ─── Concave polygon generation ───────────────────────────────────────────────

def generate_concave_polygon(n_sides, rng):
    """
    Generate a random simple concave polygon with n_sides vertices.
    - Non-symmetric (exponential angle gaps)
    - Non-self-intersecting (verified and retried)
    - At least one reflex (concave) vertex
    """
    for _ in range(400):
        # Irregular angle spacing via exponential gaps
        gaps = rng.exponential(1.0, n_sides)
        angles = np.cumsum(gaps / gaps.sum() * 2 * np.pi)
        angles -= angles[0]

        # Random radii: outer vertices vs. concave vertices pushed inward
        radii = rng.uniform(2.0, 3.5, n_sides)
        n_inner = max(1, n_sides // 3)
        inner_idx = rng.choice(n_sides, n_inner, replace=False)
        radii[inner_idx] = rng.uniform(0.4, 1.2, n_inner)

        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

        # Ensure CCW orientation
        if _signed_area(pts) < 0:
            pts = pts[::-1]

        if _is_simple(pts) and _has_concavity(pts):
            return pts

    # Reliable fallback: asymmetric star polygon
    return _star_fallback(n_sides)


def _star_fallback(n_sides):
    """Alternating-radius star polygon — always simple and concave."""
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    # Small asymmetric perturbation to break symmetry
    angles += np.linspace(0.0, 0.25, n_sides)
    radii = np.where(np.arange(n_sides) % 2 == 0, 3.0, 1.2)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    if _signed_area(pts) < 0:
        pts = pts[::-1]
    return pts


# ─── Ear-clipping triangulation ───────────────────────────────────────────────

def ear_clip(pts):
    """
    Ear-clipping triangulation of a simple CCW polygon.
    Returns a list of (i, j, k) index triples referencing pts.
    """
    n = len(pts)
    remaining = list(range(n))
    tris = []

    def is_ear(pos):
        nr = len(remaining)
        pi = remaining[(pos - 1) % nr]
        ci = remaining[pos]
        ni = remaining[(pos + 1) % nr]
        a, b, c = pts[pi], pts[ci], pts[ni]
        # Vertex must be convex in a CCW polygon
        if _cross2d(a, b, c) <= 0:
            return False
        # No other remaining vertex may lie strictly inside this triangle
        for k_pos in range(nr):
            k = remaining[k_pos]
            if k in (pi, ci, ni):
                continue
            p = pts[k]
            if (
                _cross2d(a, b, p) >= 0
                and _cross2d(b, c, p) >= 0
                and _cross2d(c, a, p) >= 0
            ):
                return False
        return True

    guard = n * n + n
    itr = 0
    while len(remaining) > 3 and itr < guard:
        itr += 1
        nr = len(remaining)
        for pos in range(nr):
            if is_ear(pos):
                pi = remaining[(pos - 1) % nr]
                ci = remaining[pos]
                ni = remaining[(pos + 1) % nr]
                tris.append((pi, ci, ni))
                remaining.pop(pos)
                break

    if len(remaining) == 3:
        tris.append(tuple(remaining))
    return tris


# ─── Hertel-Mehlhorn convex decomposition ─────────────────────────────────────

def hertel_mehlhorn(pts, triangles):
    """
    Greedily merge triangles into maximal convex polygons (Hertel-Mehlhorn).

    pts:       (n, 2) array of original polygon vertices
    triangles: list of (i, j, k) index triples from ear_clip
    Returns:   list of convex polygons, each a list of vertex indices into pts
    """
    n_pts = len(pts)

    # Canonical boundary edges of the original polygon
    boundary = {
        (min(i, (i + 1) % n_pts), max(i, (i + 1) % n_pts))
        for i in range(n_pts)
    }

    # Internal diagonals: shared by exactly 2 triangles and not on boundary
    edge_count = defaultdict(int)
    for tri in triangles:
        for k in range(3):
            e = (min(tri[k], tri[(k + 1) % 3]), max(tri[k], tri[(k + 1) % 3]))
            edge_count[e] += 1
    diagonals = [e for e, cnt in edge_count.items() if cnt == 2 and e not in boundary]

    parts = [list(tri) for tri in triangles]

    def poly_convex(poly):
        nr = len(poly)
        return all(
            _cross2d(pts[poly[(i - 1) % nr]], pts[poly[i]], pts[poly[(i + 1) % nr]]) >= -1e-9
            for i in range(nr)
        )

    def merge_parts(pa, pb, eu, ev):
        """
        Merge pa and pb by removing directed edge eu→ev (forward in pa)
        and ev→eu (forward in pb). Returns merged vertex list or None.
        """
        na, nb = len(pa), len(pb)
        try:
            ua, va = pa.index(eu), pa.index(ev)
            vb, ub = pb.index(ev), pb.index(eu)
        except ValueError:
            return None
        if (ua + 1) % na != va:
            return None
        if (vb + 1) % nb != ub:
            return None
        # Traverse pa from va (na-1 vertices, omitting the ua→va edge)
        result = [pa[(va + k) % na] for k in range(na - 1)]
        # Traverse pb from ub (nb-1 vertices, omitting the vb→ub edge)
        result += [pb[(ub + k) % nb] for k in range(nb - 1)]
        return result

    def try_diagonal(parts, diag):
        """Try to merge the two parts sharing diag. Returns (parts, success)."""
        a_val, b_val = diag

        owners = []
        for idx, part in enumerate(parts):
            nr = len(part)
            for k in range(nr):
                e = (min(part[k], part[(k + 1) % nr]), max(part[k], part[(k + 1) % nr]))
                if e == diag:
                    owners.append(idx)
                    break
        if len(owners) != 2:
            return parts, False

        ia, ib = owners
        pa, pb = parts[ia], parts[ib]
        na = len(pa)

        if a_val not in pa or b_val not in pa:
            return parts, False

        ua_pos = pa.index(a_val)
        va_pos = pa.index(b_val)

        # Determine which direction the edge runs in pa
        if (ua_pos + 1) % na == va_pos:
            eu, ev = a_val, b_val
        elif (va_pos + 1) % na == ua_pos:
            eu, ev = b_val, a_val
        else:
            return parts, False

        merged = merge_parts(pa, pb, eu, ev)
        if merged is None or not poly_convex(merged):
            return parts, False

        new_parts = [p for i, p in enumerate(parts) if i not in (ia, ib)]
        new_parts.append(merged)
        return new_parts, True

    pending = list(diagonals)
    changed = True
    while changed:
        changed = False
        for diag in list(pending):
            parts, ok = try_diagonal(parts, diag)
            if ok:
                pending.remove(diag)
                changed = True
                break  # Restart with updated parts list

    return parts


# ─── Colour generation ────────────────────────────────────────────────────────

def _distinct_colors(n):
    """Return n visually distinct RGB tuples using HSV hue rotation."""
    colors = []
    for i in range(n):
        h = (i / max(n, 1)) * 360.0
        h6 = h / 60.0
        x = 1.0 - abs(h6 % 2 - 1)
        if   h6 < 1: r, g, b = 1, x, 0
        elif h6 < 2: r, g, b = x, 1, 0
        elif h6 < 3: r, g, b = 0, 1, x
        elif h6 < 4: r, g, b = 0, x, 1
        elif h6 < 5: r, g, b = x, 0, 1
        else:         r, g, b = 1, 0, x
        # Lighten slightly for better fill visibility
        f = 0.72
        colors.append((r * f + (1 - f), g * f + (1 - f), b * f + (1 - f)))
    return colors


# ─── Demo class ───────────────────────────────────────────────────────────────

class ConvexDecompositionDemo:
    def __init__(self):
        self.plotter = BackgroundPlotter(
            window_size=(1200, 800),
            title="Convex Decomposition of Concave Polygons",
        )
        self.plotter.set_background("white")

        self._n_sides = 9
        self._seed = 7
        self._polygon = None
        self._parts = []
        self._actors = []

        self._setup_controls()
        self._regenerate()

    # ── Controls ──────────────────────────────────────────────────────────────

    def _setup_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self.plotter)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        layout.addWidget(QtWidgets.QLabel("Number of sides"))
        self._sides_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._sides_slider.setMinimum(5)
        self._sides_slider.setMaximum(24)
        self._sides_slider.setValue(self._n_sides)
        self._sides_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._sides_slider.setTickInterval(1)
        self._sides_val_label = QtWidgets.QLabel(str(self._n_sides))
        self._sides_slider.valueChanged.connect(self._on_sides_changed)
        layout.addWidget(self._sides_slider)
        layout.addWidget(self._sides_val_label)

        regen_btn = QtWidgets.QPushButton("Randomize")
        regen_btn.clicked.connect(self._on_randomize)
        layout.addWidget(regen_btn)

        self._info_label = QtWidgets.QLabel("")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        layout.addStretch(1)
        dock.setWidget(panel)
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _on_sides_changed(self, value):
        self._n_sides = value
        self._sides_val_label.setText(str(value))
        self._regenerate()

    def _on_randomize(self):
        self._seed = np.random.randint(0, 100_000)
        self._regenerate()

    # ── Generation & drawing ──────────────────────────────────────────────────

    def _regenerate(self):
        rng = np.random.RandomState(self._seed)
        self._polygon = generate_concave_polygon(self._n_sides, rng)
        tris = ear_clip(self._polygon)
        self._parts = hertel_mehlhorn(self._polygon, tris)
        self._draw()

    def _draw(self):
        for actor in self._actors:
            self.plotter.remove_actor(actor)
        self._actors.clear()

        pts = self._polygon
        n = len(pts)
        colors = _distinct_colors(len(self._parts))

        # Filled convex sub-polygons (slightly below outline)
        for color, part in zip(colors, self._parts):
            verts = pts[list(part)]
            nv = len(verts)
            pts3d = np.column_stack([verts, np.zeros(nv)])
            face = np.array([nv] + list(range(nv)), dtype=np.intp)
            mesh = pv.PolyData(pts3d, faces=face)
            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                opacity=0.70,
                show_edges=True,
                edge_color="dimgray",
                line_width=1,
            )
            self._actors.append(actor)

        # Original polygon outline (drawn slightly above fills)
        pts3d = np.column_stack([pts, np.full(n, 0.01)])
        cells = []
        for i in range(n):
            cells += [2, i, (i + 1) % n]
        outline = pv.PolyData(pts3d)
        outline.lines = np.array(cells, dtype=np.intp)
        actor = self.plotter.add_mesh(outline, color="black", line_width=3)
        self._actors.append(actor)

        # Vertex dots
        actor = self.plotter.add_mesh(
            pv.PolyData(pts3d),
            color="black",
            point_size=10,
            render_points_as_spheres=True,
        )
        self._actors.append(actor)

        # Stats
        n_reflex = sum(
            1 for i in range(n)
            if _cross2d(pts[(i - 1) % n], pts[i], pts[(i + 1) % n]) < -1e-9
        )
        self._info_label.setText(
            f"Sides: {n}\n"
            f"Reflex vertices: {n_reflex}\n"
            f"Triangles (ear-clip): {n - 2}\n"
            f"Convex parts after merge: {len(self._parts)}"
        )

        self.plotter.enable_parallel_projection()
        self.plotter.view_xy()
        self.plotter.reset_camera()
        self.plotter.render()

    def show(self):
        self.plotter.show()
        self.plotter.app.exec()


# ─── Entry points ─────────────────────────────────────────────────────────────

def main():
    demo = ConvexDecompositionDemo()
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
    "title": "Convex Decomposition",
    "description": (
        "Generate a random concave polygon and decompose it into convex sub-polygons.\n\n"
        "Features:\n"
        "- Random non-symmetric, non-self-intersecting concave polygon\n"
        "- Ear-clipping triangulation\n"
        "- Hertel-Mehlhorn convex merging\n"
        "- Each convex part shown in a distinct colour\n"
        "- Interactive number-of-sides control (5–24)\n"
        "- Randomize button for new shapes"
    ),
}


if __name__ == "__main__":
    main()
