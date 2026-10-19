from __future__ import annotations

from typing import List, Optional, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from .terrain import SteppingStone
from .miqcqp import SolveResult


def _rot2(yaw: float) -> np.ndarray:
    c = float(math.cos(yaw))
    s = float(math.sin(yaw))
    return np.array([[c, -s], [s, c]], dtype=float)


def _transform_poly(poly_local_xy: List[Tuple[float, float]], x: float, y: float, yaw: float) -> List[Tuple[float, float]]:
    R = _rot2(yaw)
    P = np.asarray(poly_local_xy, dtype=float)
    W = P @ R.T + np.array([x, y], dtype=float)
    return [(float(px), float(py)) for px, py in W]


def _make_box_vertices(center: Tuple[float, float, float], width: float, height: float,
                       thickness: float, yaw: float) -> np.ndarray:
    """Return 8 vertices of a yaw-rotated box.

    We interpret `center[2]` as the *top surface* height (matches stepping-stone JSON usage).
    The box extends downward by `thickness`.
    """
    cx, cy, cz_top = center
    w2 = 0.5 * float(width)
    h2 = 0.5 * float(height)
    z_top = float(cz_top)
    z_bot = float(cz_top) - float(thickness)

    local = np.array([
        [ w2,  h2],
        [ w2, -h2],
        [-w2, -h2],
        [-w2,  h2],
    ], dtype=float)
    R = _rot2(float(yaw))
    world_xy = local @ R.T + np.array([cx, cy], dtype=float)

    V = np.zeros((8, 3), dtype=float)
    # top face (0..3)
    V[0:4, 0:2] = world_xy
    V[0:4, 2] = z_top
    # bottom face (4..7)
    V[4:8, 0:2] = world_xy
    V[4:8, 2] = z_bot
    return V


def _box_faces_from_vertices(V8: np.ndarray) -> List[np.ndarray]:
    """Return 6 quad faces for a box, given 8 vertices (top 0..3, bottom 4..7)."""
    V = V8
    return [
        V[[0, 1, 2, 3], :],  # top
        V[[4, 5, 6, 7], :],  # bottom
        V[[0, 1, 5, 4], :],
        V[[1, 2, 6, 5], :],
        V[[2, 3, 7, 6], :],
        V[[3, 0, 4, 7], :],
    ]


def _stone_facecolor(idx: int, n: int, face_alpha: float = 0.35) -> Tuple[float, float, float, float]:
    # Highlight first/last like your other visualizer.
    if n <= 0:
        return (0.7, 0.7, 0.7, face_alpha)
    if idx == 0:
        return (0.0, 0.7, 0.2, face_alpha)
    if idx == n - 1:
        return (0.9, 0.1, 0.1, face_alpha)
    return (0.75, 0.75, 0.75, face_alpha)


def plot_plan_3d(stones: List[SteppingStone],
                 result: SolveResult,
                 out_png: Optional[str] = None,
                 *,
                 title: str = "MIQCQP footstep plan",
                 thickness: float = 0.04,
                 pad_xy: float = 0.0,
                 face_alpha: float = 0.35,
                 view_top: bool = False,
                 show: bool = True) -> None:
    """3D visualization similar to your A* visualizer.

    - Stones: yaw-rotated boxes with a constant top-surface z.
    - Footsteps: wireframe rectangles at each step's (x,y,z), colored by left/right.
    """

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    all_xyz: List[np.ndarray] = []
    n = len(stones)

    # --- stones ---
    for idx, st in enumerate(stones):
        cx, cy, cz = st.center
        w = float(st.width) + float(pad_xy)
        h = float(st.height) + float(pad_xy)
        V8 = _make_box_vertices((cx, cy, cz), w, h, thickness, st.yaw)
        faces = _box_faces_from_vertices(V8)
        poly = Poly3DCollection(faces, edgecolor="k", linewidths=0.6, zsort="min")
        poly.set_facecolor(_stone_facecolor(idx, n, face_alpha))
        ax.add_collection3d(poly)
        all_xyz.append(V8)

    # --- footsteps ---
    foot_poly_local = [(-0.075, -0.035), (-0.075, 0.035), (0.075, 0.035), (0.075, -0.035)]

    segs: List[List[Tuple[float, float, float]]] = []
    colors: List[str] = []
    act = result.active_indices if result.active_indices else list(range(len(result.footsteps)))

    for k, i in enumerate(act):
        fs = result.footsteps[i]
        side = fs.foot
        x, y, z, yaw = float(fs.x), float(fs.y), float(fs.z), float(fs.theta)

        poly_xy = _transform_poly(foot_poly_local, x, y, yaw)
        poly_3d = [(px, py, z) for (px, py) in poly_xy]
        poly_3d.append(poly_3d[0])

        # wireframe edges
        edge_color = "red" if side.upper().startswith("L") else "blue"
        for e in range(4):
            segs.append([poly_3d[e], poly_3d[e + 1]])
            colors.append(edge_color)

        # heading line
        ax.plot(
            [x, x + 0.10 * math.cos(yaw)],
            [y, y + 0.10 * math.sin(yaw)],
            [z, z],
            color="black",
            linewidth=1.0,
        )
        ax.text(x, y, z + 0.01, str(k), fontsize=8)

        all_xyz.append(np.array(poly_3d[:4], dtype=float))

    if segs:
        ax.add_collection3d(Line3DCollection(segs, colors=colors, linewidths=1.8))

    # --- axis limits / equal aspect ---
    if all_xyz:
        P = np.vstack(all_xyz)
        rng = np.ptp(P, axis=0)
        pad = 0.10 * max(float(rng[0]), float(rng[1]), float(rng[2]), 1e-6)
        ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
        ax.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)
        ax.set_zlim(P[:, 2].min() - pad, P[:, 2].max() + pad)

        xr, yr, zr = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        maxr = max(xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0])
        cx = 0.5 * (xr[0] + xr[1])
        cy = 0.5 * (yr[0] + yr[1])
        cz = 0.5 * (zr[0] + zr[1])
        ax.set_xlim(cx - maxr / 2, cx + maxr / 2)
        ax.set_ylim(cy - maxr / 2, cy + maxr / 2)
        ax.set_zlim(cz - maxr / 2, cz + maxr / 2)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.grid(False)

    ax.view_init(elev=90, azim=-90) if view_top else ax.view_init(elev=30, azim=-110)

    ax.legend(handles=[
        Patch(facecolor=(0.0, 0.7, 0.2, 0.6), edgecolor='k', label='First stone'),
        Patch(facecolor=(0.9, 0.1, 0.1, 0.6), edgecolor='k', label='Last stone'),
        Patch(facecolor="none", edgecolor="red", label="Left foot"),
        Patch(facecolor="none", edgecolor="blue", label="Right foot"),
    ], loc="upper right")

    if out_png:
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _stone_corners_xy(st: SteppingStone) -> np.ndarray:
    cx, cy, _ = st.center
    w2 = 0.5 * st.width
    h2 = 0.5 * st.height
    local = np.array([
        [ w2,  h2],
        [ w2, -h2],
        [-w2, -h2],
        [-w2,  h2],
    ], dtype=float)
    c = math.cos(st.yaw)
    s = math.sin(st.yaw)
    R = np.array([[c, -s], [s, c]], dtype=float)
    world = local @ R.T + np.array([cx, cy])
    return world


def plot_plan_2d(stones: Optional[List[SteppingStone]],
                 result: SolveResult,
                 out_png: str,
                 title: str = "MIQCQP footstep plan") -> None:
    plt.figure(figsize=(9, 7))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5)

    if stones:
        for st in stones:
            poly = _stone_corners_xy(st)
            patch = Polygon(poly, closed=True, fill=False, linewidth=1.0)
            ax.add_patch(patch)
            cx, cy, _ = st.center
            ax.text(cx, cy, st.name, fontsize=8, ha="center", va="center")

    # plot footsteps (active and trimmed)
    xs = [fs.x for fs in result.footsteps]
    ys = [fs.y for fs in result.footsteps]

    # trimmed steps in light markers
    for fs in result.footsteps:
        if fs.trim and fs.idx >= 2:
            ax.plot(fs.x, fs.y, marker="x", linestyle="None", markersize=6)
    # active sequence
    act = result.active_indices
    ax.plot([result.footsteps[i].x for i in act],
            [result.footsteps[i].y for i in act],
            marker="o", linewidth=1.5)

    # annotate feet
    for i in act:
        fs = result.footsteps[i]
        ax.text(fs.x, fs.y, f"{fs.foot}{fs.idx}", fontsize=8, ha="left", va="bottom")

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
