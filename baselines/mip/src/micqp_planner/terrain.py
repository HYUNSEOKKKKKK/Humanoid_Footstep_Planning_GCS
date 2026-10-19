from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import math
import json

import numpy as np


@dataclass(frozen=True)
class FootPose:
    x: float
    y: float
    z: float
    theta: float  # radians


@dataclass(frozen=True)
class SteppingStone:
    name: str
    center: Tuple[float, float, float]
    width: float
    height: float
    yaw: float  # radians


@dataclass(frozen=True)
class SafeRegion:
    """
    Convex region constraints in (x, y, z, theta):
        A f <= b
    where f = [x, y, z, theta]^T
    """
    name: str
    A: np.ndarray  # (m,4)
    b: np.ndarray  # (m,)


def _rect_halfspaces_xy(center_xy: Tuple[float, float], width: float, height: float, yaw: float, margin: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rectangle in XY, rotated by yaw around center.
    Returns A_xy (4,2), b_xy (4,) such that A_xy [x;y] <= b_xy.

    Local frame u,v:
        u =  cos*yaw*(x-cx) + sin*yaw*(y-cy)
        v = -sin*yaw*(x-cx) + cos*yaw*(y-cy)

    Constraints:
        |u| <= width/2 - margin
        |v| <= height/2 - margin
    """
    cx, cy = center_xy
    c = float(math.cos(yaw))
    s = float(math.sin(yaw))
    w2 = max(1e-6, 0.5 * width - margin)
    h2 = max(1e-6, 0.5 * height - margin)

    # u <= w2  =>  c x + s y <= w2 + c cx + s cy
    # -u <= w2 => -c x - s y <= w2 - c cx - s cy
    # v <= h2  => -s x + c y <= h2 + (-s) cx + c cy
    # -v <= h2 =>  s x - c y <= h2 + s cx - c cy
    A = np.array([
        [ c,  s],
        [-c, -s],
        [-s,  c],
        [ s, -c],
    ], dtype=float)
    b = np.array([
        w2 + c*cx + s*cy,
        w2 - c*cx - s*cy,
        h2 - s*cx + c*cy,
        h2 + s*cx - c*cy,
    ], dtype=float)
    return A, b


def regions_from_stepping_stones(stones: List[SteppingStone],
                                z_epsilon: float = 1e-3,
                                xy_margin: float = 0.0,
                                theta_bounds: Tuple[float, float] = (-math.pi, math.pi)) -> List[SafeRegion]:
    """
    Convert each stepping stone to one convex region A f <= b.
    We model z as (approximately) constant on the stone: z == z0 (with small tolerance).
    """
    regions: List[SafeRegion] = []
    tmin, tmax = theta_bounds

    for st in stones:
        cx, cy, cz = st.center
        Axy, bxy = _rect_halfspaces_xy((cx, cy), st.width, st.height, st.yaw, xy_margin)

        # Build full A,b on (x,y,z,theta)
        rows = []
        rhs = []

        # XY bounds
        for k in range(Axy.shape[0]):
            row = [Axy[k, 0], Axy[k, 1], 0.0, 0.0]
            rows.append(row)
            rhs.append(float(bxy[k]))

        # z approx equality: cz - eps <= z <= cz + eps
        # z <= cz + eps
        rows.append([0.0, 0.0, 1.0, 0.0])
        rhs.append(float(cz + z_epsilon))
        # -z <= -(cz - eps)  => z >= cz - eps
        rows.append([0.0, 0.0, -1.0, 0.0])
        rhs.append(float(-(cz - z_epsilon)))

        # Optional theta bounds inside region (IRIS regions include theta; for stones we keep wide)
        rows.append([0.0, 0.0, 0.0, 1.0])
        rhs.append(float(tmax))
        rows.append([0.0, 0.0, 0.0, -1.0])
        rhs.append(float(-tmin))

        A = np.array(rows, dtype=float)
        b = np.array(rhs, dtype=float)
        regions.append(SafeRegion(name=st.name, A=A, b=b))

    return regions


def infer_start_goal_from_stones(
    stones: List[SteppingStone],
    *,
    stance_width: float = 0.24,
    start_heading_mode: str = "first_to_second",
    goal_heading_mode: str = "prev_to_last",
    xy_margin: float = 0.0,
) -> Tuple[FootPose, FootPose, FootPose, FootPose]:
    """Infer start/goal foot poses from stepping stones.

    Returns:
        start_left, start_right, goal_left, goal_right

    Heading:
        * start_heading_mode:
            - "first_to_second": heading from stone_0 -> stone_1 (fallback to first->last if only 1 stone)
            - "first_to_last":   heading from stone_0 -> stone_last
        * goal_heading_mode:
            - "prev_to_last":    heading from stone_{-2} -> stone_{-1} (fallback to start heading)
            - "first_to_last":   heading from stone_0 -> stone_last

    Notes:
        - The yaw is only used to orient the stance offset and as the desired goal yaw (objective),
          not as a hard constraint unless you add explicit goal constraints.
        - We keep the inferred feet inside the stone by shrinking the offset if necessary
          (coarse check based on stone half-dimensions).
    """
    if not stones:
        raise ValueError("Cannot infer start/goal: no stones found")

    s0 = stones[0]
    sN = stones[-1]
    x0, y0, z0 = map(float, s0.center)
    xN, yN, zN = map(float, sN.center)

    def _heading(a, b) -> float:
        ax, ay = a
        bx, by = b
        dx, dy = (bx - ax), (by - ay)
        return 0.0 if abs(dx) + abs(dy) < 1e-12 else float(math.atan2(dy, dx))

    # ---- start heading ----
    if start_heading_mode == "first_to_last":
        th_start = _heading((x0, y0), (xN, yN))
    else:  # "first_to_second" (default)
        if len(stones) >= 2:
            x1, y1, _ = map(float, stones[1].center)
            th_start = _heading((x0, y0), (x1, y1))
        else:
            th_start = _heading((x0, y0), (xN, yN))

    # ---- goal heading ----
    if goal_heading_mode == "first_to_last":
        th_goal = _heading((x0, y0), (xN, yN))
    else:  # "prev_to_last" (default)
        if len(stones) >= 2:
            xp, yp, _ = map(float, stones[-2].center)
            th_goal = _heading((xp, yp), (xN, yN))
        else:
            th_goal = th_start

    def _offset_pair(xc, yc, zc, th, stone: SteppingStone):
        # heading normal (left direction)
        nx, ny = (-math.sin(th), math.cos(th))

        off = 0.5 * float(stance_width)

        # shrink offset (coarse) so both feet are likely inside the rectangle
        max_off = max(
            0.0,
            min(0.5 * float(stone.width), 0.5 * float(stone.height)) - float(xy_margin),
        )
        off = min(off, max_off)

        left  = FootPose(float(xc + off * nx), float(yc + off * ny), float(zc), float(th))
        right = FootPose(float(xc - off * nx), float(yc - off * ny), float(zc), float(th))
        return left, right

    start_left, start_right = _offset_pair(x0, y0, z0, th_start, s0)
    goal_left, goal_right   = _offset_pair(xN, yN, zN, th_goal,  sN)

    return start_left, start_right, goal_left, goal_right

def load_problem_json(path: str) -> Tuple[List[SafeRegion], FootPose, FootPose, FootPose, FootPose]:
    """Load a planning problem JSON.

    Supports:
      1) stepping stones JSON with keys:
           - stance_width (optional)
           - terrain.stones: [{name, center:[x,y,z], width, height, yaw_deg}, ...]
           - optional start_left/start_right (each [x,y,z,theta(rad)] or dict)
           - optional goal_left/goal_right (same format) OR goal (mid-pose) to be split
      2) explicit convex regions JSON with key: regions (list of {name,A,b}).

    Returns:
        regions, start_left, start_right, goal_left, goal_right
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stance_width = float(data.get("stance_width", 0.24))

    def _parse_pose(pose, default=None) -> Optional[FootPose]:
        """Accept pose as [x,y,z,theta] or {x,y,z,theta}."""
        if pose is None:
            return default
        if isinstance(pose, (list, tuple)):
            if len(pose) != 4:
                raise ValueError(f"Pose list must have 4 elements: got {pose}")
            x, y, z, th = pose
            return FootPose(float(x), float(y), float(z), float(th))
        if isinstance(pose, dict):
            return FootPose(
                x=float(pose["x"]),
                y=float(pose["y"]),
                z=float(pose.get("z", 0.0)),
                theta=float(pose.get("theta", 0.0)),
            )
        raise TypeError(f"Unsupported pose type: {type(pose)}")

    def _offset_pose(mid_xyz: Tuple[float, float, float], theta: float, signed_offset: float) -> FootPose:
        # heading normal (left direction)
        nx, ny = (-math.sin(theta), math.cos(theta))
        x, y, z = map(float, mid_xyz)
        return FootPose(float(x + signed_offset * nx), float(y + signed_offset * ny), float(z), float(theta))

    # ----------------- case (2): explicit regions -----------------
    if "regions" in data:
        regions: List[SafeRegion] = []
        for r in data["regions"]:
            name = str(r.get("name", "region"))
            A = np.array(r["A"], dtype=float)
            b = np.array(r["b"], dtype=float).reshape(-1,)
            if A.shape[1] != 4:
                raise ValueError(f"Region '{name}': A must have 4 columns for (x,y,z,theta)")
            regions.append(SafeRegion(name=name, A=A, b=b))

        start_left = _parse_pose(data.get("start_left"))
        start_right = _parse_pose(data.get("start_right"))

        goal_left = _parse_pose(data.get("goal_left"))
        goal_right = _parse_pose(data.get("goal_right"))

        if ("goal" in data) and (goal_left is None or goal_right is None):
            gmid = _parse_pose(data["goal"])
            goal_left  = goal_left  or _offset_pose((gmid.x, gmid.y, gmid.z), gmid.theta, +0.5 * stance_width)
            goal_right = goal_right or _offset_pose((gmid.x, gmid.y, gmid.z), gmid.theta, -0.5 * stance_width)

        # If start/goal not provided, we cannot infer without stones; error out.
        if start_left is None or start_right is None or goal_left is None or goal_right is None:
            raise ValueError("Explicit regions JSON requires start_left/start_right and goal/goal_left/goal_right.")
        return regions, start_left, start_right, goal_left, goal_right

    # ----------------- case (1): stepping stones -----------------
    stones_raw = data.get("terrain", {}).get("stones", [])
    if not isinstance(stones_raw, list):
        raise ValueError("terrain.stones must be a list")

    stones: List[SteppingStone] = []
    for i, st in enumerate(stones_raw):
        if not isinstance(st, dict):
            raise TypeError(f"Stone entry must be a dict, got {type(st)}")
        name = str(st.get("name", f"stone_{i+1}"))
        center = st.get("center", None)
        if center is None or len(center) != 3:
            raise ValueError(f"{name}: center must be [x,y,z]")
        cx, cy, cz = map(float, center)
        width = float(st.get("width", 0.3))
        height = float(st.get("height", 0.3))
        if "yaw_deg" in st:
            yaw = math.radians(float(st["yaw_deg"]))
        else:
            yaw = float(st.get("yaw", 0.0))  # radians
        stones.append(SteppingStone(name=name, center=(cx, cy, cz), width=width, height=height, yaw=yaw))

    regions = regions_from_stepping_stones(stones)

    # (1) explicit start/goal if provided
    start_left = _parse_pose(data.get("start_left"))
    start_right = _parse_pose(data.get("start_right"))

    goal_left = _parse_pose(data.get("goal_left"))
    goal_right = _parse_pose(data.get("goal_right"))

    # (2) if only goal mid is provided, split it
    if ("goal" in data) and (goal_left is None or goal_right is None):
        gmid = _parse_pose(data["goal"])
        goal_left  = goal_left  or _offset_pose((gmid.x, gmid.y, gmid.z), gmid.theta, +0.5 * stance_width)
        goal_right = goal_right or _offset_pose((gmid.x, gmid.y, gmid.z), gmid.theta, -0.5 * stance_width)

    # (3) infer remaining from stones
    if (start_left is None) or (start_right is None) or (goal_left is None) or (goal_right is None):
        inf_sl, inf_sr, inf_gl, inf_gr = infer_start_goal_from_stones(
            stones,
            stance_width=stance_width,
            start_heading_mode="first_to_second",
            goal_heading_mode="prev_to_last",
        )
        start_left = start_left or inf_sl
        start_right = start_right or inf_sr
        goal_left = goal_left or inf_gl
        goal_right = goal_right or inf_gr

    return regions, start_left, start_right, goal_left, goal_right

def infer_xyz_bounds_from_regions(regions: List[SafeRegion], xy_margin: float, z_margin: float) -> Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]]:
    """
    Rough AABB bounds from stepping-stone-style regions (rectangles with z-constraints).
    If your regions are arbitrary polytopes, you may want to override bounds manually.
    """
    xs, ys, zs = [], [], []
    for r in regions:
        # try to detect the z bounds from constraints: z <= ub, -z <= lbneg
        z_ubs = []
        z_lbs = []
        for row, rhs in zip(r.A, r.b):
            if np.allclose(row, [0,0,1,0]):
                z_ubs.append(rhs)
            if np.allclose(row, [0,0,-1,0]):
                z_lbs.append(-rhs)
        if z_ubs:
            zs.append(float(min(z_ubs)))
        if z_lbs:
            zs.append(float(max(z_lbs)))

        # For XY, we can't easily infer from halfspaces without LP; assume stepping-stone rectangles:
        # approximate bbox by center (from name convention unknown). We'll just skip and rely on user bounds if needed.
        # Instead, we take wide defaults centered at 0 if no info.
    if not zs:
        zs = [0.0]

    zmin, zmax = min(zs)-z_margin, max(zs)+z_margin
    # Wide XY defaults; caller can override with their terrain extents if known.
    xmin, xmax = -10.0, 10.0
    ymin, ymax = -10.0, 10.0
    return (xmin, xmax), (ymin, ymax), (zmin, zmax)
