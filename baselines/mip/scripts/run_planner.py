#!/usr/bin/env python3
from __future__ import annotations

"""Run the MIQCQP/MICO footstep planner.

This script is designed to be *PyCharm-friendly*:
- You can run it with **no command-line arguments**.
- Paths are resolved relative to the project root.
- MOSEK license is auto-detected at <repository>/.local/licenses/mosek.lic if env vars are not set.

If you want to reproduce paper-like settings later, search for "EASY_MODE" below.
"""

import os
import sys
import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

# --- project root / import path ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from micqp_planner.config import PlannerConfig
from micqp_planner.terrain import load_problem_json, FootPose, SteppingStone
from micqp_planner.plot import plot_plan_2d, plot_plan_3d


# =========================
# Defaults for "just run it"
# =========================
EASY_MODE = True
DEFAULT_TERRAIN = PROJECT_ROOT / "examples" / "stepping_stones_easy.json"
DEFAULT_OUTDIR = PROJECT_ROOT / "out"
DEFAULT_N = 10
DEFAULT_SOLVER = "mosek"   # keep "mosek" unless you have Gurobi
DEFAULT_TIME_LIMIT = 60.0
DEFAULT_SEGMENTS = 6

# Visualization defaults
PLOT_3D = True
PLOT_2D = False
VIEW_TOP = False
SHOW_PLOT = True
SAVE_PNG = True

# MOSEK license file (user-specific). You can also set env vars instead.
DEFAULT_MOSEK_LICENSE = REPOSITORY_ROOT / ".local" / "licenses" / "mosek.lic"


def _parse_pose_csv(s: str) -> FootPose:
    """Parse "x,y,z,theta" (theta in radians)."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("Pose must be 'x,y,z,theta' (theta in radians)")
    x, y, z = map(float, parts[:3])
    th = float(parts[3])
    return FootPose(x=x, y=y, z=z, theta=th)


def _load_stones_if_present(path: Path) -> Optional[List[SteppingStone]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stones_raw = data.get("terrain", {}).get("stones")
    if not stones_raw:
        return None
    stones: List[SteppingStone] = []
    for s in stones_raw:
        stones.append(
            SteppingStone(
                name=str(s.get("name", f"stone_{len(stones)+1}")),
                center=(float(s["center"][0]), float(s["center"][1]), float(s["center"][2])),
                width=float(s["width"]),
                height=float(s["height"]),
                yaw=math.radians(float(s.get("yaw_deg", 0.0))),
            )
        )
    return stones


def _bbox_from_stones(stones: List[SteppingStone]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    xs, ys, zs = [], [], []
    for st in stones:
        cx, cy, cz = st.center
        zs.append(cz)
        w2, h2 = 0.5 * st.width, 0.5 * st.height
        c, s = math.cos(st.yaw), math.sin(st.yaw)
        corners = [(w2, h2), (w2, -h2), (-w2, -h2), (-w2, h2)]
        for ux, uy in corners:
            xw = cx + c * ux - s * uy
            yw = cy + s * ux + c * uy
            xs.append(xw)
            ys.append(yw)
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def _resolve_path(p: Optional[str]) -> Path:
    if p is None or str(p).strip() == "":
        return DEFAULT_TERRAIN
    pp = Path(p)
    if pp.is_absolute():
        return pp
    # treat as project-relative by default
    return (PROJECT_ROOT / pp).resolve()


def _ensure_mosek_license_env() -> None:
    """Set env vars to point to a local MOSEK license if user hasn't set them."""
    if os.environ.get("MOSEKLM_LICENSE_FILE") or os.environ.get("MOSEK_LICENSE_FILE"):
        return
    if DEFAULT_MOSEK_LICENSE.exists():
        os.environ.setdefault("MOSEKLM_LICENSE_FILE", str(DEFAULT_MOSEK_LICENSE))
        os.environ.setdefault("MOSEK_LICENSE_FILE", str(DEFAULT_MOSEK_LICENSE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default=None, help="Terrain JSON path (project-relative or absolute)")
    ap.add_argument("--N", type=int, default=DEFAULT_N, help="Number of footsteps (including 2 fixed initial)")
    ap.add_argument("--out", default=str(DEFAULT_OUTDIR), help="Output directory")
    ap.add_argument("--solver", choices=["mosek", "gurobi"], default=DEFAULT_SOLVER)
    ap.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT, help="Time limit (s)")
    ap.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS, help="Piecewise segments for sin/cos")

    ap.add_argument("--start-left", default=None, help="Override start left pose: x,y,z,theta(rad)")
    ap.add_argument("--start-right", default=None, help="Override start right pose: x,y,z,theta(rad)")
    ap.add_argument("--goal", default=None, help="Override goal pose: x,y,z,theta(rad)")

    args = ap.parse_args()

    terrain_path = _resolve_path(args.terrain)
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- config --------
    cfg = PlannerConfig()
    cfg.solver.time_limit_s = float(args.time_limit)
    cfg.trig.num_segments = int(args.segments)

    # "Just make it run" defaults.
    if EASY_MODE:
        # Disable trim variables for stability (keeps a fixed-horizon problem).
        cfg.disable_trim = True
        cfg.enforce_trim_prefix = False
        cfg.obj.q_trim = 0.0

        # Slightly loosen reachability for stepping-stone demos.
        cfg.reach.d1 = 0.60
        cfg.reach.d2 = 0.60
        cfg.reach.max_dtheta = math.radians(90.0)

    # -------- load problem --------
    regions, start_left, start_right, goal_left, goal_right = load_problem_json(str(terrain_path))

    if args.start_left:
        start_left = _parse_pose_csv(args.start_left)
    if args.start_right:
        start_right = _parse_pose_csv(args.start_right)
    if args.goal:
        goal = _parse_pose_csv(args.goal)

    if start_left is None or start_right is None or goal is None:
        raise RuntimeError("Need start_left, start_right, and goal (either inside JSON or via CLI flags).")

    stones = _load_stones_if_present(terrain_path)

    # Absolute bounds: from stones if available; otherwise wide defaults.
    abs_bounds = None
    if stones:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = _bbox_from_stones(stones)
        abs_bounds = {
            "x": (xmin - cfg.bounds.abs_xy_margin, xmax + cfg.bounds.abs_xy_margin),
            "y": (ymin - cfg.bounds.abs_xy_margin, ymax + cfg.bounds.abs_xy_margin),
            "z": (zmin - cfg.bounds.abs_z_margin, zmax + cfg.bounds.abs_z_margin),
            "theta": (-math.pi, math.pi),
        }

    # -------- solve --------
    if args.solver == "gurobi":
        from micqp_planner.miqcqp import solve_miqcqp
        res = solve_miqcqp(regions, start_left, start_right, goal, N=int(args.N), cfg=cfg, abs_bounds=abs_bounds)
    else:
        _ensure_mosek_license_env()
        from micqp_planner.mosek_mico import solve_miqcqp_mosek
        res = solve_miqcqp_mosek(regions, start_left, start_right, goal_left, goal_right, N=int(args.N), cfg=cfg, abs_bounds=abs_bounds)

    # ---- visualization (no JSON export by default) ----
    if res.footsteps and stones:
        out_png = str(out_dir / "plan.png") if SAVE_PNG else None
        if PLOT_3D:
            plot_plan_3d(
                stones,
                res,
                out_png=out_png,
                title=f"MIQCQP plan | status={res.status} | time={res.runtime_s:.2f}s",
                thickness=0.04,
                pad_xy=0.0,
                face_alpha=0.35,
                view_top=VIEW_TOP,
                show=SHOW_PLOT,
            )
        if PLOT_2D and out_png is not None:
            plot_plan_2d(
                stones,
                res,
                out_png,
                title=f"MIQCQP plan | status={res.status} | time={res.runtime_s:.2f}s",
            )

    print(f"[MIQCQP] status={res.status} runtime={res.runtime_s:.3f}s obj={res.objective}")
    print(f"Terrain: {terrain_path}")
    if (out_dir / "plan.png").exists():
        print(f"Wrote: {out_dir / 'plan.png'}")


if __name__ == "__main__":
    main()
