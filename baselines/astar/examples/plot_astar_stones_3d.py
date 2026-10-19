from __future__ import annotations
import csv

import argparse
import math
from pathlib import Path
from typing import List, Tuple, Optional
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from ihmc_footstep_planning.parameters import DefaultFootstepPlannerParameters
from ihmc_footstep_planning.graph_search.footstep_planner_heuristic_calculator import FootstepPlannerHeuristicCalculator
from ihmc_footstep_planning.robot_side import RobotSide
from ihmc_footstep_planning.graph import DiscreteFootstep, FootstepGraphNode
from ihmc_footstep_planning.step_expansion import ParameterBasedStepExpansion
from ihmc_footstep_planning.step_cost import FootstepCostCalculator
from ihmc_footstep_planning.graph_search import AStarFootstepPlannerIterationConductor
from ihmc_footstep_planning.geometry import Pose2D, wrap_to_pi, ensure_ccw, convex_polygon_contains_polygon
from ihmc_footstep_planning.environments.stepping_stone_terrain import SteppingStoneTerrain
from ihmc_footstep_planning.step_checking.region_footstep_checker import RegionFootstepChecker
import time
from datetime import datetime


# ----------------------------
# Goal / shorthand helpers
# ----------------------------
class SimpleIdealStep:
    def __init__(self, goal_x: float, goal_y: float, nominal_len: float = 0.25, nominal_width: float = 0.20):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.nominal_len = nominal_len
        self.nominal_width = nominal_width

    def compute_ideal_step_for_node(self, stance_node) -> DiscreteFootstep:
        return self.compute_ideal_step(stance_node.get_second_step(), stance_node.get_first_step())

    def compute_ideal_step(self, stance_step: DiscreteFootstep, start_of_swing: DiscreteFootstep) -> DiscreteFootstep:
        dx = self.goal_x - stance_step.get_x()
        dy = self.goal_y - stance_step.get_y()
        yaw_to_goal = math.atan2(dy, dx)

        side = stance_step.get_robot_side().get_opposite_side()
        width = -self.nominal_width if side is RobotSide.RIGHT else self.nominal_width

        x = stance_step.get_x() + self.nominal_len * math.cos(yaw_to_goal) - width * math.sin(yaw_to_goal)
        y = stance_step.get_y() + self.nominal_len * math.sin(yaw_to_goal) + width * math.cos(yaw_to_goal)
        return DiscreteFootstep(x, y, yaw_to_goal, side)


def heading_deg_x_forward(a_xyz, b_xyz) -> float:
    ax, ay, _ = a_xyz
    bx, by, _ = b_xyz
    return math.degrees(math.atan2(by - ay, bx - ax))


def feet_from_center(center_xyz, heading_deg: float, stance_width: float):
    cx, cy, _ = center_xyz
    yaw = math.radians(heading_deg)
    lx = -math.sin(yaw)
    ly = math.cos(yaw)
    half = 0.5 * float(stance_width)
    left = (cx + half * lx, cy + half * ly)
    right = (cx - half * lx, cy - half * ly)
    return left, right


def make_entry_from_stones(stones, stance_width: float):
    sc = stones[0]["center"]
    gc = stones[-1]["center"]
    if len(stones) >= 2:
        shy = heading_deg_x_forward(stones[0]["center"], stones[1]["center"])
        ghy = heading_deg_x_forward(stones[-2]["center"], stones[-1]["center"])
    else:
        shy = ghy = 0.0
    return {
        "start": {"center": sc, "heading_deg": shy},
        "goal": {"center": gc, "heading_deg": ghy},
        "stance_width": float(stance_width),
        # 0.0 means "use heading"
        "foot_yaw_deg": {"start_L": 0.0, "start_R": 0.0, "goal_L": 0.0, "goal_R": 0.0},
    }


def _yaw_deg_or_heading(fy, key: str, heading_deg: float) -> float:
    if key not in fy:
        return heading_deg
    v = float(fy[key])
    if abs(v) < 1e-12:
        return heading_deg
    return v


def expand_shorthand(entry, default_stance_width: float):
    w = float(entry.get("stance_width", default_stance_width))

    sc = entry["start"]["center"]
    shy = float(entry["start"]["heading_deg"])
    gc = entry["goal"]["center"]
    ghy = float(entry["goal"]["heading_deg"])

    start_L, start_R = feet_from_center(sc, shy, w)
    goal_L, goal_R = feet_from_center(gc, ghy, w)

    fy = entry.get("foot_yaw_deg", {})
    sYL = math.radians(_yaw_deg_or_heading(fy, "start_L", shy))
    sYR = math.radians(_yaw_deg_or_heading(fy, "start_R", shy))
    gYL = math.radians(_yaw_deg_or_heading(fy, "goal_L", ghy))
    gYR = math.radians(_yaw_deg_or_heading(fy, "goal_R", ghy))

    return {
        "startpos_L": start_L,
        "startpos_R": start_R,
        "targetpos_L": goal_L,
        "targetpos_R": goal_R,
        "startyaw_L": sYL,
        "startyaw_R": sYR,
        "targetyaw_L": gYL,
        "targetyaw_R": gYR,
    }


def midfoot_from_two_feet(L, R) -> Pose2D:
    mx = 0.5 * (L[0] + R[0])
    my = 0.5 * (L[1] + R[1])
    cy = math.cos(L[2]) + math.cos(R[2])
    sy = math.sin(L[2]) + math.sin(R[2])
    myaw = math.atan2(sy, cy)
    return Pose2D(mx, my, myaw)


# ----------------------------
# Footstep extraction / transforms
# ----------------------------
def extract_footsteps_from_path(path: List[FootstepGraphNode]):
    steps = []
    first = path[0].get_first_step()
    second = path[0].get_second_step()
    steps.append((first.get_robot_side(), first.get_x(), first.get_y(), first.get_yaw()))
    steps.append((second.get_robot_side(), second.get_x(), second.get_y(), second.get_yaw()))
    for i in range(1, len(path)):
        s = path[i].get_second_step()
        steps.append((s.get_robot_side(), s.get_x(), s.get_y(), s.get_yaw()))
    return steps


def transform_poly(poly_xy, x: float, y: float, yaw: float) -> List[Tuple[float, float]]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return [(x + c * px - s * py, y + s * px + c * py) for (px, py) in poly_xy]


# ----------------------------
# 3D plotting helpers
# ----------------------------
def make_box_vertices(center_xyz, w: float, h: float, thickness: float, yaw_deg: float):
    cx, cy, cz = center_xyz
    w2, h2 = 0.5 * w, 0.5 * h
    top = np.array([[-w2, -h2, 0.0],
                    [ w2, -h2, 0.0],
                    [ w2,  h2, 0.0],
                    [-w2,  h2, 0.0]], dtype=float)
    th = np.deg2rad(float(yaw_deg))
    c, s = np.cos(th), np.sin(th)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]], dtype=float)
    top = (Rz @ top.T).T + np.array([cx, cy, cz])
    bottom = top.copy()
    bottom[:, 2] = cz - float(thickness)
    return np.vstack([top, bottom])  # (8,3)


def box_faces_from_vertices(V8: np.ndarray):
    top = [tuple(V8[i]) for i in range(0, 4)]
    bottom = [tuple(V8[i]) for i in range(4, 8)]
    sides = []
    for k in range(4):
        k2 = (k + 1) % 4
        sides.append([bottom[k], top[k], top[k2], bottom[k2]])
    return [top, bottom] + sides


def stone_facecolor(idx: int, n: int, centers_z: np.ndarray, face_alpha: float):
    if idx == 0:
        return (0.0, 0.7, 0.2, face_alpha)
    if idx == n - 1:
        return (0.9, 0.1, 0.1, face_alpha)

    if n >= 3:
        mid_z = centers_z[1:-1]
        z_min = float(mid_z.min())
        z_max = float(mid_z.max())
        if z_max > z_min:
            cmap = plt.get_cmap("gray")
            norm = Normalize(vmin=z_min - 0.01, vmax=z_max + 0.01)
            r, g, b, _ = cmap(norm(float(centers_z[idx])))
            return (r, g, b, face_alpha)

    return (0.6, 0.6, 0.6, face_alpha)


def find_support_stone_z(terrain: SteppingStoneTerrain,
                         foot_poly_world_xy: List[Tuple[float, float]],
                         default_z: float = 0.0) -> float:
    poly = ensure_ccw(foot_poly_world_xy)
    for stone in terrain.stones:
        stone_poly = ensure_ccw(stone.polygon_xy())
        if convex_polygon_contains_polygon(stone_poly, poly):
            return float(stone.center[2])

    fx = float(np.mean([p[0] for p in poly]))
    fy = float(np.mean([p[1] for p in poly]))
    best = None
    best_d = float("inf")
    for stone in terrain.stones:
        dx = float(stone.center[0]) - fx
        dy = float(stone.center[1]) - fy
        d = dx * dx + dy * dy
        if d < best_d:
            best_d = d
            best = stone
    return float(best.center[2]) if best is not None else float(default_z)


def print_footsteps_python(terrain: SteppingStoneTerrain, steps, foot_polys_local, run_name: str = ""):
    centers_z = np.array([float(s.center[2]) for s in terrain.stones], dtype=float) if terrain.stones else np.array([0.0])
    label = f"  # {run_name}" if run_name else ""
    print(f"\nFOOTSTEPS = [{label}")
    for side, x, y, yaw in steps:
        poly_xy = transform_poly(foot_polys_local[side], x, y, yaw)
        z = find_support_stone_z(terrain, poly_xy, default_z=float(centers_z.min()))
        side_char = side.name[0]
        yaw_deg = math.degrees(yaw)
        print(f"    ('{side_char}', {x:8.3f}, {y:8.3f}, {z:8.3f}, {yaw_deg:8.2f}),")
    print("]")


def plot_3d(terrain: SteppingStoneTerrain,
            steps,
            foot_polys_local,
            *,
            thickness: float = 0.04,
            pad_xy: float = 0.0,
            face_alpha: float = 0.35,
            save: Optional[str] = None,
            view_top: bool = False):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    stones = terrain.stones
    n = len(stones)
    centers_z = np.array([float(s.center[2]) for s in stones], dtype=float) if n else np.array([0.0])

    all_xyz = []

    # Stones as boxes
    for idx, s in enumerate(stones):
        cx, cy, cz = s.center
        w = float(s.width) + pad_xy
        h = float(s.height) + pad_xy
        yaw = float(getattr(s, "yaw_deg", 0.0))

        fc = stone_facecolor(idx, n, centers_z, face_alpha)
        V8 = make_box_vertices((cx, cy, cz), w, h, thickness, yaw)
        faces = box_faces_from_vertices(V8)
        poly = Poly3DCollection(faces, edgecolor="k", linewidths=0.6, zsort="min")
        poly.set_facecolor(fc)
        ax.add_collection3d(poly)
        all_xyz.append(V8)

    # Footsteps as wireframe on top surface z
    segs = []
    colors = []
    for k, (side, x, y, yaw) in enumerate(steps):
        poly_local = foot_polys_local[side]
        poly_xy = transform_poly(poly_local, x, y, yaw)
        z = find_support_stone_z(terrain, poly_xy, default_z=float(centers_z.min()))

        poly_3d = [(px, py, z) for (px, py) in poly_xy]
        poly_3d.append(poly_3d[0])

        for i in range(4):
            segs.append([poly_3d[i], poly_3d[i + 1]])
            colors.append("red" if side is RobotSide.LEFT else "blue")

        ax.plot([x, x + 0.10 * math.cos(yaw)], [y, y + 0.10 * math.sin(yaw)], [z, z], color="black", linewidth=1.0)
        ax.text(x, y, z + 0.01, str(k), fontsize=8)

        all_xyz.append(np.array(poly_3d[:4], dtype=float))

    ax.add_collection3d(Line3DCollection(segs, colors=colors, linewidths=1.8))

    # axis limits
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
    ax.set_title("A* footstep planner")
    ax.grid(False)

    ax.view_init(elev=90, azim=-90) if view_top else ax.view_init(elev=30, azim=-110)

    ax.legend(handles=[
        Patch(facecolor=(0.0, 0.7, 0.2), edgecolor='k', label='First stone'),
        Patch(facecolor=(0.9, 0.1, 0.1), edgecolor='k', label='Last stone'),
        Patch(facecolor="none", edgecolor="red", label="Left foot"),
        Patch(facecolor="none", edgecolor="blue", label="Right foot"),
    ], loc="upper right")

    if save:
        fig.savefig(save, dpi=220, bbox_inches="tight")
        print(f"Saved plot to: {save}")

    plt.show()


def solve_one(run_json: str, args, foot_polys, params):
    """
    run_json 하나에 대해 planning 수행하고 결과 요약을 dict로 반환.
    plot은 여기서 하지 않음(마지막에만 하려고).
    """
    terrain = SteppingStoneTerrain.from_run_json(run_json)

    stones_raw = [{"center": list(s.center), "name": s.name} for s in terrain.stones]
    entry = make_entry_from_stones(stones_raw, stance_width=args.stance_width)
    exp = expand_shorthand(entry, args.stance_width)

    startpos_L = exp["startpos_L"]; startpos_R = exp["startpos_R"]
    targetpos_L = exp["targetpos_L"]; targetpos_R = exp["targetpos_R"]
    startyaw_L = exp["startyaw_L"]; startyaw_R = exp["startyaw_R"]
    targetyaw_L = exp["targetyaw_L"]; targetyaw_R = exp["targetyaw_R"]

    goal_mid = midfoot_from_two_feet(
        (targetpos_L[0], targetpos_L[1], targetyaw_L),
        (targetpos_R[0], targetpos_R[1], targetyaw_R),
    )

    heuristic_calc = FootstepPlannerHeuristicCalculator(params)
    heuristic_calc.initialize(goal_mid)
    hfun = heuristic_calc.compute

    ideal = SimpleIdealStep(goal_mid.x, goal_mid.y, nominal_len=0.25, nominal_width=args.stance_width)
    expansion = ParameterBasedStepExpansion(params, foot_polys, ideal_step_calculator=ideal)
    expansion.initialize()

    print("num_cached_offsets =", len(expansion.x_offsets))
    from ihmc_footstep_planning.graph.lattice_point import LatticePoint
    print("grid_xy, grid_yaw =", LatticePoint.grid_size_xy, LatticePoint.grid_size_yaw)
    print("yaw_divisions =", LatticePoint.yaw_divisions)

    cost = FootstepCostCalculator(params, ideal_step_calculator=ideal, heuristics=hfun)

    checker = RegionFootstepChecker(regions_xy=terrain.polygons_xy(), foot_polygons_in_sole_frame=foot_polys)
    conductor = AStarFootstepPlannerIterationConductor(expansion, checker, cost, hfun)

    left = DiscreteFootstep(startpos_L[0], startpos_L[1], startyaw_L, RobotSide.LEFT)
    right = DiscreteFootstep(startpos_R[0], startpos_R[1], startyaw_R, RobotSide.RIGHT)
    start = FootstepGraphNode(first_step=left, second_step=right)
    conductor.initialize(start)

    # -------------------------
    # Planning loop + stats
    # -------------------------
    t0 = time.perf_counter()
    deadline = t0 + float(getattr(args, "time_limit_sec", 1e18))  # ★ 추가
    node = conductor.get_next_node()

    it = 0
    expanded_iters = 0
    best = start

    total_gen = 0
    total_valid = 0
    total_invalid = 0

    print_every = getattr(args, "print_every", 100)
    last_print = 0

    reached = False
    reached_iter = None

    tol_xy = float(getattr(args, "xy_tol", 0.05))  # m
    tol_yaw = math.radians(float(getattr(args, "yaw_tol_deg", 0.1)))  # rad

    goal_L_xy = (targetpos_L[0], targetpos_L[1])
    goal_R_xy = (targetpos_R[0], targetpos_R[1])

    goal_L_yaw = float(targetyaw_L)
    goal_R_yaw = float(targetyaw_R)

    def within_xy(step, goal_xy):
        return math.hypot(step.get_x() - goal_xy[0], step.get_y() - goal_xy[1]) <= tol_xy

    def within_yaw(step, goal_yaw):
        # wrap_to_pi는 이미 import 되어 있음: from ihmc_footstep_planning.geometry import ... wrap_to_pi ...
        err = wrap_to_pi(step.get_yaw() - goal_yaw)
        return abs(err) <= tol_yaw

    def get_graph_nodes_count():
        gnodes = None
        if hasattr(conductor, "graph"):
            g = conductor.graph
            if hasattr(g, "came_from"):
                gnodes = len(g.came_from)
            elif hasattr(g, "g_score"):
                gnodes = len(g.g_score)
            elif hasattr(g, "nodes"):
                gnodes = len(g.nodes)
        return gnodes

    stop_reason = None  # ★ 추가: 왜 멈췄는지

    # ★ while 조건에 시간 조건 추가
    while node is not None and it < args.max_iter and time.perf_counter() < deadline:
        iter_data = conductor.do_planning_iteration(node, perform_iterative_expansion=False)

        valid_now = len(iter_data.get_valid_child_nodes())
        invalid_now = len(iter_data.get_invalid_child_nodes())
        generated_now = valid_now + invalid_now

        total_gen += generated_now
        total_valid += valid_now
        total_invalid += invalid_now

        expanded_iters += 1

        if expanded_iters - last_print >= print_every:
            last_print = expanded_iters
            elapsed = time.perf_counter() - t0
            rej_pct = 100.0 * (total_invalid / total_gen) if total_gen > 0 else 0.0
            graph_nodes = get_graph_nodes_count()

            remaining = max(0.0, deadline - time.perf_counter())
            print(
                f"[{Path(run_json).name}] [iter={expanded_iters:5d}] elapsed={elapsed:7.3f}s (remain={remaining:5.2f}s)  "
                f"gen={generated_now:4d} valid={valid_now:4d} invalid={invalid_now:4d}  "
                f"tot_gen={total_gen} tot_invalid={total_invalid} (rej={rej_pct:.1f}%)  "
                f"graph_nodes={graph_nodes}"
            )

        # goal check
        s1 = node.get_first_step()
        s2 = node.get_second_step()

        if s1.get_robot_side().name == "LEFT":
            left_ok = within_xy(s1, goal_L_xy) and within_yaw(s1, goal_L_yaw)
            right_ok = within_xy(s2, goal_R_xy) and within_yaw(s2, goal_R_yaw)
        else:
            right_ok = within_xy(s1, goal_R_xy) and within_yaw(s1, goal_R_yaw)
            left_ok = within_xy(s2, goal_L_xy) and within_yaw(s2, goal_L_yaw)

        if left_ok and right_ok:
            best = node
            reached = True
            reached_iter = expanded_iters  # ★ expanded_iters 기준이 더 직관적
            print(f"[{Path(run_json).name}] Reached BOTH feet targets within {tol_xy*100:.1f} cm at iter={expanded_iters}")
            stop_reason = "reached"
            break

        # next node
        node = conductor.get_next_node()
        it += 1

    # ★ 루프 종료 이유 기록
    if stop_reason is None:
        if node is None:
            stop_reason = "open_list_empty"
        elif it >= args.max_iter:
            stop_reason = "max_iter"
        elif time.perf_counter() >= deadline:
            stop_reason = "timeout"
        else:
            stop_reason = "unknown"

    # -------------------------
    # End + summary
    # -------------------------
    elapsed = time.perf_counter() - t0
    graph_nodes = get_graph_nodes_count()

    rej_pct = 100.0 * (total_invalid / total_gen) if total_gen > 0 else 0.0
    print(
        f"[{Path(run_json).name}] child stats: total_gen={total_gen}, "
        f"total_valid={total_valid}, total_invalid={total_invalid} (rej={rej_pct:.1f}%)"
    )

    # 경로/step은 성공/실패와 상관없이 best 기준으로 뽑되,
    # 실패면 best가 start일 수도 있음
    path = conductor.graph.reconstruct_path(best)
    steps = extract_footsteps_from_path(path)

    return {
        "run_json": run_json,
        "ok": reached,
        "stop_reason": stop_reason,
        "reached_iter": reached_iter,
        "expanded_iters": expanded_iters,
        "graph_nodes": graph_nodes,
        "elapsed_sec": elapsed,
        # 추가로 통계도 반환하면 나중에 비교 실험에 유용
        "child_total_gen": total_gen,
        "child_total_valid": total_valid,
        "child_total_invalid": total_invalid,
        "child_reject_pct": rej_pct,
        "terrain": terrain,
        "steps": steps,
    }

def append_result_csv(csv_path: str, row: dict, fieldnames: list):
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    write_header = (not p.exists()) or (p.stat().st_size == 0)
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)

def main():

    # --------- fixed config (no argparse) ----------
    data_dir = Path("data/terrains/alpha/0.0")
    pattern = "run_*.json"

    result_dir = Path("outputs/astar")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 예: 20260211_101234_123456

    args = SimpleNamespace(
        data_dir=str(data_dir),
        pattern=pattern,
        stance_width=0.20,
        xy_tol=0.10,
        yaw_tol_deg=10.0,
        max_iter=50000,
        time_limit_sec=5.0,
        save="",           # 저장 안 하면 "" 유지
        view_top=False,
        thickness=0.04,
        pad_xy=0.0,
        # solve_one에서 로그 주기 조절하고 싶으면 사용
        print_every=100,
        # ★ 추가
        results_csv=str(result_dir / f"graph_based_{stamp}.csv"),
        alpha=0.0,
    )
    # ---------------------------------------------


    params = DefaultFootstepPlannerParameters()

    foot_poly = [(-0.075, -0.035), (-0.075, 0.035), (0.075, 0.035), (0.075, -0.035)]
    foot_polys = {RobotSide.LEFT: foot_poly, RobotSide.RIGHT: foot_poly}

    run_files = sorted(Path(args.data_dir).glob(args.pattern))
    if not run_files:
        raise FileNotFoundError(f"No json files matched: {Path(args.data_dir) / args.pattern}")

    results = []
    fieldnames = [
        "planner", "alpha", "t_max_sec", "run_json",
        "ok", "stop_reason", "time_sec",
        "expanded_iters", "graph_nodes",
        "child_total_gen", "child_total_valid", "child_total_invalid", "child_reject_pct",
    ]

    for rf in run_files:
        print("\n" + "=" * 80)
        print(f"Running: {rf.name}")

        res = solve_one(str(rf), args, foot_polys, params)
        # ★ CSV로 한 줄 저장
        row = {
            "planner": "Graph-based",
            "alpha": float(getattr(args, "alpha", 0.0)),
            "t_max_sec": float(getattr(args, "time_limit_sec", 0.0)),
            "run_json": Path(res["run_json"]).name,
            "ok": int(bool(res["ok"])),
            "stop_reason": res.get("stop_reason", ""),
            "time_sec": float(res["elapsed_sec"]),   # <- Ours의 time_sec랑 맞추면 좋음
            "expanded_iters": int(res.get("expanded_iters", 0)),
            "graph_nodes": ("" if res.get("graph_nodes", None) is None else int(res["graph_nodes"])),
            "child_total_gen": int(res.get("child_total_gen", 0)),
            "child_total_valid": int(res.get("child_total_valid", 0)),
            "child_total_invalid": int(res.get("child_total_invalid", 0)),
            "child_reject_pct": float(res.get("child_reject_pct", 0.0)),
        }
        append_result_csv(args.results_csv, row, fieldnames)

        results.append(res)

        print(f"=== Planning summary [{rf.name}] ===")
        print(f"ok              = {res['ok']}")
        print(f"expanded_iters  = {res['expanded_iters']}")
        print(f"graph_nodes     = {res['graph_nodes']}")
        print(f"elapsed_sec     = {res['elapsed_sec']:.6f}")
        # 추가: child 통계도 같이 보면 좋음
        print(f"child_reject(%) = {res.get('child_reject_pct', 0.0):.1f}")
        print(f"child_total_gen = {res.get('child_total_gen', 0)}")

    # ===== 전체 요약 출력 (원래 너가 쓰던 Summary 복구) =====
    ok_cnt = sum(1 for r in results if r["ok"])
    total = len(results)

    print("\n================ Summary ================")
    print(f"OK: {ok_cnt}/{total}")

    for r in results:
        name = Path(r["run_json"]).name
        if r["ok"]:
            expanded = r["expanded_iters"]
            graph_nodes = r["graph_nodes"]
            t = r["elapsed_sec"]
            rej = r.get("child_reject_pct", 0.0)
            print(f"[OK]   {name} | expanded_iters={expanded} graph_nodes={graph_nodes} time={t:.3f}s reject={rej:.1f}%")
        else:
            t = r["elapsed_sec"]
            rej = r.get("child_reject_pct", 0.0)
            print(f"[FAIL] {name} | No solution (did not reach goal within max_iter). time={t:.3f}s reject={rej:.1f}%")

    # ===== 마지막 성공 케이스만 plot =====
    last_success = None
    for r in reversed(results):
        if r["ok"]:
            last_success = r
            break

    if last_success is None:
        print("\nNo successful run to plot.")
        return

    run_name = Path(last_success["run_json"]).name
    print_footsteps_python(last_success["terrain"], last_success["steps"], foot_polys, run_name)

    save_path = args.save if str(args.save).strip() else None
    plot_3d(
        last_success["terrain"], last_success["steps"], foot_polys,
        thickness=args.thickness,
        pad_xy=args.pad_xy,
        save=save_path,
        view_top=args.view_top
    )

if __name__ == "__main__":
    main()
