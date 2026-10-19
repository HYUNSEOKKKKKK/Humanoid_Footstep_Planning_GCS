
from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, Any, Tuple

from ihmc_footstep_planning.parameters import DefaultFootstepPlannerParameters
from ihmc_footstep_planning.graph_search.footstep_planner_heuristic_calculator import FootstepPlannerHeuristicCalculator
from ihmc_footstep_planning.robot_side import RobotSide
from ihmc_footstep_planning.graph import DiscreteFootstep, FootstepGraphNode
from ihmc_footstep_planning.step_expansion import ParameterBasedStepExpansion
from ihmc_footstep_planning.step_cost import FootstepCostCalculator
from ihmc_footstep_planning.graph_search import AStarFootstepPlannerIterationConductor
from ihmc_footstep_planning.geometry import Pose2D, wrap_to_pi
from ihmc_footstep_planning.environments.stepping_stone_terrain import SteppingStoneTerrain
from ihmc_footstep_planning.step_checking.region_footstep_checker import RegionFootstepChecker


class SimpleIdealStep:
    def __init__(self, goal_x: float, goal_y: float, nominal_len: float = 0.25, nominal_width: float = 0.20):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.nominal_len = nominal_len
        self.nominal_width = nominal_width

    def compute_ideal_step_for_node(self, stance_node) -> DiscreteFootstep:
        # stance_node: FootstepGraphNode (duck-typed)
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
    # left unit vector
    lx = -math.sin(yaw)
    ly =  math.cos(yaw)
    half = 0.5 * float(stance_width)
    left = (cx + half * lx, cy + half * ly)
    right = (cx - half * lx, cy - half * ly)
    return left, right

def make_entry_from_stones(stones, stance_width: float) -> Dict[str, Any]:
    sc = stones[0]["center"]
    gc = stones[-1]["center"]
    if len(stones) >= 2:
        shy = heading_deg_x_forward(stones[0]["center"], stones[1]["center"])
        ghy = heading_deg_x_forward(stones[-2]["center"], stones[-1]["center"])
    else:
        shy = ghy = 0.0
    return {
        "start": {"center": sc, "heading_deg": shy},
        "goal":  {"center": gc, "heading_deg": ghy},
        "stance_width": float(stance_width),
        # 0.0 means "use heading" (★ 0.0 -> heading)
        "foot_yaw_deg": {"start_L": 0.0, "start_R": 0.0, "goal_L": 0.0, "goal_R": 0.0},
    }

def _yaw_deg_or_heading(fy: Dict[str, Any], key: str, heading_deg: float) -> float:
    if key not in fy:
        return heading_deg
    v = float(fy[key])
    if abs(v) < 1e-12:
        return heading_deg
    return v

def expand_shorthand(entry: Dict[str, Any], default_stance_width: float) -> Dict[str, Any]:
    w = float(entry.get("stance_width", default_stance_width))

    sc = entry["start"]["center"];  shy = float(entry["start"]["heading_deg"])
    gc = entry["goal"]["center"];   ghy = float(entry["goal"]["heading_deg"])

    start_L, start_R = feet_from_center(sc, shy, w)
    goal_L,  goal_R  = feet_from_center(gc, ghy, w)

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
        "start_heading_deg": shy,
        "goal_heading_deg": ghy,
    }

def midfoot_from_two_feet(L: Tuple[float, float, float], R: Tuple[float, float, float]) -> Pose2D:
    mx = 0.5 * (L[0] + R[0])
    my = 0.5 * (L[1] + R[1])
    cy = math.cos(L[2]) + math.cos(R[2])
    sy = math.sin(L[2]) + math.sin(R[2])
    myaw = math.atan2(sy, cy)
    return Pose2D(mx, my, myaw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_json", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "run_000.json"))
    ap.add_argument("--stance_width", type=float, default=0.20)
    ap.add_argument("--xy_tol", type=float, default=0.20)
    ap.add_argument("--yaw_tol_deg", type=float, default=15.0)
    ap.add_argument("--max_iter", type=int, default=2000)
    args = ap.parse_args()

    params = DefaultFootstepPlannerParameters()

    foot_poly = [(-0.10, -0.04), (-0.10, 0.04), (0.10, 0.04), (0.10, -0.04)]
    foot_polys = {RobotSide.LEFT: foot_poly, RobotSide.RIGHT: foot_poly}

    terrain = SteppingStoneTerrain.from_run_json(args.run_json)

    stones_raw = [{"center": list(s.center), "name": s.name} for s in terrain.stones]
    entry = make_entry_from_stones(stones_raw, stance_width=args.stance_width)
    exp = expand_shorthand(entry, args.stance_width)

    startpos_L = exp["startpos_L"]; startpos_R = exp["startpos_R"]
    targetpos_L = exp["targetpos_L"]; targetpos_R = exp["targetpos_R"]
    startyaw_L = exp["startyaw_L"]; startyaw_R = exp["startyaw_R"]
    targetyaw_L = exp["targetyaw_L"]; targetyaw_R = exp["targetyaw_R"]

    print(f"Loaded terrain: #stones={len(terrain.stones)} from {args.run_json}")

    goal_mid = midfoot_from_two_feet((targetpos_L[0], targetpos_L[1], targetyaw_L),
                                     (targetpos_R[0], targetpos_R[1], targetyaw_R))

    heuristic_calc = FootstepPlannerHeuristicCalculator(params)
    heuristic_calc.initialize(goal_mid)
    hfun = heuristic_calc.compute

    ideal = SimpleIdealStep(goal_mid.x, goal_mid.y, nominal_len=0.25, nominal_width=args.stance_width)
    expansion = ParameterBasedStepExpansion(params, foot_polys, ideal_step_calculator=ideal)
    expansion.initialize()
    cost = FootstepCostCalculator(params, ideal_step_calculator=ideal, heuristics=hfun)

    checker = RegionFootstepChecker(regions_xy=terrain.polygons_xy(), foot_polygons_in_sole_frame=foot_polys)

    conductor = AStarFootstepPlannerIterationConductor(expansion, checker, cost, hfun)

    left = DiscreteFootstep(startpos_L[0], startpos_L[1], startyaw_L, RobotSide.LEFT)
    right = DiscreteFootstep(startpos_R[0], startpos_R[1], startyaw_R, RobotSide.RIGHT)
    start = FootstepGraphNode(first_step=left, second_step=right)

    conductor.initialize(start)

    yaw_tol = math.radians(args.yaw_tol_deg)
    node = conductor.get_next_node()
    it = 0
    best = start
    best_d = float("inf")

    while node is not None and it < args.max_iter:
        conductor.do_planning_iteration(node, perform_iterative_expansion=False)
        pose = node.get_or_compute_mid_foot_pose()

        dxy = math.hypot(goal_mid.x - pose.x, goal_mid.y - pose.y)
        dyaw = abs(wrap_to_pi(goal_mid.yaw - pose.yaw))
        if dxy < best_d:
            best_d = dxy
            best = node

        if dxy < args.xy_tol and dyaw < yaw_tol:
            best = node
            print(f"Reached goal(midfoot) at iter={it}, dxy={dxy:.3f}, dyaw_deg={math.degrees(dyaw):.2f}")
            break

        node = conductor.get_next_node()
        it += 1

    path = conductor.graph.reconstruct_path(best)
    print(f"Finished: iters={it}, path_len={len(path)}, best_dxy={best_d:.3f}")
    for i, n in enumerate(path):
        p = n.get_or_compute_mid_foot_pose()
        print(i, p)

if __name__ == "__main__":
    main()
