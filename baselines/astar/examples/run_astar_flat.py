from __future__ import annotations
import math
from typing import List

from ihmc_footstep_planning.parameters import DefaultFootstepPlannerParameters
from ihmc_footstep_planning.graph_search.footstep_planner_heuristic_calculator import FootstepPlannerHeuristicCalculator
from ihmc_footstep_planning.geometry import Pose2D
from ihmc_footstep_planning.robot_side import RobotSide
from ihmc_footstep_planning.graph import DiscreteFootstep, FootstepGraphNode
from ihmc_footstep_planning.step_expansion import ParameterBasedStepExpansion
from ihmc_footstep_planning.step_cost import FootstepCostCalculator
from ihmc_footstep_planning.graph_search import AStarFootstepPlannerIterationConductor

# --- Minimal 'Ideal step' for demo: aim toward goal in stance frame
class SimpleIdealStep:
    def __init__(self, goal_x: float, goal_y: float, nominal_len: float = 0.25, nominal_width: float = 0.20):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.nominal_len = nominal_len
        self.nominal_width = nominal_width

    def compute_ideal_step(self, stance_step: DiscreteFootstep, start_of_swing: DiscreteFootstep) -> DiscreteFootstep:
        # Move nominal_len toward goal direction, keep nominal_width lateral by side.
        dx = self.goal_x - stance_step.get_x()
        dy = self.goal_y - stance_step.get_y()
        yaw_to_goal = math.atan2(dy, dx)
        # Put the foot ahead in the heading direction, with lateral offset depending on side
        # This is NOT IHMC's exact ideal step, but is a reasonable Step 0 placeholder.
        side = stance_step.get_robot_side().get_opposite_side()
        width = nominal_width = self.nominal_width
        width = -width if side is RobotSide.RIGHT else width
        x = stance_step.get_x() + self.nominal_len * math.cos(yaw_to_goal) - width * math.sin(yaw_to_goal)
        y = stance_step.get_y() + self.nominal_len * math.sin(yaw_to_goal) + width * math.cos(yaw_to_goal)
        return DiscreteFootstep(x, y, yaw_to_goal, side)

# --- Always-valid checker for demo (no collision / terrain yet)
class AlwaysValidChecker:
    def is_step_valid(self, candidate_step, stance_step, start_of_swing) -> bool:
        return True


def main():
    params = DefaultFootstepPlannerParameters()
    # Simple rectangles as foot polygons (sole frame), meters
    foot_poly = [(-0.10, -0.04), (-0.10, 0.04), (0.10, 0.04), (0.10, -0.04)]
    foot_polys = {RobotSide.LEFT: foot_poly, RobotSide.RIGHT: foot_poly}

    goal_x, goal_y = 2.0, 0.0

    expansion = ParameterBasedStepExpansion(params, foot_polys, ideal_step_calculator=None)
    expansion.initialize()

    ideal = SimpleIdealStep(goal_x, goal_y)
    heuristic_calc = FootstepPlannerHeuristicCalculator(params)
    heuristic_calc.initialize(Pose2D(goal_x, goal_y, 0.0))
    hfun = heuristic_calc.compute
    cost = FootstepCostCalculator(params, ideal_step_calculator=ideal, heuristics=hfun)

    conductor = AStarFootstepPlannerIterationConductor(expansion, AlwaysValidChecker(), cost, hfun)

    # initial stance: left at (0,0), right at (0, -0.2)
    left = DiscreteFootstep(0.0, 0.10, 0.0, RobotSide.LEFT)
    right = DiscreteFootstep(0.0, -0.10, 0.0, RobotSide.RIGHT)
    start = FootstepGraphNode(first_step=left, second_step=right)

    conductor.initialize(start)

    # Run a few iterations
    node = conductor.get_next_node()
    it = 0
    while node is not None and it < 200:
        conductor.do_planning_iteration(node, perform_iterative_expansion=False)
        # naive stop: midfoot within 0.2m
        pose = node.get_or_compute_mid_foot_pose()
        if math.hypot(goal_x - pose.x, goal_y - pose.y) < 0.2:
            path = conductor.graph.reconstruct_path(node)
            print(f"Reached near goal in {it} iterations, path length={len(path)}")
            for k, n in enumerate(path):
                p = n.get_or_compute_mid_foot_pose()
                print(k, p)
            return
        node = conductor.get_next_node()
        it += 1

    print("Did not reach goal within iteration limit.")

if __name__ == "__main__":
    main()
