from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol
import math

from ..parameters import DefaultFootstepPlannerParameters
from ..geometry import Pose2D, wrap_to_pi
from ..graph import FootstepGraphNode

def angle_diff_minus_pi_to_pi(a: float, b: float) -> float:
    """Equivalent of IHMC AngleTools.computeAngleDifferenceMinusPiToPi(a, b)."""
    return wrap_to_pi(a - b)

class BodyPathPlanHolderLike(Protocol):
    """Minimal protocol matching what FootstepPlannerHeuristicCalculator needs.

    The real IHMC type is WaypointDefinedBodyPathPlanHolder, but that's outside Step 0.
    If you later port it, implement these methods (or adapt in a small wrapper).
    """
    def get_closest_point_alpha(self, x: float, y: float) -> float: ...
    def get_segment_index_from_alpha(self, alpha: float) -> int: ...
    def get_waypoint_yaw(self, segment_index: int) -> float: ...
    def get_final_waypoint_yaw(self) -> float: ...

@dataclass
class FootstepPlannerHeuristicCalculator:
    """Port of us.ihmc.footstepPlanning.graphSearch.FootstepPlannerHeuristicCalculator.

    In IHMC, this heuristic blends:
      - initial turn distance (align to body-path posture heading)
      - walk distance (xy to goal)
      - final turn distance (align to goal yaw or final path yaw)

    Step 0.5 note:
      - We support an optional body path plan holder.
      - If it is not provided, we fall back to headings computed from the goal direction.
    """
    parameters: DefaultFootstepPlannerParameters
    body_path_plan_holder: Optional[BodyPathPlanHolderLike] = None

    _goal_pose: Pose2D = Pose2D()

    def initialize(self, goal_pose: Pose2D) -> None:
        self._goal_pose = Pose2D(goal_pose.x, goal_pose.y, goal_pose.yaw)

    def compute(self, node: FootstepGraphNode) -> float:
        mid = node.get_or_compute_mid_foot_pose()
        dx = mid.x - self._goal_pose.x
        dy = mid.y - self._goal_pose.y
        xy_dist = math.hypot(dx, dy)

        initial_turn = 0.0
        walk_dist = 0.0

        # arc length per radian: 0.5*pi*w  (same as Java)
        turn_scale = 0.5 * math.pi * self.parameters.getIdealFootstepWidth()

        if xy_dist < self.parameters.getFinalTurnProximity():
            final_turn = abs(angle_diff_minus_pi_to_pi(mid.yaw, self._goal_pose.yaw)) * turn_scale
        else:
            if self.body_path_plan_holder is not None:
                alpha = self.body_path_plan_holder.get_closest_point_alpha(mid.x, mid.y)
                seg = self.body_path_plan_holder.get_segment_index_from_alpha(alpha)
                desired_heading = self.body_path_plan_holder.get_waypoint_yaw(seg)
                final_heading = self.body_path_plan_holder.get_final_waypoint_yaw()
            else:
                # Fallback: use direction-to-goal as the desired posture heading
                desired_heading = math.atan2(self._goal_pose.y - mid.y, self._goal_pose.x - mid.x)
                final_heading = desired_heading

            initial_turn = abs(angle_diff_minus_pi_to_pi(mid.yaw, desired_heading)) * turn_scale
            walk_dist = xy_dist
            final_turn = abs(angle_diff_minus_pi_to_pi(final_heading, self._goal_pose.yaw)) * turn_scale

        return self.parameters.getAStarHeuristicsWeight() * (initial_turn + walk_dist + final_turn)
