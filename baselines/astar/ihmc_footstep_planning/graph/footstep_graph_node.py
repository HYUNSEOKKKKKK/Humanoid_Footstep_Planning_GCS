from __future__ import annotations
from dataclasses import dataclass
from ..geometry import Pose2D, interpolate, interpolate_angle, wrap_to_pi
from ..robot_side import RobotSide
from .discrete_footstep import DiscreteFootstep

@dataclass(frozen=True)
class FootstepGraphNode:
    # Mirrors FootstepGraphNode.java:
    # - firstStep: start-of-swing step at this node
    # - secondStep: stance step at this node (child nodes' firstStep equals this)
    first_step: DiscreteFootstep
    second_step: DiscreteFootstep

    def __post_init__(self):
        if self.first_step.get_robot_side() == self.second_step.get_robot_side():
            raise ValueError("FootstepGraphNode given two steps on the same side!")

    def get_first_step(self) -> DiscreteFootstep:
        return self.first_step

    def get_second_step(self) -> DiscreteFootstep:
        return self.second_step

    def get_first_step_side(self) -> RobotSide:
        return self.first_step.get_robot_side()

    def get_second_step_side(self) -> RobotSide:
        return self.second_step.get_robot_side()

    def get_or_compute_mid_foot_pose(self) -> Pose2D:
        # Mid foot pose is derived on demand in Java and cached; here it's cheap, we compute each time.
        x = interpolate(self.second_step.get_x(), self.first_step.get_x(), 0.5)
        y = interpolate(self.second_step.get_y(), self.first_step.get_y(), 0.5)
        yaw = interpolate_angle(self.second_step.get_yaw(), self.first_step.get_yaw(), 0.5)
        return Pose2D(x, y, yaw)

    def get_stance_angle(self) -> float:
        return abs(wrap_to_pi(self.first_step.get_yaw() - self.second_step.get_yaw()))

    def __str__(self) -> str:
        return f"Footstep graph node:\n\t First {self.first_step}\n\t Second {self.second_step}"
