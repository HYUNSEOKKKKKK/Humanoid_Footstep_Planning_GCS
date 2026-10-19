from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional
import math

from ..geometry import Transform2D, polygon_transform, distance_between_convex_polygons
from ..robot_side import RobotSide
from .discrete_footstep import DiscreteFootstep

# In IHMC this is RigidBodyTransform (SE3). For Step 0 we keep a small subset:
@dataclass
class SnapTransform:
    # Minimal SE(3) representation used by the cost calculator:
    # - translation (x,y,z)
    # - orientation in yaw/pitch/roll
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

def get_step_transform(step: DiscreteFootstep) -> Transform2D:
    return Transform2D(step.get_x(), step.get_y(), step.get_yaw())

def get_snapped_step_transform(step: DiscreteFootstep, snap_transform: Optional[SnapTransform] = None) -> SnapTransform:
    # If no snap transform is given, treat as flat/unsnapped.
    if snap_transform is None:
        return SnapTransform(step.get_x(), step.get_y(), 0.0, step.get_yaw(), 0.0, 0.0)
    # Compose: snap_transform is defined in world in IHMC logs; we assume it already encodes final pose.
    return snap_transform

def get_foot_polygon_in_world(step: DiscreteFootstep, polygon_in_sole_frame: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    # Sole frame polygon -> world via step pose (x,y,yaw)
    T = get_step_transform(step)
    return polygon_transform(polygon_in_sole_frame, T)

def compute_distance_between_foot_polygons(stepA: DiscreteFootstep,
                                          stepB: DiscreteFootstep,
                                          foot_polygons_in_sole_frame: Dict[RobotSide, List[Tuple[float,float]]]) -> float:
    polyA = get_foot_polygon_in_world(stepA, foot_polygons_in_sole_frame[stepA.get_robot_side()])
    polyB = get_foot_polygon_in_world(stepB, foot_polygons_in_sole_frame[stepB.get_robot_side()])
    return distance_between_convex_polygons(polyA, polyB)

def construct_step_in_previous_step_frame(step_length: float, step_width: float, step_yaw: float, previous_step: DiscreteFootstep) -> DiscreteFootstep:
    # Equivalent to ParameterBasedStepExpansion.constructNodeInPreviousNodeFrame()
    c = math.cos(previous_step.get_yaw())
    s = math.sin(previous_step.get_yaw())
    # rotate (step_length, step_width) by previous yaw
    dx = c*step_length - s*step_width
    dy = s*step_length + c*step_width
    return DiscreteFootstep(previous_step.get_x() + dx,
                           previous_step.get_y() + dy,
                           step_yaw + previous_step.get_yaw(),
                           previous_step.get_robot_side().get_opposite_side())
