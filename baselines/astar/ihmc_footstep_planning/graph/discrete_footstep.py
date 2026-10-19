from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from ..robot_side import RobotSide
from .lattice_point import LatticePoint

@dataclass(frozen=True)
class DiscreteFootstep:
    lattice_point: LatticePoint
    robot_side: RobotSide
    # In the Java code this stores snapping data; here we keep it generic.
    snap_data: Optional[object] = None

    def __init__(self, x: float | int, y: float | int, yaw: float | int = 0.0, robot_side: RobotSide = RobotSide.LEFT):
        if isinstance(x, LatticePoint):
            lp = x
            rs = y  # type: ignore
            object.__setattr__(self, "lattice_point", lp)
            object.__setattr__(self, "robot_side", rs)
            object.__setattr__(self, "snap_data", None)
        elif isinstance(x, int) and isinstance(y, int) and isinstance(yaw, int):
            object.__setattr__(self, "lattice_point", LatticePoint(x, y, yaw))
            object.__setattr__(self, "robot_side", robot_side)
            object.__setattr__(self, "snap_data", None)
        else:
            object.__setattr__(self, "lattice_point", LatticePoint(float(x), float(y), float(yaw)))
            object.__setattr__(self, "robot_side", robot_side)
            object.__setattr__(self, "snap_data", None)

    def set_snap_data(self, snap_data: object) -> "DiscreteFootstep":
        # Return a new instance (immutability-friendly)
        fs = DiscreteFootstep(self.lattice_point, self.robot_side)
        object.__setattr__(fs, "snap_data", snap_data)
        return fs

    def clear_snap_data(self) -> "DiscreteFootstep":
        return DiscreteFootstep(self.lattice_point, self.robot_side)

    def has_snap_data(self) -> bool:
        return self.snap_data is not None

    def get_snap_data(self):
        return self.snap_data

    def get_x(self) -> float:
        return self.lattice_point.get_x()

    def get_y(self) -> float:
        return self.lattice_point.get_y()

    def get_yaw(self) -> float:
        return self.lattice_point.get_yaw()

    def get_x_index(self) -> int:
        return self.lattice_point.get_x_index()

    def get_y_index(self) -> int:
        return self.lattice_point.get_y_index()

    def get_yaw_index(self) -> int:
        return self.lattice_point.get_yaw_index()

    def get_robot_side(self) -> RobotSide:
        return self.robot_side

    def compute_yaw_index_distance(self, other: "DiscreteFootstep") -> int:
        d = abs(self.get_yaw_index() - other.get_yaw_index())
        return min(d, LatticePoint.yaw_divisions - d)

    def compute_xy_manhattan_distance(self, other: "DiscreteFootstep") -> int:
        return abs(self.get_x_index() - other.get_x_index()) + abs(self.get_y_index() - other.get_y_index())

    def compute_manhattan_distance(self, other: "DiscreteFootstep") -> int:
        return self.compute_xy_manhattan_distance(other) + self.compute_yaw_index_distance(other)

    def equal_position(self, other: "DiscreteFootstep") -> bool:
        return (self.get_x_index() == other.get_x_index()) and (self.get_y_index() == other.get_y_index())

    def __str__(self) -> str:
        return f"Node: x={self.get_x():.3f}, y={self.get_y():.3f}, yaw={self.get_yaw():.3f}, side={self.robot_side.get_lower_case_name()}"
