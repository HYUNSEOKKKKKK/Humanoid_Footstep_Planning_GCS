from __future__ import annotations
from enum import Enum

class RobotSide(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    def get_opposite_side(self) -> "RobotSide":
        return RobotSide.RIGHT if self is RobotSide.LEFT else RobotSide.LEFT

    def negate_if_right_side(self, value: float) -> float:
        return -value if self is RobotSide.RIGHT else value

    def get_lower_case_name(self) -> str:
        return "left" if self is RobotSide.LEFT else "right"
