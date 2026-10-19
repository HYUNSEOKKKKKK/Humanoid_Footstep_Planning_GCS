from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class LatticePoint:
    # Mirrors IHMC defaults in LatticePoint.java
    grid_size_xy: float = 0.05   #5cm (논문 동일)
    yaw_divisions: int = 36    #10도 (논문 은 10도 )
    grid_size_yaw: float = 2.0 * math.pi / yaw_divisions

    x_index: int = 0
    y_index: int = 0
    yaw_index: int = 0

    def __init__(self, x: float | int, y: float | int, yaw: float | int = 0):
        # Overload:
        # - If x,y,yaw are ints => indices
        # - Else floats => quantize into indices
        if isinstance(x, int) and isinstance(y, int) and isinstance(yaw, int):
            object.__setattr__(self, "x_index", x)
            object.__setattr__(self, "y_index", y)
            object.__setattr__(self, "yaw_index", self._wrap_yaw_index(yaw))
        else:
            xi = int(round(float(x) / self.grid_size_xy))
            yi = int(round(float(y) / self.grid_size_xy))
            yiaw = self._quantize_yaw(float(yaw))
            object.__setattr__(self, "x_index", xi)
            object.__setattr__(self, "y_index", yi)
            object.__setattr__(self, "yaw_index", yiaw)

    @classmethod
    def _wrap_yaw_index(cls, idx: int) -> int:
        return idx % cls.yaw_divisions

    @classmethod
    def _quantize_yaw(cls, yaw: float) -> int:
        # map to [-pi,pi), then quantize to divisions
        a = (yaw + math.pi) % (2.0 * math.pi) - math.pi
        idx = int(round(a / cls.grid_size_yaw))
        return cls._wrap_yaw_index(idx)

    def get_x_index(self) -> int:
        return self.x_index

    def get_y_index(self) -> int:
        return self.y_index

    def get_yaw_index(self) -> int:
        return self.yaw_index

    def get_x(self) -> float:
        return self.grid_size_xy * self.x_index

    def get_y(self) -> float:
        return self.grid_size_xy * self.y_index

    def get_yaw(self) -> float:
        return self.grid_size_yaw * self.yaw_index
