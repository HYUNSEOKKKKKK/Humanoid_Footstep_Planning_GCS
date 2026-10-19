
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import json
import math

from ..geometry import ensure_ccw

@dataclass
class Stone:
    name: str
    center: Tuple[float, float, float]
    width: float
    height: float
    yaw_deg: float

    def polygon_xy(self) -> List[Tuple[float, float]]:
        cx, cy, _ = self.center
        hx = 0.5 * float(self.width)
        hy = 0.5 * float(self.height)

        # local rectangle corners (CCW)
        local = [(-hx, -hy), (-hx, hy), (hx, hy), (hx, -hy)]

        yaw = math.radians(float(self.yaw_deg))
        c = math.cos(yaw); s = math.sin(yaw)

        world = []
        for x, y in local:
            wx = cx + c * x - s * y
            wy = cy + s * x + c * y
            world.append((wx, wy))

        return ensure_ccw(world)

@dataclass
class SteppingStoneTerrain:
    stones: List[Stone]

    @staticmethod
    def from_run_json(path: str) -> "SteppingStoneTerrain":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        stones_raw = data["terrain"]["stones"]
        stones: List[Stone] = []
        for s in stones_raw:
            stones.append(Stone(
                name=str(s.get("name", "stone")),
                center=tuple(s["center"]),
                width=float(s["width"]),
                height=float(s["height"]),
                yaw_deg=float(s.get("yaw_deg", 0.0)),
            ))
        return SteppingStoneTerrain(stones)

    def polygons_xy(self) -> List[List[Tuple[float, float]]]:
        return [s.polygon_xy() for s in self.stones]
