from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from ..graph import DiscreteFootstep
from ..robot_side import RobotSide
from ..graph.discrete_footstep_tools import get_foot_polygon_in_world
from ..geometry import convex_polygon_contains_polygon, ensure_ccw


# -------------------- small convex geometry helpers --------------------
def _poly_area(poly: List[Tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _is_left_of_edge(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float], eps: float) -> bool:
    # clip polygon must be CCW, so "inside" is left side of each directed edge (a->b)
    px, py = p
    ax, ay = a
    bx, by = b
    return _cross(bx - ax, by - ay, px - ax, py - ay) >= -eps


def _segment_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
    eps: float,
) -> Tuple[float, float]:
    # intersection of line p1->p2 with line a->b (as infinite lines),
    # used in Sutherland–Hodgman; assumes they are not parallel in typical cases.
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = a
    x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < eps:
        # almost parallel: return midpoint as a safe fallback
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (px, py)


def _convex_clip(subject: List[Tuple[float, float]], clip: List[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
    # Sutherland–Hodgman polygon clipping for convex clip polygon (CCW).
    out = subject
    if len(out) < 3 or len(clip) < 3:
        return []

    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        inp = out
        out = []
        if not inp:
            break

        s = inp[-1]
        s_inside = _is_left_of_edge(s, a, b, eps)
        for e in inp:
            e_inside = _is_left_of_edge(e, a, b, eps)
            if e_inside:
                if not s_inside:
                    out.append(_segment_intersection(s, e, a, b, eps))
                out.append(e)
            elif s_inside:
                out.append(_segment_intersection(s, e, a, b, eps))
            s, s_inside = e, e_inside

    return out


# -------------------- checker --------------------
@dataclass
class RegionFootstepChecker:
    regions_xy: List[List[Tuple[float, float]]]
    foot_polygons_in_sole_frame: Dict[RobotSide, List[Tuple[float, float]]]

    eps: float = 1e-9

    # 논문 재현용: partial foothold 허용 (support area 기준)
    allow_partial: bool = True
    min_support_ratio: float = 0.6  # 발 면적 대비 교차 면적 비율 (튜닝 포인트)
    min_support_area: float = 0.0     # 절대 면적 하한(필요하면 사용)

    # --- stats ---
    n_checked: int = field(default=0, init=False)
    n_passed: int = field(default=0, init=False)
    n_failed: int = field(default=0, init=False)

    # last query (디버깅용)
    last_best_ratio: float = field(default=0.0, init=False)
    last_best_area: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.regions_xy = [ensure_ccw(r) for r in self.regions_xy]

    def reset_stats(self) -> None:
        self.n_checked = 0
        self.n_passed = 0
        self.n_failed = 0
        self.last_best_ratio = 0.0
        self.last_best_area = 0.0

    def snapshot_stats(self) -> Tuple[int, int, int]:
        return (self.n_checked, self.n_passed, self.n_failed)

    def _best_support_ratio(self, foot_poly_world: List[Tuple[float, float]]) -> Tuple[float, float]:
        foot_area = _poly_area(foot_poly_world)
        if foot_area <= self.eps:
            return 0.0, 0.0

        best_area = 0.0
        for region in self.regions_xy:
            inter = _convex_clip(foot_poly_world, region, eps=self.eps)
            a = _poly_area(inter)
            if a > best_area:
                best_area = a

        return best_area / foot_area, best_area

    def is_step_valid(self,
                      candidate_step: DiscreteFootstep,
                      stance_step: DiscreteFootstep,
                      start_of_swing: DiscreteFootstep) -> bool:
        self.n_checked += 1

        foot_poly = self.foot_polygons_in_sole_frame[candidate_step.get_robot_side()]
        poly_world = get_foot_polygon_in_world(candidate_step, foot_poly)

        # 1) 빠른 경로: 완전 포함이면 OK
        for region in self.regions_xy:
            if convex_polygon_contains_polygon(region, poly_world, eps=self.eps):
                self.last_best_ratio = 1.0
                self.last_best_area = _poly_area(poly_world)
                self.n_passed += 1
                return True

        # 2) partial foothold 허용 시: intersecting area 기준
        if self.allow_partial:
            ratio, area = self._best_support_ratio(poly_world)
            self.last_best_ratio = ratio
            self.last_best_area = area
            if ratio + self.eps >= self.min_support_ratio and area + self.eps >= self.min_support_area:
                self.n_passed += 1
                return True

        self.last_best_ratio = 0.0
        self.last_best_area = 0.0
        self.n_failed += 1
        return False
