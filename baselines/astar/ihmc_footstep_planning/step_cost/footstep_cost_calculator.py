from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Tuple
import math

from ..parameters import DefaultFootstepPlannerParameters
from ..graph import DiscreteFootstep, FootstepGraphNode
from ..robot_side import RobotSide
from ..geometry import wrap_to_pi, ensure_ccw
from ..graph.discrete_footstep_tools import get_foot_polygon_in_world


# ---- optional hooks (stubs) ----
class FootstepSnapperReadOnly:
    def snap_footstep(self, footstep: DiscreteFootstep):
        raise NotImplementedError


class EnvironmentHandler:
    def has_terrain_map_data(self) -> bool:
        return False

    def get_traversability_score(self, x: float, y: float) -> float:
        return 1.0


class IdealStepCalculatorInterface:
    def compute_ideal_step(self, stance_step: DiscreteFootstep, start_of_swing: DiscreteFootstep) -> DiscreteFootstep:
        raise NotImplementedError


# -------------------- convex geometry helpers (same as checker, duplicated for independence) --------------------
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
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = a
    x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < eps:
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (px, py)


def _convex_clip(subject: List[Tuple[float, float]], clip: List[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
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


def _angle_mean(yaw1: float, yaw2: float) -> float:
    # average angle using unit circle
    c = math.cos(yaw1) + math.cos(yaw2)
    s = math.sin(yaw1) + math.sin(yaw2)
    if abs(c) < 1e-12 and abs(s) < 1e-12:
        return wrap_to_pi(0.5 * (yaw1 + yaw2))
    return math.atan2(s, c)


@dataclass
class FootstepCostCalculator:
    """
    논문 Step Cost 구조에 최대한 맞춘 버전:
      - nominal mid-stance pose 간 거리
      - height change
      - yaw change
      - foothold quality: partial support penalty + roll/pitch penalty
      - + cost per step
    (논문 G. Edge Scoring, Step Cost 설명 기반) :contentReference[oaicite:4]{index=4}
    """
    parameters: DefaultFootstepPlannerParameters

    # 기존 코드 호환을 위해 남겨둠(논문 Step Cost 자체에는 ideal step 필요 없음)
    ideal_step_calculator: Optional[IdealStepCalculatorInterface] = None
    heuristics: Optional[Callable[[FootstepGraphNode], float]] = None

    # foothold quality 계산을 위해 필요
    foot_polygons_in_sole_frame: Optional[Dict[RobotSide, List[Tuple[float, float]]]] = None
    regions_xy: Optional[List[List[Tuple[float, float]]]] = None

    # (선택) stepping-stone 같이 z가 필요하면 외부에서 제공
    height_provider: Optional[Callable[[DiscreteFootstep], float]] = None
    # (선택) roll/pitch(스냅 결과)를 주고 싶으면 제공
    roll_pitch_provider: Optional[Callable[[DiscreteFootstep], Tuple[float, float]]] = None

    environment_handler: Optional[EnvironmentHandler] = None

    # 논문 “foothold quality”의 partial area 페널티 가중치(튜닝 포인트)
    partial_area_weight: float = 0.5

    eps: float = 1e-9

    def __post_init__(self):
        if self.regions_xy is not None:
            self.regions_xy = [ensure_ccw(r) for r in self.regions_xy]

    def _support_ratio(self, step: DiscreteFootstep) -> float:
        if self.foot_polygons_in_sole_frame is None or self.regions_xy is None:
            return 1.0  # 정보 없으면 fully supported로 간주

        foot_poly = self.foot_polygons_in_sole_frame[step.get_robot_side()]
        poly_world = get_foot_polygon_in_world(step, foot_poly)
        foot_area = _poly_area(poly_world)
        if foot_area <= self.eps:
            return 0.0

        best_area = 0.0
        for region in self.regions_xy:
            inter = _convex_clip(poly_world, region, eps=self.eps)
            a = _poly_area(inter)
            if a > best_area:
                best_area = a

        return max(0.0, min(1.0, best_area / foot_area))

    def _height(self, step: DiscreteFootstep) -> float:
        return float(self.height_provider(step)) if self.height_provider is not None else 0.0

    def _roll_pitch(self, step: DiscreteFootstep) -> Tuple[float, float]:
        if self.roll_pitch_provider is None:
            return 0.0, 0.0
        r, p = self.roll_pitch_provider(step)
        return float(r), float(p)

    @staticmethod
    def _midstance_pose(a: DiscreteFootstep, b: DiscreteFootstep, za: float, zb: float) -> Tuple[float, float, float, float]:
        mx = 0.5 * (a.get_x() + b.get_x())
        my = 0.5 * (a.get_y() + b.get_y())
        mz = 0.5 * (za + zb)
        myaw = _angle_mean(a.get_yaw(), b.get_yaw())
        return mx, my, mz, myaw

    def compute_cost(self, candidate_step: DiscreteFootstep, stance_step: DiscreteFootstep, start_of_swing: DiscreteFootstep) -> float:
        # heights (optional)
        z_start = self._height(start_of_swing)
        z_stance = self._height(stance_step)
        z_cand = self._height(candidate_step)

        # nominal mid-stance poses:
        m0x, m0y, m0z, m0yaw = self._midstance_pose(start_of_swing, stance_step, z_start, z_stance)
        m1x, m1y, m1z, m1yaw = self._midstance_pose(stance_step, candidate_step, z_stance, z_cand)

        # 1) distance between mid-stance poses (논문 핵심) :contentReference[oaicite:5]{index=5}
        dist = math.hypot(m1x - m0x, m1y - m0y)

        # 2) height change
        dz = (m1z - m0z)

        # 3) yaw change
        dyaw = wrap_to_pi(m1yaw - m0yaw)

        # 4) foothold quality: partial support + roll/pitch requirement :contentReference[oaicite:6]{index=6}
        support = self._support_ratio(candidate_step)
        roll, pitch = self._roll_pitch(candidate_step)

        edge_cost = 0.0

        # distance weight: 기존 파라미터를 그대로 쓰기 위해 forward_weight를 “거리 스케일”로 사용
        edge_cost += dist * self.parameters.getForwardWeight()

        # height up/down weight는 기존 방식 유지
        if dz >= 0.0:
            edge_cost += abs(dz) * self.parameters.getStepUpWeight()
        else:
            edge_cost += abs(dz) * self.parameters.getStepDownWeight()

        edge_cost += abs(dyaw) * self.parameters.getYawWeight()

        # partial area penalty: (1 - support_ratio)
        edge_cost += (1.0 - support) * self.partial_area_weight

        # roll/pitch penalty
        edge_cost += abs(roll) * self.parameters.getRollWeight()
        edge_cost += abs(pitch) * self.parameters.getPitchWeight()

        # static cost per step :contentReference[oaicite:7]{index=7}
        edge_cost += self.parameters.getCostPerStep()

        # (선택) traversability cost는 그대로 유지 (논문에서도 “foothold quality”/지형 quality 항이 존재)
        if self.environment_handler is not None and self.environment_handler.has_terrain_map_data():
            traversability_cost = 1.0 - self.environment_handler.get_traversability_score(candidate_step.get_x(), candidate_step.get_y())
            traversability_scale = 0.8
            traversability_cost = max(0.0, min(1.0, traversability_cost))
            edge_cost += traversability_cost * traversability_scale

        return edge_cost
