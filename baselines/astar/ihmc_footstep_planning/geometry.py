from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable, List, Tuple
import itertools

def wrap_to_pi(angle: float) -> float:
    # Equivalent to AngleTools.computeAngleDifferenceMinusPiToPi for diffs
    a = (angle + math.pi) % (2.0 * math.pi) - math.pi
    # Normalize -pi to pi (keep -pi)
    return a

def interpolate(a: float, b: float, alpha: float) -> float:
    return (1.0 - alpha) * a + alpha * b

def interpolate_angle(a: float, b: float, alpha: float) -> float:
    # Shortest-path interpolation on circle
    d = wrap_to_pi(b - a)
    return wrap_to_pi(a + alpha * d)

@dataclass(frozen=True)
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

@dataclass(frozen=True)
class Transform2D:
    # 2D rigid transform: R(yaw) and translation (x,y)
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def apply(self, p: Tuple[float, float]) -> Tuple[float, float]:
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        px, py = p
        return (self.x + c * px - s * py, self.y + s * px + c * py)

    def inverse(self) -> "Transform2D":
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        # inv: R^T, t' = -R^T t
        ix = -(c * self.x + s * self.y)
        iy = -(-s * self.x + c * self.y)
        return Transform2D(ix, iy, -self.yaw)

    def compose(self, other: "Transform2D") -> "Transform2D":
        # self ∘ other (apply other then self)
        ox, oy = self.apply((other.x, other.y))
        return Transform2D(ox, oy, wrap_to_pi(self.yaw + other.yaw))

def polygon_transform(points: Iterable[Tuple[float, float]], T: Transform2D) -> List[Tuple[float, float]]:
    return [T.apply(p) for p in points]

def _dot(a, b) -> float:
    return a[0]*b[0] + a[1]*b[1]

def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1])

def _norm2(a) -> float:
    return a[0]*a[0] + a[1]*a[1]

def _seg_distance(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    # distance from point p to segment ab
    ab = _sub(b, a)
    ap = _sub(p, a)
    denom = _norm2(ab)
    if denom <= 1e-12:
        return math.sqrt(_norm2(_sub(p, a)))
    t = _dot(ap, ab) / denom
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    proj = (a[0] + t*ab[0], a[1] + t*ab[1])
    return math.sqrt(_norm2(_sub(p, proj)))

def _ccw(a, b, c) -> bool:
    return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])

def segments_intersect(a,b,c,d) -> bool:
    # Proper intersection test (including collinear overlaps is not needed for our use;
    # we handle 'touching' conservatively as intersecting via SAT below.)
    return (_ccw(a,c,d) != _ccw(b,c,d)) and (_ccw(a,b,c) != _ccw(a,b,d))

def convex_polygon_axes(poly: List[Tuple[float,float]]):
    # outward normals of edges (not necessarily unit length)
    n = len(poly)
    for i in range(n):
        p = poly[i]
        q = poly[(i+1)%n]
        e = (q[0]-p[0], q[1]-p[1])
        # normal
        yield (-e[1], e[0])

def project(poly, axis):
    # axis can be non-unit
    dots = [_dot(p, axis) for p in poly]
    return min(dots), max(dots)

def convex_polygons_intersect(polyA: List[Tuple[float,float]], polyB: List[Tuple[float,float]]) -> bool:
    # Separating Axis Theorem
    for axis in itertools.chain(convex_polygon_axes(polyA), convex_polygon_axes(polyB)):
        minA, maxA = project(polyA, axis)
        minB, maxB = project(polyB, axis)
        if maxA < minB or maxB < minA:
            return False
    return True

def distance_between_convex_polygons(polyA: List[Tuple[float,float]], polyB: List[Tuple[float,float]]) -> float:
    # If they intersect => 0
    if convex_polygons_intersect(polyA, polyB):
        return 0.0
    # Otherwise min of point-to-segment distances (O(n^2), fine for small polygons)
    min_d = float("inf")
    for polyP, polyQ in ((polyA, polyB), (polyB, polyA)):
        m = len(polyP)
        n = len(polyQ)
        for p in polyP:
            for j in range(n):
                a = polyQ[j]
                b = polyQ[(j+1)%n]
                d = _seg_distance(p, a, b)
                if d < min_d:
                    min_d = d
    return float(min_d)


# --- Convex polygon containment helpers (CCW polygons) ---

def polygon_signed_area(poly: List[Tuple[float, float]]) -> float:
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area

def ensure_ccw(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    # Return a CCW-ordered polygon (does not modify input)
    if polygon_signed_area(poly) < 0.0:
        return list(reversed(poly))
    return list(poly)

def convex_polygon_contains_point(poly: List[Tuple[float, float]], p: Tuple[float, float], eps: float = 1e-9) -> bool:
    # Half-plane test for convex CCW polygon
    x, y = p
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # left-of test: cross(edge, p - v1) >= 0
        if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) < -eps:
            return False
    return True

def convex_polygon_contains_polygon(container: List[Tuple[float, float]],
                                    containee: List[Tuple[float, float]],
                                    eps: float = 1e-9) -> bool:
    container_ccw = ensure_ccw(container)
    for v in containee:
        if not convex_polygon_contains_point(container_ccw, v, eps=eps):
            return False
    return True
