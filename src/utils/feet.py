import math
from typing import List, Tuple

def _rot2d(deg: float):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return ((c, -s), (s, c))

def _matvec2(R, v):
    return (R[0][0]*v[0] + R[0][1]*v[1],
            R[1][0]*v[0] + R[1][1]*v[1])

def feet_from_center(center_xyz: List[float],
                     heading_deg: float,
                     stance_width: float,
                     left_is_pos_local_y: bool = True) -> Tuple[List[float], List[float]]:
    """center와 heading으로부터 L/R 발 좌표만 계산 (yaw는 좌표 계산에 사용 안 함)."""
    cx, cy, cz = center_xyz
    R = _rot2d(heading_deg)
    off = stance_width * 0.5

    # 로컬 +y가 왼발이면 왼발=+off, 오른발=-off (반대면 스왑)
    off_L_local = (0.0, +off) if left_is_pos_local_y else (0.0, -off)
    off_R_local = (0.0, -off) if left_is_pos_local_y else (0.0, +off)

    dxL, dyL = _matvec2(R, off_L_local)
    dxR, dyR = _matvec2(R, off_R_local)

    pos_L = [cx + dxL, cy + dyL, cz]
    pos_R = [cx + dxR, cy + dyR, cz]
    return pos_L, pos_R
