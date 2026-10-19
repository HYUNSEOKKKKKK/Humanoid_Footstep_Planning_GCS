from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from .terrain import Vertex


def compute_centroids_from_vertices(vertices: List[Vertex]) -> Tuple[np.ndarray, np.ndarray]:
    """
    vertices: List of (3x4) arrays, each column is a corner
    return:
      centroid_mat: (3, N) where each column is [cx; cy; cz]
      base_param:   (3N, 1) = reshape(centroid_mat, [], 1)
    """
    N = len(vertices)
    if N == 0:
        return np.zeros((3, 0)), np.zeros((0, 1))
    C = np.zeros((3, N), dtype=float)
    for i, vtx in enumerate(vertices):
        c = np.mean(vtx, axis=1)
        C[:, i] = c
    base_param = C.reshape(-1, 1)
    return C, base_param


def centroid_and_scaled_vertices(vertices: List[Vertex], scale: float = 1.0):
    """Return centroids (3,N) and centered+scaled vertices list (3x4 each)."""
    N = len(vertices)
    C = np.zeros((3, N), dtype=float)
    v_scaled: List[Vertex] = []
    for i, v in enumerate(vertices):
        c = np.mean(v, axis=1)
        C[:, i] = c
        v_scaled.append(scale * (v - c.reshape(3, 1)))
    return C, v_scaled


def _get_R_world_to_ref_safe() -> Optional[np.ndarray]:
    """
    순환 임포트를 피하기 위해 런타임에 params.get_R_world_to_ref만 임포트.
    실패하면 None 반환(=회전 미적용).
    """
    # 1) 절대 경로 시도 (src/model/params.py가 있는 구조)
    try:
        from src.model.params import get_R_world_to_ref  # type: ignore
        return get_R_world_to_ref()
    except Exception:
        pass
    # 2) 패키지 상대 경로 시도 (src/graph/centroid.py → ..model.params)
    try:
        from ..model.params import get_R_world_to_ref  # type: ignore
        return get_R_world_to_ref()
    except Exception:
        pass
    # 3) 대체 경로 시도 (실행 맥락에 따라)
    try:
        from model.params import get_R_world_to_ref  # type: ignore
        return get_R_world_to_ref()
    except Exception:
        pass
    # 4) 모두 실패 → 회전 미적용
    return None


import numpy as np
from typing import Tuple

def compute_edge_directions(
    centroid_xy: np.ndarray,   # (N, 2), 각 행 = 버텍스의 (x, y)
    edge: np.ndarray,          # (E, 2), 각 행 = (from, to)
    eps: float = 1e-8,
    assume_zero_based: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    엣지 e=(u->v)에 대해, centroid(v) - centroid(u)를 정규화한 단위방향벡터 (ux, uy)를 반환.
    - 좌표계 변환 없음. 입력된 centroid_xy 기준 그대로 계산.
    - 두 점이 너무 가깝다면(has_ref[e]=0), (ux,uy)=(0,0)으로 둠.
    Returns:
        ux: (E,), uy: (E,), has_ref: (E,)  (유효 방향이면 1, 아니면 0)
    """
    C = np.asarray(centroid_xy, dtype=float)
    E = np.asarray(edge, dtype=int)

    if C.ndim != 2 or C.shape[1] != 2:
        raise ValueError("centroid_xy must be shape (N, 2)")
    if E.ndim != 2 or E.shape[1] != 2:
        raise ValueError("edge must be shape (E, 2)")

    if not assume_zero_based:
        E = E - 1

    u = E[:, 0]
    v = E[:, 1]
    dx = C[v, 0] - C[u, 0]
    dy = C[v, 1] - C[u, 1]
    nrm = np.sqrt(dx*dx + dy*dy)

    has_ref = (nrm > eps).astype(np.int32)
    ux = np.zeros_like(dx, dtype=float)
    uy = np.zeros_like(dy, dtype=float)
    m = has_ref.astype(bool)
    ux[m] = dx[m] / nrm[m]
    uy[m] = dy[m] / nrm[m]
    return ux, uy, has_ref

