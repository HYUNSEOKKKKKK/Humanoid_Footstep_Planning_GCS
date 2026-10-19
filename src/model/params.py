# src/model/params.py
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np


__all__ = [
    "pack_relaxed_params_for_model",
    "pack_fixedpath_params_for_model",
    "prepare_problem_params",
    "pack_base_param",
    "get_R_world_to_ref",

]

# -----------------------------------------------------------------------------
# Global reference frame cache: World → Ref (2x2 rotation)
# -----------------------------------------------------------------------------
_REF_R_WORLD_TO_REF: Optional[np.ndarray] = None


def _rot_from_start_target(startpos_xy: np.ndarray, targetpos_xy: np.ndarray) -> np.ndarray:
    """
    start→target 방향이 Ref의 +x축이 되도록 하는 회전행렬 R_world_to_ref (2x2)를 생성.
    입력은 (2,) 또는 (3,) 가능하나, xy만 사용.
    """
    startpos_xy = np.asarray(startpos_xy, float).reshape(-1)[:2]
    targetpos_xy = np.asarray(targetpos_xy, float).reshape(-1)[:2]
    refdir = targetpos_xy - startpos_xy
    refangle = float(np.arctan2(refdir[1], refdir[0]))  # world에서 본 각도
    c, s = np.cos(refangle), np.sin(refangle)
    # R_ref_to_world = [[c,-s],[s,c]]
    # 우리가 필요한 건 world→ref 이므로 전치
    R_world_to_ref = np.array([[c, s], [-s, c]], dtype=float).T  # 동일하게 [[c, s],[-s, c]]
    return R_world_to_ref


def set_global_reference_from_startRL(
    startpos_L: np.ndarray,
    startpos_R: np.ndarray,
    targetpos_L: np.ndarray,
    targetpos_R: np.ndarray,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Global frame 정의:
      +y_ref = start_R -> start_L (정규화)
      +x_ref = +y_ref 를 시계방향 90° 회전한 벡터 (b, -a) if y_ref=(a,b)

    반환: R_world_to_ref (2x2).  world 좌표 w 를 ref 좌표 r 로 바꿀 때 r = R_world_to_ref @ w.
    targetpos_* 인자는 프레임 정의에 사용하지 않으며, 시그니처 호환용으로만 받습니다.
    """
    # 입력을 2D로 취함
    pL = np.asarray(startpos_L, float).reshape(-1)[:2]
    pR = np.asarray(startpos_R, float).reshape(-1)[:2]

    # +y_ref = R -> L
    vec_y = pL - pR
    n = np.linalg.norm(vec_y)
    if n < eps:
        # 두 발이 거의 동일 위치면 기본축 사용
        y_ref = np.array([0.0, 1.0], dtype=float)
    else:
        y_ref = vec_y / n

    # +x_ref = +y_ref 의 시계방향 90도 회전: (y, -x)
    x_ref = np.array([y_ref[1], -y_ref[0]], dtype=float)

    # world -> ref : r = [ x_ref^T ; y_ref^T ] w
    R_w2r = np.array([
        [x_ref[0], x_ref[1]],
        [y_ref[0], y_ref[1]],
    ], dtype=float)

    # 전역 보관
    global _REF_R_WORLD_TO_REF
    _REF_R_WORLD_TO_REF = R_w2r
    return R_w2r



def get_R_world_to_ref() -> Optional[np.ndarray]:
    """
    전역으로 설정된 월드→Ref 회전행렬(2x2)을 반환. 아직 설정 전이면 None.
    """
    return _REF_R_WORLD_TO_REF


# -----------------------------------------------------------------------------
# Packing helpers (원본 유지)
# -----------------------------------------------------------------------------
def pack_relaxed_params_for_model(
    centroid_mat: np.ndarray,
    n_edge: int,
    M_large: float,
    start_L: np.ndarray,
    start_R: np.ndarray,
    target_L: np.ndarray,
    target_R: np.ndarray,
    pref_gap: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    1단계(릴랙스드)용 파라미터 묶음.
    """
    params: Dict[str, Any] = {
        "geometry": np.asarray(centroid_mat, float),
        "y_relax": np.zeros((int(n_edge), 1), dtype=float),  # 1단계는 전부 0으로 시작
        "M_scalar": float(M_large),
        "start_L": np.asarray(start_L, float).reshape(-1),
        "start_R": np.asarray(start_R, float).reshape(-1),
        "target_L": np.asarray(target_L, float).reshape(-1),
        "target_R": np.asarray(target_R, float).reshape(-1),
    }
    if pref_gap is not None:
        params["pref_gap"] = np.asarray(pref_gap, float).reshape(-1)
    return params


def pack_fixedpath_params_for_model(
    centroid_mat: np.ndarray,
    y_relax: np.ndarray,
    M_small: float,
    start_L: np.ndarray,
    start_R: np.ndarray,
    target_L: np.ndarray,
    target_R: np.ndarray,
    pref_gap: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    2단계(경로 고정)용 파라미터 묶음.

    Notes
    -----
    - y_relax : (n_edge, 1) 형태. 선택된 경로의 엣지에만 1, 나머지는 0.
    - M_small : 아주 작은 Big-M(사실상 equality 강제용).
    """
    params: Dict[str, Any] = {
        "geometry": np.asarray(centroid_mat, float),
        "y_relax": np.asarray(y_relax, float).reshape(-1, 1),
        "M_scalar": float(M_small),
        "start_L": np.asarray(start_L, float).reshape(-1),
        "start_R": np.asarray(start_R, float).reshape(-1),
        "target_L": np.asarray(target_L, float).reshape(-1),
        "target_R": np.asarray(target_R, float).reshape(-1),
    }
    if pref_gap is not None:
        params["pref_gap"] = np.asarray(pref_gap, float).reshape(-1)
    return params


# -----------------------------------------------------------------------------
# Problem preparation (전역 Ref 설정을 사이드이펙트로 추가)
# -----------------------------------------------------------------------------


def prepare_problem_params(
    startpos_L: np.ndarray,
    startpos_R: np.ndarray,
    targetpos_L: np.ndarray,
    targetpos_R: np.ndarray,
    startyaw_L: float,   # 이미 ref 기준 각도 [rad]
    startyaw_R: float,   # 이미 ref 기준 각도 [rad]
    targetyaw_L: float,  # 이미 ref 기준 각도 [rad]
    targetyaw_R: float,  # 이미 ref 기준 각도 [rad]
) -> Dict[str, np.ndarray]:
    """
    Global frame 고정 규칙:
      +y_ref = start_R -> start_L
      +x_ref = 위를 시계방향 90°
    입력 yaw_* 는 이미 'ref x축 기준 각도'라고 가정.
    => 5D 포즈 (x, y, z, cos, sin)에서 cos/sin은 단순히 cos(yaw), sin(yaw)로 설정.
    (target 위치/자세는 frame 정의에 영향 없음)
    """

    # 전역 Ref 설정 (사이드 이펙트용: 다른 모듈에서 world→ref 쓰면 필요)
    # 반환값은 쓰지 않지만 호출은 유지
    _ = set_global_reference_from_startRL(
        startpos_L, startpos_R, targetpos_L, targetpos_R
    )

    def _pose5(pos, yaw_ref_rad: float):
        pos = np.asarray(pos, float).reshape(-1)
        c = float(np.cos(yaw_ref_rad))
        s = float(np.sin(yaw_ref_rad))
        # 수치 안정화를 위한 재정규화(옵션이지만 안전함)
        r = float(np.hypot(c, s)) or 1.0
        c /= r; s /= r
        return np.array([pos[0], pos[1], pos[2], c, s], dtype=float)

    start_L_5d = _pose5(startpos_L, startyaw_L)
    start_R_5d = _pose5(startpos_R, startyaw_R)
    target_L_5d = _pose5(targetpos_L, targetyaw_L)
    target_R_5d = _pose5(targetpos_R, targetyaw_R)

    return dict(
        start_L=start_L_5d,
        start_R=start_R_5d,
        target_L=target_L_5d,
        target_R=target_R_5d,
    )




def pack_base_param(centroid: np.ndarray) -> np.ndarray:
    """
    MATLAB 호환: base_param = reshape(centroid, [], 1)

    Parameters
    ----------
    centroid : (3, N) ndarray

    Returns
    -------
    (3N, 1) ndarray
    """
    if centroid is None:
        return np.zeros((0, 1))
    return np.asarray(centroid, float).reshape(-1, 1)
