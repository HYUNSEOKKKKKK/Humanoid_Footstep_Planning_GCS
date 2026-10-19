from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class FootSeries:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    c: np.ndarray  # cos(yaw)  <-- 이제 global ref와 동일 좌표계로 간주
    s: np.ndarray  # sin(yaw)

@dataclass
class ProcessedSolution:
    L: FootSeries
    R: FootSeries
    edge_idx_path: List[int]        # 경로에 해당하는 엣지 인덱스(0-based)
    foot_L_full: np.ndarray         # (N_L_full, 5) pos[3] + yaw[2]
    foot_R_full: np.ndarray         # (N_R_full, 5)

def _edge_path_to_indices(edge_1b: np.ndarray, path_1b: List[int]) -> List[int]:
    """1-based edge/경로로부터 0-based 엣지 인덱스 리스트를 만든다."""
    idx_map: Dict[tuple[int, int], int] = {}
    for i in range(edge_1b.shape[0]):
        u, v = int(edge_1b[i, 0]), int(edge_1b[i, 1])
        idx_map[(u, v)] = i
    idxs: List[int] = []
    for i in range(len(path_1b) - 1):
        u, v = int(path_1b[i]), int(path_1b[i + 1])
        if (u, v) not in idx_map:
            raise RuntimeError(f"경로에 없는 엣지 ({u}->{v})")
        idxs.append(idx_map[(u, v)])
    return idxs

def _unit2(v: np.ndarray) -> np.ndarray:
    """2D 벡터 정규화. 영벡터면 그대로 반환."""
    n = float(np.hypot(v[0], v[1]))
    return v if n == 0.0 else v / n

def _normalize_cs_rows(X: np.ndarray) -> np.ndarray:
    """(N,5) 배열의 마지막 두 열(c,s)을 행별 정규화."""
    if X.size == 0:
        return X
    Y = np.array(X, dtype=float, copy=True)
    n = np.hypot(Y[:, 3], Y[:, 4])
    n[n == 0.0] = 1.0
    Y[:, 3] /= n
    Y[:, 4] /= n
    return Y

def postprocess_solution(
    ex: Dict[str, Any],
    edge_1b: np.ndarray,
    relaxed_path_1b: List[int],
    startpos_L: np.ndarray,
    startpos_R: np.ndarray,
    targetpos_L: np.ndarray,
    targetpos_R: np.ndarray,
    target_L_5d: np.ndarray,
    target_R_5d: np.ndarray,
    eps: float = 1e-9,
) -> ProcessedSolution:
    """
    solve 이후 추출한 ex(dict)와 경로/시작/목표 정보를 받아
    '현재 global ref(= start_R→start_L 규칙)'와 동일한 좌표계에서
    발 시퀀스(샘플링 적용)를 반환.

    ex: extract_solution(V, IDX) 결과
        keys: y(E,), z_L(E,5), z_R(E,5), zp_L(E,5), zp_R(E,5)
    edge_1b: (E,2) 1-based edge list
    relaxed_path_1b: [노드1, 노드2, ...] 1-based
    """
    y = np.asarray(ex["y"], float).reshape(-1)
    z_L = np.asarray(ex["z_L"], float)
    z_R = np.asarray(ex["z_R"], float)
    # print(y)
    # 1) 경로 엣지 인덱스(0-based)
    edge_idx_path = _edge_path_to_indices(edge_1b, relaxed_path_1b)
    # print(edge_idx_path)
    # 2) 분수 복원 x = z / y (경로 순서대로)
    yy = np.clip(y[edge_idx_path].reshape(-1, 1), eps, None)
    x_L_path = z_L[edge_idx_path] / yy  # (N_path, 5)  = [x,y,z, cos_ref, sin_ref]
    x_R_path = z_R[edge_idx_path] / yy
    # print(edge_idx_path)
    # 3) yaw 변환 제거: 이미 global ref와 동일 좌표계로 사용
    #    단, 수치적 안정성을 위해 (cos, sin)만 정규화
    foot_L = _normalize_cs_rows(x_L_path)
    foot_R = _normalize_cs_rows(x_R_path)

    # 타깃 포즈도 동일하게 추가 (정규화)
    tgtL_row = np.r_[target_L_5d[:3], _unit2(target_L_5d[3:5])]
    tgtR_row = np.r_[target_R_5d[:3], _unit2(target_R_5d[3:5])]
    foot_L = np.vstack([foot_L, tgtL_row]) if foot_L.size else tgtL_row.reshape(1, 5)
    foot_R = np.vstack([foot_R, tgtR_row]) if foot_R.size else tgtR_row.reshape(1, 5)
    # print(foot_L)
    # print(y)
    def _dedup_preserve_order(idx_list):
        seen = set()
        out = []
        for i in idx_list:
            if i not in seen:
                out.append(i)
                seen.add(i)
        return np.array(out, dtype=int)

    nL = foot_L.shape[0]
    nR = foot_R.shape[0]

    # 안전장치: nL == nR 가정
    assert nL == nR, f"Expected nL == nR, but got nL={nL}, nR={nR}"
    n = nL

    if n == 0:
        Ls = np.zeros((0, 5))
        Rs = np.zeros((0, 5))

    else:
        if n % 2 == 0:
            # ===== 짝수 개: 기존 규칙 유지 =====
            if y[0] > 0.5:
                # L: [0] + 홀수(1,3,5,...) + [마지막]
                odd_idx = np.arange(1, n, 2)
                L_idx = _dedup_preserve_order([0, *odd_idx.tolist(), n - 1])

                # R: 짝수(0,2,4,...) + (마지막 추가)
                even_idx0 = np.arange(0, n, 2)
                R_idx = even_idx0.tolist()
                R_idx.append(n - 1)

                Ls = foot_L[L_idx, :]
                Rs = foot_R[_dedup_preserve_order(R_idx), :]

            else:
                # 좌우 규칙 뒤바꿈 (기존 동작 유지)
                # L: 짝수(0,2,4,...) + (마지막 추가)
                even_idx0 = np.arange(0, n, 2)
                L_idx = even_idx0.tolist()
                L_idx.append(n - 1)

                # R: 짝수(0,2,4,...) + (마지막 추가)
                R_idx = even_idx0.tolist()
                R_idx.append(n - 1)

                Ls = foot_L[_dedup_preserve_order(L_idx), :]
                Rs = foot_R[_dedup_preserve_order(R_idx), :]

        else:
            # ===== 홀수 개: 새 규칙 적용 일부 =====
            if y[0] > 0.5:
                # 기존 규칙 유지
                # L: [0] + 홀수(1,3,5,...) + [마지막]
                odd_idx = np.arange(1, n, 2)
                L_idx = _dedup_preserve_order([0, *odd_idx.tolist(), n - 1])

                # R: 짝수(0,2,4,...)  (n-1은 홀수라 자동 미포함)
                even_idx0 = np.arange(0, n, 2)

                Ls = foot_L[L_idx, :]
                Rs = foot_R[even_idx0, :]

            else:

                # ✅ 변경: 짝수 인덱스(0,2,4,...)만 사용

                even_idx0 = np.arange(0, n, 2)  # 예: n=9 → [0,2,4,6,8]

                Ls = foot_L[even_idx0, :]

                Rs = foot_R[even_idx0, :]  # L/R 길이와 타임스텝 정렬 유지가 필요하면 함께 적용

    L_series = FootSeries(x=Ls[:, 0], y=Ls[:, 1], z=Ls[:, 2], c=Ls[:, 3], s=Ls[:, 4])
    R_series = FootSeries(x=Rs[:, 0], y=Rs[:, 1], z=Rs[:, 2], c=Rs[:, 3], s=Rs[:, 4])

    return ProcessedSolution(
        L=L_series,
        R=R_series,
        edge_idx_path=edge_idx_path,
        foot_L_full=foot_L,
        foot_R_full=foot_R,
    )

