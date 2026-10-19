# src/utils/udp.py
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import socket

from .postprocess import ProcessedSolution  # proc.L / proc.R 사용


def _build_side_matrix(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, c: np.ndarray, s: np.ndarray,
    start_center: np.ndarray, yaw_offset: float,
    indicator: float, roll_val: float = 0.0, pitch_val: float = 0.0,
) -> np.ndarray:
    """
    한 쪽(L 또는 R) 발 시퀀스를 MATLAB 형식(7 x N)으로 구성.
    indicator: Left=+1, Right=-1
    좌표변환: start_center 기준 로컬 → (x',y',z') = (y_local, -x_local, z_local)
    yaw: atan2(s, c) - yaw_offset
    """
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)
    c = np.asarray(c, float).reshape(-1)
    s = np.asarray(s, float).reshape(-1)
    n = x.size

    # yaw 계산 및 오프셋 보정
    yaw = np.arctan2(s, c) - float(yaw_offset)

    # 로컬 기준으로 이동
    x_loc = x - float(start_center[0])
    y_loc = y - float(start_center[1])
    z_loc = z - float(start_center[2])

    # 좌표축 스왑/부호 (MATLAB 코드와 동일)
    x_final = y_loc
    y_final = -x_loc
    z_final = z_loc

    # 7 x n
    mat = np.vstack([
        np.full(n, indicator, dtype=float),   # 1 or -1
        x_final,
        y_final,
        z_final,
        np.full(n, roll_val, dtype=float),
        np.full(n, pitch_val, dtype=float),
        yaw,
    ])
    return mat

def build_combined_from_proc_DDR(
    proc: "ProcessedSolution",
    y: np.ndarray,
    startpos_L: np.ndarray, startpos_R: np.ndarray,
    startyaw_L: float, startyaw_R: float,
    interleave_from_y0: Optional[bool] = None,   # None이면 y[0] > 0.49 사용
) -> Tuple[np.ndarray, bool]:
    """
    DDR 전송용 간소화 버전:
      - proc.foot_L_full / proc.foot_R_full 의 첫 행만 사용하여
        각 측 1스텝씩(총 2컬럼)의 combined(7x2)를 생성.
      - 시작 발은 y[0] > 0.49 규칙(또는 interleave_from_y0)으로 결정.

    반환:
      combined: (7, 0|1|2)  # 정상이면 (7,2)
      first_is_left: bool
    """
    # --- 안전 보정: 2D startpos면 z=0으로 3D 패딩 ---
    def _to3(p):
        p = np.asarray(p, float).reshape(-1)
        if p.size == 2:
            return np.array([p[0], p[1], 0.0], float)
        if p.size == 3:
            return p.astype(float)
        raise ValueError(f"startpos must be length 2 or 3, got size={p.size}")

    startpos3_L = _to3(startpos_L)
    startpos3_R = _to3(startpos_R)

    # --- 기준 프레임(원점/회전) ---
    start_center = (startpos3_L + startpos3_R) / 2.0
    yaw_offset = (float(startyaw_L) + float(startyaw_R)) / 2.0

    # --- 데이터 소스: full 테이블 첫 행만 사용 ---
    if proc.foot_L_full is None or proc.foot_L_full.shape[0] == 0:
        raise ValueError("proc.foot_L_full is empty")
    if proc.foot_R_full is None or proc.foot_R_full.shape[0] == 0:
        raise ValueError("proc.foot_R_full is empty")

    # 각 측 1행: [x,y,z,c,s]
    xL0, yL0, zL0, cL0, sL0 = [float(v) for v in proc.foot_L_full[0, :5]]
    xR0, yR0, zR0, cR0, sR0 = [float(v) for v in proc.foot_R_full[0, :5]]

    # --- 7×1로 변환 (좌/우 각각) ---
    # _build_side_matrix는 기존 함수(그대로 사용)
    footsteps_L = _build_side_matrix(
        np.array([xL0]), np.array([yL0]), np.array([zL0]),
        np.array([cL0]), np.array([sL0]),
        start_center, yaw_offset, indicator=+1.0
    )  # (7,1)

    footsteps_R = _build_side_matrix(
        np.array([xR0]), np.array([yR0]), np.array([zR0]),
        np.array([cR0]), np.array([sR0]),
        start_center, yaw_offset, indicator=-1.0
    )  # (7,1)

    # --- 시작 발 결정 ---
    if interleave_from_y0 is None:
        yy = np.asarray(y, float).reshape(-1)
        first_is_left = bool(yy.size > 0 and yy[0] > 0.49)
    else:
        first_is_left = bool(interleave_from_y0)

    # --- 인터리브 (항상 최대 2컬럼: L,R or R,L) ---
    cols = []
    if first_is_left:
        cols.append(footsteps_L)  # L 먼저
        cols.append(footsteps_R)
    else:
        cols.append(footsteps_R)  # R 먼저
        cols.append(footsteps_L)

    combined = np.concatenate(cols, axis=1) if cols else np.zeros((7, 0), float)
    return combined, first_is_left

def build_combined_from_proc(
    proc: ProcessedSolution,
    y: np.ndarray,
    startpos_L: np.ndarray, startpos_R: np.ndarray,
    startyaw_L: float, startyaw_R: float,
    drop_first_column_if_ge2: bool = True,
    interleave_from_y0: Optional[bool] = None,  # None이면 y[0] > 0.49 사용
    *,
    use_only_first_full_rows: bool = False,      # ★ 추가: True면 foot_*_full의 첫 행만 사용
) -> Tuple[np.ndarray, bool]:
    """
    후처리 결과(proc)와 시작 포즈로부터 combined(7 x M) 행렬 생성.
    반환: (combined, first_is_left)

    use_only_first_full_rows=True이면
      - proc.foot_L_full[0], proc.foot_R_full[0] (각각 1행의 [x,y,z,c,s])
        만을 사용해 각 측 1스텝씩(= 총 2컬럼)만 전송용으로 구성.
    """
    import numpy as np

    # --- 공통: 기준 프레임 설정 (MATLAB 동일) ---
    start_center = (np.asarray(startpos_L, float).reshape(3) +
                    np.asarray(startpos_R, float).reshape(3)) / 2.0
    yaw_offset = (float(startyaw_L) + float(startyaw_R)) / 2.0

    # --- 데이터 소스 선택 ---
    if use_only_first_full_rows:
        # 안전장치: full 테이블이 비어있지 않은지 확인
        if proc.foot_L_full is None or proc.foot_L_full.shape[0] == 0:
            raise ValueError("proc.foot_L_full is empty")
        if proc.foot_R_full is None or proc.foot_R_full.shape[0] == 0:
            raise ValueError("proc.foot_R_full is empty")

        # 각 측 1개 스텝만 뽑아서 _build_side_matrix로 7×1 만들기
        xL0, yL0, zL0, cL0, sL0 = [float(v) for v in proc.foot_L_full[0, :5]]
        xR0, yR0, zR0, cR0, sR0 = [float(v) for v in proc.foot_R_full[0, :5]]

        footsteps_L = _build_side_matrix(
            np.array([xL0]), np.array([yL0]), np.array([zL0]),
            np.array([cL0]), np.array([sL0]),
            start_center, yaw_offset, indicator=+1.0
        )  # shape: (7,1)

        footsteps_R = _build_side_matrix(
            np.array([xR0]), np.array([yR0]), np.array([zR0]),
            np.array([cR0]), np.array([sR0]),
            start_center, yaw_offset, indicator=-1.0
        )  # shape: (7,1)

        # 첫 열 제거 규칙은 의미 없음(각 1열뿐)이므로 스킵
        nL = footsteps_L.shape[1]  # 1
        nR = footsteps_R.shape[1]  # 1

    else:
        # 기존 동작: proc.L/R 시퀀스 전체 사용
        footsteps_L = _build_side_matrix(
            proc.L.x, proc.L.y, proc.L.z, proc.L.c, proc.L.s,
            start_center, yaw_offset, indicator=+1.0
        )
        footsteps_R = _build_side_matrix(
            proc.R.x, proc.R.y, proc.R.z, proc.R.c, proc.R.s,
            start_center, yaw_offset, indicator=-1.0
        )

        # 첫 열 제거 규칙 (양쪽 다 2개 이상이면 첫 번째 열 drop)
        if drop_first_column_if_ge2 and footsteps_L.shape[1] >= 2:
            footsteps_L = footsteps_L[:, 1:]
        if drop_first_column_if_ge2 and footsteps_R.shape[1] >= 2:
            footsteps_R = footsteps_R[:, 1:]
        nL = footsteps_L.shape[1]
        nR = footsteps_R.shape[1]

    # --- interleave 시작측 결정 (기존 규칙 유지) ---
    if interleave_from_y0 is None:
        y = np.asarray(y, float).reshape(-1)
        first_is_left = bool(y.size > 0 and y[0] > 0.49)
    else:
        first_is_left = bool(interleave_from_y0)

    # --- interleave: L,R 또는 R,L ---
    cols = []
    if first_is_left:
        # L → R (이번 요구 조건에선 총 2컬럼)
        for i in range(max(nL, nR)):
            if i < nL: cols.append(footsteps_L[:, i:i+1])
            if i < nR: cols.append(footsteps_R[:, i:i+1])
    else:
        # R → L
        for i in range(max(nL, nR)):
            if i < nR: cols.append(footsteps_R[:, i:i+1])
            if i < nL: cols.append(footsteps_L[:, i:i+1])

    combined = np.concatenate(cols, axis=1) if cols else np.zeros((7, 0), dtype=float)
    return combined, first_is_left


def make_udp_payload(
    combined: np.ndarray,
    drop_last_column: bool = True,
) -> Tuple[np.ndarray, int, np.ndarray, bytes]:
    """
    MATLAB 동일 포맷의 UDP 페이로드 생성.
    - 마지막 열 제거 옵션 (combined = combined[:, :-1])
    - flat_data = [N; combined(:)]  (column-major)
    - 바이트스트림: float64 (little-endian)
    Returns: (combined, N, flat_data, byte_stream)
    """
    if drop_last_column and combined.shape[1] >= 1:
        combined = combined[:, :-1]

    N = int(combined.shape[1])

    # column-major flatten (MATLAB(:)와 동일)
    flat = combined.astype(np.float64, copy=False).reshape(-1, order="F")

    # 첫 원소에 N을 넣고 나머지 데이터 뒤따르게 구성 (float64, little-endian)
    flat_data = np.empty(1 + flat.size, dtype="<f8")
    flat_data[0] = float(N)
    flat_data[1:] = flat

    byte_stream = flat_data.tobytes(order="C")
    return combined, N, flat_data, byte_stream


def send_udp(byte_stream: bytes, host: str, port: int, timeout: float = 1.0) -> int:
    """
    UDP로 바이트스트림 전송. 반환: 보낸 바이트 수.
    """
    import os
    if os.environ.get("FOOTSTEP_ENABLE_ROBOT_UDP") != "1":
        return 0
    if len(byte_stream) > 65000:
        # 일반적인 MTU를 넘을 수 있음—필요하면 조각내기 구현 검토
        print(f"[WARN] UDP payload is large: {len(byte_stream)} bytes")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(timeout)
        sent = sock.sendto(byte_stream, (host, port))
    finally:
        sock.close()
    return sent


def prepare_and_send_udp(
    proc: ProcessedSolution,
    y: np.ndarray,
    startpos_L: np.ndarray, startpos_R: np.ndarray,
    startyaw_L: float, startyaw_R: float,
    host: str, port: int,
    *, drop_first_column_if_ge2: bool = True,
    drop_last_column: bool = True,
    interleave_from_y0: Optional[bool] = None,
) -> Tuple[int, np.ndarray]:
    """
    전체 파이프라인: (proc,y,시작포즈) → combined → 바이트스트림 → UDP 전송
    반환: (보낸 바이트 수, combined)
    """
    combined, _ = build_combined_from_proc(
        proc, y, startpos_L, startpos_R, startyaw_L, startyaw_R,
        drop_first_column_if_ge2=drop_first_column_if_ge2,
        interleave_from_y0=interleave_from_y0,
    )
    combined, N, flat_data, byte_stream = make_udp_payload(
        combined, drop_last_column=drop_last_column
    )

    # 필요하면 여기서 flat_data 프린트 가능
    # import numpy as _np
    # _np.set_printoptions(precision=6, suppress=True, linewidth=220)
    # print(f"[UDP] N={N}, flat_data length={flat_data.size}")
    # print(flat_data)

    sent = send_udp(byte_stream, host, port)
    return sent, combined
