from __future__ import annotations
import time
from typing import Dict, Any, Sequence, Tuple
import numpy as np
# NOTE: 이 파일은 MOSEK Fusion의 Model/Variable/Parameter를 '사용'하지만,
# 굳이 타입 임포트는 하지 않습니다(의존성 격리를 위해).
# solve_relaxed for MOSEK Optimizer API
import numpy as np
import mosek as ms
from typing import Any, Dict, Tuple, Optional


def snapshot_solution_ddr_optapi(M, V, IDX):
    """
    현재 M의 해에서 DDR 변수들을 numpy로 긁어와 dict로 반환.
    - V: 슬라이스(slices)가 들어있어야 함 (y, z_L, z_R, zp_L, zp_R, rho_L, rho_R)
    - IDX: n_edge 필요
    """
    import numpy as np
    import mosek

    E = int(IDX["n_edge"])

    # 1) 전 연속변수 해 (dense) 읽기
    nvar = M.getnumvar()
    xx = [0.0] * nvar
    # interior-point 해 우선, 없으면 basic로 폴백
    try:
        M.getxx(mosek.soltype.itr, xx)
    except mosek.Error:
        M.getxx(mosek.soltype.bas, xx)
    x = np.asarray(xx, dtype=float)

    def _sl(slc):  # slice -> np.array
        return x[slc.start:slc.stop].copy()

    def _edge5(slc):  # (E*5,) -> (E,5)
        flat = _sl(slc)
        if flat.size == 0:
            return np.zeros((E, 5), dtype=float)
        return flat.reshape(E, 5)

    def _edge1(slc):  # (E,) -> (E,)
        flat = _sl(slc)
        if flat.size == 0:
            return np.zeros((E,), dtype=float)
        return flat

    ex = dict(
        y=_edge1(V["y"]),
        z_L=_edge5(V["z_L"]),
        z_R=_edge5(V["z_R"]),
        zp_L=_edge5(V["zp_L"]),
        zp_R=_edge5(V["zp_R"]),
        rho_L=_edge1(V["rho_L"]),
        rho_R=_edge1(V["rho_R"]),
    )
    return ex

def _fix_var_to_value(var, value, *, warm_start: bool = True):
    """
    Fusion 변수 var을 value로 '고정'하려는 유틸.
    ⚠️ Fusion은 생성 후 경계(lower/upper) 변경 API가 없음.
    -> 실사용은 warm-start( setLevel )만 지원.
    -> 진짜 고정은 모델 제약으로(예: ||y - y_relax||_inf <= M_scalar).
    """
    if not hasattr(var, "getShape"):
        raise TypeError("Unsupported variable type in _fix_var_to_value")
    arr = np.asarray(value, dtype=float).reshape(var.getShape())
    if warm_start:
        var.setLevel(arr)
    else:
        raise NotImplementedError(
            "Fusion 변수 경계는 생성 후 변경 불가. "
            "고정하려면 제약(예: M_scalar≈0 + y_relax=value)으로 처리하세요."
        )


def set_parameters(M, V: Dict[str, Any], param_dict: Dict[str, Any]):
    """
    모델 파라미터/워밍스타트 세팅.
    - start_*, target_*: Parameter.setValue
    - y_relax, M_scalar : Parameter.setValue
    - fix_y            : Variable.setLevel (워밍스타트만)
    """
    # Pose parameters
    if "start_L" in param_dict:
        V["start_L"].setValue(np.asarray(param_dict["start_L"], float).reshape(-1))
    if "start_R" in param_dict:
        V["start_R"].setValue(np.asarray(param_dict["start_R"], float).reshape(-1))
    if "target_L" in param_dict:
        V["target_L"].setValue(np.asarray(param_dict["target_L"], float).reshape(-1))
    if "target_R" in param_dict:
        V["target_R"].setValue(np.asarray(param_dict["target_R"], float).reshape(-1))

    # Relax/fixing parameters
    if "y_relax" in param_dict:
        V["y_relax"].setValue(np.asarray(param_dict["y_relax"], float).reshape(-1))
    if "M_scalar" in param_dict:
        V["M_scalar"].setValue(float(param_dict["M_scalar"]))

    # (옵션) y 워밍스타트
    if "fix_y" in param_dict and param_dict["fix_y"] is not None:
        yval = np.asarray(param_dict["fix_y"], float).reshape(-1)
        V["y"].setLevel(yval)   # warm start만, 하드 고정 아님


def set_parameters_optapi(M, V, params):
    """
    Optimizer API에서 파라미터 값을 제약/경계/계수에 반영.
    기대 keys:
      - start_L, start_R, target_L, target_R: 길이 5
      - y_relax: 길이 E
      - M_scalar: float
    """
    import mosek
    meta = V["meta"]
    y_sl = V["y"]

    # --- (A) ||y - y_relax||_inf <= M_scalar  → ub 갱신 ---
    y_relax = np.asarray(params["y_relax"], float)
    M_scalar = float(params["M_scalar"])

    A_rows_upper = meta["A_rows_upper"]  # +y_e <= y_relax[e] + M
    A_rows_lower = meta["A_rows_lower"]  # -y_e <= -(y_relax[e] - M)

    for e, r in enumerate(A_rows_upper):
        ub = y_relax[e] + M_scalar
        M.putconbound(r, mosek.boundkey.up, -0.0, float(ub))
    for e, r in enumerate(A_rows_lower):
        ub = -(y_relax[e] - M_scalar)
        M.putconbound(r, mosek.boundkey.up, -0.0, float(ub))

    # --- (B) source/target flow: y 계수 갱신 ---
    start_L = np.asarray(params["start_L"], float)
    start_R = np.asarray(params["start_R"], float)
    target_L = np.asarray(params["target_L"], float)
    target_R = np.asarray(params["target_R"], float)
    # ===== (J) 5x5 row0 상수 RHS 주입 =====
    y0 = y_sl.start
    E11, E12 = meta["E11"], meta["E12"]
    E21, E22 = meta["E21"], meta["E22"]


    # 소스 엣지의 개수와 행 인덱스 목록 (빌드 시 저장)
    rows_L_X01 = meta.get("J_src_rows_L_X01", [])
    rows_L_X02 = meta.get("J_src_rows_L_X02", [])
    rows_R_X01 = meta.get("J_src_rows_R_X01", [])
    rows_R_X02 = meta.get("J_src_rows_R_X02", [])

    # c0_g, s0_g 계산: [c0_g, s0_g] = E * [c; s]
    cL0 = float(E11*start_L[3] + E12*start_L[4])
    sL0 = float(E21*start_L[3] + E22*start_L[4])
    cR0 = float(E11*start_R[3] + E12*start_R[4])
    sR0 = float(E21*start_R[3] + E22*start_R[4])

    # 등식 RHS 업데이트
    for r in rows_L_X01: M.putconbound(r, mosek.boundkey.fx, cL0, cL0)
    for r in rows_L_X02: M.putconbound(r, mosek.boundkey.fx, sL0, sL0)
    for r in rows_R_X01: M.putconbound(r, mosek.boundkey.fx, cR0, cR0)
    for r in rows_R_X02: M.putconbound(r, mosek.boundkey.fx, sR0, sR0)

    # src
    src_edges = meta["src_edges"]
    rows_src_zL = meta["B_rows_src_zL"]  # list[list[row]]
    rows_src_zR = meta["B_rows_src_zR"]
    for i, e in enumerate(src_edges):
        ycol = y0 + int(e)
        for k in range(5):
            M.putaij(rows_src_zL[i][k], ycol, -float(start_L[k]))
            M.putaij(rows_src_zR[i][k], ycol, -float(start_R[k]))

    # sink
    sink_edges = meta["sink_edges"]
    rows_sink_zpL = meta["B_rows_sink_zpL"]
    rows_sink_zpR = meta["B_rows_sink_zpR"]
    for i, e in enumerate(sink_edges):
        ycol = y0 + int(e)
        for k in range(5):
            M.putaij(rows_sink_zpL[i][k], ycol, -float(target_L[k]))
            M.putaij(rows_sink_zpR[i][k], ycol, -float(target_R[k]))
    # ===== (K) 7x7 row0 상수 RHS 주입 =====
    # E는 meta["E**"]에 이미 저장되어 있다고 가정 (위에서 처리)
    start_L = np.asarray(params["start_L"], float)
    start_R = np.asarray(params["start_R"], float)
    target_L = np.asarray(params["target_L"], float)
    target_R = np.asarray(params["target_R"], float)

    E11 = V["meta"]["E11"]; E12 = V["meta"]["E12"]
    E21 = V["meta"]["E21"]; E22 = V["meta"]["E22"]

    # 시작 yaw 전역 성분
    cL0 = float(E11*start_L[3] + E12*start_L[4])
    sL0 = float(E21*start_L[3] + E22*start_L[4])
    cR0 = float(E11*start_R[3] + E12*start_R[4])
    sR0 = float(E21*start_R[3] + E22*start_R[4])

    for r in V["meta"].get("K_src_rows_XL2_01", []): M.putconbound(r, mosek.boundkey.fx, float(start_L[0]), float(start_L[0]))
    for r in V["meta"].get("K_src_rows_XL2_02", []): M.putconbound(r, mosek.boundkey.fx, float(start_L[1]), float(start_L[1]))
    for r in V["meta"].get("K_src_rows_XL2_03", []): M.putconbound(r, mosek.boundkey.fx, cL0, cL0)
    for r in V["meta"].get("K_src_rows_XL2_04", []): M.putconbound(r, mosek.boundkey.fx, sL0, sL0)

    for r in V["meta"].get("K_src_rows_XR2_01", []): M.putconbound(r, mosek.boundkey.fx, float(start_R[0]), float(start_R[0]))
    for r in V["meta"].get("K_src_rows_XR2_02", []): M.putconbound(r, mosek.boundkey.fx, float(start_R[1]), float(start_R[1]))
    for r in V["meta"].get("K_src_rows_XR2_03", []): M.putconbound(r, mosek.boundkey.fx, cR0, cR0)
    for r in V["meta"].get("K_src_rows_XR2_04", []): M.putconbound(r, mosek.boundkey.fx, sR0, sR0)

    # 타깃 위치
    for r in V["meta"].get("K_sink_rows_XL2_05", []): M.putconbound(r, mosek.boundkey.fx, float(target_R[0]), float(target_R[0]))
    for r in V["meta"].get("K_sink_rows_XL2_06", []): M.putconbound(r, mosek.boundkey.fx, float(target_R[1]), float(target_R[1]))
    for r in V["meta"].get("K_sink_rows_XR2_05", []): M.putconbound(r, mosek.boundkey.fx, float(target_L[0]), float(target_L[0]))
    for r in V["meta"].get("K_sink_rows_XR2_06", []): M.putconbound(r, mosek.boundkey.fx, float(target_L[1]), float(target_L[1]))


def solve_relaxed(M, V: Dict[str, Any], base_param=None, M_large: float = 1e3, log_level: int = 0):
    """1차: y_relax=0, M_scalar=M_large 상태에서 풀이."""
    M.setLogHandler(None if log_level == 0 else print)
    M.solve()

    pstat = str(M.getProblemStatus())
    sstat = str(M.getPrimalSolutionStatus())

    obj = None
    try:
        obj = float(M.primalObjValue())
    except Exception:
        try:
            M.writeTask("relaxed_problem.opf")
            print("[MOSEK] wrote relaxed_problem.opf for inspection")
        except Exception:
            pass

    return {"pstatus": pstat, "sstatus": sstat}, obj


import mosek

def solve_relaxed_ddr_optapi(task, V, IDX=None, PAR=None, M_large: float = 1e3, log_level: int = 0):
    """
    1차 릴랙스 풀이 (Optimizer API).
    - Fusion의 solve_relaxed(M, V, M_large, log_level)와 1:1 의미로 동작
    - 여기서는 파라미터 세팅이 이미 끝났다고 가정하고 바로 optimize()만 수행
    - log_level=0이면 로그 끄고, >0이면 표준출력으로 로그 전달
    반환: (info_dict, obj)
      info_dict = {"pstatus": str(prosta), "sstatus": str(solsta)}
      obj: primal objective (float) or None on failure
    """
    # ---- LOG 설정 ----
    if log_level == 0:
        # 완전 침묵
        task.putintparam(mosek.iparam.log, 0)
        # stream 핸들러 제거(안 붙여도 되지만 안전하게)
        task.set_Stream(mosek.streamtype.log, None)
    else:
        task.putintparam(mosek.iparam.log, 1)
        def _printer(msg):
            print(msg, end="")
        task.set_Stream(mosek.streamtype.log, _printer)

    # ---- Optimize ----
    task.optimize()

    # ---- 상태 조회 (interior solution 우선) ----
    # 연속(릴랙스) 문제이므로 itr 해가 정상. 없으면 bas/itg 순으로 폴백.
    soltype_order = [mosek.soltype.itr, mosek.soltype.bas, mosek.soltype.itg]
    prosta = None
    solsta = None
    used_soltype = None
    for st in soltype_order:
        try:
            prosta = task.getprosta(st)
            solsta = task.getsolsta(st)
            used_soltype = st
            break
        except Exception:
            continue

    pstat_str = str(prosta) if prosta is not None else "Unknown"
    sstat_str = str(solsta) if solsta is not None else "Unknown"

    # ---- 목적함수 ----
    obj = None
    if used_soltype is not None:
        try:
            obj = float(task.getprimalobj(used_soltype))
        except Exception:
            obj = None

    # 실패 시 문제 덤프 (Fusion의 writeTask와 동일 목적)
    if obj is None:
        try:
            task.writedata("relaxed_problem.opf")
            print("[MOSEK] wrote relaxed_problem.opf for inspection")
        except Exception:
            pass

    return {"pstatus": pstat_str, "sstatus": sstat_str}, obj



def solve_relaxed_optapi(M, V, log_level=0):
    import mosek

    # 로그
    M.putintparam(mosek.iparam.log, int(log_level))

    # 최적화
    M.optimize()

    # 해 추출 헬퍼
    def _get_slice(slc):
        idx0, idx1 = slc.start, slc.stop
        arr = [0.0] * (idx1 - idx0)
        M.getxxslice(mosek.soltype.itr, idx0, idx1, arr)
        return np.array(arr)

    sol = {}
    for key in ["y", "z_L", "z_R", "zp_L", "zp_R",
                "rho_L", "rho_R", "rhoz_L", "rhoz_R"]:
        sl = V.get(key)
        if isinstance(sl, slice):
            sol[key] = _get_slice(sl)

    # 목적값
    obj = M.getprimalobj(mosek.soltype.itr)

    return sol, obj

def set_parameters_ddr_optapi(task, V, IDX, PAR, params):
    """
    Fusion의 set_parameters와 1:1 의미:
      params = {
        "geometry": (2*Vn,), "start_L":(5,), "start_R":(5,),
        "target_L":(5,), "target_R":(5,),
        "M_scalar": float, "y_relax": (E,)
      }
    - PAR 딕셔너리에 값을 반영
    - 파라미터에 의존하는 제약/경계/계수 갱신
    """
    import numpy as np
    E = IDX["n_edge"]

    # ---- 0) PAR 업데이트 ----
    if "geometry" in params:
        PAR["geometry"] = np.asarray(params["geometry"], float).reshape(-1)
    if "start_L" in params:  PAR["start_L"]  = np.asarray(params["start_L"],  float).reshape(-1)
    if "start_R" in params:  PAR["start_R"]  = np.asarray(params["start_R"],  float).reshape(-1)
    if "target_L" in params: PAR["target_L"] = np.asarray(params["target_L"], float).reshape(-1)
    if "target_R" in params: PAR["target_R"] = np.asarray(params["target_R"], float).reshape(-1)
    if "M_scalar" in params: PAR["M_scalar"] = float(params["M_scalar"])
    if "y_relax"  in params: PAR["y_relax"]  = np.asarray(params["y_relax"],  float).reshape(E)

    # ---- (A) ||y - y_relax||_inf <= M_scalar : UB 갱신 ----
    if IDX.get("con_A_inf") is not None:
        rows_pos = IDX["con_A_inf"]["row_ub_pos"]  # y - yrelax <= M
        rows_neg = IDX["con_A_inf"]["row_ub_neg"]  # -y + yrelax <= M
        yrelax = PAR["y_relax"]; Mscal = float(PAR["M_scalar"])
        for e, r in enumerate(rows_pos):
            task.putconbound(r, ms.boundkey.up, 0.0, yrelax[e] + Mscal)
        for e, r in enumerate(rows_neg):
            task.putconbound(r, ms.boundkey.up, 0.0, -yrelax[e] + Mscal)

    # ---- (B) source/target flow: 계수 갱신 (start/target이 바뀔 수 있으므로) ----
    # 저장 구조: con_flow = {"src_rows_L":[], "src_rows_R":[], "sink_rows_L":[], "sink_rows_R":[]}
    # 생성 순서: for e in src_e: for k in 0..4: row 추가 → 인덱싱으로 역추적
    cf = IDX.get("con_flow")
    if cf is not None:
        # 준비
        startL = PAR["start_L"]; startR = PAR["start_R"]
        targL  = PAR["target_L"]; targR  = PAR["target_R"]
        y_cols = list(range(V["y"].start, V["y"].stop))

        # src: zL(e,k) - startL[k]*y[e] = 0
        rows = cf.get("src_rows_L", [])
        for idx, r in enumerate(rows):
            e = idx // 5; k = idx % 5
            y_col = y_cols[IDX["src_e"][e]] if "src_e" in IDX else y_cols[ np.where(IDX["outV"]==0)[0][e] ]
            task.putaij(r, y_col, -float(startL[k]))

        rows = cf.get("src_rows_R", [])
        for idx, r in enumerate(rows):
            e = idx // 5; k = idx % 5
            y_col = y_cols[IDX["src_e"][e]] if "src_e" in IDX else y_cols[ np.where(IDX["outV"]==0)[0][e] ]
            task.putaij(r, y_col, -float(startR[k]))

        # sink: zpL(e,k) - targetL[k]*y[e] = 0
        rows = cf.get("sink_rows_L", [])
        for idx, r in enumerate(rows):
            e = idx // 5; k = idx % 5
            sink_edges = IDX["sink_e"] if "sink_e" in IDX else np.where(IDX["inV"] == IDX["vertex_number"]-1)[0]
            y_col = y_cols[ sink_edges[e] ]
            task.putaij(r, y_col, -float(targL[k]))

        rows = cf.get("sink_rows_R", [])
        for idx, r in enumerate(rows):
            e = idx // 5; k = idx % 5
            sink_edges = IDX["sink_e"] if "sink_e" in IDX else np.where(IDX["inV"] == IDX["vertex_number"]-1)[0]
            y_col = y_cols[ sink_edges[e] ]
            task.putaij(r, y_col, -float(targR[k]))

    # ---- (F) region ineq: geometry 바뀌면 y계수 (-(b + A g)) 갱신 ----
    # 저장 구조: con_region = [ { "rows":[...], "A":A, "b":b, "g_vid":v, "y_col": <int> }, ... ]
    if IDX.get("con_region") is not None and PAR.get("geometry") is not None:
        geom = PAR["geometry"].reshape(-1)
        for blk in IDX["con_region"]:
            rows = blk["rows"]; A = blk["A"]; b = blk["b"]; v = blk["g_vid"]
            y_col = blk.get("y_col", None)
            if y_col is None:
                # 빌드 단계에서 y_col을 저장하지 않았다면, 추적 필요.
                # 지금 구조에선 y_scalar 컬럼은 e별로 다르므로, 빌더에서 반드시 blk["y_col"]을 저장하도록 패치 권장.
                continue
            gx, gy = geom[2*v], geom[2*v+1]
            for r_local, r in enumerate(rows):
                a0 = float(A[r_local,0]); a1 = float(A[r_local,1]); beta = float(b[r_local])
                coeff = -(beta + a0*gx + a1*gy)
                task.putaij(r, y_col, coeff)


def solve_with_fixed_path(M, V: Dict[str, Any], y_relax_vec,
                          M_small: float = 1e-8,
                          base_param=None, log_level: int = 0,
                          *,
                          return_used: bool = False,
                          on_solved=None):
    """
    2차: 경로 고정 → y_relax=path, M_scalar≈0 (사실상 y≈y_relax).
    확장:
      - on_solved(M_used, V_used): 풀이 직후 콜백(스냅샷에 사용)
      - return_used=True: (status, obj, M_used, V_used) 반환
    """
    import numpy as np

    # --- 파라미터/워밍스타트 ---
    yv = np.asarray(y_relax_vec, float).reshape(-1)
    if "y_relax" in V:   V["y_relax"].setValue(yv)
    if "M_scalar" in V:  V["M_scalar"].setValue(float(M_small))
    if "y" in V:         V["y"].setLevel(yv)  # warm start

    # 로깅
    M.setLogHandler(None if log_level == 0 else print)

    # --- 여기서는 M, V 그대로 사용(별도 작은 모델을 만들지 않음) ---
    M_used, V_used = M, V

    # --- 풀이 ---
    M_used.solve()

    # --- 풀이 직후 콜백(스냅샷은 여기서!) ---
    if on_solved is not None:
        try:
            on_solved(M_used, V_used)
        except Exception as _:
            # 스냅샷에서 실패하더라도 본 함수는 계속 진행
            pass

    # --- 상태/목표값 수집 ---
    pstat = str(M_used.getProblemStatus())
    sstat = str(M_used.getPrimalSolutionStatus())
    status = {"pstatus": pstat, "sstatus": sstat}

    obj = None
    try:
        obj = float(M_used.primalObjValue())
    except Exception:
        try:
            M_used.writeTask("fixedpath_problem.opf")
            print("[MOSEK] wrote fixedpath_problem.opf for inspection")
        except Exception:
            pass

    if return_used:
        return status, obj, M_used, V_used
    else:
        return status, obj



def choose_best_solution(candidates: Sequence[Tuple[Dict[str, Any], float]]):
    """여러 trial이 있을 때 최소 cost 선택 (지금은 첫 번째 반환)"""
    if not candidates:
        raise RuntimeError("No feasible solution candidates.")
    return candidates[0]


def get_relaxed_y(V: Dict[str, Any]):
    """Fusion solve 후 relaxed y 값을 (n_edge,) numpy array로 반환"""
    y = np.asarray(V["y"].level(), dtype=float)
    return y.reshape(-1)

def _pick_soltype_for_read(task):
    # 연속 릴랙스는 interior(itr)가 기본. 없으면 bas → itg 순으로 폴백
    for st in (mosek.soltype.itr, mosek.soltype.bas, mosek.soltype.itg):
        try:
            task.getsolsta(st)  # 존재 여부 체크 목적
            return st
        except Exception:
            pass
    return None

def get_relaxed_y_ddr_optapi(task, V):
    """
    Fusion의 get_relaxed_y(V)와 동일 의미:
    solve 후 y의 level을 (n_edge,) numpy array로 반환
    """
    st = _pick_soltype_for_read(task)
    if st is None:
        return np.zeros(V["y"].stop - V["y"].start, dtype=float)

    a, b = V["y"].start, V["y"].stop
    buf = [0.0] * (b - a)
    task.getxxslice(st, a, b, buf)
    return np.asarray(buf, dtype=float).reshape(-1)


# def extract_solution(V: Dict[str, Any], IDX: Dict[str, Any]):
#     """Fusion 변수에서 numpy로 값 추출."""
#     n = int(IDX["n_edge"])
#     get = lambda k: np.asarray(V[k].level(), dtype=float)
#
#     y    = get("y").reshape(n)
#     z_L  = get("z_L").reshape(n, 5)
#     zp_L = get("zp_L").reshape(n, 5)
#     z_R  = get("z_R").reshape(n, 5)
#     zp_R = get("zp_R").reshape(n, 5)
#
#     rho_L  = get("rho_L").reshape(n)
#     rho_R  = get("rho_R").reshape(n)
#     rhoz_L = get("rhoz_L").reshape(n)
#     rhoz_R = get("rhoz_R").reshape(n)
#
#     return dict(y=y, z_L=z_L, zp_L=zp_L, z_R=z_R, zp_R=zp_R,
#                 rho_L=rho_L, rho_R=rho_R, rhoz_L=rhoz_L, rhoz_R=rhoz_R)

def extract_solution(V, IDX):
    import numpy as np
    n = int(IDX["n_edge"])

    def get_mat(name, shape):
        if name not in V: return np.full(shape, np.nan)
        try: return np.asarray(V[name].level(), dtype=float).reshape(*shape)
        except Exception: return np.full(shape, np.nan)

    def get_list(name):
        L = V.get(name, [])
        out = []
        for var in L:
            try: out.append(float(var.level()[0]))
            except Exception: out.append(np.nan)
        return np.asarray(out, dtype=float)

    # --- PSD helpers ---
    def vec_to_sym_matrix(vals, dim):
        vals = np.asarray(vals, float).ravel()
        if vals.size == dim*dim:
            return vals.reshape(dim, dim, order='F')  # Fusion은 column-major
        if vals.size == dim*(dim+1)//2:              # 삼각 저장일 때 복원
            A = np.zeros((dim, dim), float); k = 0
            for j in range(dim):
                for i in range(j, dim):
                    A[i, j] = A[j, i] = vals[k]; k += 1
            return A
        return None  # 예외적 형상

    def get_psd_list(name, dim):
        L = V.get(name)
        if L is None: return None
        mats = []
        for var in L:
            try:
                vals = var.level()   # 인자 없이!
                A = vec_to_sym_matrix(vals, dim)
            except Exception:
                A = None
            mats.append(A)
        return mats  # 길이 n_edge, 각 원소는 np.ndarray(dim,dim) 또는 None

    # --- 기존 ---
    y    = get_mat("y",    (n,))
    z_L  = get_mat("z_L",  (n,5))
    zp_L = get_mat("zp_L", (n,5))
    z_R  = get_mat("z_R",  (n,5))
    zp_R = get_mat("zp_R", (n,5))
    rho_L  = get_mat("rho_L",  (n,))
    rho_R  = get_mat("rho_R",  (n,))
    rhoz_L = get_mat("rhoz_L", (n,))
    rhoz_R = get_mat("rhoz_R", (n,))

    ux    = get_list("ux")
    uy    = get_list("uy")
    s_abs = get_list("s_abs")


    # --- PSD 스냅샷 추가 ---
    X_L  = get_psd_list("X_L",  5)
    X_R  = get_psd_list("X_R",  5)
    X_L2 = get_psd_list("X_L2", 7)
    X_R2 = get_psd_list("X_R2", 7)

    return dict(
        y=y, z_L=z_L, zp_L=zp_L, z_R=z_R, zp_R=zp_R,
        rho_L=rho_L, rho_R=rho_R, rhoz_L=rhoz_L, rhoz_R=rhoz_R,
        ux=ux, uy=uy, s_abs=s_abs,
        X_L=X_L, X_R=X_R, X_L2=X_L2, X_R2=X_R2
    )



def recover_x_from_z_over_y(z: np.ndarray, y: np.ndarray, tol: float = 1e-9):
    """
    MATLAB의 x = z ./ y 와 동일.
    z: (E,5), y: (E,)
    선택된 엣지(y>0.49)만 안전하게 나눠서 (N_sel,5) 반환.
    """
    idx = np.where(y > 0.49)[0]
    if idx.size == 0:
        return idx, np.zeros((0, 5))
    yy = y[idx].reshape(-1, 1)
    yy = np.clip(yy, tol, None)
    return idx, z[idx] / yy
