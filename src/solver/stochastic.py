# stochastic.py
from __future__ import annotations
import mosek.fusion as mf
import time
from . import relaxation_max, relaxation_prob, path_to_y, extract_solution, set_parameters, solve_with_fixed_path
from ..model import pack_fixedpath_params_for_model
from src.solver.utils import set_parameters_optapi, set_parameters_ddr_optapi, snapshot_solution_ddr_optapi
from typing import Any, Dict, Tuple, Optional
import numpy as np
import mosek as ms
from typing import Any, Dict, Tuple
from src.solver.utils import solve_relaxed_optapi

def get_var_slice(V: Dict[str, Any], name: str, M: Tuple[ms.Env, ms.Task] = None) -> np.ndarray:
    """Fusion/OptAPI 겸용 변수 읽기: V[name]이 Fusion Variable이면 level(), OptAPI면 (start,n)로 해에서 슬라이스."""
    slot = V[name]
    # Fusion
    if hasattr(slot, "level"):
        return np.asarray(slot.level(), dtype=float).reshape(-1)
    # OptAPI
    if isinstance(slot, tuple) and len(slot) == 2:
        s, n = slot
        xx = V.get("__LAST_SOL__")
        if xx is None:
            if M is None:
                raise RuntimeError("OptAPI: M이 없고 V['__LAST_SOL__']도 없습니다.")
            env, task = M
            xx_full = [0.0] * task.getnumvar()
            task.getxx(ms.soltype.itr, xx_full)
            xx = np.asarray(xx_full, dtype=float)
        return xx[s:s+n].copy()
    raise TypeError(f"V['{name}'] 형식을 알 수 없습니다.")

def _is_feasible_opt(pstatus, sstatus):
    """
    pstatus/sstatus는 문자열로 넘어온다고 가정(Fusion과 동일).
    MOSEK 11에는 near_optimal이 없으므로 hasattr로 안전 체크.
    """
    good = {str(ms.solsta.optimal), str(ms.solsta.prim_feas)}

    # 일부 버전에서만 존재하는 상태들을 옵션으로 추가
    for name in ("near_optimal", "near_prim_feas"):
        if hasattr(ms.solsta, name):
            good.add(str(getattr(ms.solsta, name)))

    return sstatus in good



def _is_feasible(sol: Dict[str, str], obj: Optional[float]) -> bool:
    if obj is None or not np.isfinite(obj):
        return False
    p = sol.get("pstatus", "").lower()
    s = sol.get("sstatus", "").lower()
    if "infeasible" in p:
        return False
    if s in ("unknown", "undefined"):
        return False
    return True


def _level_safe(var):
    """
    Fusion 변수 값을 안전하게 읽기.
    - PSD/행렬 변수는 인자 없는 level()이 가장 안정적
    - 실패하면 솔루션 타입을 순회 (연속 변수에서만 의미 있는 경우가 많음)
    """
    try:
        vals = var.level()  # 인자 없이!
        return np.array(vals)
    except Exception:
        pass

    for st in (mf.SolutionType.Interior, mf.SolutionType.Default, mf.SolutionType.Basic):
        try:
            vals = var.level(st)
            return np.array(vals)
        except Exception:
            pass
    return None


def _vec_to_sym_matrix(vals: np.ndarray, n: int):
    """
    Fusion의 PSD 변수 레벨을 dense 대칭행렬로 복원.
    - full (n*n) 또는 packed tri (n*(n+1)//2) 지원
    """
    vals = np.asarray(vals, float).ravel()
    if vals.size == n * n:  # full vectorization (Fortran order)
        return vals.reshape(n, n, order='F')
    if vals.size == n * (n + 1) // 2:  # packed lower/upper → symmetric
        A = np.zeros((n, n), float)
        k = 0
        for j in range(n):
            for i in range(j, n):
                A[i, j] = A[j, i] = vals[k]
                k += 1
        return A
    return None


def _psd_to_matrix_list(var_list, n: int):
    """PSD 변수 리스트 -> dense 행렬 리스트 (읽기 실패 시 None)"""
    mats = []
    for var in var_list:
        vals = _level_safe(var)
        if vals is None:
            mats.append(None)
            continue
        A = _vec_to_sym_matrix(vals, n)
        mats.append(A)
    return mats


def _solve_fixed_once(
    M, V, IDX, *,
    centroid_mat, y_relax, M_small,
    start_L, start_R, target_L, target_R,
    log_level: int = 0,
):
    """
    한 경로에 대해 파라미터 세팅 → solve → 즉시 스냅샷(ex) 구성.
    - on_solved 콜백에서 X_*를 level()로 바로 복사해 snap에 넣음
    - 이후 extract_solution로 나머지 변수 스냅샷을 만들고 snap을 합침
    """
    # NearOptimal/Feasible 도 읽기 허용
    M.acceptedSolutionStatus(mf.AccSolutionStatus.Feasible)

    # 고정 경로 파라미터 세팅
    params = pack_fixedpath_params_for_model(
        centroid_mat=centroid_mat, y_relax=y_relax, M_small=M_small,
        start_L=start_L, start_R=start_R, target_L=target_L, target_R=target_R
    )
    set_parameters(M, V, {**params, "fix_y": y_relax})

    # --- PSD 스냅샷을 담아둘 컨테이너 ---
    snap: Dict[str, Any] = {}

    # --- solve 직후 호출되어 PSD를 바로 복사하는 콜백 ---
    def _cb(M_used, V_used):
        # V_used에 X_*가 있으면 바로 읽어 스냅샷
        X_L_list  = V_used.get("X_L",  None) if isinstance(V_used, dict) else None
        X_R_list  = V_used.get("X_R",  None) if isinstance(V_used, dict) else None
        X_L2_list = V_used.get("X_L2", None) if isinstance(V_used, dict) else None
        X_R2_list = V_used.get("X_R2", None) if isinstance(V_used, dict) else None

        if X_L_list  is not None: snap["X_L"]  = _psd_to_matrix_list(X_L_list,  5)
        if X_R_list  is not None: snap["X_R"]  = _psd_to_matrix_list(X_R_list,  5)
        if X_L2_list is not None: snap["X_L2"] = _psd_to_matrix_list(X_L2_list, 7)
        if X_R2_list is not None: snap["X_R2"] = _psd_to_matrix_list(X_R2_list, 7)

    # --- 풀기 (solve_with_fixed_path가 콜백을 호출하도록) ---
    sol, obj, M_used, V_used = solve_with_fixed_path(
        M, V, y_relax, M_small=M_small, log_level=log_level,
        return_used=True, on_solved=_cb
    )

    if not _is_feasible(sol, obj):
        return sol, obj, None

    # 나머지 변수 스냅샷 (가능하면 V_used 기반)
    try:
        ex = extract_solution(V_used, IDX)
    except Exception:
        ex = extract_solution(V, IDX)

    # PSD 스냅샷 합치기
    ex.update(snap)
    return sol, obj, ex

def _solve_fixed_once_optapi(
    M, V, IDX, *,
    centroid_mat, y_relax, M_small,
    start_L, start_R, target_L, target_R,
    log_level: int = 0,
):
    """
    Optimizer API(Task) 버전.
    - 고정 경로(y 고정/근처) 파라미터 적용 -> solve -> 스냅샷(ex) 구성
    - Fusion의 acceptedSolutionStatus 등은 사용 안 하고, solsta/prosta로 판별
    """
    import mosek, numpy as np

    # 1) 파라미터 세팅 (fixed-path용)
    params = pack_fixedpath_params_for_model(
        centroid_mat=centroid_mat, y_relax=y_relax, M_small=M_small,
        start_L=start_L, start_R=start_R, target_L=target_L, target_R=target_R
    )
    # 기존 set_parameters에서 (A)(B) 등 갱신
    set_parameters_optapi(M, V, params)

    # 2) y 고정(또는 근처): y_e ∈ [fix - M_small, fix + M_small] ∩ [0,1]
    # --- 추가: y_relax를 1D로 평탄화 ---
    y_fix = np.asarray(y_relax, dtype=float).reshape(-1)

    # 2) y 고정(또는 근처): y_e ∈ [fix - M_small, fix + M_small] ∩ [0,1]
    y_sl  = V["y"]
    y_idx = list(range(y_sl.start, y_sl.stop))

    lb_np = np.maximum(0.0, y_fix - float(M_small))
    ub_np = np.minimum(1.0, y_fix + float(M_small))

    # 길이 검증(안전)
    assert len(y_idx) == lb_np.size == ub_np.size, \
        f"length mismatch: {len(y_idx)} vs {lb_np.size} vs {ub_np.size}"

    # 파이썬 float 리스트로 변환
    lb = [float(v) for v in lb_np.ravel()]
    ub = [float(v) for v in ub_np.ravel()]

    n = len(y_idx)
    M.putvarboundlist(y_idx, [mosek.boundkey.ra] * n, lb, ub)

    def _is_ok_solsta(solsta):
        import mosek
        ok = (solsta == mosek.solsta.optimal)
        # MIP일 경우 대비(아닐 가능성이 높지만 무해)
        ok = ok or (hasattr(mosek.solsta, "integer_optimal") and solsta == mosek.solsta.integer_optimal)
        # 일부 버전에서만 존재하는 near_optimal 지원(있으면 허용)
        if hasattr(mosek.solsta, "near_optimal"):
            ok = ok or (solsta == mosek.solsta.near_optimal)
        return ok

    # 3) 풀기
    M.putintparam(mosek.iparam.log, int(log_level))
    M.optimize()
    # solve_relaxed_optapi / _solve_fixed_once_optapi 공통 패치
    prosta = M.getprosta(mosek.soltype.itr)
    solsta = M.getsolsta(mosek.soltype.itr)
    if not _is_ok_solsta(solsta):
        raise RuntimeError(f"MOSEK solve failed: prosta={prosta}, solsta={solsta}")

    # 4) 해 읽기 (연속 변수)
    def _read_slice(slc: slice):
        arr = [0.0]*(slc.stop - slc.start)
        M.getxxslice(mosek.soltype.itr, slc.start, slc.stop, arr)
        return np.asarray(arr, dtype=float)

    sol = {}
    for key in ["y","z_L","z_R","zp_L","zp_R","rho_L","rho_R","rhoz_L","rhoz_R"]:
        if isinstance(V.get(key), slice):
            sol[key] = _read_slice(V[key])

    # 5) PSD(bar) 변수 읽기 → 대칭행렬로 복원
    def _read_bar_matrix(j: int) -> np.ndarray:
        # barvar j의 차원
        dim = M.getdimbarvarj(j)
        # 하삼각 packed 길이 = dim*(dim+1)//2
        x = [0.0]*(dim*(dim+1)//2)
        M.getbarxj(mosek.soltype.itr, j, x)
        # packed -> full symmetric
        A = np.zeros((dim, dim), dtype=float)
        idx = 0
        for col in range(dim):
            for row in range(col, dim):
                A[row, col] = x[idx]
                A[col, row] = x[idx]
                idx += 1
        return A

    # 6) ex 구성: (E,5) 행렬로 표준화해서 넣기 + y는 (E,)
    E = int(IDX["n_edge"])

    def _as_E5(name):
        """sol[name] -> (E,5) float array"""
        v = sol.get(name)
        if v is None:
            return None
        arr = np.asarray(v, dtype=float)
        if arr.ndim == 1:
            # 벡터(E*5,)로 온 경우
            assert arr.size == E * 5, f"{name}: size {arr.size} != E*5 ({E * 5})"
            return arr.reshape(E, 5)
        elif arr.ndim == 2:
            # 이미 2D인 경우 (E,5) 가정 (안전상 클립)
            assert arr.shape[1] >= 5, f"{name}: expected >=5 columns, got {arr.shape}"
            if arr.shape[1] != 5:
                arr = arr[:, :5]
            assert arr.shape[0] == E, f"{name}: expected {E} rows, got {arr.shape[0]}"
            return arr.astype(float, copy=False)
        else:
            raise ValueError(f"{name}: unexpected ndim={arr.ndim}")

    def _as_E1(name):
        """sol[name] -> (E,) float array"""
        v = sol.get(name)
        if v is None:
            return None
        arr = np.asarray(v, dtype=float).reshape(-1)
        assert arr.size == E, f"{name}: expected length {E}, got {arr.size}"
        return arr

    ex = {
        # 연속 변수 (표준화 후)
        "y": _as_E1("y"),
        "z_L": _as_E5("z_L"),
        "z_R": _as_E5("z_R"),
        "zp_L": _as_E5("zp_L"),
        "zp_R": _as_E5("zp_R"),
        "rho_L": _as_E1("rho_L"),
        "rho_R": _as_E1("rho_R"),
        "rhoz_L": _as_E1("rhoz_L"),
        "rhoz_R": _as_E1("rhoz_R"),
    }

    # 7) PSD(bar) 변수도 그대로 추가 (이전에 읽은 것 재사용)
    if "X_L" in V and isinstance(V["X_L"], list):
        ex["X_L"] = [_read_bar_matrix(j) for j in V["X_L"]]
    if "X_R" in V and isinstance(V["X_R"], list):
        ex["X_R"] = [_read_bar_matrix(j) for j in V["X_R"]]
    if "X_L2" in V and isinstance(V["X_L2"], list):
        ex["X_L2"] = [_read_bar_matrix(j) for j in V["X_L2"]]
    if "X_R2" in V and isinstance(V["X_R2"], list):
        ex["X_R2"] = [_read_bar_matrix(j) for j in V["X_R2"]]

    # 8) 목적값
    obj = M.getprimalobj(mosek.soltype.itr)

    return sol, obj, ex


def run_trials_select_best(
    M, V, IDX, *,
    centroid_mat, y_relaxed_val, edge, vertex_number, n_edge,
    M_small, start_L, start_R, target_L, target_R,
    trials: int = 50,
    time_budget_sec=None,
    temp: float = 1.0, greedy_first: bool = True,
    print_fn=print,
    early_stop: bool = True,
):
    best_obj = np.inf
    best = None
    feasible_cnt = 0
    attempted_cnt = 0

    # ===== time budget helper =====
    t0 = time.perf_counter()

    def _budget_enabled():
        return (time_budget_sec is not None) and (float(time_budget_sec) > 0.0)

    def _time_over():
        return _budget_enabled() and ((time.perf_counter() - t0) >= float(time_budget_sec))

    def _try_path(path, tag: str):
        nonlocal best_obj, best, feasible_cnt, attempted_cnt

        attempted_cnt += 1
        y_relax = path_to_y(edge, path, n_edge)

        sol, obj, ex = _solve_fixed_once(
            M, V, IDX,
            centroid_mat=centroid_mat, y_relax=y_relax, M_small=M_small,
            start_L=start_L, start_R=start_R, target_L=target_L, target_R=target_R
        )
        if not _is_feasible(sol, obj):
            return False

        feasible_cnt += 1

        if obj < best_obj:
            best_obj = obj
            best = dict(path=path, y_relax=y_relax, sol=sol, ex=ex)
        return True

    # ===== 단일 트라이얼 =====
    if trials <= 1:
        if greedy_first:
            path0 = relaxation_max(y_relaxed_val, edge, vertex_number)
            ok = _try_path(path0, "greedy")
            if not ok:
                raise RuntimeError("Greedy fixed-path infeasible.")
        else:
            try:
                path1 = relaxation_prob(y_relaxed_val, edge, vertex_number, temperature=float(temp))
            except Exception as e:
                if print_fn: print_fn(f"[prob-1] relaxation_prob failed: {e}, fallback greedy")
                path1 = relaxation_max(y_relaxed_val, edge, vertex_number)
            ok = _try_path(path1, "prob-1")
            if not ok:
                raise RuntimeError("Probabilistic fixed-path infeasible.")

        # 단일이라도 budget 정보 출력(원하면 제거 가능)
        if _budget_enabled() and print_fn:
            elapsed = time.perf_counter() - t0
            print_fn(f"[trial_budget] attempted={attempted_cnt}, elapsed={elapsed:.3f}s / budget={float(time_budget_sec):.3f}s")
        return best_obj, best

    # ===== 다중 트라이얼 =====
    rest = trials

    # (A) greedy first
    if greedy_first:
        try:
            path0 = relaxation_max(y_relaxed_val, edge, vertex_number)
            ok0 = _try_path(path0, "greedy")
            rest -= 1

            # greedy 수행 후 시간 초과면 여기서 종료
            if _time_over():
                if best is None:
                    raise RuntimeError("Time budget exceeded and no feasible solution found (after greedy).")
                if _budget_enabled() and print_fn:
                    elapsed = time.perf_counter() - t0
                    print_fn(f"[trial_budget] stop after greedy: attempted={attempted_cnt}, elapsed={elapsed:.3f}s / budget={float(time_budget_sec):.3f}s")
                if print_fn: print_fn(f"[stochastic] best of {feasible_cnt}/{attempted_cnt} = {best_obj:.6f}")
                return best_obj, best

            if early_stop and ok0:
                return best_obj, best

        except Exception as e:
            if print_fn: print_fn(f"[greedy] failed: {e}")

    # (B) stochastic trials
    temp = float(temp)
    for t in range(rest):
        # 다음 trial 시작 전에 budget 체크
        if _time_over():
            break

        try:
            path_t = relaxation_prob(y_relaxed_val, edge, vertex_number, temperature=temp)
        except Exception as e:
            if print_fn: print_fn(f"[trial {t}] relaxation_prob failed: {e}, fallback greedy")
            path_t = relaxation_max(y_relaxed_val, edge, vertex_number)

        ok_t = _try_path(path_t, f"trial {t}")

        if early_stop and ok_t:
            if print_fn: print_fn(f"[early_stop] feasible at trial {t} → return")
            return best_obj, best

        # solve가 끝난 직후에도 budget 체크(다음 trial로 넘어가기 전에 끊기)
        if _time_over():
            break

    if best is None:
        raise RuntimeError("No feasible solution among trials (or within time budget).")

    if _budget_enabled() and print_fn:
        elapsed = time.perf_counter() - t0
        print_fn(f"[trial_budget] attempted={attempted_cnt}/{trials}, elapsed={elapsed:.3f}s / budget={float(time_budget_sec):.3f}s")

    if print_fn: print_fn(f"[stochastic] best of {feasible_cnt}/{attempted_cnt} = {best_obj:.6f}")
    return best_obj, best


def run_trials_select_best_optapi_timelimit(
    M, V, IDX, *,
    centroid_mat, y_relaxed_val, edge, vertex_number, n_edge,
    M_small, start_L, start_R, target_L, target_R,
    trials: int = 5, temp: float = 1.0, greedy_first: bool = True,
    print_fn=print,
    early_stop: bool = True,
    time_budget_sec: float | None = None,   # ✅ 추가
):
    t0 = time.perf_counter()                # ✅ 추가
    best_obj = np.inf
    best = None
    feasible_cnt = 0

    def _time_up() -> bool:
        return (time_budget_sec is not None) and ((time.perf_counter() - t0) >= float(time_budget_sec))

    def _try_path(path, tag: str):
        nonlocal best_obj, best, feasible_cnt
        if _time_up():
            return False

        y_relax = path_to_y(edge, path, n_edge)

        try:
            sol, obj, ex = _solve_fixed_once_optapi(
                M, V, IDX,
                centroid_mat=centroid_mat, y_relax=y_relax, M_small=M_small,
                start_L=start_L, start_R=start_R, target_L=target_L, target_R=target_R
            )
        except Exception as e:
            if print_fn:
                print_fn(f"[{tag}] solve failed: {e}")
            return False

        if not _is_feasible(sol, obj):
            return False

        feasible_cnt += 1
        if obj < best_obj:
            best_obj = obj
            best = dict(path=path, y_relax=y_relax, sol=sol, ex=ex)
        return True

    # ===== 단일 트라이얼 =====
    if trials <= 1:
        if greedy_first:
            path0 = relaxation_max(y_relaxed_val, edge, vertex_number)
            ok = _try_path(path0, "greedy")
            if not ok:
                raise RuntimeError("Greedy fixed-path infeasible.")
        else:
            try:
                path1 = relaxation_prob(y_relaxed_val, edge, vertex_number, temperature=float(temp))
            except Exception as e:
                if print_fn: print_fn(f"[prob-1] relaxation_prob failed: {e}, fallback greedy")
                path1 = relaxation_max(y_relaxed_val, edge, vertex_number)
            ok = _try_path(path1, "prob-1")
            if not ok:
                raise RuntimeError("Probabilistic fixed-path infeasible.")
        return best_obj, best

    # ===== 다중 트라이얼 =====
    rest = trials
    if greedy_first:
        try:
            path0 = relaxation_max(y_relaxed_val, edge, vertex_number)
            ok0 = _try_path(path0, "greedy")
            rest -= 1
            if early_stop and ok0:
                if print_fn: print_fn("[early_stop] greedy feasible → return")
                return best_obj, best
        except Exception as e:
            if print_fn: print_fn(f"[greedy] failed: {e}")

    temp = float(temp)
    for t in range(rest):
        if _time_up():
            if print_fn:
                print_fn(f"[budget] time_budget_sec={time_budget_sec:.3f}s hit → stop trials")
            break

        try:
            path_t = relaxation_prob(y_relaxed_val, edge, vertex_number, temperature=temp)
        except Exception as e:
            if print_fn: print_fn(f"[trial {t}] relaxation_prob failed: {e}, fallback greedy")
            path_t = relaxation_max(y_relaxed_val, edge, vertex_number)

        ok_t = _try_path(path_t, f"trial {t}")
        if early_stop and ok_t:
            if print_fn: print_fn(f"[early_stop] feasible at trial {t} → return")
            return best_obj, best

    if best is None:
        raise RuntimeError("No feasible solution among trials.")
    if print_fn: print_fn(f"[stochastic] best of {feasible_cnt}/{trials} = {best_obj:.6f}")
    return best_obj, best


def run_trials_select_best_optapi(
    M, V, IDX, *,
    centroid_mat, y_relaxed_val, edge, vertex_number, n_edge,
    M_small, start_L, start_R, target_L, target_R,
    trials: int = 5, temp: float = 1.0, greedy_first: bool = True,
    print_fn=print,
    early_stop: bool = True,
):
    best_obj = np.inf
    best = None
    feasible_cnt = 0

    def _try_path(path, tag: str):
        nonlocal best_obj, best, feasible_cnt
        y_relax = path_to_y(edge, path, n_edge)

        try:
            sol, obj, ex = _solve_fixed_once_optapi(
                M, V, IDX,
                centroid_mat=centroid_mat, y_relax=y_relax, M_small=M_small,
                start_L=start_L, start_R=start_R, target_L=target_L, target_R=target_R
            )
        except Exception as e:
            # ✅ 여기서 예외를 먹고 "이번 path 실패"로 처리
            if print_fn:
                print_fn(f"[{tag}] solve failed: {e}")
            return False

        if not _is_feasible(sol, obj):
            return False

        feasible_cnt += 1
        if obj < best_obj:
            best_obj = obj
            best = dict(path=path, y_relax=y_relax, sol=sol, ex=ex)
        return True

    # ===== 단일 트라이얼 =====
    if trials <= 1:
        if greedy_first:
            path0 = relaxation_max(y_relaxed_val, edge, vertex_number)
            ok = _try_path(path0, "greedy")
            if not ok:
                raise RuntimeError("Greedy fixed-path infeasible.")
        else:
            try:
                path1 = relaxation_prob(y_relaxed_val, edge, vertex_number, temperature=float(temp))
            except Exception as e:
                if print_fn: print_fn(f"[prob-1] relaxation_prob failed: {e}, fallback greedy")
                path1 = relaxation_max(y_relaxed_val, edge, vertex_number)
            ok = _try_path(path1, "prob-1")
            if not ok:
                raise RuntimeError("Probabilistic fixed-path infeasible.")
        return best_obj, best

    # ===== 다중 트라이얼 =====
    # if print_fn: print_fn(f"Total trials = {trials}")

    rest = trials
    if greedy_first:
        try:
            path0 = relaxation_max(y_relaxed_val, edge, vertex_number)
            ok0 = _try_path(path0, "greedy")
            rest -= 1
            if early_stop and ok0:
                if print_fn: print_fn("[early_stop] greedy feasible → return")
                return best_obj, best
        except Exception as e:
            if print_fn: print_fn(f"[greedy] failed: {e}")

    temp = float(temp)
    for t in range(rest):
        try:
            path_t = relaxation_prob(y_relaxed_val, edge, vertex_number, temperature=temp)
        except Exception as e:
            if print_fn: print_fn(f"[trial {t}] relaxation_prob failed: {e}, fallback greedy")
            path_t = relaxation_max(y_relaxed_val, edge, vertex_number)
        ok_t = _try_path(path_t, f"trial {t}")
        if early_stop and ok_t:
            if print_fn: print_fn(f"[early_stop] feasible at trial {t} → return")
            return best_obj, best

    if best is None:
        raise RuntimeError("No feasible solution among trials.")
    if print_fn: print_fn(f"[stochastic] best of {feasible_cnt}/{trials} = {best_obj:.6f}")
    return best_obj, best


def _pick_soltype_for_read(task):
    # 연속 릴랙스는 interior(itr)가 기본. 없으면 bas → itg 순으로 폴백
    for st in (ms.soltype.itr, ms.soltype.bas, ms.soltype.itg):
        try:
            task.getsolsta(st)  # 존재 여부 체크 목적
            return st
        except Exception:
            pass
    return None

def _snapshot_solution_optapi(task, V):
    """ Fusion의 ex dict와 동일 키로 값 스냅샷 (shape: E×5 등) """
    st = _pick_soltype_for_read(task)
    if st is None:
        return {}

    # 1) 전체 해 벡터 x_full 가져오기
    numvar = task.getnumvar()
    buf = [0.0] * numvar
    task.getxx(st, buf)
    x_full = np.asarray(buf, dtype=float)   # <-- 전체 primal 변수

    # 2) 기존 slice 함수는 x_full에서 잘라 쓰도록 변경
    def _slice_to_np(slc, width=None):
        a, b = slc.start, slc.stop
        arr = x_full[a:b].copy()
        if width is not None:
            E = (b - a) // width
            arr = arr.reshape(E, width)
        return arr

    ex = dict(
        y      = _slice_to_np(V["y"]),
        z_L    = _slice_to_np(V["z_L"],  width=5),
        z_R    = _slice_to_np(V["z_R"],  width=5),
        zp_L   = _slice_to_np(V["zp_L"], width=5),
        zp_R   = _slice_to_np(V["zp_R"], width=5),
        rho_L  = _slice_to_np(V["rho_L"]),
        rho_R  = _slice_to_np(V["rho_R"]),
        x_full = x_full,                 # ★ 추가: 전체 해 벡터
    )
    return ex


def _solve_fixed_once_ddr_optapi(
    task, V, IDX, *, centroid_mat, y_relax, M_small,
    start_L, start_R, target_L, target_R, log_level=0
):
    # ---- geometry interleave: [x0,y0,x1,y1,...] ----
    Vn = centroid_mat.shape[1]          # centroid_mat: (2 or 3, Vn)
    geom_vec = np.empty(2 * Vn, dtype=float)
    geom_vec[0::2] = centroid_mat[0, :] # x0,x1,...
    geom_vec[1::2] = centroid_mat[1, :] # y0,y1,...

    # 파라미터 세팅 (Fusion과 동일 키)
    params = {
        "geometry": geom_vec,           # ★ 수정: reshape(-1) 대신 interleave
        "start_L": start_L, "start_R": start_R,
        "target_L": target_L, "target_R": target_R,
        "M_scalar": float(M_small),
        "y_relax":  y_relax.reshape(-1),
    }
    set_parameters_ddr_optapi(task, V, IDX, PAR=V.get("meta", {}).get("PAR", None) or {}, params=params)
    # ↑ V에 PAR를 보관하지 않았다면, build 함수에서 넘겨받은 PAR를 외부 스코프에서 전달해도 OK.
    #   (함수 시그니처에 PAR를 추가하는 게 가장 깔끔합니다: set_parameters_ddr_optapi(task, V, IDX, PAR, params))

    # 로그
    if log_level == 0:
        task.putintparam(ms.iparam.log, 0)
        task.set_Stream(ms.streamtype.log, None)
    else:
        task.putintparam(ms.iparam.log, 1)
        task.set_Stream(ms.streamtype.log, lambda s: print(s, end=""))

    # 풀기
    task.optimize()

    # 상태/객체
    st = _pick_soltype_for_read(task)
    prosta = task.getprosta(st) if st is not None else None
    solsta = task.getsolsta(st) if st is not None else None
    obj = None
    if st is not None:
        try:
            obj = float(task.getprimalobj(st))
        except Exception:
            obj = None

    # 해 스냅샷
    ex = _snapshot_solution_optapi(task, V)

    info = {"pstatus": str(prosta), "sstatus": str(solsta)}
    return info, obj, ex


def run_trials_select_best_ddr_optapi(
    task, V, IDX, *,
    centroid_mat, y_relaxed_val, edge, vertex_number, n_edge,
    M_small, start_L, start_R, target_L, target_R,
    trials: int = 5, temp: float = 1.0, greedy_first: bool = True,
    print_fn=print,
    early_stop: bool = True,
):
    best_obj = np.inf
    best = None
    feasible_cnt = 0

    def _try_path(path, tag: str):
        nonlocal best_obj, best, feasible_cnt
        y_relax = path_to_y(edge, path, n_edge)

        sol, obj, ex = _solve_fixed_once_ddr_optapi(
            task, V, IDX,
            centroid_mat=centroid_mat, y_relax=y_relax, M_small=M_small,
            start_L=start_L, start_R=start_R, target_L=target_L, target_R=target_R,
            log_level=0
        )

        if not _is_feasible_opt(sol.get("pstatus"), sol.get("sstatus")) or obj is None:
            # if print_fn: print_fn(f"[{tag}] ", y_relax.T, "infeasible")
            return False

        feasible_cnt += 1
        # if print_fn: print_fn(f"[{tag}] cost={obj:.6f}")

        if obj < best_obj:
            best_obj = obj
            best = dict(path=path, y_relax=y_relax, sol=sol, ex=ex)
        return True

    # ===== 단일 트라이얼 =====
    if trials <= 1:
        if greedy_first:
            path0 = relaxation_max(y_relaxed_val, edge, vertex_number)
            ok = _try_path(path0, "greedy")
            if not ok:
                raise RuntimeError("Greedy fixed-path infeasible.")
        else:
            try:
                path1 = relaxation_prob(y_relaxed_val, edge, vertex_number, temperature=float(temp))
            except Exception as e:
                if print_fn: print_fn(f"[prob-1] relaxation_prob failed: {e}, fallback greedy")
                path1 = relaxation_max(y_relaxed_val, edge, vertex_number)
            ok = _try_path(path1, "prob-1")
            if not ok:
                raise RuntimeError("Probabilistic fixed-path infeasible.")
        return best_obj, best

    # ===== 다중 트라이얼 =====
    rest = trials
    if greedy_first:
        try:
            path0 = relaxation_max(y_relaxed_val, edge, vertex_number)
            ok0 = _try_path(path0, "greedy")
            rest -= 1
            if early_stop and ok0:
                # if print_fn: print_fn("[early_stop] greedy feasible → return")
                return best_obj, best
        except Exception as e:
            if print_fn: print_fn(f"[greedy] failed: {e}")

    temp = float(temp)
    for t in range(rest):
        try:
            path_t = relaxation_prob(y_relaxed_val, edge, vertex_number, temperature=temp)
        except Exception as e:
            if print_fn: print_fn(f"[trial {t}] relaxation_prob failed: {e}, fallback greedy")
            path_t = relaxation_max(y_relaxed_val, edge, vertex_number)
        ok_t = _try_path(path_t, f"trial {t}")
        if early_stop and ok_t:
            if print_fn: print_fn(f"[early_stop] feasible at trial {t} → return")
            return best_obj, best

    if best is None:
        raise RuntimeError("No feasible solution among trials.")
    if print_fn: print_fn(f"[stochastic] best of {feasible_cnt}/{trials} = {best_obj:.6f}")
    return best_obj, best

