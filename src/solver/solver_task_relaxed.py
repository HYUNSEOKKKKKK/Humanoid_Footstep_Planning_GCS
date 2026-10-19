from __future__ import annotations
import numpy as np
import mosek

def update_task_params_relaxed(task, idx, params):
    """
    Fusion의 set_parameters(...) 대응.
    - start_*, target_*: '상수 변수' bounds를 fx로 세팅
    - y_relax, M_scalar: fx로 세팅
    - (옵션) fix_y: warm-start만 (bounds는 건드리지 않음)
    """
    def fix_vec(var_idx, vec):
        vec = np.asarray(vec, float).reshape(-1)
        for i, val in enumerate(vec):
            task.putvarbound(int(var_idx[i]), mosek.boundkey.fx, float(val), float(val))

    # Pose params
    if "start_L" in params: fix_vec(idx["start_L"], params["start_L"])
    if "start_R" in params: fix_vec(idx["start_R"], params["start_R"])
    if "target_L" in params: fix_vec(idx["target_L"], params["target_L"])
    if "target_R" in params: fix_vec(idx["target_R"], params["target_R"])

    # Relax/fixing params
    if "y_relax" in params: fix_vec(idx["y_relax"], params["y_relax"])
    if "M_scalar" in params:
        task.putvarbound(int(idx["M_scalar"][0]), mosek.boundkey.fx,
                         float(params["M_scalar"]), float(params["M_scalar"]))

    # (옵션) y warm-start
    if "fix_y" in params and params["fix_y"] is not None:
        yval = np.asarray(params["fix_y"], float).reshape(-1)
        # Optimizer API에선 '레벨'만 넣는 별도 함수는 없음.
        # 대신 프리솔브 전에 MIP가 아니므로 초깃값 영향 제한적. 필요하면
        # basis/solution injection을 고려. 여기선 생략.
        pass


def solve_relaxed_task(task, idx, log_level: int = 0):
    # 로그 레벨
    task.putintparam(mosek.iparam.log, log_level)
    # 최적화
    task.optimize()

    # 솔루션 타입: 연속 SOCP -> interior(itr)
    sol = mosek.soltype.itr
    xx = [0.0] * task.getnumvar()
    task.getxx(sol, xx)
    xx = np.asarray(xx, float)

    # 목적값
    obj = task.getprimalobj(sol)

    # 필요한 것만 패킹 (여기선 y만 우선)
    sol_dict = {
        "y": xx[idx["y"]],
        "x_all": xx,  # 원하면 나중에 z_L/z_R 등도 꺼내서 넣자
    }
    return obj, sol_dict


def get_relaxed_y_task(sol_dict, idx):
    return np.asarray(sol_dict["y"], float).reshape(-1)
