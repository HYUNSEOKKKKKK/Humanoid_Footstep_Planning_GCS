import numpy as np
from gcs_task import GcsTask

def build_gcs_model_short_optapi_C(v, edge, step_span, alpha, mode_list, x_ref, y_ref):
    task = GcsTask()
    task.build(v.astype(np.float64, copy=False),
               edge.astype(np.int32, copy=False),
               float(step_span), float(alpha),
               mode_list.astype(np.int32, copy=False),
               np.asarray(x_ref, dtype=np.float64),
               np.asarray(y_ref, dtype=np.float64))
    # 기존 함수가 반환하던 튜플 형태를 흉내 (M,V,IDX,PAR,build_time)
    M = task
    V = None  # C 버전에서는 task 내부에 인덱스/패턴을 모두 보관
    IDX = {"n_edge": int(edge.shape[0])}
    PAR = {}
    build_time = 0.0
    return M, V, IDX, PAR, build_time

def set_parameters_optapi_C(M, V, params_relaxed: dict):
    # M은 GcsTask 인스턴스
    M.set_params(params_relaxed)

def solve_relaxed_optapi_C(M, V, log_level=0):
    out = M.solve(int(log_level))
    return out, float(out.get("obj", np.nan))
