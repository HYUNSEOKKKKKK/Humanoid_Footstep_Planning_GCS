import numpy as np
from gcs_c_adapter import (
    build_gcs_model_short_optapi_C,
    set_parameters_optapi_C,
    solve_relaxed_optapi_C
)


def run_wiretest():
    # 가짜 데이터
    E = 3
    v = np.zeros((4,4,2), float)  # 미니 모델에선 사용 안함
    edge = np.vstack([np.arange(E), np.arange(1, E+1)]).T
    edge = np.clip(edge, 0, E-1).astype(np.int32)
    mode_list = np.zeros((E,), np.int32)
    x_ref = np.array([1.0, 0.0])
    y_ref = np.array([0.0, 1.0])

    step_span = 0.5
    alpha = 0.0

    M, V, IDX, PAR, _ = build_gcs_model_short_optapi_C(
        v, edge, step_span, alpha, mode_list, x_ref, y_ref
    )

    params = {"M_large": 1e3}
    set_parameters_optapi_C(M, V, params)

    sol, obj = solve_relaxed_optapi_C(M, V, log_level=0)
    print("obj:", obj)
    print("zL:", sol["zL"])
    print("zR:", sol["zR"])

if __name__ == "__main__":
    run_wiretest()
