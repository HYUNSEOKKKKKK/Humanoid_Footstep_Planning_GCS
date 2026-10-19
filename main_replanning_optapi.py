# main_longfootsteps_optapi.py
import time
import argparse

from main import load_experiment  # 네 main.py 안의 함수
from src.experiments import list_experiments
from src.terrain_preset import load_terrain
from src.utils import expand_shorthand, postprocess_solution

from src.graph import build_graph, compute_centroids_from_vertices
from src.model import build_gcs_model_short, pack_relaxed_params_for_model, prepare_problem_params
from src.solver import (
    set_parameters, solve_relaxed, get_relaxed_y,
    run_trials_select_best, set_parameters_optapi, solve_relaxed_optapi, run_trials_select_best_optapi,
)
from src.model import build_gcs_model_short_optapi
from src.viz.feet import plot_feet_3d

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=10_000, suppress=True, precision=6)

def foot_yaw_from_key(key: str) -> float:
    # 방향키(left/right/up)
    if key == "left":
        return 45.0
    if key == "right":
        return -45.0
    if key == "up":
        return 0.0

    # 숫자패드 (NumPad)
    if key == "4":      # left (강회전)
        return 45.0
    if key == "6":      # right (강회전)
        return -45.0
    if key == "7":      # left (약회전)
        return 20.0
    if key == "9":      # right (약회전)
        return -20.0
    if key == "8":      # straight
        return 0.0
    if key == "3":      # back right (약회전)
        return 20.0
    if key == "1":      # back left (약회전)
        return -20.0
    # 기본값
    return 0.0

def infer_first_is_left_from_proc(proc, eps: float = 1e-6) -> bool:
    """
    proc의 좌/우 시퀀스를 보고 '처음으로 실제로 움직인' 발을 찾아서
    True(왼발 먼저) / False(오른발 먼저) 반환.
    - 움직임 판정은 (x,y,z) 변화량 중 하나라도 eps 초과이면 '움직임'으로 간주.
    - 둘 다 동시에 움직였거나 판단 불가면 안전하게 왼발 먼저로 fallback.
    """
    import numpy as np

    xL, yL, zL = np.asarray(proc.L.x, float), np.asarray(proc.L.y, float), np.asarray(proc.L.z, float)
    xR, yR, zR = np.asarray(proc.R.x, float), np.asarray(proc.R.y, float), np.asarray(proc.R.z, float)

    tL = None
    for t in range(1, len(xL)):
        if (abs(xL[t] - xL[t-1]) > eps) or (abs(yL[t] - yL[t-1]) > eps) or (abs(zL[t] - zL[t-1]) > eps):
            tL = t; break

    tR = None
    for t in range(1, len(xR)):
        if (abs(xR[t] - xR[t-1]) > eps) or (abs(yR[t] - yR[t-1]) > eps) or (abs(zR[t] - zR[t-1]) > eps):
            tR = t; break

    # 판정 로직
    if (tL is None) and (tR is None):
        return True                 # 둘 다 변화가 없으면 왼발 먼저로
    if (tL is not None) and (tR is None):
        return True                 # 왼발만 먼저 움직임
    if (tL is None) and (tR is not None):
        return False                # 오른발만 먼저 움직임
    # 둘 다 있다면 더 이른 쪽
    if tL < tR:
        return True
    if tR < tL:
        return False
    # 같은 프레임에 동시에 움직인 경우: 안전하게 왼발로
    return True

def first_is_left_from_yvec(y_like) -> bool:
    """
    y_like[0]이 1이면 왼발 먼저(True), 0이면 오른발 먼저(False).
    그 외 값/비어있음은 안전하게 왼발(True)로 폴백.
    """
    import numpy as np
    y = np.asarray(y_like).ravel()
    if y.size == 0:
        return True
    v = int(round(float(y[0])))
    if v == 1:
        return True
    if v == 0:
        return False
    return True  # 예외값 폴백

def get_y_vector_from_proc(proc):
    """
    proc["y"] 또는 proc.y 둘 다 지원. 없으면 None.
    """
    if isinstance(proc, dict) and "y" in proc:
        return proc["y"]
    if hasattr(proc, "y"):
        return getattr(proc, "y")
    # postprocess_solution 결과가 proc.L/ proc.R만 있는 구조면 ex_after["y"]를 보관해서 넘겨줘야 함.
    return None

def decide_first_is_left(proc, fallback_infer=False) -> bool:
    yvec = get_y_vector_from_proc(proc)
    if yvec is not None:
        return first_is_left_from_yvec(yvec)
    if fallback_infer:
        # y가 없으면 (x,y,z) 움직임으로 추론하는 백업 로직 사용하려면 True로.
        return infer_first_is_left_from_proc(proc)
    # 기본 폴백은 왼발 먼저
    return True

# === 새로 추가: 애니메이션 중 키 입력으로 중단 ===
def animate_until_interruption(
    proc,
    *,
    fig,
    ax,
    startpos_L,
    startpos_R,
    interval: float = 0.05,
    label_prefix: str = "",
    matlab_style: bool = True,
    first_is_left: bool | None = None,
):
    """
    plot_feet_3d의 animate 모션과 동일한 느낌으로 프레임을 진행.
    - 좌/우 번갈아 1스텝씩 추가해서 그림 (first_is_left 제어)
    - plot_feet_3d의 내부 스타일(사각형, yaw 화살표, grid, top-view 등) 그대로 사용
      -> 매 프레임마다 plot_feet_3d(...)를 '부분 시퀀스'로 호출 (animate=False, accumulate=False)
    - 애니메이션 도중 키가 눌리면 즉시 정지:
        - Enter/Return 또는 ALLOWED_KEYS(any 방향키)
        - 정지 시, 현재 프레임의 L/R 마지막 인덱스와 좌표를 캡처해서 반환
    반환 dict 예:
    {
        "interrupted": True/False,
        "k": 현재 프레임(step) 수 (1..N),
        "key": 눌린 키 or None,
        # 캡처 세부 (있을 때만)
        "kL": int, "kR": int, "active_foot": "L"/"R",
        "xL": float, "yL": float, "zL": float, "cL": float, "sL": float,
        "xR": float, "yR": float, "zR": float, "cR": float, "sR": float,
    }
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # === 데이터 꺼내기 ===
    xL = np.asarray(proc.L.x, float); yL = np.asarray(proc.L.y, float)
    zL = np.asarray(proc.L.z, float); cL = np.asarray(proc.L.c, float); sL = np.asarray(proc.L.s, float)
    xR = np.asarray(proc.R.x, float); yR = np.asarray(proc.R.y, float)
    zR = np.asarray(proc.R.z, float); cR = np.asarray(proc.R.c, float); sR = np.asarray(proc.R.s, float)

    NL, NR = len(xL), len(xR)
    if first_is_left is None:
        first_is_left = decide_first_is_left(proc, fallback_infer=True)

    # === 상태 ===
    iL = iR = 0                   # 현재까지 그려진 길이
    turn_left = bool(first_is_left)
    result = {"interrupted": False, "k": 0, "key": None}
    _stop_flag = {"stop": False}

    # === 키 핸들러: Enter/Return 또는 방향키(ALLOWED_KEYS) 누르면 즉시 정지 ===
    def _on_key(ev):
        key = (ev.key or "").replace("shift+", "")
        if key in ALLOWED_KEYS or key in ("enter", "return"):
            result["key"] = key
            _stop_flag["stop"] = True

    cid = fig.canvas.mpl_connect("key_press_event", _on_key)

    # === 루프: 좌/우 번갈아 1스텝씩 증가 ===
    while (iL < NL) or (iR < NR):
        highlight = None
        # 번갈아가며 한 발만 1 증가
        if turn_left:
            if iL < NL:
                iL += 1
                highlight = "L"
            elif iR < NR:
                iR += 1
                highlight = "R"
        else:
            if iR < NR:
                iR += 1
                highlight = "R"
            elif iL < NL:
                iL += 1
                highlight = "L"

        # 라벨: 총 해결시간/스텝 같은 prefix 뒤에 현재 step 수 추가 (원하면 label_prefix만 사용)
        steps_now = iL + iR
        time_steps_label = f"{label_prefix} (steps: {steps_now})" if label_prefix else f"steps: {steps_now}"

        # === 프레임 그리기 ===
        # plot_feet_3d의 내부 스타일을 그대로 쓰기 위해 매 프레임마다 '부분 시퀀스'로 호출
        ax.cla()  # 프레임마다 리프레시(중복 아티스트 방지)
        plot_feet_3d(
            xL[:iL], yL[:iL], zL[:iL], cL[:iL], sL[:iL],
            xR[:iR], yR[:iR], zR[:iR], cR[:iR], sR[:iR],
            startpos_L=startpos_L, startpos_R=startpos_R,
            ax=ax,
            accumulate=False,           # 프레임마다 다시 그림
            show=False,                 # 여기서는 블로킹 X
            matlab_style=matlab_style,
            label=time_steps_label,
        )
        # 마지막 디딘 발만 살짝 강조(큰 점)
        try:
            if highlight == "L" and iL > 0:
                ax.scatter([xL[iL-1]], [yL[iL-1]], [zL[iL-1]], s=40, c="#F58518", edgecolors="k", linewidths=0.8)
            elif highlight == "R" and iR > 0:
                ax.scatter([xR[iR-1]], [yR[iR-1]], [zR[iR-1]], s=40, c="#4C78A8", edgecolors="k", linewidths=0.8)
        except Exception:
            pass

        fig.canvas.draw(); fig.canvas.flush_events()
        plt.pause(interval)

        result["k"] = steps_now

        # === 인터럽트 체크 ===
        if _stop_flag["stop"]:
            # 현재 프레임의 마지막 인덱스(0-based) 계산
            kL = max(iL - 1, 0) if NL > 0 else None
            kR = max(iR - 1, 0) if NR > 0 else None
            active = highlight or ("L" if first_is_left else "R")

            # 캡처 패킷 구성
            capture = {
                "kL": kL, "kR": kR, "active_foot": active,
                "xL": float(xL[kL]) if (NL > 0 and kL is not None) else None,
                "yL": float(yL[kL]) if (NL > 0 and kL is not None) else None,
                "zL": float(zL[kL]) if (NL > 0 and kL is not None) else None,
                "cL": float(cL[kL]) if (NL > 0 and kL is not None) else None,
                "sL": float(sL[kL]) if (NL > 0 and kL is not None) else None,
                "xR": float(xR[kR]) if (NR > 0 and kR is not None) else None,
                "yR": float(yR[kR]) if (NR > 0 and kR is not None) else None,
                "zR": float(zR[kR]) if (NR > 0 and kR is not None) else None,
                "cR": float(cR[kR]) if (NR > 0 and kR is not None) else None,
                "sR": float(sR[kR]) if (NR > 0 and kR is not None) else None,
            }
            # ✅ 여기서 바로 start(center, heading_deg) 계산해 넣기
            if (capture["xL"] is not None) and (capture["xR"] is not None):
                import numpy as np
                Lx, Ly, Lz = capture["xL"], capture["yL"], capture["zL"]
                Rx, Ry, Rz = capture["xR"], capture["yR"], capture["zR"]

                cx, cy, cz = 0.5 * (Lx + Rx), 0.5 * (Ly + Ry), 0.5 * (Lz + Rz)
                vRL = np.array([Lx - Rx, Ly - Ry], float)
                n = np.linalg.norm(vRL)
                y_ref = np.array([0.0, 1.0]) if n < 1e-6 else vRL / n
                x_ref = np.array([y_ref[1], -y_ref[0]])
                heading_deg = float(np.degrees(np.arctan2(x_ref[1], x_ref[0])) % 360.0)

                capture["start"] = {"center": [cx, cy, cz], "heading_deg": heading_deg}
            else:
                capture["start"] = None

            result.update(capture)
            result["interrupted"] = True
            break

        # 다음엔 반대 발
        turn_left = not turn_left

    # 이벤트 핸들러 해제
    try:
        fig.canvas.mpl_disconnect(cid)
    except Exception:
        pass

    return result

def make_start_from_xy(Lx, Ly, Lz, Rx, Ry, Rz, cL=None, sL=None, cR=None, sR=None):
    import numpy as np
    cx, cy, cz = 0.5*(Lx+Rx), 0.5*(Ly+Ry), 0.5*(Lz+Rz)
    vRL = np.array([Lx - Rx, Ly - Ry], float)
    n = np.linalg.norm(vRL)
    y_ref = np.array([0.0,1.0]) if n < 1e-6 else vRL / n
    x_ref = np.array([y_ref[1], -y_ref[0]])
    heading_deg = float(np.degrees(np.arctan2(x_ref[1], x_ref[0])) % 360.0)
    return {"center":[cx,cy,cz], "heading_deg": heading_deg}

# === 새로 추가: 프레임 k의 L/R → 새 start(dict)로 변환 ===
def make_start_from_frame(proc, k: int) -> dict:
    import numpy as np
    idx = k - 1

    Lx, Ly, Lz = float(proc.L.x[idx]), float(proc.L.y[idx]), float(proc.L.z[idx])
    Rx, Ry, Rz = float(proc.R.x[idx]), float(proc.R.y[idx]), float(proc.R.z[idx])
    print(Lx,Ly)
    print(Rx,Ry)
    # 몸 중심
    cx, cy, cz = (0.5 * (Lx + Rx), 0.5 * (Ly + Ry), 0.5 * (Lz + Rz))
    print(cx,cy)
    # 로컬 기준축: y_ref = R→L, x_ref = 전/후
    vRL = np.array([Lx - Rx, Ly - Ry], float)

    y_ref = vRL / np.linalg.norm(vRL)
    x_ref = np.array([y_ref[1], -y_ref[0]])  # ⟂

    # ✅ heading = x_ref(전/후) 기준 각도
    heading_rad = np.arctan2(x_ref[1], x_ref[0])
    heading_deg = float(np.rad2deg(heading_rad)) % 360.0
    print(heading_deg)
    return {"center": [cx, cy, cz], "heading_deg": heading_deg}


ALLOWED_KEYS = ("left", "right", "up", "4", "6", "7", "8", "9","3","1")

def wait_for_direction_key(fig) -> str:
    result = {"key": None}
    def on_key(e):
        key = (e.key or "").replace("shift+", "")
        if key in ALLOWED_KEYS:
            result["key"] = key
    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    while result["key"] is None and plt.fignum_exists(fig.number):
        plt.pause(0.02)
    fig.canvas.mpl_disconnect(cid)
    return result["key"]


def goal_from_start_and_key(
    start_dict: dict,
    key: str,
    step_len: float = 1.8,
    turn_deg: float = 40.0,       # 강회전(←/→/4/6)
    small_turn_deg: float = 20.0  # 약회전(7/9)
) -> dict:
    sx, sy, sz = map(float, start_dict["center"][:3])
    h0 = float(start_dict.get("heading_deg", 0.0))

    if key in ("left", "4"):            # 강 좌
        h_new = h0 + turn_deg
    elif key in ("right", "6"):         # 강 우
        h_new = h0 - turn_deg
    elif key == "7":                    # 약 좌
        h_new = h0 + small_turn_deg
    elif key == "9":                    # 약 우
        h_new = h0 - small_turn_deg
    elif key in ("up", "8"):            # 직진
        h_new = h0
    elif key == "3":
        h_new = h0 + small_turn_deg
    elif key == "1":
        h_new = h0 - small_turn_deg
    else:
        raise ValueError(f"unsupported key: {key}")

    if key in ("3","1"):
        step_len = -0.4

    rad = np.deg2rad(h_new)
    dx = step_len * np.cos(rad)
    dy = step_len * np.sin(rad)
    return {"center": [sx + dx, sy + dy, sz], "heading_deg": h_new % 360.0}



def solve_once(exp, T, step_span, alpha, M_large, M_small):
    """한 번의 실험을 풀고, 후처리에 필요한 값들과 타이밍을 반환"""
    # === 그래프 생성 ===
    t0 = time.perf_counter()
    copy_counts = T["copy_counts"] if T.get("copy_counts") is not None else [5] * len(T["stones"])
    G = build_graph(
        exp["startpos_L"], exp["startpos_R"],
        exp["targetpos_L"], exp["targetpos_R"],
        step_span=step_span,
        copy_counts=copy_counts,
        edge_rule="general",
        terrain_stones=T["stones"],
    )
    v, edge, mode_list = G["v"], G["edge"], G["mode_list"]
    vertex_number, n_edge = G["vertex_number"], G["n_edge"]
    t_graph = time.perf_counter() - t0

    # === 모델 생성 ===
    t1 = time.perf_counter()
    # === 포즈 파라미터 ===
    params_pose = prepare_problem_params(
        exp["startpos_L"], exp["startpos_R"], exp["targetpos_L"], exp["targetpos_R"],
        exp.get("startyaw_L", 0.0), exp.get("startyaw_R", 0.0),
        exp.get("targetyaw_L", 0.0), exp.get("targetyaw_R", 0.0)
    )
    start_L = params_pose["start_L"]; start_R = params_pose["start_R"]
    target_L = params_pose["target_L"]; target_R = params_pose["target_R"]

    # === centroid ===
    centroid_mat, _ = compute_centroids_from_vertices(v)

    # === global ref 축 === (R→L 방향 = y_ref, 그 직교가 x_ref)
    pL = np.asarray(exp["startpos_L"], float).reshape(-1)[:2]
    pR = np.asarray(exp["startpos_R"], float).reshape(-1)[:2]
    vRL = pL - pR
    if np.linalg.norm(vRL) < 1e-12:
        y_ref = np.array([0.0, 1.0])
    else:
        y_ref = vRL / np.linalg.norm(vRL)
    x_ref = np.array([y_ref[1], -y_ref[0]])


    M, V, IDX, PAR, build_time = build_gcs_model_short_optapi(
        v, edge, step_span, alpha, mode_list, x_ref, y_ref
    )

    t_build = time.perf_counter() - t1
    t2 = time.perf_counter()

    params_relaxed = pack_relaxed_params_for_model(
        centroid_mat=centroid_mat, n_edge=IDX["n_edge"], M_large=M_large,
        start_L=start_L, start_R=start_R, target_L=target_L, target_R=target_R,
    )
    # x_ref, y_ref는 현재 버전에선 불필요 (무시됨)

    set_parameters_optapi(M, V, params_relaxed)

    sol_relaxed, obj1 = solve_relaxed_optapi(M, V, log_level=0)
    t_relaxed = time.perf_counter() - t2
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =

    # === 경로 선택 + 2차 ===
    t3 = time.perf_counter()
    # y_relaxed_val = get_relaxed_y(V)
    y_relaxed_val = sol_relaxed["y"]
    print("y_relaxed_val", y_relaxed_val)
    try:
        best_obj, best = run_trials_select_best_optapi(
            M, V, IDX,
            centroid_mat=centroid_mat,
            y_relaxed_val=y_relaxed_val,
            edge=edge,
            vertex_number=vertex_number,
            n_edge=n_edge,
            M_small=1e-6,
            start_L=start_L, start_R=start_R,
            target_L=target_L, target_R=target_R,
            trials=1,  # greedy만 1회
            temp=1.0,
            greedy_first=True,
        )
    except RuntimeError:
        # 확률 1회씩 최대 N번 시도, 첫 feasible이면 즉시 반환
        N = 30
        best_obj = None;
        best = None
        for i in range(N):
            try:
                cand_obj, cand = run_trials_select_best_optapi(
                    M, V, IDX,
                    centroid_mat=centroid_mat,
                    y_relaxed_val=y_relaxed_val,
                    edge=edge,
                    vertex_number=vertex_number,
                    n_edge=n_edge,
                    M_small=1e-6,
                    start_L=start_L, start_R=start_R,
                    target_L=target_L, target_R=target_R,
                    trials=1,  # ★ 한 번만
                    temp=0.8 + 0.1 * i,  # 온도 살짝 변주(선택)
                    greedy_first=False,  # ★ 확률 1회
                    early_stop=True,  # ★ 단일이므로 의미 없지만 시그니처 일치
                )
                best_obj, best = cand_obj, cand
                break
            except RuntimeError:
                continue
        if best is None:
            raise RuntimeError("No feasible fixed-path solution found in fallback trials.")

    t_fixed = time.perf_counter() - t3

    return {
        "ex": best["ex"],                 # ex_after
        "edge": edge,
        "relaxed_path": best["path"],
        "target_L": target_L,
        "target_R": target_R,
        "times": {"graph": t_graph, "build": t_build, "relaxed": t_relaxed, "fixed": t_fixed},
        "obj_relaxed": obj1,
        "obj_fixed": best_obj,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", type=str, default="default_flat_Shortsequence")
    parser.add_argument("--terrain-file", type=str, default=None)
    parser.add_argument("--experiments-file", type=str, default="configs/experiments_Shortsequence.json")
    parser.add_argument("--period", type=float, default=0.0, help="seconds between runs (auto loop)")
    parser.add_argument("--step_span", type=float, default=0.35)
    parser.add_argument("--alpha_deg", type=float, default=50.0)
    parser.add_argument("--M_large", type=float, default=1e3)
    # (옵션) 키 동작 파라미터를 고정값으로 둘 거면 아래 두 줄은 생략해도 됨
    parser.add_argument("--step_len", type=float, default=0.5, help="distance advanced for each key press")
    parser.add_argument("--turn_deg", type=float, default=45.0, help="heading change for left/right")
    args = parser.parse_args()

    # Terrain 고정 로드
    T = load_terrain(args.terrain, args.terrain_file)
    alpha = float(args.alpha_deg) * np.pi / 180.0

    # ====== 초기 start/goal: JSON의 첫 케이스에서 'start'만 1회 가져옴 ======
    first_name = list_experiments(args.experiments_file)[0]
    raw0 = load_experiment(first_name, args.experiments_file)
    current_start = raw0["start"]
    curr_fyaw_start = {"L": 0, "R": 0}
    # 초기 goal은 'up'(회전 없음) 규칙으로 step_len만큼 전진
    current_goal  = goal_from_start_and_key(current_start, "up", step_len=args.step_len, turn_deg=args.turn_deg)
    curr_fyaw_goal = {"L": 0, "R": 0}

    # 플롯 준비(누적)
    plt.ion()
    fig = plt.figure("Online footstep planning (DLR - keyboard: left/up/right)")
    if fig.axes:
        ax = fig.axes[0]
    else:
        ax = fig.add_subplot(111, projection="3d")


    run_idx = 0

    # ====== 메인 루프: 창이 살아있는 동안 반복 ======
    while plt.fignum_exists(fig.number):
        # shorthand 확장 대신, 현재 start/goal로 exp 딕셔너리 직접 구성
        exp = {
            "start": current_start,
            "goal": current_goal,
            "foot_yaw_deg": {
                "start_L": curr_fyaw_start["L"],  # 항상 0.0
                "start_R": curr_fyaw_start["R"],  # 항상 0.0
                "goal_L": curr_fyaw_goal["L"],  # 키에서 온 값(둘이 동일)
                "goal_R": curr_fyaw_goal["R"],  # 키에서 온 값(둘이 동일)
            },
        }

        expx = expand_shorthand(exp, 0.25)

        # ===== 최적화 수행 =====
        info = solve_once(
            expx, T,
            step_span=args.step_span,
            alpha=alpha,
            M_large=args.M_large,
            M_small=1e-6
        )
        print("z_L =", info["ex"]["z_L"])
        print("zp_L =", info["ex"]["zp_L"])
        print("XL =", info["ex"]["X_L"])

        # === 후처리 ===
        proc = postprocess_solution(
            info["ex"], info["edge"], info["relaxed_path"],
            expx["startpos_L"], expx["startpos_R"],
            expx["targetpos_L"], expx["targetpos_R"],
            info["target_L"], info["target_R"],
        )

        # print (proc)
        # 풀이시간/스텝수 라벨
        tms = info["times"]
        solve_time = tms["graph"] + tms["build"] + tms["relaxed"] + tms["fixed"]
        print(tms["graph"], tms["build"], tms["relaxed"], tms["fixed"])
        time_steps_label = f"{solve_time:.3f}s"

        # 🔧 마지막 값(타겟) 제거 — 세그먼트 중복 방지
        # proc.L.x, proc.L.y, proc.L.z, proc.L.c, proc.L.s = proc.L.x[:-1], proc.L.y[:-1], proc.L.z[:-1], proc.L.c[:-1], proc.L.s[:-1]
        # proc.R.x, proc.R.y, proc.R.z, proc.R.c, proc.R.s = proc.R.x[:-1], proc.R.y[:-1], proc.R.z[:-1], proc.R.c[:-1], proc.R.s[:-1]


        # y[0] 규칙으로 first_is_left 계산
        yvec = np.array(info["ex"]["y"], dtype=float)
        fisl = (yvec is not None and len(yvec) > 0 and float(yvec[0]) > 0.5)
        # print(fisl)
        # ★ 변경점: 애니메이션으로 그리고, 중간에 키가 들어오면 즉시 멈춤
        result = animate_until_interruption(
            proc, fig=fig, ax=ax,
            startpos_L=expx["startpos_L"], startpos_R=expx["startpos_R"],
            interval=0.01,  # 프레임 간 간격 (원하면 args.period 활용도 가능)
            label_prefix=time_steps_label, matlab_style=True,
            first_is_left=fisl,
        )

        # 애니메이션이 중단되지 않은 경우: 마지막 프레임을 누적 상태로 남기고 다음 단계 진행
        # (원하면 여기서 한 번 더 accumulate=True로 고정 플롯 해도 됨)
        fig.canvas.draw(); fig.canvas.flush_events()

        # # 콘솔 로그
        # print(f"[run{run_idx:02d}] graph={tms['graph']:.3f}s, build={tms['build']:.3f}s, "
        #       f"relaxed={tms['relaxed']:.3f}s, fixed={tms['fixed']:.3f}s, "
        #       f"obj_relaxed={info['obj_relaxed']:.3f}, obj_fixed={info['obj_fixed']:.3f}")

        # ★ 여기부터: '애니메이션 중' 들어온 키(enter/방향키) 처리
        if result["interrupted"]:

            if result.get("start") is not None:
                new_start = result["start"]
            else:
                # (폴백) 어쩔 수 없을 때 저장된 좌표로 생성
                new_start = make_start_from_xy(
                    result["xL"], result["yL"], result["zL"],
                    result["xR"], result["yR"], result["zR"]
                )


            # --- new_start 디버그 표시 (초록 점 + 헤딩 화살표) ---
            sx, sy, sz = map(float, new_start["center"][:3])
            ax.scatter([sx], [sy], [sz],
                       s=18, marker='o', c='g', edgecolors='k', linewidths=0.6, zorder=5)
            # 헤딩 시각화(선택): 로컬 전/후 방향 화살표
            h_rad = np.deg2rad(float(new_start.get("heading_deg", 0.0)))
            ux, uy = np.cos(h_rad), np.sin(h_rad)
            arrow_len = getattr(args, "arrow_len", 0.25)  # 없으면 0.25m
            ax.quiver(sx, sy, sz, ux, uy, 0.0, length=arrow_len, normalize=True, zorder=6)
            fig.canvas.draw();
            fig.canvas.flush_events()

            # 키가 enter/return이면, 방향키를 추가로 받아서 goal을 정함
            key_first = result["key"]
            if key_first in ("enter", "return"):
                key_dir = wait_for_direction_key(fig)  # 방향키 입력 대기
                if key_dir is None:
                    break
            else:
                # 애니메이션 도중에 바로 방향키가 눌린 경우
                key_dir = key_first

            # 새 goal 생성 (기준은 방금 캡처한 new_start)
            next_goal = goal_from_start_and_key(
                new_start, key_dir, step_len=args.step_len, turn_deg=args.turn_deg
            )
            yaw_val = foot_yaw_from_key(key_dir)
            next_fyaw_goal = {"L": yaw_val, "R": yaw_val}

            # 시각적으로 표시(빨간 점)
            gx, gy, gz = map(float, next_goal["center"][:3])
            ax.scatter([gx], [gy], [gz], s=10, marker='o', c='r', edgecolors='k', linewidths=0.6)
            fig.canvas.draw(); fig.canvas.flush_events()

            # 롤링 업데이트: ★ 핵심
            current_start = new_start
            curr_fyaw_start = {"L": 0.0, "R": 0.0}
            current_goal = next_goal
            curr_fyaw_goal = next_fyaw_goal

            run_idx += 1
            # 인터럽트 즉시 다음 루프로(대기시간 없이)
            continue


        # ===== 키 입력 → 다음 goal 생성 (항상 현재 start 기준) =====
        key = wait_for_direction_key(fig)  # 'left' / 'up' / 'right'
        if key is None:
            break

        # 🔧 여기! 다음 세그먼트의 "기준 start"는 현재 current_goal 이어야 함
        base_start_for_next = current_goal

        next_goal = goal_from_start_and_key(
            base_start_for_next, key, step_len=args.step_len, turn_deg=45.0, small_turn_deg=20.0
        )

        yaw_val = foot_yaw_from_key(key)
        next_fyaw_goal = {"L": yaw_val, "R": yaw_val}

        # 키 입력으로 생성된 "다음 goal" 위치를 빨간 점으로 표시(누적)
        gx, gy, gz = map(float, next_goal["center"][:3])
        ax.scatter([gx], [gy], [gz], s=10, marker='o', c='r', edgecolors='k', linewidths=0.6)
        fig.canvas.draw();
        fig.canvas.flush_events()

        # ===== 롤링 =====
        current_start = base_start_for_next  # (= 이전 루프의 current_goal)
        curr_fyaw_start = {"L": 0.0, "R": 0.0}  # start yaw는 항상 0°
        current_goal = next_goal  # 새 goal
        curr_fyaw_goal = next_fyaw_goal  # 키에 맞춘 goal yaw

        run_idx += 1
        time.sleep(args.period)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
