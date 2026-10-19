# main_longfootsteps_optapi.py
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from main import load_experiment  # 네 main.py 안의 함수
from src.experiments import list_experiments
from src.terrain_preset import load_terrain
from src.utils import expand_shorthand, postprocess_solution

from src.graph import build_graph, compute_centroids_from_vertices
from src.model import build_gcs_model_short, pack_relaxed_params_for_model, prepare_problem_params
from src.solver import (
    set_parameters, solve_relaxed, get_relaxed_y,
    run_trials_select_best
)
from src.viz.feet import plot_feet_3d


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

    # === 모델 생성 ===
    t1 = time.perf_counter()
    M, V, IDX, PAR, build_time = build_gcs_model_short(v, edge, step_span, alpha, mode_list, x_ref, y_ref)
    t_build = time.perf_counter() - t1

    # === 1차: 릴랙스드 ===
    params_relaxed = pack_relaxed_params_for_model(
        centroid_mat=centroid_mat, n_edge=n_edge, M_large=M_large,
        start_L=start_L, start_R=start_R, target_L=target_L, target_R=target_R,
    )
    params_relaxed["x_ref"] = x_ref
    params_relaxed["y_ref"] = y_ref
    set_parameters(M, V, params_relaxed)

    t2 = time.perf_counter()
    sol_relaxed, obj1 = solve_relaxed(M, V, M_large=M_large, log_level=0)
    t_relaxed = time.perf_counter() - t2

    # === 경로 선택 + 2차 ===
    y_relaxed_val = get_relaxed_y(V)
    t3 = time.perf_counter()
    try:
        best_obj, best = run_trials_select_best(
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
                cand_obj, cand = run_trials_select_best(
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
    parser.add_argument("--terrain", type=str, default="default_flat_DLR")
    parser.add_argument("--terrain-file", type=str, default=None)
    parser.add_argument("--experiments-file", type=str, default="configs/experiments_DLR.json")
    parser.add_argument("--exp-seq", type=str, nargs="+", default=None,
                        help="experiment names to run in order")
    parser.add_argument("--period", type=float, default=0.5,
                        help="seconds between runs (auto loop)")
    parser.add_argument("--step_span", type=float, default=0.33)
    parser.add_argument("--alpha_deg", type=float, default=30.0)
    parser.add_argument("--M_large", type=float, default=1e3)
    args = parser.parse_args()

    # Terrain 고정 로드
    T = load_terrain(args.terrain, args.terrain_file)

    # 실행할 실험 시퀀스
    if args.exp_seq is None:
        exp_names = list_experiments(args.experiments_file)  # 파일 순서대로
    else:
        exp_names = args.exp_seq

    alpha = float(args.alpha_deg) * np.pi / 180.0

    # 플롯 준비(누적)
    plt.ion()
    fig = plt.figure("Online footstep planning (Drawing 'DLR')")
    if fig.axes:
        ax = fig.axes[0]
    else:
        ax = fig.add_subplot(111, projection="3d")

    next_goal_artists = []  # 프리뷰(점/화살표) 핸들 리스트
    run_idx = 0

    for run_idx, name in enumerate(exp_names):
        name = exp_names[run_idx % len(exp_names)]

        # shorthand 확장
        raw = load_experiment(name, args.experiments_file)
        exp = expand_shorthand(raw, 0.25)

        # ===== 최적화 수행 =====
        info = solve_once(
            exp, T,
            step_span=args.step_span,
            alpha=alpha,
            M_large=args.M_large,
            M_small=1e-6
        )

        # ===== 후처리 =====
        proc = postprocess_solution(
            info["ex"], info["edge"], info["relaxed_path"],
            exp["startpos_L"], exp["startpos_R"],
            exp["targetpos_L"], exp["targetpos_R"],
            info["target_L"], info["target_R"],
        )

        # 풀이시간/스텝수 라벨
        tms = info["times"]
        solve_time = tms["graph"] + tms["build"] + tms["relaxed"] + tms["fixed"]
        # 필요 시 -2 유지 (네 코드에 맞춤). 원하면 그냥 len(L)+len(R)로 바꿔도 됨.
        n_steps = int(len(proc.L.x) + len(proc.R.x) - 2)
        time_steps_label = f"{solve_time:.3f}s / {n_steps} steps"

        # ===== 누적 플롯 =====
        plot_feet_3d(
            proc.L.x, proc.L.y, proc.L.z, proc.L.c, proc.L.s,
            proc.R.x, proc.R.y, proc.R.z, proc.R.c, proc.R.s,
            matlab_style=True,
            startpos_L=exp["startpos_L"], startpos_R=exp["startpos_R"],
            ax=ax,                 # 같은 축 재사용
            accumulate=True,       # 누적
            show=False,            # 블록 금지
            label=time_steps_label # 범례: 시간/스텝수
        )
        fig.canvas.draw(); fig.canvas.flush_events()

        # 콘솔 로그
        print(f"[{name}] graph={tms['graph']:.3f}s, build={tms['build']:.3f}s, "
              f"relaxed={tms['relaxed']:.3f}s, fixed={tms['fixed']:.3f}s, "
              f"obj_relaxed={info['obj_relaxed']:.3f}, obj_fixed={info['obj_fixed']:.3f}")

        # ===== 다음 테스트 goal center + heading(화살표) 프리뷰 =====
        next_idx = (run_idx + 1) % len(exp_names)
        next_name = exp_names[next_idx]

        # raw JSON 그대로 읽기 (expand_shorthand 필요 없음)
        raw_next = load_experiment(next_name, args.experiments_file)
        goal = raw_next["goal"]

        # 이전 프리뷰 제거
        for a in next_goal_artists:
            try:
                a.remove()
            except Exception:
                pass
        next_goal_artists = []

        # goal.center -> 점 위치
        cx, cy = float(goal["center"][0]), float(goal["center"][1])
        cz = float(goal["center"][2]) if len(goal["center"]) > 2 else 0.0

        # goal.heading_deg -> 화살표 방향 (전역 x축 기준, CCW+)
        theta = np.deg2rad(float(goal.get("heading_deg", 0.0)))
        u, v, w = np.cos(theta), np.sin(theta), 0.0

        # 점 + 화살표
        pt = ax.scatter([cx], [cy], [cz], s=70, marker='o', c='r',
                        edgecolors='k', linewidths=0.6)
        arrow = ax.quiver(cx, cy, cz, u, v, w,
                          length=0.25, normalize=True, color='r', linewidths=1.8)

        next_goal_artists = [pt, arrow]
        fig.canvas.draw();
        fig.canvas.flush_events()

        # ===== 주기 대기 =====
        time.sleep(args.period)

        # 대기 끝: 프리뷰 제거
        for a in next_goal_artists:
            try: a.remove()
            except Exception: pass
        next_goal_artists = []

        # 다음 테스트로
        run_idx += 1
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
