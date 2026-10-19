from __future__ import annotations
import time
import numpy as np
import mosek.fusion as mf
from src.experiments import load_experiment, list_experiments
from src.graph import build_graph, compute_centroids_from_vertices
from src.model import (
    build_gcs_model,
    pack_relaxed_params_for_model, pack_fixedpath_params_for_model,
    prepare_problem_params, pack_base_param,
)
from src.solver import (
    set_parameters, solve_relaxed, solve_with_fixed_path,
    get_relaxed_y, extract_solution, recover_x_from_z_over_y,
    relaxation_max, path_to_y, pick_path_from_relaxation,
)
from src.viz import plot_feet_3d
from src.utils import (tic, toc, normalize_cos_sin, postprocess_solution, prepare_and_send_udp,
    build_combined_from_proc, make_udp_payload, send_udp,
)
from src.terrain_preset import load_terrain, list_terrains
from src.solver import relaxation_prob, run_trials_select_best
from src.utils import expand_shorthand
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_experiment(exp_name: str, experiments_file: str | None):
    path = Path(experiments_file or "configs/experiments.json")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if exp_name not in data:
        raise KeyError(f"Experiment '{exp_name}' not found in {path}")
    return data[exp_name]  # ★ 여기서는 '그대로' 반환만

def main():


    parser = argparse.ArgumentParser(description="GCS Footstep Planning (Python Fusion)")

    # Start/Goal preset
    parser.add_argument("--experiment", type=str, default="test_01",
                        help="experiment preset name in configs/experiments.json")
    parser.add_argument("--experiments-file", type=str, default=None,
                        help="path to experiments.json (optional)")
    parser.add_argument("--list-experiments", action="store_true",
                        help="list available experiments and exit")

    # Terrain preset
    parser.add_argument("--terrain", type=str, default="default_flat_",
                        help="terrain preset name in configs/terrain.json")
    parser.add_argument("--terrain-file", type=str, default=None,
                        help="path to terrain.json (optional)")
    parser.add_argument("--list-terrains", action="store_true",
                        help="list available terrains and exit")

    # 하이퍼파라미터
    parser.add_argument("--copy_counts", type=int, nargs="+", default=[5, 5], help="copy counts per region")
    parser.add_argument("--step_span", type=float, default=0.33, help="max step span (scalar)")
    parser.add_argument("--alpha_deg", type=float, default=30.0, help="yaw rotation input in degrees")
    parser.add_argument("--M_large", type=float, default=1e3, help="Big-M for relaxed solve")
    parser.add_argument("--M_small", type=float, default=1e-6, help="Small-M for equality-like enforcement")
    parser.add_argument("--trials", type=int, default=50, help="number of relaxed trials (unused placeholder)")


    # 시각화 토글
    parser.add_argument("--plot", action="store_true", help="plot 3D footsteps")
    parser.add_argument("--stance-width", type=float, default=0.25,
                        help="stance width for shorthand experiments")


    args = parser.parse_args()

    # 프리셋 목록만 출력하고 종료
    if args.list_experiments:
        names = list_experiments(args.experiments_file)
        print("Available experiments:")
        for n in names:
            print("  -", n)
        return

    if args.list_terrains:
        names = list_terrains(args.terrain_file)
        print("Available terrains:")
        for n in names:
            print("  -", n)
        return

    # ===== 실험 프리셋 로드 (축약 포맷 → 완전 포맷으로 '항상' 변환) =====
    exp_raw = load_experiment(args.experiment, args.experiments_file)  # 기존 헬퍼 그대로 사용
    exp = expand_shorthand(exp_raw, args.stance_width)                 # ✅ 딱 이 한 줄로 변환

    print(f"[preset] using experiment: {args.experiment}")
    startpos_L = exp["startpos_L"]; startpos_R = exp["startpos_R"]
    targetpos_L = exp["targetpos_L"]; targetpos_R = exp["targetpos_R"]
    # ✔ 발 자체 yaw는 *_deg_* 키에서 직접 읽어 사용 (대부분 0)
    startyaw_L  = exp["startyaw_L"];  startyaw_R  = exp["startyaw_R"]
    targetyaw_L = exp["targetyaw_L"]; targetyaw_R = exp["targetyaw_R"]

    # ===== Terrain 프리셋 로드 (절대 건드리지 않음) =====
    T = load_terrain(args.terrain, args.terrain_file)
    print(f"[terrain] using: {T['name']}")
    terrain_stones = T["stones"]

    # ===== 하이퍼파라미터 =====
    step_span   = float(args.step_span)
    alpha       = float(args.alpha_deg) * np.pi / 180.0
    # terrain에 copy_counts가 있으면 그걸 우선 사용, 없으면 CLI 값 사용
    copy_counts = T["copy_counts"] if T.get("copy_counts") is not None else list(args.copy_counts)

    # ===== vertex 이름 배열 생성 =====
    v_names = []
    for stone, count in zip(terrain_stones, copy_counts):
        v_names.extend([stone["name"]] * count)
    print(f"[terraixxn] {v_names}")

    # 길이 검증
    if len(copy_counts) != len(terrain_stones):
        raise ValueError(f"copy_counts({len(copy_counts)}) must match #stones({len(terrain_stones)}).")

    M_large = float(args.M_large)
    M_small = float(args.M_small)

    overall_t0 = tic()

    # ----- 그래프 생성 -----
    t_graph = tic()
    G = build_graph(
        startpos_L, startpos_R, targetpos_L, targetpos_R,
        step_span=step_span, copy_counts=copy_counts, edge_rule="general",
        terrain_stones=terrain_stones,   # ★ terrain 전달
    )
    v = G["v"]; edge = G["edge"]; mode_list = G["mode_list"]
    vertex_number = G["vertex_number"]; n_edge = G["n_edge"]
    toc(t_graph, "Making graph")

    for i, verts in enumerate(v[:]):
        print(f"v[{i}] corners (x,y,z) each column:")
        print(np.round(verts, 3))



    # ----- 파라미터(시작/목표 5D) 준비 -----
    params_pose = prepare_problem_params(
        startpos_L, startpos_R, targetpos_L, targetpos_R,
        startyaw_L, startyaw_R, targetyaw_L, targetyaw_R
    )
    start_L = params_pose["start_L"]; start_R = params_pose["start_R"]
    target_L = params_pose["target_L"]; target_R = params_pose["target_R"]

    print(start_L, start_R, target_L, target_R)
    # ----- centroid / base_param 계산 (MATLAB의 reshape(centroid,[],1) 대응) -----
    centroid_mat, base_param = compute_centroids_from_vertices(v)

    startL_val = np.asarray(startpos_L, float).reshape(-1)
    startR_val = np.asarray(startpos_R, float).reshape(-1)

    # global ref 축 계산
    pL, pR = startL_val[:2], startR_val[:2]
    vRL = pL - pR
    if np.linalg.norm(vRL) < 1e-12:
        y_ref = np.array([0.0, 1.0])
    else:
        y_ref = vRL / np.linalg.norm(vRL)
    x_ref = np.array([y_ref[1], -y_ref[0]])
    print(x_ref, y_ref)

    # ----- 모델 생성 (한 번만) -----
    t_parameter = tic()
    M, V, IDX, PAR, build_time = build_gcs_model(v, edge, step_span, alpha, mode_list, x_ref, y_ref)
    toc(t_parameter, "making optimization problem")

    # ==================== 1차: 릴랙스드 ====================
    t_solving = tic()
    params_relaxed = pack_relaxed_params_for_model(
        centroid_mat=centroid_mat, n_edge=n_edge, M_large=M_large,
        start_L=start_L, start_R=start_R, target_L=target_L, target_R=target_R,
    )
    params_relaxed["x_ref"] = x_ref
    params_relaxed["y_ref"] = y_ref

    set_parameters(M, V, params_relaxed)
    sol_relaxed, obj1 = solve_relaxed(M, V, M_large=M_large, log_level=0)
    print(f"Cost (relaxed path): {obj1:.3f}")
    toc(t_solving, "solving time(relaxed path)")

    # ====================
    # y (relaxed) 확인
    ex_relax = extract_solution(V, IDX)
    print("=== y  ===")
    print(ex_relax["y"])

    # ==================== 경로 선택 + 2차 (greedy 먼저, 이후 확률) ====================
    y_relaxed_val = get_relaxed_y(V)  # (n_edge,)

    best_obj, best = run_trials_select_best(
        M, V, IDX,
        centroid_mat=centroid_mat,
        y_relaxed_val=y_relaxed_val,
        edge=edge,
        vertex_number=vertex_number,
        n_edge=n_edge,
        M_small=M_small,
        start_L=start_L, start_R=start_R,
        target_L=target_L, target_R=target_R,
        trials=int(args.trials),
        temp=getattr(args, "temp", 1.0),
        greedy_first=True,  # ← 요청하신 대로 greedy를 항상 첫 trial로 수행
    )

    relaxed_path = best["path"]
    y_relax = best["y_relax"]
    sol_fixed = best["sol"]
    ex_after = best["ex"]


    # # ----- 후처리 / 결과 추출 -----
    t_post = tic()

    proc = postprocess_solution(
        ex_after, edge, relaxed_path,
        startpos_L, startpos_R, targetpos_L, targetpos_R,
        target_L, target_R,
    )
    toc(t_post, "postprocess")
    print(f"Total main.py running Time: {time.perf_counter() - overall_t0:.4f} seconds")

    # y_relax[0] == 1 → 왼발 먼저, == 0 → 오른발 먼저
    first_is_left = bool(y_relax[0] == 1)

    plot_feet_3d(
        proc.L.x, proc.L.y, proc.L.z, proc.L.c, proc.L.s,
        proc.R.x, proc.R.y, proc.R.z, proc.R.c, proc.R.s,
        matlab_style=True,
        startpos_L=startpos_L, startpos_R=startpos_R,
        animate=True, interval=0.1,
        first_is_left=first_is_left,  # ★ 여기서 제어
        show=True,
    )


if __name__ == "__main__":
    main()



