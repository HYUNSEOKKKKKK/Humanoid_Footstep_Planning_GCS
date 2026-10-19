from __future__ import annotations
import time
import numpy as np
from collections import Counter
from src.graph import compute_centroids_from_vertices
from src.model import (
    build_gcs_model_wfc,
    pack_relaxed_params_for_model,
    prepare_problem_params,
)
from src.solver import (
    set_parameters, solve_relaxed, get_relaxed_y, extract_solution,
)
from src.viz import plot_feet_3d_wfc
from src.utils import (tic, toc, postprocess_solution   )
from src.solver import run_trials_select_best
from src.graph.wavefunctioncollapse import build_graph_wfc, build_heightmap, plot_heightmap_3d
import matplotlib.pyplot as plt
import math, numpy as np
import meshcat.geometry as mcg
import meshcat.transformations as mct

def overlay_feet_meshcat_simple(
    vis, proc,
    *,
    x_ref, y_ref,                 # ★ plot_feet_3d_wfc와 동일 ref축 전달
    forward_axis="x",             # ★ 동일 옵션
    invert_yaw=False,             # ★ 동일 옵션
    foot_len=0.08, foot_wid=0.04, thickness=0.01, z_lift=0.04,
    color_L=0xE74C3C,  # 왼발=빨강
    color_R=0x2E86DE,  # 오른발=파랑
):
    # ref축 정규화 + 우수(right-handed) 보장 (plot_feet_3d_wfc와 동일)
    xr = np.asarray(x_ref, float).reshape(2)
    yr = np.asarray(y_ref, float).reshape(2)
    xr /= (np.linalg.norm(xr) + 1e-12)
    yr /= (np.linalg.norm(yr) + 1e-12)
    if xr[0]*yr[1] - xr[1]*yr[0] < 0:
        xr = -xr

    def yaw_from_cs_ref(c, s):
        # world_dir = c*x_ref + s*y_ref
        vx = float(c)*xr[0] + float(s)*yr[0]
        vy = float(c)*xr[1] + float(s)*yr[1]
        ang = math.atan2(vy, vx)
        if invert_yaw:
            ang = -ang
        # forward_axis에 따라 박스의 '길이축'을 어느 축에 맞출지 결정
        if forward_axis == "y":
            ang += math.pi/2.0
        # 정규화
        ang = (ang + math.pi) % (2*math.pi) - math.pi
        return ang

    def place_box(name, x, y, z, yaw, color):
        T = mct.translation_matrix([float(x), float(y), float(z) + z_lift]) @ \
            mct.rotation_matrix(float(yaw), [0, 0, 1])
        vis[name].set_object(
            mcg.Box([float(foot_len), float(foot_wid), float(thickness)]),
            mcg.MeshPhongMaterial(color=int(color))
        )
        vis[name].set_transform(T)

    n = len(proc.L.x)
    for i in range(n):
        # Left (RED)
        place_box(f"feet/L_{i:03d}",
                  proc.L.x[i], proc.L.y[i], proc.L.z[i],
                  yaw_from_cs_ref(proc.L.c[i], proc.L.s[i]),
                  color_L)
        # Right (BLUE)
        place_box(f"feet/R_{i:03d}",
                  proc.R.x[i], proc.R.y[i], proc.R.z[i],
                  yaw_from_cs_ref(proc.R.c[i], proc.R.s[i]),
                  color_R)

def main():
    # ======= EDIT HERE: 하드코딩 파라미터 =======
    W, H, tile_size = 4, 4, 1.0
    SEED = None

    STEP_SPAN = 0.5
    STANCE_WIDTH = 0.25
    ALPHA_DEG = 30.0

    M_LARGE = 1e3
    M_SMALL = 1e-6
    TRIALS   = 20
    TEMP     = 1.0

    PLOT_WFC = True   # WFC 파이프라인 중간 플롯

    # ===========================================
    alpha = np.deg2rad(ALPHA_DEG)
    overall_t0 = tic()

    # ----- 그래프 생성 -----
    t_graph = tic()
    G = build_graph_wfc(
        step_span=STEP_SPAN,
        W=W, H=H, tile_size=tile_size,
        stance_width=STANCE_WIDTH,
        seed=SEED,
        plot=PLOT_WFC, plt_show=PLOT_WFC
    )
    v = G["v"]; edge = G["edge"]; mode_list = G["mode_list"]
    vertex_number = G["vertex_number"]; n_edge = G["n_edge"]
    direction_list = G["direction_list"]

    toc(t_graph, "Making graph (WFC)")
    # === Nodes overview: idx | mode | dir | name (from grid) ===
    grid = G["grid"]  # 2D tile names
    Hgrid = len(grid)
    Wgrid = len(grid[0])

    def _center_xyz(vi):
        """v[i]가 사각형(3x4)이든 점(3,)이든 center (x,y[,z]) 반환"""
        a = np.asarray(vi)
        if a.ndim == 2 and a.shape[1] >= 4:  # corners
            cx = float(0.25 * a[0].sum())
            cy = float(0.25 * a[1].sum())
            cz = float(0.25 * a[2].sum()) if a.shape[0] >= 3 else None
            return cx, cy, cz
        # point-like
        cx = float(a[0])
        cy = float(a[1])
        cz = float(a[2]) if a.size >= 3 else None
        return cx, cy, cz

    def _xy_to_rc(x, y, H, tile_size):
        # y-up → grid (row, col)
        r = int(np.floor((H * tile_size - y) / tile_size))
        c = int(np.floor(x / tile_size))
        r = max(0, min(Hgrid - 1, r))
        c = max(0, min(Wgrid - 1, c))
        return r, c

    n = len(v)
    assert n == len(mode_list) == len(direction_list), \
        f"length mismatch: |v|={len(v)}, |mode|={len(mode_list)}, |dir|={len(direction_list)}"

    print("\n[Nodes] idx | mode | dir | name | center")
    for i, (m, d) in enumerate(zip(mode_list, direction_list)):
        cx, cy, cz = _center_xyz(v[i])
        r, c = _xy_to_rc(cx, cy, H, tile_size)  # ← main의 H, tile_size 사용
        name = grid[r][c]
        dstr = d if d is not None else "-"
        cstr = f"({cx:.3f}, {cy:.3f}" + (f", {cz:.3f})" if cz is not None else ")")
        print(f"[{i:03d}]  {m:>3}   {dstr:>2}   {name:<10}  center={cstr}")

    # yaw(deg) -> rad
    to_rad = np.deg2rad
    params_pose = prepare_problem_params(
        G["startpos_L"], G["startpos_R"], G["targetpos_L"], G["targetpos_R"],
        to_rad(G["startyaw_L"]), to_rad(G["startyaw_R"]),
        to_rad(G["targetyaw_L"]), to_rad(G["targetyaw_R"]),
    )
    start_L = params_pose["start_L"]; start_R = params_pose["start_R"]
    target_L = params_pose["target_L"]; target_R = params_pose["target_R"]
    print(start_L, start_R, target_L, target_R)
    # ----- centroid / base_param -----
    centroid_mat, base_param = compute_centroids_from_vertices(v)

    # ----- global ref 축 -----
    pL = np.asarray(G["startpos_L"][:2], float)
    pR = np.asarray(G["startpos_R"][:2], float)
    y_ref = pL - pR
    n = np.linalg.norm(y_ref)
    y_ref = np.array([0.0, 1.0]) if n < 1e-12 else y_ref / n
    x_ref = np.array([y_ref[1], -y_ref[0]], float)
    if x_ref[0]*y_ref[1] - x_ref[1]*y_ref[0] < 0:  # 우수 좌표계 보장
        x_ref = -x_ref
    print("x_ref:", x_ref, "y_ref:", y_ref)

    # ----- 모델 생성 -----
    t_parameter = tic()
    M, V, IDX, PAR, build_time = build_gcs_model_wfc(
        v, edge, STEP_SPAN, alpha, mode_list, x_ref, y_ref, direction_list
    )
    toc(t_parameter, "making optimization problem")

    # ===== 1차: 릴랙스드 =====
    t_solving = tic()
    params_relaxed = pack_relaxed_params_for_model(
        centroid_mat=centroid_mat, n_edge=n_edge, M_large=M_LARGE,
        start_L=start_L, start_R=start_R, target_L=target_L, target_R=target_R,
    )
    set_parameters(M, V, params_relaxed)
    sol_relaxed, obj1 = solve_relaxed(M, V, M_large=M_LARGE, log_level=0)
    print(f"Cost (relaxed path): {obj1:.3f}")
    toc(t_solving, "solving time(relaxed path)")

    # ===== 경로 선택 + 2차 =====
    y_relaxed_val = get_relaxed_y(V)
    best_obj, best = run_trials_select_best(
        M, V, IDX,
        centroid_mat=centroid_mat,
        y_relaxed_val=y_relaxed_val,
        edge=edge, vertex_number=vertex_number, n_edge=n_edge,
        M_small=M_SMALL,
        start_L=start_L, start_R=start_R, target_L=target_L, target_R=target_R,
        trials=TRIALS, temp=TEMP, greedy_first=True,
    )
    relaxed_path, ex_after = best["path"], best["ex"]

    # ----- 후처리 -----
    t_post = tic()
    proc = postprocess_solution(
        ex_after, edge, relaxed_path,
        G["startpos_L"], G["startpos_R"], G["targetpos_L"], G["targetpos_R"],
        target_L, target_R,
    )
    toc(t_post, "postprocess")
    print(f"Total main.py running Time: {time.perf_counter() - overall_t0:.4f} seconds")

    # ----- 플롯(one-shot) -----
    plot_feet_3d_wfc(
        proc.L.x, proc.L.y, proc.L.z, proc.L.c, proc.L.s,
        proc.R.x, proc.R.y, proc.R.z, proc.R.c, proc.R.s,
        x_ref_override=x_ref, y_ref_override=y_ref,
        world_size=(W*tile_size, H*tile_size),  # 자동 범위 쓰려면 이 줄 삭제
        forward_axis="x", invert_yaw=False,
        show=False,
    )
    plt.draw(); plt.pause(0.01)

    # ... feet matplotlib 그림 (show=False)까지 그대로 ...

    grid = G["grid"]
    s = tuple(G.get("start_rc"))
    t = tuple(G.get("goal_rc"))

    # vis = plot_heightmap_3d(
    #     None, None, None,
    #     tile_grid=grid,
    #     tile_size=tile_size, h_step=0.2, stair_slices=2,
    #     elev=50, azim=240, start=s, goal=t,
    #     marker_size_ratio=0.35, marker_lift=0.10
    # )
    # # plot_heightmap_3d 끝에 `return vis`가 있어야 합니다.
    #
    # overlay_feet_meshcat_simple(
    #     vis, proc,
    #     x_ref=x_ref, y_ref=y_ref,  # ★ 동일 ref축
    #     forward_axis="x", invert_yaw=False,  # ★ plot_feet_3d_wfc와 동일
    #     foot_len=0.24, foot_wid=0.10, thickness=0.01, z_lift=0.02,
    #     color_L=0xE74C3C, color_R=0x2E86DE
    # )

    plt.show()

if __name__ == "__main__":
    main()



