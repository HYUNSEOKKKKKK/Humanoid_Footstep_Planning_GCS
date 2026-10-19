from __future__ import annotations
import time
import mosek.fusion as mf
import math
from src.experiments import load_experiment, list_experiments
from src.graph import build_graph, compute_centroids_from_vertices
from src.model import (
    build_gcs_model,
    pack_relaxed_params_for_model, pack_fixedpath_params_for_model,
    prepare_problem_params, pack_base_param, build_gcs_model_short,
)
from src.solver import (
    set_parameters, solve_relaxed, solve_with_fixed_path,
    get_relaxed_y, extract_solution, recover_x_from_z_over_y,
    relaxation_max, path_to_y, pick_path_from_relaxation,
)
from src.viz import plot_feet_3d_VAE
from src.utils import (tic, toc, normalize_cos_sin, postprocess_solution, prepare_and_send_udp,
    build_combined_from_proc, make_udp_payload, send_udp,
)
from src.terrain_preset import load_terrain, list_terrains, load_terrain_yaw
from src.solver import relaxation_prob, run_trials_select_best
from src.utils import expand_shorthand
import argparse, json
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import csv
from datetime import datetime


def plot_vertices_xy(v_list, *, ax=None, facecolor=None, edgecolor="k",
                     alpha=0.25, linewidth=1.0, annotate=False,
                     title="Vertices (top view)",
                     points=None, points_label="points", points_style=None):
    """
    v_list: 각 원소가 3x4 (x,y,z; 4 corners as columns) 배열인 리스트/배열
    points: (N,2) or (2,N) or (x_list, y_list) 형태 허용
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    for i, verts in enumerate(v_list):
        verts = np.asarray(verts)
        if verts.shape == (4, 3):
            verts = verts.T
        xy = verts[:2, :].T

        poly = Polygon(
            xy, closed=True,
            facecolor=(facecolor if facecolor is not None else "C0"),
            edgecolor=edgecolor, alpha=alpha, linewidth=linewidth
        )
        ax.add_patch(poly)

        if annotate:
            cx, cy = xy.mean(axis=0)
            ax.text(cx, cy, str(i), ha="center", va="center", fontsize=8)

    # --- points overlay ---
    if points is not None:
        if isinstance(points, tuple) and len(points) == 2:
            px, py = np.asarray(points[0]), np.asarray(points[1])
        else:
            P = np.asarray(points)
            if P.shape[0] == 2 and P.ndim == 2:
                px, py = P[0], P[1]
            else:
                px, py = P[:, 0], P[:, 1]

        style = dict(s=35, marker="o", zorder=5)
        if points_style:
            style.update(points_style)
        ax.scatter(px, py, label=points_label, **style)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid(True, linestyle=":")
    ax.autoscale()
    return fig, ax



# ----- 1) 본체: 이미 로드된 stones 리스트를 바로 그린다 -----
def plot_stones_3d(stones: List[Dict[str, Any]],
                   *,
                   thickness: float = 0.04,
                   face_alpha: float = 0.35,
                   edgecolor: str = 'k',
                   ax=None,
                   colorbar: bool = True,
                   view_top: bool = False,
                   title: str = " GCS footstep planner "):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch, Polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    created_ax = False

    # ---------------------------
    # axes/figure 준비
    # ---------------------------
    if ax is None:
        if view_top:
            fig, ax = plt.subplots(figsize=(8, 6))   # 2D
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
        created_ax = True
    else:
        fig = ax.figure

    n = len(stones)
    if n == 0:
        if created_ax:
            plt.tight_layout(); plt.show()
        return [], ax

    centers_z = np.array([float(s["center"][2]) for s in stones], dtype=float)

    # ---------------------------
    # 가운데 스톤들 z범위로 그레이스케일 설정
    # ---------------------------
    use_colorbar = False
    if n >= 3 and colorbar:
        mid_z = centers_z[1:-1]
        if mid_z.max() > mid_z.min():
            use_colorbar = True
            z_min, z_max = -0.15, 0.15
        else:
            z_min, z_max = float(mid_z.min()), float(mid_z.max())
    else:
        z_min, z_max = 0.0, 1.0

    cmap = plt.get_cmap('gray')
    norm = Normalize(vmin=z_min - 0.01, vmax=z_max + 0.01)

    def gray_by_z(cz):
        if n < 3:
            return (0.6, 0.6, 0.6, face_alpha)
        r, g, b, _ = cmap(norm(cz))
        return (r, g, b, face_alpha)

    # ============================================================
    #  (A) view_top=True  -> 2D top-down 렌더링
    # ============================================================
    if view_top:
        # 사용자가 3D ax를 넘겨도 main 수정 없이 동작하도록:
        #  - 3D ax는 숨기고
        #  - fig에 2D ax를 캐시해서 그쪽에 그림
        if ax is not None:
            pos = ax.get_position()
        else:
            pos = None

        # fig에 캐시된 2D ax가 있으면 재사용
        ax2d = getattr(fig, "_gcs_top2d_ax", None)
        if ax2d is None or ax2d not in fig.axes:
            if pos is not None:
                ax2d = fig.add_axes(pos)    # 같은 위치에 2D axes 생성
            else:
                ax2d = fig.add_subplot(111)
            fig._gcs_top2d_ax = ax2d
        else:
            # 위치가 바뀌었을 수 있으니 동기화
            if pos is not None:
                ax2d.set_position(pos)

        # 기존 ax(대개 3D)는 숨김
        if ax is not None and ax is not ax2d:
            ax.set_visible(False)
        ax2d.set_visible(True)

        # 2D 그리기 시작
        ax2d.cla()
        artists = []
        all_xy = []

        def top_face_xy(center, w, h, yaw_deg=0.0):
            cx, cy, _ = center
            w2, h2 = 0.5 * w, 0.5 * h
            V = np.array([[-w2, -h2],
                          [ w2, -h2],
                          [ w2,  h2],
                          [-w2,  h2]], dtype=float)
            th = np.deg2rad(float(yaw_deg))
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s],
                          [s,  c]], dtype=float)
            V = (R @ V.T).T
            V[:, 0] += float(cx)
            V[:, 1] += float(cy)
            return V

        for idx, s in enumerate(stones):
            cx, cy, cz = s["center"]
            w = float(s["width"])
            h = float(s["height"])
            yaw = float(s.get("yaw_deg", 0.0))

            if idx == 0:
                facecolor = (0.0, 0.7, 0.2, face_alpha)  # Start
            elif idx == n - 1:
                facecolor = (0.9, 0.1, 0.1, face_alpha)  # Goal
            else:
                facecolor = gray_by_z(float(cz))

            Vxy = top_face_xy((cx, cy, cz), w, h, yaw)
            poly2d = Polygon(Vxy, closed=True, facecolor=facecolor,
                             edgecolor=edgecolor, linewidth=0.8)
            ax2d.add_patch(poly2d)
            artists.append(poly2d)
            all_xy.append(Vxy)

        # 축 범위 자동 + equal aspect
        if all_xy:
            P = np.vstack(all_xy)
            rng = np.ptp(P, axis=0)
            pad = 0.1 * max(float(rng[0]), float(rng[1]), 1e-6)
            ax2d.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
            ax2d.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)

        ax2d.set_aspect('equal', adjustable='box')
        ax2d.grid(False)


        # 컬러바
        if n >= 3 and use_colorbar:
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            cbar = fig.colorbar(mappable, ax=ax2d, fraction=0.046, pad=0.04)
            cbar.set_label('Height (z)', rotation=90)

        return artists, ax2d

    # ============================================================
    #  (B) view_top=False  -> 기존 3D 렌더링 유지
    # ============================================================

    # 혹시 view_top=True로 그렸던 2D ax가 있으면 숨김
    ax2d = getattr(fig, "_gcs_top2d_ax", None)
    if ax2d is not None and ax2d in fig.axes:
        ax2d.set_visible(False)
    ax.set_visible(True)

    def make_box_vertices(center, w, h, t, yaw_deg=0.0):
        cx, cy, cz = center
        w2, h2 = 0.5 * w, 0.5 * h
        top = np.array([[-w2, -h2, 0.0],
                        [ w2, -h2, 0.0],
                        [ w2,  h2, 0.0],
                        [-w2,  h2, 0.0]], dtype=float)
        th = np.deg2rad(float(yaw_deg))
        c, s = np.cos(th), np.sin(th)
        Rz = np.array([[c, -s, 0.0],
                       [s,  c, 0.0],
                       [0.0,0.0,1.0]], dtype=float)
        top = (Rz @ top.T).T + np.array([cx, cy, cz])
        bottom = top.copy(); bottom[:, 2] = cz - t
        return np.vstack([top, bottom])

    def box_faces_from_vertices(V8):
        top = [tuple(V8[i]) for i in range(0,4)]
        bottom = [tuple(V8[i]) for i in range(4,8)]
        sides = []
        for k in range(4):
            k2 = (k + 1) % 4
            sides.append([bottom[k], top[k], top[k2], bottom[k2]])
        return [top, bottom] + sides

    artists = []
    all_xyz = []
    #padding
    for idx, s in enumerate(stones):
        cx, cy, cz = s["center"]
        w = float(s["width"])
        h = float(s["height"])
        yaw = float(s.get("yaw_deg", 0.0))

        if idx == 0:
            facecolor = (0.0, 0.7, 0.2, face_alpha)
        elif idx == n - 1:
            facecolor = (0.9, 0.1, 0.1, face_alpha)
        else:
            facecolor = gray_by_z(centers_z[idx])

        V8 = make_box_vertices((cx, cy, cz), w, h, thickness, yaw)
        faces = box_faces_from_vertices(V8)
        poly = Poly3DCollection(faces, edgecolor=edgecolor, linewidths=0.8, zsort='min')
        poly.set_facecolor(facecolor)
        poly.set_array(None); poly.set_cmap(None)

        ax.add_collection3d(poly)
        artists.append(poly)
        all_xyz.append(V8)

    if all_xyz:
        P = np.vstack(all_xyz)
        rng = np.ptp(P, axis=0)
        pad = 0.1 * max(float(rng[0]), float(rng[1]), float(rng[2]), 1e-6)
        ax.set_xlim(P[:,0].min()-pad, P[:,0].max()+pad)
        ax.set_ylim(P[:,1].min()-pad, P[:,1].max()+pad)
        ax.set_zlim(P[:,2].min()-pad, P[:,2].max()+pad)

    xr, yr, zr = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    maxr = max(xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0])
    cx = 0.5*(xr[0]+xr[1]); cy = 0.5*(yr[0]+yr[1]); cz = 0.5*(zr[0]+zr[1])
    ax.set_xlim(cx-maxr/2, cx+maxr/2)
    ax.set_ylim(cy-maxr/2, cy+maxr/2)
    ax.set_zlim(cz-maxr/3, cz+maxr/3)
    # ax.set_box_aspect((cx, cy, cz))
    # ax.set_axis_off()

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.grid(False)
    ax.view_init(elev=15, azim=-115)

    legend_patches = []
    if n >= 1:
        legend_patches.append(Patch(facecolor=(0.0, 0.7, 0.2), edgecolor='k', label='Start'))
    if n >= 2:
        legend_patches.append(Patch(facecolor=(0.9, 0.1, 0.1), edgecolor='k', label='Goal'))
    if legend_patches:
        ax.legend(handles=legend_patches, loc='best')

    if n >= 3 and use_colorbar:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Height (z)', rotation=90)

    return artists, ax

# -------- rotation (z-axis) --------
def _Rz_deg(deg: float) -> np.ndarray:
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0.0],
                     [sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)
def _foot_poly_xy(x, y, c, s, w=0.10, h=0.20):
    """중심(x,y)와 yaw=(c,s)로 2D 발 사각형 꼭짓점 4개를 CCW로 반환 (shape: (4,2))."""
    # 로컬 코너 (폭 w: x축, 길 h: y축으로 가정)
    half_w, half_h = 0.5*w, 0.5*h
    local = np.array([[-half_w, -half_h],
                      [ half_w, -half_h],
                      [ half_w,  half_h],
                      [-half_w,  half_h]], dtype=float)  # (4,2)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    world = (local @ R.T) + np.array([x, y])
    return world  # (4,2)



def rotate_vertices_with_yaw(v_list, terrain_stones, copy_counts):
    """
    v_list: List of (3,4) vertex arrays from build_graph
    terrain_stones: [{'center':..., 'width':..., 'height':..., 'yaw_deg':...}, ...]
    copy_counts: 각 stone 복제 개수
    """
    rotated_v = []
    # copy_counts를 따라 stone 인덱스를 확장 (예: [5,5,5] → [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2])
    stone_idx_expanded = [i for i, c in enumerate(copy_counts) for _ in range(c)]

    for vtx, stone_idx in zip(v_list, stone_idx_expanded):
        stone = terrain_stones[stone_idx]
        yaw_deg = float(stone.get("yaw_deg", 0.0))
        if abs(yaw_deg) < 1e-9:
            rotated_v.append(vtx)  # 회전 불필요
            continue

        cx, cy, cz = stone["center"]
        th = math.radians(yaw_deg)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]])

        # 회전: (x−cx, y−cy)를 회전 → 다시 중심 더하기
        v_shifted = vtx.copy()
        v_shifted[0, :] -= cx
        v_shifted[1, :] -= cy
        v_rot = R @ v_shifted
        v_rot[0, :] += cx
        v_rot[1, :] += cy
        rotated_v.append(v_rot)

    return rotated_v


def heading_deg_x_forward(c0, c1):
    dx = float(c1[0]) - float(c0[0])
    dy = float(c1[1]) - float(c0[1])
    if abs(dx) + abs(dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dy, dx))  # 표준형 (+x 전진 기준)

def make_entry_from_stones(stones, stance_width):
    sc = stones[0]["center"]; gc = stones[-1]["center"]
    if len(stones) >= 2:
        shy = heading_deg_x_forward(stones[0]["center"], stones[1]["center"])
        ghy = heading_deg_x_forward(stones[-2]["center"], stones[-1]["center"])
    else:
        shy = ghy = 0.0
    return {
        "start": {"center": sc, "heading_deg": shy},
        "goal":  {"center": gc, "heading_deg": ghy},
        "stance_width": float(stance_width),
        "foot_yaw_deg": {"start_L":0.0,"start_R":0.0,"goal_L":0.0,"goal_R":0.0},
    }



def main():

    parser = argparse.ArgumentParser(description="GCS Footstep Planning (Python Fusion)")

    # 하이퍼파라미터
    parser.add_argument("--copy_counts", type=int, nargs="+", default=[5,4,4,5], help="copy counts per region")
    parser.add_argument("--step_span", type=float, default=0.5, help="max step span (scalar)")
    parser.add_argument("--alpha_deg", type=float, default=30.0, help="yaw rotation input in degrees")
    parser.add_argument("--M_large", type=float, default=1e3, help="Big-M for relaxed solve")
    parser.add_argument("--M_small", type=float, default=1e-6, help="Small-M for equality-like enforcement")
    parser.add_argument("--trials", type=int, default=100, help="number of relaxed trials (unused placeholder)")
    parser.add_argument("--trial_budget_sec", type=float, default=18.3, help="time budget (sec) for trial loop; <=0 disables")

    # 시각화 토글
    parser.add_argument("--stance-width", type=float, default=0.25, help="stance width for shorthand experiments")

    # json 파일 관련 부분
    parser.add_argument("--terrain_dir", type=str, default="data/terrains/alpha",
                        help="directory containing generated terrain json files")
    parser.add_argument("--terrain_key", type=str, default="terrain",
                        help="top-level key name inside each json")

    args = parser.parse_args()

    terrain_dir = Path(args.terrain_dir)
    json_files = sorted(terrain_dir.glob("*.json"))
    if len(json_files) == 0:
        raise FileNotFoundError(f"No .json files found in: {terrain_dir}")

    # ✅ 파일별 결과 모아두기
    results = []
    last_plot_payload = None  # 마지막 성공 케이스 저장용

    # =========================
    # ✅ terrain 하나당 최적화 1회
    # =========================
    for k, jp in enumerate(json_files):
        print(f"\n=== [{k+1}/{len(json_files)}] Loaded: {jp.name} | key='{args.terrain_key}' ===")

        try:
            # 0) Load
            T = load_terrain_yaw(name=args.terrain_key, path=jp)
            terrain_stones = T["stones"]
            print(f"    #stones={len(terrain_stones)}")

            # 1) expand_shorthand용 entry 생성
            entry = make_entry_from_stones(terrain_stones, stance_width=args.stance_width)

            # 2) 기존 파이프라인 그대로 사용
            exp = expand_shorthand(entry, args.stance_width)
            print(f"From VAE terrain ({jp.name}), start/goal auto-set")

            startpos_L = exp["startpos_L"]
            startpos_R = exp["startpos_R"]
            targetpos_L = exp["targetpos_L"]
            targetpos_R = exp["targetpos_R"]
            startyaw_L = exp["startyaw_L"]
            startyaw_R = exp["startyaw_R"]
            targetyaw_L = exp["targetyaw_L"]
            targetyaw_R = exp["targetyaw_R"]

            # ===== 하이퍼파라미터 =====
            step_span   = float(args.step_span)
            alpha       = float(args.alpha_deg) * np.pi / 180.0

            # terrain에 copy_counts가 있으면 그걸 우선 사용, 없으면 CLI 값 사용
            copy_counts = T["copy_counts"] if T.get("copy_counts") is not None else list(args.copy_counts)

            # 길이 검증
            if len(copy_counts) != len(terrain_stones):
                raise ValueError(f"[{jp.name}] copy_counts({len(copy_counts)}) must match #stones({len(terrain_stones)}).")

            # ===== vertex 이름 배열 생성 =====
            v_names = []
            for stone, count in zip(terrain_stones, copy_counts):
                v_names.extend([stone["name"]] * count)
            print(f"[terrain] v_names len={len(v_names)}")

            M_large = float(args.M_large)
            M_small = float(args.M_small)

            overall_t0 = tic()

            # ----- 그래프 생성 -----
            t_graph = tic()
            G = build_graph(
                startpos_L, startpos_R, targetpos_L, targetpos_R,
                step_span=step_span, copy_counts=copy_counts, edge_rule="general",
                terrain_stones=terrain_stones,
            )
            v = G["v"]; edge = G["edge"]; mode_list = G["mode_list"]
            vertex_number = G["vertex_number"]; n_edge = G["n_edge"]
            toc(t_graph, f"Making graph ({jp.name})")

            # ✨ 회전 후처리 추가
            v = rotate_vertices_with_yaw(v, terrain_stones, copy_counts)

            # ----- 파라미터(시작/목표 5D) 준비 -----
            params_pose = prepare_problem_params(
                startpos_L, startpos_R, targetpos_L, targetpos_R,
                startyaw_L, startyaw_R, targetyaw_L, targetyaw_R
            )
            start_L = params_pose["start_L"]; start_R = params_pose["start_R"]
            target_L = params_pose["target_L"]; target_R = params_pose["target_R"]
            print(start_L, start_R, target_L, target_R)
            # ----- centroid / base_param 계산 -----
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

            # ----- 모델 생성 (terrain마다 새로) -----
            t_parameter = tic()
            M, V, IDX, PAR, build_time = build_gcs_model(v, edge, step_span, alpha, mode_list, x_ref, y_ref)
            toc(t_parameter, f"making optimization problem ({jp.name})")

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
            toc(t_solving, f"solving time(relaxed path) ({jp.name})")
            print(f"Cost (relaxed path): {obj1:.3f}")

            # y (relaxed)
            ex_relax = extract_solution(V, IDX)
            y_relaxed_val = get_relaxed_y(V)

            # ==================== 경로 선택 + 2차 ====================
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
                time_budget_sec=float(args.trial_budget_sec),
                temp=getattr(args, "temp", 1.0),
                greedy_first=True,
            )

            relaxed_path = best["path"]
            ex_after = best["ex"]

            # ----- 후처리 / 결과 추출 -----
            t_post = tic()
            proc = postprocess_solution(
                ex_after, edge, relaxed_path,
                startpos_L, startpos_R, targetpos_L, targetpos_R,
                target_L, target_R,
            )
            toc(t_post, f"postprocess ({jp.name})")

            total_sec = time.perf_counter() - overall_t0
            print(f"Total ({jp.name}) running Time: {total_sec:.4f} seconds")
            print(f"Cost (fixed path): {best_obj:.3f}")

            # ✅ 마지막 성공 케이스만 plot하기 위해 저장
            last_plot_payload = {
                "file": jp.name,
                "terrain_stones": terrain_stones,
                "proc": proc,
                "startpos_L": startpos_L,
                "startpos_R": startpos_R,
            }

            # ✅ 결과 저장(요약)
            results.append({
                "file": jp.name,
                "ok": True,
                "obj_relaxed": float(obj1),
                "obj_fixed": float(best_obj),
                "time_sec": float(total_sec),
            })

        except Exception as e:
            print(f"[SKIP] {jp.name}: {e}")
            results.append({
                "file": jp.name,
                "ok": False,
                "error": str(e),
            })
            continue

    # ===== 전체 요약 =====
    print("\n================ Summary ================")
    ok_cnt = sum(1 for r in results if r.get("ok"))
    print(f"OK: {ok_cnt}/{len(results)}")
    for r in results:
        if r.get("ok"):
            print(f"[OK] {r['file']} | relaxed={r['obj_relaxed']:.3f} fixed={r['obj_fixed']:.3f} time={r['time_sec']:.3f}s")
        else:
            print(f"[FAIL] {r['file']} | {r.get('error')}")

    # ===== 마지막 한개는 늘 plot 하기  =====
    if last_plot_payload is None:
        raise RuntimeError("No successful terrain to plot (all failed).")
    fig1 = plt.figure(figsize=(8, 6))
    ax3d = fig1.add_subplot(111, projection='3d')

    plot_stones_3d(
        last_plot_payload["terrain_stones"],
        thickness=0.04, face_alpha=0.25, edgecolor='k',
        ax=ax3d
    )

    proc = last_plot_payload["proc"]

    plot_feet_3d_VAE(
        proc.L.x, proc.L.y, proc.L.z, proc.L.c, proc.L.s,
        proc.R.x, proc.R.y, proc.R.z, proc.R.c, proc.R.s,
        matlab_style=True,
        startpos_L=last_plot_payload["startpos_L"],
        startpos_R=last_plot_payload["startpos_R"],
        ax=ax3d,
        accumulate=True,
        show=False
    )
    plt.show()
    # (A) 이 파이썬 파일이 있는 폴더 기준으로 results 폴더 생성
    SCRIPT_DIR = Path(__file__).resolve().parent
    OUT_DIR = SCRIPT_DIR / "results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # (B) 파일명: 날짜시간 붙여서 덮어쓰기 방지 (원하면 고정 이름으로 바꿔도 됨)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"vae_batch_results_{stamp}.csv"

    # (C) 저장
    fieldnames = ["file", "ok", "obj_relaxed", "obj_fixed", "time_sec", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            if isinstance(row["ok"], bool):
                row["ok"] = int(row["ok"])
            w.writerow(row)

    print(f"[saved] CSV results -> {csv_path}")

if __name__ == "__main__":
    main()



