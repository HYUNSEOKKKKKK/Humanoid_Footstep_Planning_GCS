# main_VAE_optapi.py
from __future__ import annotations

import time
import math
import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =========================
# Project imports (OptAPI)
# =========================
from src.terrain_preset import load_terrain_yaw
from src.utils import tic, toc, expand_shorthand, postprocess_solution
from src.graph import build_graph, compute_centroids_from_vertices
from src.model import (
    prepare_problem_params,
    pack_relaxed_params_for_model,
    build_gcs_model_VAE_optapi
)
from src.solver import (
    set_parameters_optapi,
    solve_relaxed_optapi,
    run_trials_select_best_optapi,
    run_trials_select_best_optapi_timelimit
)
from src.viz import plot_feet_3d_VAE


# ============================================================
# Plot helpers (same as your Fusion main style)
# ============================================================

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


def plot_stones_3d(stones: List[Dict[str, Any]],
                   *,
                   thickness: float = 0.04,
                   face_alpha: float = 0.35,
                   edgecolor: str = 'k',
                   ax=None,
                   colorbar: bool = True,
                   view_top: bool = False,
                   title: str = " GCS footstep planner "):
    """
    너가 쓰던 Fusion main의 plot_stones_3d를 그대로 가져옴.
    (3D 또는 top-down 2D 둘 다 지원)
    """
    created_ax = False

    if ax is None:
        if view_top:
            fig, ax = plt.subplots(figsize=(8, 6))
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

    # =========================
    # (A) top-down 2D
    # =========================
    if view_top:
        if ax is not None:
            pos = ax.get_position()
        else:
            pos = None

        ax2d = getattr(fig, "_gcs_top2d_ax", None)
        if ax2d is None or ax2d not in fig.axes:
            if pos is not None:
                ax2d = fig.add_axes(pos)
            else:
                ax2d = fig.add_subplot(111)
            fig._gcs_top2d_ax = ax2d
        else:
            if pos is not None:
                ax2d.set_position(pos)

        if ax is not None and ax is not ax2d:
            ax.set_visible(False)
        ax2d.set_visible(True)

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

        if all_xy:
            P = np.vstack(all_xy)
            rng = np.ptp(P, axis=0)
            pad = 0.1 * max(float(rng[0]), float(rng[1]), 1e-6)
            ax2d.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
            ax2d.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)

        ax2d.set_aspect('equal', adjustable='box')
        ax2d.grid(False)

        if n >= 3 and use_colorbar:
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            cbar = fig.colorbar(mappable, ax=ax2d, fraction=0.046, pad=0.04)
            cbar.set_label('Height (z)', rotation=90)

        return artists, ax2d

    # =========================
    # (B) 3D rendering
    # =========================
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
    ccx = 0.5*(xr[0]+xr[1]); ccy = 0.5*(yr[0]+yr[1]); ccz = 0.5*(zr[0]+zr[1])
    ax.set_xlim(ccx-maxr/2, ccx+maxr/2)
    ax.set_ylim(ccy-maxr/2, ccy+maxr/2)
    ax.set_zlim(ccz-maxr/3, ccz+maxr/3)

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


def heading_deg_x_forward(c0, c1):
    dx = float(c1[0]) - float(c0[0])
    dy = float(c1[1]) - float(c0[1])
    if abs(dx) + abs(dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


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
        "foot_yaw_deg": {"start_L": 0.0, "start_R": 0.0, "goal_L": 0.0, "goal_R": 0.0},
    }


def rotate_vertices_with_yaw(v_list, terrain_stones, copy_counts):
    """
    v_list: List of (3,4) vertex arrays from build_graph
    terrain_stones: [{'center':..., 'width':..., 'height':..., 'yaw_deg':...}, ...]
    copy_counts: 각 stone 복제 개수
    """
    rotated_v = []
    stone_idx_expanded = [i for i, c in enumerate(copy_counts) for _ in range(c)]

    for vtx, stone_idx in zip(v_list, stone_idx_expanded):
        stone = terrain_stones[stone_idx]
        yaw_deg = float(stone.get("yaw_deg", 0.0))
        if abs(yaw_deg) < 1e-9:
            rotated_v.append(vtx)
            continue

        cx, cy, cz = stone["center"]
        th = math.radians(yaw_deg)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]])

        v_shifted = vtx.copy()
        v_shifted[0, :] -= cx
        v_shifted[1, :] -= cy
        v_rot = R @ v_shifted
        v_rot[0, :] += cx
        v_rot[1, :] += cy
        rotated_v.append(v_rot)

    return rotated_v


# ============================================================
# Main (OptAPI batch runner for VAE terrains)
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GCS Footstep Planning (MOSEK Optimizer API) - VAE batch")

    # 하이퍼파라미터
    parser.add_argument("--copy_counts", type=int, nargs="+", default=None, help="copies per stone; default: 5 at endpoints, 4 in between")
    parser.add_argument("--step_span", type=float, default=0.7, help="max step span (scalar)")
    parser.add_argument("--alpha_deg", type=float, default=30.0, help="yaw rotation input in degrees")
    parser.add_argument("--M_large", type=float, default=1e3, help="Big-M for relaxed solve")
    parser.add_argument("--M_small", type=float, default=1e-6, help="Small-M for equality-like enforcement")

    # trial loop
    parser.add_argument(
        "--trials", type=int, default=100, help="number of fixed-path trials")
    parser.add_argument("--trial_budget_sec", type=float, default=5.0, help="time budget (sec) for trial loop; <=0 disables")
    parser.add_argument("--temp", type=float, default=1.0, help="sampling temperature for path picking")

    # shorthand
    parser.add_argument("--stance-width", type=float, default=0.25, help="stance width for shorthand experiments")

    # VAE json dir
    parser.add_argument("--terrain_dir", type=str, default="data/terrains/alpha/0.0",
                        help="directory containing generated terrain json files")
    parser.add_argument("--terrain_key", type=str, default="terrain",
                        help="top-level key name inside each json")

    parser.add_argument("--terrain", type=Path, help="plan one terrain JSON instead of scanning a directory")
    parser.add_argument("--max-terrains", type=int, default=1, help="number of inputs; 0 selects the whole directory")
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gcs"))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if not args.output_dir.is_absolute():
        args.output_dir = root / args.output_dir
    if args.output_dir.resolve().is_relative_to((root / "data").resolve()):
        parser.error("select an output directory outside preserved data/")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_terrains < 0:
        parser.error("--max-terrains must be nonnegative")
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.no_show:
        plt.switch_backend("Agg")

    terrain_dir = Path(args.terrain_dir)
    if not terrain_dir.is_absolute():
        terrain_dir = root / terrain_dir
    if args.terrain is not None:
        terrain = args.terrain if args.terrain.is_absolute() else root / args.terrain
        if not terrain.is_file():
            parser.error(f"terrain not found: {terrain}")
        json_files = [terrain]
    else:
        json_files = sorted(terrain_dir.glob("*.json"))
        if args.max_terrains:
            json_files = json_files[:args.max_terrains]
    if len(json_files) == 0:
        raise FileNotFoundError(f"No .json files found in: {terrain_dir}")

    results = []
    last_plot_payload = None

    for k, jp in enumerate(json_files):
        print(f"\n=== [{k+1}/{len(json_files)}] Loaded: {jp.name} | key='{args.terrain_key}' ===")

        try:
            # 0) Load terrain json
            T = load_terrain_yaw(name=args.terrain_key, path=jp)
            terrain_stones = T["stones"]
            print(f"    #stones={len(terrain_stones)}")

            # 1) entry 만들고 shorthand 확장
            entry = make_entry_from_stones(terrain_stones, stance_width=args.stance_width)
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

            step_span = float(args.step_span)
            alpha = float(args.alpha_deg) * np.pi / 180.0

            copy_counts = T.get("copy_counts")
            if copy_counts is None:
                copy_counts = list(args.copy_counts) if args.copy_counts is not None else [5] + [4] * (len(terrain_stones) - 2) + [5]
            if len(copy_counts) != len(terrain_stones):
                raise ValueError(f"[{jp.name}] copy_counts({len(copy_counts)}) must match #stones({len(terrain_stones)}).")

            M_large = float(args.M_large)
            M_small = float(args.M_small)

            overall_t0 = tic()
            budget_total = float(args.trial_budget_sec)  # 의미: JSON 1개당 총 예산
            t_json0 = time.perf_counter()
            deadline = t_json0 + budget_total

            # 2) Graph build
            t_graph = tic()
            G = build_graph(
                startpos_L, startpos_R, targetpos_L, targetpos_R,
                step_span=step_span, copy_counts=copy_counts, edge_rule="general",
                terrain_stones=terrain_stones,
            )
            v = G["v"]; edge = G["edge"]; mode_list = G["mode_list"]
            vertex_number = G["vertex_number"]; n_edge = G["n_edge"]
            toc(t_graph, f"Making graph ({jp.name})")

            # 3) yaw 회전 후처리 (vertex polygon rotate)
            v = rotate_vertices_with_yaw(v, terrain_stones, copy_counts)

            # 4) Pose params (5D)
            params_pose = prepare_problem_params(
                startpos_L, startpos_R, targetpos_L, targetpos_R,
                startyaw_L, startyaw_R, targetyaw_L, targetyaw_R
            )
            start_L = params_pose["start_L"]; start_R = params_pose["start_R"]
            target_L = params_pose["target_L"]; target_R = params_pose["target_R"]

            # 5) centroid
            centroid_mat, _base_param = compute_centroids_from_vertices(v)

            # 6) global ref axis (y_ref = R->L)
            startL_val = np.asarray(startpos_L, float).reshape(-1)
            startR_val = np.asarray(startpos_R, float).reshape(-1)
            pL, pR = startL_val[:2], startR_val[:2]
            vRL = pL - pR
            if np.linalg.norm(vRL) < 1e-12:
                y_ref = np.array([0.0, 1.0])
            else:
                y_ref = vRL / np.linalg.norm(vRL)
            x_ref = np.array([y_ref[1], -y_ref[0]])

            # 7) Build model (OptAPI)
            t_build = tic()
            M, V, IDX, PAR, build_time = build_gcs_model_VAE_optapi(v, edge, step_span, alpha, mode_list, x_ref, y_ref)
            toc(t_build, f"making optimization problem (OptAPI) ({jp.name})")

            # ====================
            # (1) Relaxed solve
            # ====================
            t_relaxed = tic()
            params_relaxed = pack_relaxed_params_for_model(
                centroid_mat=centroid_mat,
                n_edge=IDX["n_edge"],
                M_large=M_large,
                start_L=start_L, start_R=start_R,
                target_L=target_L, target_R=target_R,
            )
            set_parameters_optapi(M, V, params_relaxed)
            sol_relaxed, obj1 = solve_relaxed_optapi(M, V, log_level=0)
            toc(t_relaxed, f"solving time(relaxed path) OptAPI ({jp.name})")
            print(f"Cost (relaxed path): {float(obj1):.6f}")

            y_relaxed_val = np.asarray(sol_relaxed["y"], float).ravel()

            # ====================
            # (2) Fixed path (trials)  -- per-JSON deadline enforced
            # ====================
            t_fixed = tic()

            # ✅ JSON당 전체 budget 중 "남은 시간"만 fixed에 할당
            remain = deadline - time.perf_counter()
            if remain <= 0:
                raise TimeoutError(f"[{jp.name}] time budget exceeded before fixed-path (budget={budget_total:.3f}s).")

            try:
                best_obj, best = run_trials_select_best_optapi_timelimit(
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
                    time_budget_sec=float(remain),  # ✅ 핵심: remain
                    temp=float(args.temp),
                    greedy_first=True,
                    early_stop=True,  # best 찾고 싶으면 False, "첫 feasible만"이면 True
                )
            except RuntimeError as e:
                # ✅ 해가 없으면 여기서 끝. fallback으로 더 돌지 않음(= 시간 지킴)
                raise RuntimeError(f"[{jp.name}] fixed-path failed within remaining budget={remain:.3f}s: {e}")

            toc(t_fixed, f"solving time(fixed path) OptAPI ({jp.name})")

            relaxed_path = best["path"]
            ex_after = best["ex"]

            # ====================
            # Postprocess
            # ====================
            t_post = tic()
            proc = postprocess_solution(
                ex_after, edge, relaxed_path,
                startpos_L, startpos_R, targetpos_L, targetpos_R,
                target_L, target_R,
            )
            toc(t_post, f"postprocess ({jp.name})")

            # ----- footstep sequence 출력 -----
            first_is_left = bool(np.asarray(best["y_relax"], float).ravel()[0] > 0.5)
            Lx, Ly, Lz, Lc, Ls = proc.L.x, proc.L.y, proc.L.z, proc.L.c, proc.L.s
            Rx, Ry, Rz, Rc, Rs = proc.R.x, proc.R.y, proc.R.z, proc.R.c, proc.R.s
            NL, NR = len(Lx), len(Rx)
            iL = iR = 0
            turn_left = first_is_left
            footsteps = []
            while iL < NL or iR < NR:
                if turn_left:
                    if iL < NL:
                        yaw_deg = float(np.degrees(np.arctan2(Ls[iL], Lc[iL])))
                        footsteps.append(('L', float(Lx[iL]), float(Ly[iL]), float(Lz[iL]), yaw_deg))
                        iL += 1
                    elif iR < NR:
                        yaw_deg = float(np.degrees(np.arctan2(Rs[iR], Rc[iR])))
                        footsteps.append(('R', float(Rx[iR]), float(Ry[iR]), float(Rz[iR]), yaw_deg))
                        iR += 1
                else:
                    if iR < NR:
                        yaw_deg = float(np.degrees(np.arctan2(Rs[iR], Rc[iR])))
                        footsteps.append(('R', float(Rx[iR]), float(Ry[iR]), float(Rz[iR]), yaw_deg))
                        iR += 1
                    elif iL < NL:
                        yaw_deg = float(np.degrees(np.arctan2(Ls[iL], Lc[iL])))
                        footsteps.append(('L', float(Lx[iL]), float(Ly[iL]), float(Lz[iL]), yaw_deg))
                        iL += 1
                turn_left = not turn_left
            print(f"DEFAULT_FOOTSTEPS = [  # {jp.name}")
            for side, x, y, z, yaw in footsteps:
                print(f"    ('{side}', {x:7.3f}, {y:7.3f}, {z:7.3f}, {yaw:7.2f}),")
            print("]")

            total_sec = time.perf_counter() - overall_t0
            print(f"Total ({jp.name}) running Time: {total_sec:.4f} seconds")
            print(f"Cost (fixed path): {float(best_obj):.6f}")

            last_plot_payload = {
                "file": jp.name,
                "terrain_stones": terrain_stones,
                "proc": proc,
                "startpos_L": startpos_L,
                "startpos_R": startpos_R,
                "footsteps": footsteps,
                "source_path": str(jp),
            }

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

    # =========================
    # Summary
    # =========================
    print("\n================ Summary ================")
    ok_cnt = sum(1 for r in results if r.get("ok"))
    print(f"OK: {ok_cnt}/{len(results)}")
    for r in results:
        if r.get("ok"):
            print(f"[OK] {r['file']} | relaxed={r['obj_relaxed']:.3f} fixed={r['obj_fixed']:.3f} time={r['time_sec']:.3f}s")
        else:
            print(f"[FAIL] {r['file']} | {r.get('error')}")

    # =========================
    # Plot last success
    # =========================
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
    fig1.savefig(args.output_dir / "plan.png", dpi=160, bbox_inches="tight")
    import json
    plan = {"terrain": {"stones": last_plot_payload["terrain_stones"]},
            "source_terrain": last_plot_payload["source_path"],
            "units": {"position": "m", "yaw": "deg"},
            "footsteps": [{"foot": side, "x": x, "y": y, "z": z, "yaw_deg": yaw}
                          for side, x, y, z, yaw in last_plot_payload["footsteps"]]}
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if not args.no_show:
        plt.show()
    plt.close(fig1)

    # =========================
    # Save CSV
    # =========================
    SCRIPT_DIR = Path(__file__).resolve().parent
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"vae_batch_results_optapi_{stamp}.csv"

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
