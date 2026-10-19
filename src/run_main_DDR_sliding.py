# src/run_main_DDR_sliding.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.terrain_ddr import Terrain_GCS_DDR
from src.graph.centroid import compute_centroids_from_vertices
from src.solver import (
    set_parameters, solve_relaxed, run_trials_select_best,
    extract_solution, get_relaxed_y
)
from src.model.gcs_model_ddr import build_gcs_model_ddr   # ← 경로 주의
from src.viz.viz_ddr import plot_stone_map, compute_foot_rects, draw_heading_arrow, upsert_highlight
from src.viz.kinematic import  prepare_3d_axes,draw_robot_from_ddr


def _ensure_ddr_layers(ax2d):
    """정적 레이어(배경/스톤맵) 1회만 생성 + 동적 핸들 컨테이너 준비."""
    if not hasattr(ax2d, "_ddr_initialized"):
        # 배경/스톤맵은 '한 번만' 그림
        from src.viz.viz_ddr import plot_stone_map
        plot_stone_map(
            ax2d,
            window_labels=['All','All','All'],  # 아무거나 초기값
            last_key=None,
            bg_image='./pump_dancepad.png',
        )
        ax2d.set_aspect('equal', adjustable='box')
        ax2d.set_xlim(-0.4, 0.4); ax2d.set_ylim(-0.4, 0.4)
        ax2d.set_xticks([]); ax2d.set_yticks([])
        ax2d._ddr_initialized = True

    if not hasattr(ax2d, "_ddr_dynamic"):
        ax2d._ddr_dynamic = dict(patches=[], lines=[], texts=[], artists=[])

def _clear_ddr_dynamic(ax2d):
    """동적 요소만 제거 (배경/스톤맵/이미지 유지)."""
    d = getattr(ax2d, "_ddr_dynamic", None)
    if not d:
        return
    for h in d["patches"] + d["lines"] + d["texts"] + d["artists"]:
        try:
            h.remove()
        except Exception:
            pass
    d["patches"].clear(); d["lines"].clear(); d["texts"].clear(); d["artists"].clear()


def _pose_from_prev(prev, default_pos, default_yaw):
    """
    prev가 2D(x,y)면 위치만, 4D(x,y,cos,sin) 또는 5D(x,y,z,cos,sin)면
    위치+yaw까지 복원해서 반환.
    """
    import numpy as np
    if prev is None:
        return default_pos, default_yaw
    arr = np.asarray(prev, float).reshape(-1)
    pos = arr[:2]
    if arr.size >= 5:
        c, s = arr[3], arr[4]
        yaw = np.arctan2(s, c)
    elif arr.size >= 4:
        c, s = arr[2], arr[3]
        yaw = np.arctan2(s, c)
    else:
        yaw = default_yaw
    return pos, yaw


def _pack_pose5(pos2, yaw, z=0.0):
    """[x, y, z, cos, sin]"""
    return np.array([pos2[0], pos2[1], float(z), np.cos(yaw), np.sin(yaw)], float)

def _debug_dump_graph(v, edge, mode_list):
    import numpy as np
    edge = np.asarray(edge, dtype=int)
    Vn = len(v); E = edge.shape[0]
    print(f"\n=== DUMP: vertices (V={Vn}) ===")
    for i, vert in enumerate(v, start=1):
        cx = float(np.mean(vert[0, :])); cy = float(np.mean(vert[1, :]))
        mode = mode_list[i - 1] if i - 1 < len(mode_list) else "?"
        print(f"[v {i:>3}] mode={mode} center=({cx:.3f}, {cy:.3f})")
        print(vert[:2, :])
    print(f"\n=== DUMP: edges (E={E}) ===")
    for i, (u, w) in enumerate(edge.astype(int), start=1):
        mode_to = mode_list[w - 1] if w - 1 < len(mode_list) else "?"
        print(f"[e {i:>3}] {u} -> {w}  (to mode={mode_to})")
    print("=================================\n")

def _pairs_from_path(ex, edge_1based, path_1based):
    """경로대로 next-state들(L_next, R_next)을 (L,R) 튜플 리스트로 반환. 각 벡터는 5D."""
    import numpy as np
    edge_1based = np.asarray(edge_1based, dtype=int)
    y = ex["y"].reshape(-1)
    zp_L = ex["zp_L"]  # (E,5)
    zp_R = ex["zp_R"]  # (E,5)
    pairs = []
    for i in range(len(path_1based) - 1):
        u = int(path_1based[i]); v = int(path_1based[i + 1])
        idx = np.where((edge_1based[:, 0] == u) & (edge_1based[:, 1] == v))[0]
        if idx.size == 0:
            continue
        e = int(idx[0])
        yy = max(1e-9, float(y[e]))
        L_next = zp_L[e, :] / yy  # 5D
        R_next = zp_R[e, :] / yy  # 5D
        pairs.append((L_next, R_next))
    return pairs


def run_main_DDR_sliding(label_window, prev_center_L=None, prev_center_R=None, prev_mover=None, ax2d=None, ax3d=None):
    # ===== 1) 시작 포즈 (5D: z=0 포함). prev가 있으면 시작 좌표로 덮어쓰기 =====
    startpos_L = np.array([-0.16, 0.0], float)
    startpos_R = np.array([0.16, 0.0], float)
    startyaw_L = startyaw_R = 9 * np.pi / 18
    startpos_L, startyaw_L = _pose_from_prev(prev_center_L, startpos_L, startyaw_L)
    startpos_R, startyaw_R = _pose_from_prev(prev_center_R, startpos_R, startyaw_R)

    targetpos_L = np.array([-0.16, 0.0], float)
    targetpos_R = np.array([0.16, 0.0], float)
    targetyaw_L = 9 * np.pi / 18
    targetyaw_R = 9 * np.pi / 18

    # targetpos_L = np.array([0.45 - 0.113, 0.45], float); targetyaw_L = 9 * np.pi / 18
    # targetpos_R = np.array([0.45 + 0.113, 0.45], float); targetyaw_R = 9 * np.pi / 18

    start_L  = _pack_pose5(startpos_L,  startyaw_L, z=0.0)  # ✅ 5D
    start_R  = _pack_pose5(startpos_R,  startyaw_R, z=0.0)  # ✅ 5D
    target_L = _pack_pose5(targetpos_L, targetyaw_L, z=0.0) # ✅ 5D
    target_R = _pack_pose5(targetpos_R, targetyaw_R, z=0.0) # ✅ 5D
    alpha = 9 * np.pi / 18

    # ===== 2) Terrain (시작 Do='All') =====
    First, Second, Third = label_window
    start = First
    goal = 'C'
    T = Terrain_GCS_DDR(start, First, Second, Third, goal)
    v = T.get_vertices()
    edge = np.asarray(T.get_edges(), dtype=int)
    mode_list = T.get_mode_list()
    _debug_dump_graph(v, edge, mode_list)

    v_names = [s.label for s in T.stepping_stones]
    # ===== 3) centroid =====
    centroid_mat, _ = compute_centroids_from_vertices(v)  # (2,V)

    # ===== 4) 모델 생성 & 1차 릴랙스 =====
    M, V, IDX, PAR, _ = build_gcs_model_ddr(
        v, edge, alpha, mode_list,
        start_L, start_R, target_L, target_R,
        v_names=v_names,
    )

    # 파라미터 세팅 (항상 y_relax, M_scalar 키를 넣어 KeyError 예방)
    params_relaxed = {
        "geometry": centroid_mat.reshape(-1),
        "start_L": start_L, "start_R": start_R,
        "target_L": target_L, "target_R": target_R,
        "M_scalar": 1.0,
        "y_relax": np.zeros(IDX["n_edge"]),
    }
    set_parameters(M, V, params_relaxed)

    # 릴랙스드 풀이
    sol_relaxed, _ = solve_relaxed(M, V, M_large=1e3, log_level=0)

    # ===== 5) 경로 선택 + 2차 고정 경로 풀이 =====
    y_relaxed_val = get_relaxed_y(V)
    best_obj, best = run_trials_select_best(
        M, V, IDX,
        centroid_mat=centroid_mat,
        y_relaxed_val=y_relaxed_val,
        edge=edge,
        vertex_number=IDX["vertex_number"],
        n_edge=IDX["n_edge"],
        M_small=1e-6,
        start_L=start_L, start_R=start_R,
        target_L=target_L, target_R=target_R,
        trials=1, temp=1.0, greedy_first=True
    )
    relaxed_path = best["path"]
    ex_after = best["ex"]

    #for debugging
    E = 17  # edge 개수
    def _get_17x5(key):
        if isinstance(ex_after, dict) and key in ex_after:
            arr = np.asarray(ex_after[key], dtype=float)
        else:
            arr = np.asarray(V[key].level(), dtype=float)
        return arr.reshape(E, 5)

    zL = _get_17x5("z_L")
    zpL = _get_17x5("zp_L")
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("z_L (17x5):\n", zL)
    print("\nzp_L (17x5):\n", zpL)

    # ===== 6) 시각화 =====
    # --- 2D 축 준비 ---
    if ax2d is None:
        ax2d = plt.gca()
    _ensure_ddr_layers(ax2d)   # 배경/스톤맵 1회만
    _clear_ddr_dynamic(ax2d)   # 지난 프레임의 동적 요소들만 제거
    d = ax2d._ddr_dynamic      # 새로 그린 핸들을 여기에 모아둔다

    # ---- 현재(두 발) — 진하게 + 테두리 + 화살표 ----
    polyL0 = compute_foot_rects([start_L[0]], [start_L[1]], [start_L[3]], [start_L[4]])[0]
    polyR0 = compute_foot_rects([start_R[0]], [start_R[1]], [start_R[3]], [start_R[4]])[0]
    hL = ax2d.fill(polyL0[:, 0], polyL0[:, 1], color='r', alpha=0.95, edgecolor='k', linewidth=2, zorder=5)
    hR = ax2d.fill(polyR0[:, 0], polyR0[:, 1], color='b', alpha=0.95, edgecolor='k', linewidth=2, zorder=5)
    d["patches"].extend(hL + hR)

    # 화살표(annotate 핸들을 직접 받아서 관리)
    def _annot(ax, x, y, c, s, length=0.12, color='k', alpha=1.0, zorder=7, lw=1.6, mscale=18):
        n = float(np.hypot(c, s))
        if n < 1e-9:
            return None
        ux, uy = c/n, s/n
        return ax.annotate('', xy=(x + length*ux, y + length*uy), xytext=(x, y),
                           arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                           mutation_scale=mscale, alpha=alpha),
                           zorder=zorder)

    aL = _annot(ax2d, start_L[0], start_L[1], start_L[3], start_L[4], zorder=7)
    aR = _annot(ax2d, start_R[0], start_R[1], start_R[3], start_R[4], zorder=7)
    if aL: d["artists"].append(aL)
    if aR: d["artists"].append(aR)

    # --- 직전 프레임에서 움직였던 발 표식 ---
    if prev_mover == "L":
        dot = ax2d.plot([float(start_L[0])], [float(start_L[1])], 'ko', markersize=6, zorder=8)[0]
        d["lines"].append(dot)
    elif prev_mover == "R":
        dot = ax2d.plot([float(start_R[0])], [float(start_R[1])], 'ko', markersize=6, zorder=8)[0]
        d["lines"].append(dot)

    # ---- 미래 — 경로 기반 next states (각 5D) ----
    pairs = _pairs_from_path(ex_after, edge, relaxed_path)  # [(L_next5D, R_next5D), ...]

    def pose_str5(p):
        x, y, z, c, s = map(float, p)
        yaw = np.degrees(np.arctan2(s, c))
        return f"x={x:.3f}, y={y:.3f}, cos={c:.3f}, sin={s:.3f}, yaw={yaw:.2f}°"

    print(f"[now] L: {pose_str5(start_L)} | R: {pose_str5(start_R)}")

    def _fade(n, a0=0.45, a1=0.15):
        if n <= 0: return []
        t = np.linspace(0, 1, n)
        return list(a0 + (a1 - a0) * t)

    # 마지막 to-vertex가 Do이면 해당 스텝은 시각화 제외
    last_to_mode = mode_list[int(relaxed_path[-1]) - 1] if len(relaxed_path) > 1 else None
    pairs_to_draw = pairs[:-1] if last_to_mode == "Do" else pairs

    alphas = _fade(len(pairs_to_draw), 0.35, 0.10)
    for (Lk, Rk), a in zip(pairs_to_draw, alphas):
        pL = compute_foot_rects([Lk[0]], [Lk[1]], [Lk[3]], [Lk[4]])[0]
        pR = compute_foot_rects([Rk[0]], [Rk[1]], [Rk[3]], [Rk[4]])[0]
        hL = ax2d.fill(pL[:, 0], pL[:, 1], color='r', alpha=a, edgecolor='none', zorder=4)
        hR = ax2d.fill(pR[:, 0], pR[:, 1], color='b', alpha=a, edgecolor='none', zorder=4)
        d["patches"].extend(hL + hR)
        a1 = _annot(ax2d, Lk[0], Lk[1], Lk[3], Lk[4], alpha=a, zorder=5)
        a2 = _annot(ax2d, Rk[0], Rk[1], Rk[3], Rk[4], alpha=a, zorder=5)
        if a1: d["artists"].append(a1)
        if a2: d["artists"].append(a2)

    # ---- Step labels: 1,2,3… (L/R는 색으로 구분) ----
    k = 1
    for i in range(len(relaxed_path) - 1):
        v_to = int(relaxed_path[i + 1])  # 1-based
        mode_to = mode_list[v_to - 1]    # "SL" | "SR" | "Do"
        if mode_to not in ("SL", "SR"):
            continue
        Ln, Rn = pairs[i]
        pos = (float(Ln[0]), float(Ln[1])) if mode_to == "SL" else (float(Rn[0]), float(Rn[1]))
        txt = ax2d.text(pos[0], pos[1] - 0.08, f"{k}", ha='center', va='bottom',
                        fontsize=9, fontweight=('bold' if k == 1 else 'normal'),
                        color='k', zorder=6)
        d["texts"].append(txt)
        k += 1

    current_key = label_window[0]
    upsert_highlight(ax2d, current_key)

    if ax3d is None:
        fig = plt.gcf()
        ax3d = fig.add_subplot(122, projection='3d')  # prepare_3d_axes 없이도 OK

    # 프레임마다 싹 초기화
    ax3d.cla()
    ax3d.view_init(elev=35, azim=-40)

    # 축 범위/비율
    ax3d.set_xlim(-0.4, 0.4)
    ax3d.set_ylim(-0.4, 0.4)
    ax3d.set_zlim(-0.05, 1.0)
    ax3d.set_box_aspect([1, 1, 1])

    # 라벨/눈금/그리드 끄고 축선만 남기기
    ax3d.set_xlabel('');
    ax3d.set_ylabel('');
    ax3d.set_zlabel('')
    ax3d.set_xticks([]);
    ax3d.set_yticks([]);
    ax3d.set_zticks([])
    ax3d.grid(False)
    try:
        for axis in (ax3d.w_xaxis, ax3d.w_yaxis, ax3d.w_zaxis):
            axis.set_pane_color((1, 1, 1, 0))  # 배경판 투명
            axis.line.set_color('k')  # 축선 색/두께
            axis.line.set_linewidth(1.2)
        ax3d.patch.set_alpha(0.0)
    except Exception:
        pass

    # (옵션) z=-0.01 평면에 stone_map 사각형들
    STONE_MAP = {
        'DL': dict(center=(-0.21, -0.17), width=0.21, height=0.25),
        'DR': dict(center=(0.21, -0.17), width=0.21, height=0.25),
        'C': dict(center=(0.0, 0.0), width=0.21, height=0.21),
        'UL': dict(center=(-0.21, 0.17), width=0.21, height=0.25),
        'UR': dict(center=(0.21, 0.17), width=0.21, height=0.25),
    }
    COLOR_MAP = {
        'UL': ((1.0, 0.0, 0.0), 0.25),
        'UR': ((1.0, 0.0, 0.0), 0.25),
        'C': ((1.0, 1.0, 0.0), 0.25),
        'DL': ((0.0, 0.0, 1.0), 0.25),
        'DR': ((0.0, 0.0, 1.0), 0.25),
        '_default': ((0.0, 0.0, 0.0), 0.15),
    }
    polys, facecolors = [], []
    for k, st in STONE_MAP.items():
        cx, cy = st['center'];
        w, h = st['width'], st['height']
        x0, x1 = cx - w / 2, cx + w / 2
        y0, y1 = cy - h / 2, cy + h / 2
        z = -0.01
        polys.append([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)])
        color, alpha = COLOR_MAP.get(k, COLOR_MAP['_default'])
        facecolors.append((*color, alpha))
    coll = Poly3DCollection(polys, facecolors=facecolors,
                            edgecolors='k', linewidths=1.0, zorder=-1_000_000)
    ax3d.add_collection3d(coll)

    # 로봇 그리기
    draw_robot_from_ddr(
        ax3d,
        xL=float(start_L[0]), yL=float(start_L[1]), zL=float(start_L[2]),
        cL=float(start_L[3]), sL=float(start_L[4]),
        xR=float(start_R[0]), yR=float(start_R[1]), zR=float(start_R[2]),
        cR=float(start_R[3]), sR=float(start_R[4]),
        roll=0.0, pitch=0.0,
        z_pelvis=0.80, hip_offset=0.15,
        line_width=2, line_color='k'
    )

    plt.draw()
    plt.pause(0.001)

    # --- (NEW) 다음 프레임용 prev_mover 계산: step1 모드로 결정 ---
    mover_step1 = None
    if len(relaxed_path) > 1:
        mode1 = mode_list[int(relaxed_path[1]) - 1]  # "SL" / "SR" / "Do"
        if mode1 == "SL":
            mover_step1 = "L"
        elif mode1 == "SR":
            mover_step1 = "R"
    # -----------------------------------------------------------------
    # ---- 다음 호출용 prev 갱신 (5D 그대로) ----
    if len(pairs) >= 1:
        nextL, nextR = pairs[0]
        new_prev_L = nextL.copy()
        new_prev_R = nextR.copy()
    else:
        new_prev_L = start_L.copy()
        new_prev_R = start_R.copy()

    return new_prev_L, new_prev_R, mover_step1
