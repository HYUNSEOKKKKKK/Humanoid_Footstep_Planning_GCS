
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from src.graph.WFCtoGCS import build_terrain_from_WFC_outputs
from src.graph.WFCtoGCS_adapter import build_label_order_tree_from_start, build_G_GCS_vertices_strict_with_order, build_G_GCS_edges_from_Gk_directed
# 시각화/그래프용
import re
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import matplotlib.patches as patches
# === 3D: Meshcat 버전 ===
import meshcat
import meshcat.geometry as mcg
from collections import deque
from typing import Dict, List, Set, Hashable
import meshcat.transformations as tf
from typing import Dict, Any, List, Tuple,Set, Optional
from collections import deque, defaultdict

from typing import List, Sequence, Optional, Tuple
from typing import Dict, Any
import numpy as np, random, math
import math
import numpy as np

def _unit(v):
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-12 else v

def _angle_to_cardinal(theta_rad: float) -> str:
    # θ: x+축=0°, CCW(+). 4방위로 양자화.
    deg = (math.degrees(theta_rad) + 360.0) % 360.0
    if 45 <= deg < 135:   return "N"
    if 135 <= deg < 225:  return "W"
    if 225 <= deg < 315:  return "S"
    return "E"

def _get_group_centroids_y_up(node_info, labels, tile_size, H):
    centroids = []
    for nid in labels:
        info = node_info[nid]
        if "bbox" in info:
            x0,y0,x1,y1 = info["bbox"]
            cx, cy = 0.5*(x0+x1), 0.5*(y0+y1)
        elif "cells" in info:
            xs, ys = [], []
            for (r,c) in info["cells"]:
                xs.append((c+0.5)*tile_size)
                ys.append((r+0.5)*tile_size)
            cx = float(np.mean(xs)); cy = float(np.mean(ys))
        else:
            cx, cy = 0.0, 0.0
        # y-up 변환
        cy = H*tile_size - cy
        centroids.append((cx, cy))
    return np.asarray(centroids, dtype=float)

def _compute_group_flow_cardinals(E_groups, group_centroids, start_g, goal_g, n_groups):
    # E_groups: (E,2) 배열 (u->v)
    preds = [[] for _ in range(n_groups)]
    succs = [[] for _ in range(n_groups)]
    for (u,v) in E_groups:
        preds[v].append(u)
        succs[u].append(v)

    flow_card = [None]*n_groups
    for g in range(n_groups):
        if g in (start_g, goal_g):
            flow_card[g] = None
            continue

        gc = group_centroids[g]
        Ps = preds[g]; Ss = succs[g]

        if Ps and Ss:
            pm = group_centroids[Ps].mean(axis=0)
            sm = group_centroids[Ss].mean(axis=0)
            vec = sm - pm
        elif Ss:
            sm = group_centroids[Ss].mean(axis=0)
            vec = sm - gc
        elif Ps:
            pm = group_centroids[Ps].mean(axis=0)
            vec = gc - pm
        else:
            flow_card[g] = None
            continue

        vec = _unit(vec)
        flow_card[g] = _angle_to_cardinal(math.atan2(vec[1], vec[0]))
    return flow_card

def _plot_wfc_pipeline(
    grid, G_nav, G_new, G_compact, Gk, node_info, s, t,
):
    fig, axs = plt.subplots(3, 2, figsize=(8, 12), constrained_layout=True)
    gs = axs[0, 0].get_gridspec()
    for ax in axs[0, :]:
        ax.remove()
    ax_top = fig.add_subplot(gs[0, :])

    # 1) 지형
    plot_wfc_terrain(grid, ax=ax_top)

    # 2) 네비 그래프 & 도달가능 서브그래프
    plot_nav_graph(grid, G_nav, path=None, start=s, goal=t, ax=axs[1, 0])
    plot_nav_graph(grid, G_new, path=None, start=s, goal=t, ax=axs[1, 1])

    # 3) 압축 그래프 & ST 커널
    plot_compact_graph(grid, G_compact, node_info, draw_node_labels=True, ax=axs[2, 0])
    overlay_start_goal_on_ax(axs[2, 0], s, t, use_tile_centers=False)
    plot_compact_graph(grid, Gk, node_info, draw_node_labels=True, ax=axs[2, 1])
    overlay_start_goal_on_ax(axs[2, 1], s, t, use_tile_centers=False)

    return fig, axs


def build_graph_wfc(
    *args,
    W: int = 5, H: int = 4,
    seed=None,
    grid=None,
    tile_size: float = 1.0,
    floor_z: dict | None = None,
    first_is_left: bool = True,
    span_stride: float | None = None,
    reorder_for_front_to_back: bool = True,
    stance_width: float | None = None,
    # ▼ 추가
    plot: bool = False,           # 그림 그릴지 여부
    return_fig: bool = False,     # fig/axs를 결과 dict에 담아줄지
    plt_show: bool = True,        # 즉시 plt.show() 호출할지
    **kwargs,
) -> Dict[str, Any]:
    # ---- 0) grid 생성 (없으면 WFC) ----
    if grid is None:
        tiles = [F0, F1, SX_L0R1, SX_L1R0, SY_U0D1, SY_U1D0]
        rules = make_default_rules()
        wfc = WFC(W, H, tiles, rules, seed=seed,
                  boundary="floors_only", cluster_bias={"floor1": 0.7})
        grid = wfc.collapse()

    # ---- 1) nav → reachable → compact → ST kernel ----
    G_nav = build_nav_graph_simple(grid)
    s, t, _ = sample_start_goal_with_path_random(
        grid, G_nav, floors_only=True, floor_names=("floor0","floor1"),
        rng=random.Random(42), max_trials=10000
    )
    G_new = build_reachable_subgraph(G_nav, s)
    node_info, cell_to_node = make_floor_rects_from_Gnew(grid, G_new, floor_names=("floor0","floor1"))
    G_compact = compact_graph_from_rects(grid, G_new, node_info, cell_to_node)
    start_id  = map_point_to_compact_node(cell_to_node, s)
    goal_id   = map_point_to_compact_node(cell_to_node, t)
    Gk = trim_leaves_iterative_undirected(
        prune_to_st_kernel_undirected(G_compact, start_id, goal_id),
        start_id, goal_id
    )

    # ---- 2) label order (start 앞, goal 뒤) + (옵션) 재정렬 ----
    order_raw   = build_label_order_tree_from_start(Gk, start=start_id, goal=goal_id)
    label_order = normalize_label_order(order_raw)
    if reorder_for_front_to_back:
        label_order = reorder_label_order_for_front_to_back(Gk, label_order, start=start_id, goal=goal_id)

    # ---- 3) 그룹 v / 그룹 엣지(앞→뒤) ----
    pack = build_G_GCS_vertices_strict_with_order(node_info, label_order, tile_size=tile_size, floor_z=floor_z)
    E_groups = build_G_GCS_edges_from_Gk_directed(Gk, label_order, start_label=start_id, goal_label=goal_id)

    # ---- 4) 복제 + 모드 (길이 기반) ----
    if span_stride is None:
        span_stride = kwargs.get("step_span", 1.0)
    v, mode_list, group_map, _ = expand_vertices_with_modes_from_edges(
        pack["v"], E_groups, start_id=start_id, goal_id=goal_id, labels=pack["labels"],
        first_is_left=first_is_left, span_stride=span_stride,
    )

    # ✅ 추가: v를 y-up(GCS)로 변환 (y' = H*tile_size - y)
    Ht = H * tile_size
    v = [np.array([vert[0], Ht - vert[1], vert[2]], dtype=float) for vert in v]


    # ---- 5) 확장 엣지 (소스 SL→목적 첫 SR / 소스 SR→목적 첫 SL, goal 내부 D 수렴) ----
    labels = pack["labels"]
    start_group = labels.index(start_id)
    goal_group  = len(labels) - 1 - labels[::-1].index(goal_id)

    # === direction_list 생성 (expand_* 와 동일 논리) ===
    # 1) 1-based → 0-based
    E0 = np.asarray(E_groups, int) - 1  # ← 핵심

    # 2) 그룹 중심 (pack["v"]는 그룹 사각형; expand_*와 같은 공식)
    centers = np.array([[0.25 * rect[0].sum(), 0.25 * rect[1].sum()]
                        for rect in pack["v"]], dtype=float)

    #    y-up 좌표계에 맞추려면 y 뒤집기 (v 를 y-up으로 바꾸셨으므로 방향도 맞춰줍니다)
    centers[:, 1] = H * tile_size - centers[:, 1]

    n_groups = len(pack["v"])

    # 3) 인접 리스트
    outN = [[] for _ in range(n_groups)]
    inN  = [[] for _ in range(n_groups)]
    for u, w in E0:
        if 0 <= u < n_groups and 0 <= w < n_groups:
            outN[u].append(w); inN[w].append(u)

    # 4) 글로벌 백업 방향: start → goal
    gdir = centers[goal_group] - centers[start_group]
    gdir /= (np.linalg.norm(gdir) + 1e-12)

    # 5) 그룹별 진행 벡터: 이웃 평균(없으면 gdir)
    dirs = np.zeros((n_groups, 2), float)
    for g in range(n_groups):
        neigh = outN[g] if outN[g] else inN[g]
        if neigh:
            d = (centers[neigh] - centers[g]).mean(axis=0)
            nrm = np.linalg.norm(d)
            dirs[g] = d / nrm if nrm > 1e-12 else gdir
        else:
            dirs[g] = gdir


    def _angle_to_octant(theta_rad: float) -> str:
        deg = (math.degrees(theta_rad) + 360.0) % 360.0

        def in_range(d, a, b):
            # [a, b) on a circle (wrap-aware)
            return (a <= d < b) if a <= b else (d >= a or d < b)

        # 카디널: 폭 10°(±5°), 대각: 남는 80°
        sectors = [
            ("E", 345.0, 15.0),  # 355°~360° and 0°~5°
            ("NE", 15.0, 75.0),
            ("N", 75.0, 105.0),
            ("NW", 105.0, 165.0),
            ("W", 165.0, 195.0),
            ("SW", 195.0, 255.0),
            ("S", 255.0, 285.0),
            ("SE", 285.0, 345.0),
        ]
        for label, a, b in sectors:
            if in_range(deg, a, b):
                return label
        return "E"  # fallback (이론상 도달 안 함)

    EPS = 1e-12
    NONE_LABEL = "None"

    # (A) 라벨 기준으로 start==goal인지 판단 (동일 라벨이 여러 인덱스에 등장하는 케이스 포함)
    same_label = (start_id == goal_id)
    sg_indices = set()
    if same_label:
        sg_indices = {i for i, lb in enumerate(labels) if lb == start_id}

    # (B) 각 그룹 중심이 속한 타일 이름을 구해둔다 (y-up -> grid row 변환)
    def _xy_to_rc(x, y, H, ts):
        r = int(np.floor((H * ts - y) / ts))
        c = int(np.floor(x / ts))
        r = max(0, min(len(grid) - 1, r))
        c = max(0, min(len(grid[0]) - 1, c))
        return r, c

    group_tile = []
    for g in range(n_groups):
        cx, cy = float(centers[g, 0]), float(centers[g, 1])
        r, c = _xy_to_rc(cx, cy, H, tile_size)
        name = grid[r][c]
        group_tile.append(name)

    def _is_stair_y(name: str) -> bool:
        return name.startswith("SY_") or name.startswith("stair_y_")

    def _is_stair_x(name: str) -> bool:
        return name.startswith("SX_") or name.startswith("stair_x_")

    group_card = [NONE_LABEL] * n_groups  # 기본은 "None"
    for g in range(n_groups):
        # start==goal 라벨에 해당하는 모든 그룹은 "None"
        if g in sg_indices:
            group_card[g] = NONE_LABEL
            continue

        vec = None
        if g == start_group:
            neigh = outN[g]
            if len(neigh) > 0:
                vec = centers[neigh].mean(axis=0) - centers[g]
        elif g == goal_group:
            neigh = inN[g]
            if len(neigh) > 0:
                vec = centers[g] - centers[neigh].mean(axis=0)
        else:
            neigh = outN[g] if len(outN[g]) > 0 else inN[g]
            if len(neigh) > 0:
                vec = (centers[neigh] - centers[g]).mean(axis=0)

        # 폴백: 벡터가 없거나 너무 짧으면 전역 gdir 사용
        if vec is None or float(np.linalg.norm(vec)) < EPS:
            vec = gdir

        # --- ★ 계단이면 축방향으로 스냅 ---
        name = group_tile[g]
        if _is_stair_y(name):
            # y축 진행부호만 본다
            group_card[g] = "N" if vec[1] > 0 else "S"
            continue
        if _is_stair_x(name):
            # x축 진행부호만 본다
            group_card[g] = "E" if vec[0] > 0 else "W"
            continue

        # 일반 타일은 8방위 양자화
        theta = math.atan2(vec[1], vec[0])
        group_card[g] = _angle_to_octant(theta)

    # 7) 확장 정점에 브로드캐스트 (group_map은 1-based 인덱스)
    direction_list = [NONE_LABEL] * len(v)
    for g, idxs_1based in enumerate(group_map):
        for idx in idxs_1based:
            direction_list[idx - 1] = group_card[g]




    edge = build_expanded_edges_from_groups(
        group_map, mode_list, E_groups,
        start_group=start_group, goal_group=goal_group
    ).astype(int)


    # ---- 6) start/goal L/R 발 (x,y,z) + yaw(deg) 구성 ----
    def rc_to_xy(rc):
        r, c = rc
        return ( (c + 0.5) * tile_size, (r + 0.5) * tile_size )

    def z_from_rc(rc):
        r, c = rc
        name = grid[r][c]
        if floor_z is None:
            # 기본 높이
            return 0.0 if name == "floor0" else 0.2
        return float(floor_z.get(name, 0.0))

    start_xy = rc_to_xy(s);
    goal_xy = rc_to_xy(t)
    z_s = z_from_rc(s);
    z_g = z_from_rc(t)

    # ✅ 추가: 센터를 y-up 좌표로 변환 (원점 이동 포함)
    start_xy = (start_xy[0], H * tile_size - start_xy[1])
    goal_xy = (goal_xy[0], H * tile_size - goal_xy[1])

    # --- start/goal 발 배치 yaw: 시작= start_pos → (첫 후속 그룹 중심),
    #                             끝  = (마지막 선행 그룹 중심) → goal_pos ---
    # --- avg-direction yaw for start & goal ---
    EPS = 1e-12

    def _succ_list(E0, u):
        return [int(w) for (uu, w) in E0 if int(uu) == int(u)]

    def _pred_list(E0, v):
        return [int(u) for (u, vv) in E0 if int(vv) == int(v)]

    def _avg_yaw_from_point_to_groups(point_xy, group_idxs, centers):
        """point -> 평균(centers[group_idxs]) 방향의 각도(rad) 또는 None"""
        if not group_idxs:
            return None
        P = np.asarray(point_xy, float)
        C = centers[np.asarray(group_idxs, int)]  # (k,2)
        vecs = C - P  # (k,2)
        mean_vec = vecs.mean(axis=0)
        n = float(np.linalg.norm(mean_vec))
        if n < EPS:
            return None
        return math.atan2(mean_vec[1], mean_vec[0])

    def _avg_yaw_from_groups_to_point(group_idxs, point_xy, centers):
        """평균( point - centers[group_idxs] ) 방향의 각도(rad) 또는 None"""
        if not group_idxs:
            return None
        P = np.asarray(point_xy, float)
        C = centers[np.asarray(group_idxs, int)]  # (k,2)
        vecs = P - C  # (k,2)
        mean_vec = vecs.mean(axis=0)
        n = float(np.linalg.norm(mean_vec))
        if n < EPS:
            return None
        return math.atan2(mean_vec[1], mean_vec[0])

    def _fallback_yaw(start_xy, goal_xy, gdir):
        dx, dy = (goal_xy[0] - start_xy[0]), (goal_xy[1] - start_xy[1])
        if abs(dx) + abs(dy) >= EPS:
            return math.atan2(dy, dx)
        return math.atan2(gdir[1], gdir[0]) if np.linalg.norm(gdir) > EPS else 0.0

    # 1) 이웃 목록
    succs = _succ_list(E0, start_group)  # 시작 그룹의 모든 후속 이웃
    preds = _pred_list(E0, goal_group)  # 목표 그룹의 모든 선행 이웃

    # 2) 평균 방향 yaw
    yaw_start = _avg_yaw_from_point_to_groups(start_xy, succs, centers)
    yaw_goal = _avg_yaw_from_groups_to_point(preds, goal_xy, centers)

    # 3) 폴백 (이웃 없거나 평균 벡터가 0에 가까운 경우)
    if yaw_start is None:
        yaw_start = _fallback_yaw(start_xy, goal_xy, gdir)
    if yaw_goal is None:
        yaw_goal = _fallback_yaw(start_xy, goal_xy, gdir)


    # stance width
    if stance_width is None:
        stance_width = kwargs.get("stance_width", 0.176)  # 기본값(예시)

    # 좌/우 발 위치 (왼쪽 법선 n = (-sinθ, cosθ))
    def lr_feet(center_xy, z, yaw_rad, sw):
        n = (-math.sin(yaw_rad), math.cos(yaw_rad))
        off = 0.5 * sw
        L = (center_xy[0] + off*n[0], center_xy[1] + off*n[1], z)
        R = (center_xy[0] - off*n[0], center_xy[1] - off*n[1], z)
        return np.array(L, float), np.array(R, float)

    # (그대로 유지) 발 좌표는 진행방향 yaw_start, yaw_goal 로 계산
    startpos_L, startpos_R = lr_feet(start_xy, z_s, yaw_start, stance_width)
    targetpos_L, targetpos_R = lr_feet(goal_xy, z_g, yaw_goal, stance_width)

    # --- 각도 래핑 헬퍼 (라디안/도) ---
    def _wrap_pi(a: float) -> float:
        # [-pi, pi)
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _wrap_deg(d: float) -> float:
        # [-180, 180)
        return (d + 180.0) % 360.0 - 180.0

    # (위에서 yaw_start, yaw_goal 은 라디안으로 이미 계산됨)
    dtheta = _wrap_pi(yaw_goal - yaw_start)  # 라디안 상대각
    dtheta_deg = _wrap_deg(math.degrees(dtheta))  # 도(deg) 상대각

    # ✅ 출력 yaw: 스타트는 0 유지, 타겟은 스타트 대비 상대각
    startyaw_L = 0.0
    startyaw_R = 0.0
    targetyaw_L = dtheta_deg
    targetyaw_R = dtheta_deg


    # --- 결과 패키징 ---
    out = {
        "v": v,
        "edge": edge,
        "mode_list": list(mode_list),
        "vertex_number": len(v),
        "n_edge": int(edge.shape[0]),
        "startpos_L": startpos_L, "startpos_R": startpos_R,
        "targetpos_L": targetpos_L, "targetpos_R": targetpos_R,
        "startyaw_L": startyaw_L, "startyaw_R": startyaw_R,
        "targetyaw_L": targetyaw_L, "targetyaw_R": targetyaw_R,
        "direction_list": direction_list,
        # ▼ meshcat heightmap용 추가 필드
        "grid": grid,  # 2D 타일 네임 배열 (floor0/floor1/stair_x/y 등)
        "start_rc": s,  # (row, col)
        "goal_rc": t,  # (row, col)
    }

    # --- (옵션) 플로팅: main의 3x2 구성 그대로 ---
    if plot:
        fig, axs = _plot_wfc_pipeline(grid, G_nav, G_new, G_compact, Gk, node_info, s, t)
        if plt_show:
            plt.show(block=False)
            plt.pause(0.001)
        if return_fig:
            out["fig"] = fig
            out["axs"] = axs

    return out



def build_expanded_edges_from_groups(
    group_map: List[Sequence[int]],      # 그룹 g -> 확장 정점 1-based 인덱스 목록
    mode_list: Sequence[str],            # 확장 정점별 모드 ("D","SL","SR")
    E_groups: np.ndarray,                # 그룹 간 방향 엣지 (1-based, shape=(M,2))
    *,
    start_group: int,                    # 0-based
    goal_group: int,                     # 0-based
) -> np.ndarray:
    edges = set()

    def first_in_group_by_mode(g: int, target: str) -> Optional[int]:
        for vid in group_map[g]:
            if mode_list[vid - 1] == target:
                return vid
        return None

    # ① 그룹 내부 엣지
    for g, ids in enumerate(group_map):
        k = len(ids)
        if k == 0:
            continue

        # 체인
        for j in range(k - 1):
            edges.add((int(ids[j]), int(ids[j + 1])))

        if g == start_group and k >= 2:
            # 시작 그룹: D -> SL / D -> SR 분기
            d_id  = ids[0]
            sl_id = first_in_group_by_mode(g, "SL")
            sr_id = first_in_group_by_mode(g, "SR")
            if sl_id is not None: edges.add((int(d_id), int(sl_id)))
            if sr_id is not None: edges.add((int(d_id), int(sr_id)))

        if g == goal_group and k >= 2:
            # 마지막 그룹: 모두 마지막 D로 수렴
            d_last = ids[-1]
            for u in ids[:-1]:
                edges.add((int(u), int(d_last)))

    # ② 그룹 간 엣지 (모드 매핑 규칙: SL->첫 SR, SR->첫 SL)
    E_arr = np.asarray(E_groups, dtype=int)
    for u1, w1 in E_arr:
        gu = u1 - 1  # 0-based
        gw = w1 - 1

        src_ids = list(group_map[gu])
        # 시작 그룹이면 맨 앞 D는 외부 연결에서 제외 (내부 분기 후 외부로 나감)
        if gu == start_group and src_ids:
            src_ids = src_ids[1:]

        if not src_ids or not group_map[gw]:
            continue

        dst_first_sl = first_in_group_by_mode(gw, "SL")
        dst_first_sr = first_in_group_by_mode(gw, "SR")

        # 목적 그룹에 SL/SR 둘 다 없을 극단 케이스 폴백
        fallback = group_map[gw][0] if (dst_first_sl is None and dst_first_sr is None) else None

        for s in src_ids:
            sm = mode_list[s - 1]
            if sm == "SL":
                t = dst_first_sr or dst_first_sl or fallback
            elif sm == "SR":
                t = dst_first_sl or dst_first_sr or fallback
            else:
                # (비정상) 중간/마지막 그룹에 D가 소스로 올 일은 없지만 방어적으로 skip
                continue
            if t is not None:
                edges.add((int(s), int(t)))

    return np.asarray(sorted(edges), dtype=int).reshape(-1, 2)

def _bfs_dist(Gk, src, nodes_set):
    INF = 10**9
    dist = {u: INF for u in nodes_set}
    if src not in dist:  # src가 이번 label_order에 없으면 그대로 INF
        return dist
    q = deque([src]); dist[src] = 0
    # 무향 인접
    adj = defaultdict(set)
    for u, Ns in Gk.items():
        for v in Ns:
            if u in nodes_set and v in nodes_set:
                adj[u].add(v); adj[v].add(u)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == INF:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def reorder_label_order_for_front_to_back(Gk, label_order, *, start, goal):
    """
    앞→뒤 정향을 그대로 쓰되, 중간 노드 순서를 (dS 오름, dG 내림)으로 재정렬.
    start는 맨 앞, goal은 맨 뒤에 고정.
    """
    nodes = list(label_order)
    S = set(nodes)
    dS = _bfs_dist(Gk, start, S)
    dG = _bfs_dist(Gk, goal,  S)
    start_only = nodes[0] if nodes else start
    goal_only  = nodes[-1] if nodes else goal

    mids = [u for u in nodes if u not in (start_only, goal_only)]
    # 타이브레이커는 문자열화
    mids.sort(key=lambda u: (dS.get(u, 10**9), -dG.get(u, -10**9), str(u)))
    return [start_only] + mids + [goal_only]


Node = Hashable  # ('rect', idx), ('stair',(r,c)) 등 해시 가능한 타입


def bfs(G: Dict[Node, List[Node]], start: Node) -> Set[Node]:
    """무향 그래프: start에서 도달 가능한 노드 집합"""
    if start not in G:
        return set()
    vis: Set[Node] = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in G.get(u, []):
            if v not in vis:
                vis.add(v); q.append(v)
    return vis

def prune_to_st_kernel_undirected(
    G: Dict[Node, List[Node]],
    s: Node,
    t: Node,
) -> Dict[Node, List[Node]]:
    """
    s에서 도달 가능 집합 Rs, t에서 도달 가능 집합 Rt를 구해
    K = Rs ∩ Rt 만 남기는 커널 프루닝 (무향 가정).
    모든 s–t 경로 보존.
    """
    Rs = bfs(G, s)
    Rt = bfs(G, t)
    keep = Rs & Rt
    if not keep:
        return {}

    # keep 내부만 남기고, 이웃도 keep으로 필터링 + 양방향 대칭 보정
    H: Dict[Node, List[Node]] = {u: [] for u in keep}
    for u in keep:
        for v in G.get(u, []):
            if v in keep and v != u:
                H[u].append(v)
                if u not in H[v]:
                    H[v].append(u)  # 대칭 보정
    # 중복 제거
    for u in H:
        H[u] = list(dict.fromkeys(H[u]))
    return H

def trim_leaves_iterative_undirected(
    G: Dict[Node, List[Node]],
    s: Node,
    t: Node,
) -> Dict[Node, List[Node]]:
    """
    s,t를 제외한 차수<=1(막다른) 노드를 반복 제거.
    모든 s–t 경로 보존.
    """
    H: Dict[Node, List[Node]] = {u: list(dict.fromkeys(nbrs)) for u, nbrs in G.items()}
    deg = {u: len(H[u]) for u in H}

    from collections import deque
    q = deque([u for u, d in deg.items() if u not in (s, t) and d <= 1])

    removed: Set[Node] = set()
    while q:
        u = q.popleft()
        if u in removed or u in (s, t):
            continue
        if deg.get(u, 0) > 1:
            continue

        # 이웃에서 u 제거
        for v in H.get(u, []):
            if v in removed:
                continue
            if v in H:
                H[v] = [x for x in H[v] if x != u]
                deg[v] = len(H[v])
                if v not in (s, t) and deg[v] <= 1:
                    q.append(v)

        # u 제거
        removed.add(u)
        H.pop(u, None)
        deg.pop(u, None)

    # 정리: 존재 노드만 이웃으로
    for u in list(H.keys()):
        H[u] = [v for v in H[u] if v in H]
    return H

_MESH_VIS = None  # 뷰어 재사용 (여러 번 열리는 것 방지)
def reachable_nodes(G: Dict[Coord, List[Coord]], start: Coord) -> Set[Coord]:
    """무향 그래프 G에서 start로부터 도달 가능한 노드 집합."""
    if start not in G:
        return set()
    seen: Set[Coord] = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in G.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen

def induced_subgraph(G: Dict[Coord, List[Coord]], nodes: Set[Coord]) -> Dict[Coord, List[Coord]]:
    """노드 집합 nodes에 대한 유도 부분 그래프(양방향, 중복 제거)."""
    G_new: Dict[Coord, List[Coord]] = {}
    for u in nodes:
        # 이웃도 nodes 안에 있는 것만, 중복 없이
        nbrs = [v for v in G.get(u, []) if v in nodes]
        # 필요하면 set으로 중복 제거 후 순서 보존 간단 처리
        seen = set()
        G_new[u] = [v for v in nbrs if not (v in seen or seen.add(v))]
    return G_new

def normalize_label_order(order):
    # order가 dict(layers=...)이든 list든 모두 평탄화해 1차 리스트로 변환
    if isinstance(order, dict) and "layers" in order:
        return [lab for layer in order["layers"] for lab in layer]
    elif isinstance(order, (list, tuple)):
        return list(order)
    else:
        raise ValueError("order 형태를 인식할 수 없습니다.")

def build_reachable_subgraph(G: Dict[Coord, List[Coord]], start: Coord) -> Dict[Coord, List[Coord]]:
    """start에서 닿을 수 있는 노드만 남긴 부분 그래프."""
    R = reachable_nodes(G, start)
    return induced_subgraph(G, R)

# ---------- 직사각형 분해 ----------
def rect_partition_squareish(
    mask: List[List[bool]],
    *,
    alpha: float = 0.6,     # ↑ 정사각형 선호도 (0이면 순수 면적 우선)
    min_side: int = 1,      # 최소 변 길이
) -> List[Tuple[int,int,int,int]]:
    """
    True 셀들을 직사각형들 (r0, c0, h, w) 로 분해.
    - 정사각형에 가까운 확장을 선호하도록 점수 기반으로 한 칸씩 키움.
    - 그리드 좌표계: r=행(y), c=열(x)
    """
    H = len(mask); W = len(mask[0]) if H else 0
    used = [[False]*W for _ in range(H)]

    def can_fill(r0, c0, h, w) -> bool:
        if r0 < 0 or c0 < 0 or r0+h > H or c0+w > W: return False
        for rr in range(r0, r0+h):
            row = mask[rr]
            if any((not row[cc]) or used[rr][cc] for cc in range(c0, c0+w)):
                return False
        return True

    def aspect(h, w) -> float:
        a, b = (h, w) if h >= w else (w, h)
        return a / max(1, b)

    def score(h, w) -> float:
        A = h * w
        asp = aspect(h, w)
        return A / (1.0 + alpha * (asp - 1.0))

    rects: List[Tuple[int,int,int,int]] = []

    # 반복: 남은 True 셀 중 가장 위-왼쪽부터 직사각형 하나 성장시키기
    while True:
        seed = None
        for r in range(H):
            row = mask[r]
            for c in range(W):
                if row[c] and not used[r][c]:
                    seed = (r, c)
                    break
            if seed: break
        if not seed:
            break  # 더 이상 남은 True 없음

        r0, c0 = seed
        h, w = 1, 1  # 최소 1x1로 시작
        if not can_fill(r0, c0, h, w):
            # 이론상 안 걸려야 하지만 안전장치
            used[r0][c0] = True
            rects.append((r0, c0, 1, 1))
            continue

        # 확장 루프: 한 번에 1행 or 1열씩 확장하되, 점수 개선되는 쪽 선택
        improved = True
        while improved:
            improved = False
            best_h, best_w = h, w
            best_s = score(h, w)

            # 세로(아래)로 +1행 확장 가능?
            if can_fill(r0, c0, h+1, w) and (h+1) >= min_side and w >= min_side:
                s_down = score(h+1, w)
                if s_down > best_s:
                    best_s = s_down
                    best_h, best_w = h+1, w
                    improved = True

            # 가로(오른쪽)로 +1열 확장 가능?
            if can_fill(r0, c0, h, w+1) and h >= min_side and (w+1) >= min_side:
                s_right = score(h, w+1)
                if s_right > best_s:
                    best_s = s_right
                    best_h, best_w = h, w+1
                    improved = True

            h, w = best_h, best_w

        # 확정: 사용 처리
        for rr in range(r0, r0+h):
            for cc in range(c0, c0+w):
                used[rr][cc] = True
        rects.append((r0, c0, h, w))

    return rects

# ---------- floor 타일을 직사각형 노드로 병합 ----------
def make_floor_rects_from_Gnew(
    grid: List[List[str]],
    G_new: Dict[Coord, List[Coord]],
    floor_names: Tuple[str,str] = ("floor0","floor1"),
    *,
    tile_size: float = 1.0,                      # ★ 추가
    floor_z: dict = {"floor0": 0.0, "floor1": 0.2},  # ★ 추가
    y_up: bool = True,                           # (원하면 그대로 유지/사용)
):
    H = len(grid); W = len(grid[0]) if H else 0
    F0, F1 = floor_names
    def is_floor(name: str): return name == F0 or name == F1

    # G_new 안의 floor만 True인 마스크
    mask0 = [[False]*W for _ in range(H)]
    mask1 = [[False]*W for _ in range(H)]
    for (r,c) in G_new.keys():
        name = grid[r][c]
        if name == F0: mask0[r][c] = True
        elif name == F1: mask1[r][c] = True

    rects0 = rect_partition_squareish(mask0)
    rects1 = rect_partition_squareish(mask1)

    # 노드ID 부여: 직사각형 노드는 ('rect', idx)
    node_info = {}  # node_id -> dict(meta)
    cell_to_node: Dict[Coord, Tuple[str,int]] = {}

    def _add_rect_node(r0, c0, h, w, floor_name, idx):
        node_id = ('rect', idx)
        info = {"type":"rect","floor":floor_name,"r0":r0,"c0":c0,"h":h,"w":w}

        # ★ 여기서 연속좌표로 직사각형 꼭짓점 생성
        # 타일 경계: x: [c0, c0+w], y: [r0, r0+h]
        x0 = c0 * tile_size
        x1 = (c0 + w) * tile_size
        if y_up:
            y0 = r0 * tile_size
            y1 = (r0 + h) * tile_size
        else:
            # 화면 좌표처럼 r가 아래로 증가하는 체계를 세계 좌표 y-상승으로 바꾸고 싶다면:
            y0 = (H - (r0 + h)) * tile_size
            y1 = (H - r0) * tile_size

        z = float(floor_z.get(floor_name, 0.0))

        # CCW: (x0,y0) → (x0,y1) → (x1,y1) → (x1,y0)
        rect2 = [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
        rect3 = [[x0, y0, z], [x0, y1, z], [x1, y1, z], [x1, y0, z]]

        info["rect"] = rect2          # 4x2
        info["rect_3d"] = rect3       # 4x3
        info["z"] = z                 # 참고로 z도 넣어둠

        node_info[node_id] = info

        # 셀 매핑
        for rr in range(r0, r0+h):
            for cc in range(c0, c0+w):
                # floor 마스크로 채워진 셀만 매핑
                cell_to_node[(rr,cc)] = node_id

    idx = 0
    for (r0,c0,h,w) in rects0:
        _add_rect_node(r0, c0, h, w, F0, idx); idx += 1

    for (r0,c0,h,w) in rects1:
        _add_rect_node(r0, c0, h, w, F1, idx); idx += 1

    # 계단 노드 추가: G_new 안의 stair_* 는 개별 노드('stair',(r,c))
    for (r,c) in G_new.keys():
        name = grid[r][c]
        if name.startswith("stair_"):
            node_id = ('stair', (r,c))
            node_info[node_id] = {"type":"stair","r":r,"c":c,"name":name}
            cell_to_node[(r,c)] = node_id

    return node_info, cell_to_node


# ---------- 압축 그래프 만들기 ----------
def compact_graph_from_rects(
    grid: List[List[str]],
    G_new: Dict[Coord, List[Coord]],
    node_info: Dict, cell_to_node: Dict[Coord, Tuple],
):
    # 인접 리스트
    Gc: Dict[Tuple, Set[Tuple]] = {nid: set() for nid in node_info.keys()}

    for u, nbrs in G_new.items():
        nu = cell_to_node.get(u)
        if nu is None:  # 안전장치
            continue
        for v in nbrs:
            nv = cell_to_node.get(v)
            if nv is None or nv == nu:
                continue
            Gc[nu].add(nv)
            Gc[nv].add(nu)

    # list로 변환
    Gc_list: Dict[Tuple, List[Tuple]] = {k: sorted(list(v)) for k,v in Gc.items()}
    return Gc_list

# ---------- 편의: start/goal의 새 노드ID ----------
def map_point_to_compact_node(cell_to_node: Dict[Coord, Tuple], rc: Coord) -> Optional[Tuple]:
    return cell_to_node.get(rc)

def _get_vis():
    global _MESH_VIS
    if _MESH_VIS is None:
        _MESH_VIS = meshcat.Visualizer().open()  # 브라우저 뷰어 열림
    return _MESH_VIS

def _tile_side_heights(name: str, h_step: float):
    """계단 라벨에서 양쪽/위아래 높이 decode. 없으면 None 반환."""
    if name.startswith("stair_x"):
        m = re.search(r"L([01])R([01])", name)
        if m:
            zL = 0.0 if int(m.group(1)) == 0 else h_step
            zR = 0.0 if int(m.group(2)) == 0 else h_step
            return ("x", zL, zR)
        return ("x", 0.5*h_step, 0.5*h_step)  # 라벨 정보 없으면 중앙값
    if name.startswith("stair_y"):
        m = re.search(r"U([01])D([01])", name)
        if m:
            zU = 0.0 if int(m.group(1)) == 0 else h_step
            zD = 0.0 if int(m.group(2)) == 0 else h_step
            return ("y", zU, zD)
        return ("y", 0.5*h_step, 0.5*h_step)
    return None

def _tile_edge_height(name: str, h_step: float, side: str) -> float:
    """
    side: 'L','R','U','D'  (좌/우/상/하 엣지의 상판 높이)
    """
    if name == "floor0": return 0.0
    if name == "floor1": return h_step
    info = _tile_side_heights(name, h_step)
    if info is None:
        return 0.5*h_step
    axis, a, b = info  # a,b는 (좌,우) 또는 (상,하)
    if axis == "x":
        if side == 'L': return a
        if side == 'R': return b
        return (a + b) * 0.5
    else:  # axis == 'y'
        if side == 'U': return a
        if side == 'D': return b
        return (a + b) * 0.5



def add_side_walls_meshcat(vis, root, tile_grid, *, tile_size=1.0, h_step=0.2,
                           stair_slices=4, wall_color=0x737373, wall_thickness=None):
    """
    - 그리드 외곽 경계 벽
    - 이웃 타일 경계에서 높이 차이만큼 세로 벽
    - 계단 슬라이스 사이 내부 라이저
    """
    import meshcat.geometry as g
    import meshcat.transformations as tf
    H, W = len(tile_grid), len(tile_grid[0])
    t = wall_thickness or max(0.02*tile_size, 0.01)

    # A) 그리드 외곽
    for c in range(W):
        # 윗변(y=0): 타일(0,c) 의 위엣지
        zU = _tile_edge_height(tile_grid[0][c], h_step, 'U')
        if zU > 0.0:
            x = (c + 0.5)*tile_size; y = 0.0
            h = zU; node = f"{root}/walls/boundary/U_{c}"
            size = [tile_size, t, h]
            T = tf.translation_matrix([x, y - 0.5*t, 0.5*h])
            vis[node].set_object(g.Box(size), g.MeshPhongMaterial(color=wall_color))
            vis[node].set_transform(T)
        # 아랫변(y=H): 타일(H-1,c) 의 아래엣지
        zD = _tile_edge_height(tile_grid[H-1][c], h_step, 'D')
        if zD > 0.0:
            x = (c + 0.5)*tile_size; y = H*tile_size
            h = zD; node = f"{root}/walls/boundary/D_{c}"
            size = [tile_size, t, h]
            T = tf.translation_matrix([x, y + 0.5*t, 0.5*h])
            vis[node].set_object(g.Box(size), g.MeshPhongMaterial(color=wall_color))
            vis[node].set_transform(T)
    for r in range(H):
        # 좌변(x=0): 타일(r,0) 의 왼엣지
        zL = _tile_edge_height(tile_grid[r][0], h_step, 'L')
        if zL > 0.0:
            x = 0.0; y = (r + 0.5)*tile_size
            h = zL; node = f"{root}/walls/boundary/L_{r}"
            size = [t, tile_size, h]
            T = tf.translation_matrix([x - 0.5*t, y, 0.5*h])
            vis[node].set_object(g.Box(size), g.MeshPhongMaterial(color=wall_color))
            vis[node].set_transform(T)
        # 우변(x=W): 타일(r,W-1) 의 오른엣지
        zR = _tile_edge_height(tile_grid[r][W-1], h_step, 'R')
        if zR > 0.0:
            x = W*tile_size; y = (r + 0.5)*tile_size
            h = zR; node = f"{root}/walls/boundary/R_{r}"
            size = [t, tile_size, h]
            T = tf.translation_matrix([x + 0.5*t, y, 0.5*h])
            vis[node].set_object(g.Box(size), g.MeshPhongMaterial(color=wall_color))
            vis[node].set_transform(T)

    # B) 이웃 타일 사이 경계
    # 가로 경계: (r,c) | (r,c+1) 사이
    for r in range(H):
        for c in range(W-1):
            left = tile_grid[r][c]; right = tile_grid[r][c+1]
            zL = _tile_edge_height(left, h_step, 'R')
            zR = _tile_edge_height(right, h_step, 'L')
            if abs(zL - zR) < 1e-9:  # 높이 동일 → 벽 없음
                continue
            z0 = min(zL, zR); dh = abs(zL - zR)
            x = (c + 1) * tile_size; y = (r + 0.5)*tile_size
            size = [t, tile_size, dh]
            node = f"{root}/walls/h/r{r}_c{c}"
            T = tf.translation_matrix([x, y, z0 + 0.5*dh])
            vis[node].set_object(g.Box(size), g.MeshPhongMaterial(color=wall_color))
            vis[node].set_transform(T)

    # 세로 경계: (r,c) ─ (r+1,c) 사이
    for r in range(H-1):
        for c in range(W):
            up = tile_grid[r][c]; down = tile_grid[r+1][c]
            zU = _tile_edge_height(up, h_step, 'D')
            zD = _tile_edge_height(down, h_step, 'U')
            if abs(zU - zD) < 1e-9:
                continue
            z0 = min(zU, zD); dh = abs(zU - zD)
            x = (c + 0.5)*tile_size; y = (r + 1)*tile_size
            size = [tile_size, t, dh]
            node = f"{root}/walls/v/r{r}_c{c}"
            T = tf.translation_matrix([x, y, z0 + 0.5*dh])
            vis[node].set_object(g.Box(size), g.MeshPhongMaterial(color=wall_color))
            vis[node].set_transform(T)

    # C) 계단 내부 라이저 (슬라이스 간)
    for r in range(H):
        for c in range(W):
            name = tile_grid[r][c]
            info = _tile_side_heights(name, h_step)
            if info is None:
                continue
            axis, a, b = info
            if axis == "x":
                # 슬라이스 경계 x = c + i/s
                for i in range(stair_slices-1):
                    u0 = (i + 0.5) / stair_slices
                    u1 = (i + 1.5) / stair_slices
                    z0 = (1.0 - u0)*a + u0*b
                    z1 = (1.0 - u1)*a + u1*b
                    if abs(z0 - z1) < 1e-9:
                        continue
                    x = (c + (i+1)/stair_slices)*tile_size
                    y = (r + 0.5)*tile_size
                    dh = abs(z1 - z0); zmin = min(z0, z1)
                    size = [t, tile_size, dh]
                    node = f"{root}/walls/stair_x/r{r}_c{c}_i{i}"
                    T = tf.translation_matrix([x, y, zmin + 0.5*dh])
                    vis[node].set_object(g.Box(size), g.MeshPhongMaterial(color=wall_color))
                    vis[node].set_transform(T)
            else:
                # axis == 'y' : 슬라이스 경계 y = r + i/s
                for i in range(stair_slices-1):
                    v0 = (i + 0.5) / stair_slices
                    v1 = (i + 1.5) / stair_slices
                    z0 = (1.0 - v0)*a + v0*b
                    z1 = (1.0 - v1)*a + v1*b
                    if abs(z0 - z1) < 1e-9:
                        continue
                    x = (c + 0.5)*tile_size
                    y = (r + (i+1)/stair_slices)*tile_size
                    dh = abs(z1 - z0); zmin = min(z0, z1)
                    size = [tile_size, t, dh]
                    node = f"{root}/walls/stair_y/r{r}_c{c}_i{i}"
                    T = tf.translation_matrix([x, y, zmin + 0.5*dh])
                    vis[node].set_object(g.Box(size), g.MeshPhongMaterial(color=wall_color))
                    vis[node].set_transform(T)


def _tile_top_z_at_xy(tile_grid, r, c, x, y, tile_size, h_step):
    """(r,c) 타일 내 (x,y)에서 상판 z (계단은 선형 보간)."""
    name = tile_grid[r][c]
    if name == "floor0": return 0.0
    if name == "floor1": return h_step
    info = _tile_side_heights(name, h_step)
    if info is None: return 0.0
    axis, a, b = info
    if axis == "x":
        u = (x - c*tile_size) / tile_size
        u = np.clip(u, 0.0, 1.0)
        return (1.0 - u)*a + u*b
    else:
        v = (y - r*tile_size) / tile_size
        v = np.clip(v, 0.0, 1.0)
        return (1.0 - v)*a + v*b


def _camera_rotation_lookat(cam_pos, target, up=(0,0,1)):
    """three.js 규약(-Z가 보는 방향)에 맞춰 cam_pos가 target을 보도록 회전행렬 생성"""
    cp = np.asarray(cam_pos, float)
    tg = np.asarray(target, float)
    up = np.asarray(up, float)

    fwd = tg - cp
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        fwd = np.array([0, 0, -1.0])  # fallback
    else:
        fwd /= n
    z_cam = -fwd                      # 카메라의 -Z가 바라보는 방향
    x_cam = np.cross(up, z_cam)
    nx = np.linalg.norm(x_cam)
    if nx < 1e-9:                     # up과 평행하면 up을 바꿔줌
        up = np.array([0,1,0])
        x_cam = np.cross(up, z_cam)
        nx = np.linalg.norm(x_cam)
    x_cam /= nx
    y_cam = np.cross(z_cam, x_cam)

    R = np.eye(4)
    R[0:3,0] = x_cam
    R[0:3,1] = y_cam
    R[0:3,2] = z_cam
    return R


def plot_heightmap_3d(
    X, Y, Z, *,                    # (호환성) 무시
    tile_grid,
    tile_size=1.0,
    elev=50,                       # (Meshcat에선 미사용: 수동으로 돌리면 됨)
    azim=200,                      # (Meshcat에선 미사용)
    h_step=0.2,
    stair_slices=4,                # 계단 슬라이스 수
    draw_edges=False,              # (미사용) Meshcat은 기본 재질 경계가 잘 보임
    start=None,                    # (r,c)
    goal=None,                     # (r,c)
    marker_size_ratio=0.30,        # 메쉬 마커 사이즈 비율
    marker_lift=0.02               # 상판보다 살짝 띄우기
):
    vis = _get_vis()
    root = "wfc_map"
    # 깨끗이 비우고 다시 그리기
    vis[root].delete()

    H, W = len(tile_grid), len(tile_grid[0])
    # Y축 반전 행렬(스케일) + 위로 보정 이동(2D와 동일한 방향성 맞춤)
    S = np.diag([1.0, -1.0, 1.0, 1.0])  # ← Y만 -1로 반전
    T = tf.translation_matrix([0.0, H * tile_size, 0.0])  # ← 반전 후 위로 올려서 (0..H*tile) 범위로
    vis[root].set_transform(T @ S)

    # ── 타일 상판 표현: 얇은 판(BOX)을 상판 z에 맞춰 슬라이스로 배치 ──
    plate_thickness = max(0.01*tile_size, 0.015)  # 얇은 상판 두께
    # 색상
    col_floor0 = 0xFFFFFF
    col_floor1 = 0xB3B3B3
    col_stair  = 0xCCCCCC

    for r in range(H):
        for c in range(W):
            name = tile_grid[r][c]
            cx = (c + 0.5) * tile_size
            cy = (r + 0.5) * tile_size

            if name == "floor0" or name == "floor1":
                ztop = 0.0 if name == "floor0" else h_step
                size = [tile_size, tile_size, plate_thickness]
                T = tf.translation_matrix([cx, cy, ztop + 0.5*plate_thickness])
                color = col_floor0 if name == "floor0" else col_floor1
                node = f"{root}/tile_{r}_{c}"
                vis[node].set_object(mcg.Box(size), mcg.MeshPhongMaterial(color=color))
                vis[node].set_transform(T)

            elif name.startswith("stair_"):
                info = _tile_side_heights(name, h_step)
                if info is None:
                    # 정보 없으면 단일 중앙 슬랩
                    zmid = 0.5 * h_step
                    size = [tile_size, tile_size, plate_thickness]
                    T = tf.translation_matrix([cx, cy, zmid + 0.5*plate_thickness])
                    node = f"{root}/tile_{r}_{c}"
                    vis[node].set_object(mcg.Box(size), mcg.MeshPhongMaterial(color=col_stair))
                    vis[node].set_transform(T)
                else:
                    axis, a, b = info
                    if axis == "x":
                        # x 방향으로 stair_slices 개 슬랩
                        dx = tile_size / stair_slices
                        for i in range(stair_slices):
                            u = (i + 0.5) / stair_slices
                            ztop = (1.0 - u)*a + u*b
                            xmid = (c + u) * tile_size
                            size = [dx, tile_size, plate_thickness]
                            T = tf.translation_matrix([xmid, cy, ztop + 0.5*plate_thickness])
                            node = f"{root}/tile_{r}_{c}/slab_{i}"
                            vis[node].set_object(mcg.Box(size), mcg.MeshPhongMaterial(color=col_stair))
                            vis[node].set_transform(T)
                    else:
                        # y 방향으로 stair_slices 개 슬랩
                        dy = tile_size / stair_slices
                        for i in range(stair_slices):
                            v = (i + 0.5) / stair_slices
                            ztop = (1.0 - v)*a + v*b
                            ymid = (r + v) * tile_size
                            size = [tile_size, dy, plate_thickness]
                            T = tf.translation_matrix([cx, ymid, ztop + 0.5*plate_thickness])
                            node = f"{root}/tile_{r}_{c}/slab_{i}"
                            vis[node].set_object(mcg.Box(size), mcg.MeshPhongMaterial(color=col_stair))
                            vis[node].set_transform(T)
    # ── (상판 생성 끝) → 옆면 추가 ──
    add_side_walls_meshcat(vis, root, tile_grid,
                           tile_size=tile_size, h_step=h_step,
                           stair_slices=stair_slices,
                           wall_color=0x737373, wall_thickness=None)



    # ── Start / Goal 마커 (항상 위로 보이도록 depthTest 끄기 + renderOrder 크게) ──
    def place_marker(rc, color_hex, name):
        r, c = rc
        cx = (c + 0.5) * tile_size
        cy = (r + 0.5) * tile_size
        zc = _tile_top_z_at_xy(tile_grid, r, c, cx, cy, tile_size, h_step) + marker_lift
        radius = 0.4 * marker_size_ratio * tile_size
        node = f"{root}/markers/{name}"

        vis[node].set_object(mcg.Sphere(radius), mcg.MeshPhongMaterial(color=color_hex))
        vis[node].set_transform(tf.translation_matrix([cx, cy, zc]))

        # ✅ 항상 위로: 노드 메서드는 (property, value) 2개만!
        vis[node].set_property("material.depthTest", False)  # 잘못: ("material","depthTest",False)
        vis[node].set_property("renderOrder", 10000)
        # (옵션) z-fighting 완화용
        vis[node].set_property("material.polygonOffset", True)
        vis[node].set_property("material.polygonOffsetFactor", -2)
        vis[node].set_property("material.polygonOffsetUnits", -2)

    if start is not None:
        place_marker(start, 0x2ECC71, "start")
    if goal is not None:
        place_marker(goal, 0xE74C3C, "goal")

    # 배경/그리드 (선택)
    vis["/Background"].set_property("visible", False)
    vis["/Grid"].set_property("visible", False)
    vis["/Axes"].set_property("visible", False)

    return vis

def tile_top_z_at_xy(tile_grid, r, c, x, y, tile_size, h_step):
    name = tile_grid[r][c]
    if name == "floor0": return 0.0
    if name == "floor1": return h_step
    u = (x - c*tile_size) / tile_size
    v = (y - r*tile_size) / tile_size
    u = min(max(u, 0.0), 1.0); v = min(max(v, 0.0), 1.0)
    if name.startswith("stair_x"):
        m = re.search(r"L([01])R([01])", name)
        if m:
            zL = 0.0 if int(m.group(1)) == 0 else h_step
            zR = 0.0 if int(m.group(2)) == 0 else h_step
            return (1.0 - u)*zL + u*zR
        return 0.5*h_step
    if name.startswith("stair_y"):
        m = re.search(r"U([01])D([01])", name)
        if m:
            zU = 0.0 if int(m.group(1)) == 0 else h_step
            zD = 0.0 if int(m.group(2)) == 0 else h_step
            return (1.0 - v)*zU + v*zD
        return 0.5*h_step
    return 0.0


Coord = Tuple[int, int]

def build_nav_graph_simple(
    grid: List[List[str]],
    *,
    floor_names: Tuple[str, str] = ("floor0", "floor1"),
    stair_x_prefixes: Tuple[str, ...] = ("stair_x",),
    stair_y_prefixes: Tuple[str, ...] = ("stair_y",),
) -> Dict[Coord, List[Coord]]:
    """
    노드: grid의 모든 (r,c)
    엣지 규칙:
      - floor0/floor1: 같은 floor 이름인 4-이웃과만 연결
      - stair_x*: (r,c) ↔ (r, c-1), (r,c) ↔ (r, c+1)
      - stair_y*: (r,c) ↔ (r-1, c), (r,c) ↔ (r+1, c)
    ※ 계단은 노드이며, 계단-이웃 연결 시 이웃 타입은 제한하지 않음(간단화).
    """
    H = len(grid)
    W = len(grid[0]) if H else 0
    F0, F1 = floor_names

    def inb(r, c): return 0 <= r < H and 0 <= c < W
    def is_floor(name): return name == F0 or name == F1
    def is_stair_x(name): return any(name.startswith(p) for p in stair_x_prefixes)
    def is_stair_y(name): return any(name.startswith(p) for p in stair_y_prefixes)

    G: Dict[Coord, List[Coord]] = {(r, c): [] for r in range(H) for c in range(W)}

    def add_edge(u: Coord, v: Coord):
        if v not in G[u]: G[u].append(v)
        if u not in G[v]: G[v].append(u)

    # 1) floor 엣지: 같은 floor 4-이웃만
    dirs = [(-1,0), (0,1), (1,0), (0,-1)]
    for r in range(H):
        for c in range(W):
            me = grid[r][c]
            if is_floor(me):
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if inb(nr, nc) and grid[nr][nc] == me:
                        add_edge((r,c), (nr,nc))

    # 2) stair 엣지: x는 좌/우, y는 상/하
    for r in range(H):
        for c in range(W):
            name = grid[r][c]
            if is_stair_x(name):
                if inb(r, c-1): add_edge((r,c), (r, c-1))
                if inb(r, c+1): add_edge((r,c), (r, c+1))
            if is_stair_y(name):
                if inb(r-1, c): add_edge((r,c), (r-1, c))
                if inb(r+1, c): add_edge((r,c), (r+1, c))

    return G


def plot_nav_graph(
    grid: List[List[str]],
    G: Dict[Coord, List[Coord]],
    *,
    path: Optional[List[Coord]] = None,
    start: Optional[Coord] = None,
    goal: Optional[Coord] = None,
    ax=None,
):
    H = len(grid); W = len(grid[0]) if H else 0

    def color_of(name: str) -> str:
        if name == "floor0": return "#FFFFFF"  # white
        if name == "floor1": return "#C0C0C0"  # gray
        if name.startswith("stair_x"): return "#FF6666"  # light red
        if name.startswith("stair_y"): return "#FFA500"  # orange
        return "#999999"

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, W*0.6), max(4, H*0.6)))
        created = True

    # ── 엣지(중복 제거) + 엣지에 실제 등장한 노드 수집 ──
    seen = set()
    nodes_with_edges = set()
    for u, nbrs in G.items():
        for v in nbrs:
            a, b = tuple(sorted([u, v]))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            (r1, c1), (r2, c2) = a, b
            ax.plot([c1, c2], [r1, r2], lw=1, color="k", alpha=0.6, zorder=1)
            nodes_with_edges.add(a); nodes_with_edges.add(b)

    # ── 노드: 엣지에 등장한 노드만(고립노드 제외). 단, start/goal은 원하면 보이게 유지 ──
    xs, ys, cs = [], [], []
    for r in range(H):
        for c in range(W):
            cell = (r, c)
            keep = (cell in nodes_with_edges) or (start == cell) or (goal == cell)
            if not keep:
                continue
            xs.append(c); ys.append(r); cs.append(color_of(grid[r][c]))
    ax.scatter(xs, ys, s=40, c=cs, edgecolors="k", linewidths=0.3, zorder=3)

    # # 경로 (원하면 주석 해제)
    # if path and len(path) >= 2:
    #     px = [c for (r, c) in path]; py = [r for (r, c) in path]
    #     ax.plot(px, py, linewidth=3, alpha=0.9, zorder=4)

    # start/goal
    if start:
        ax.scatter([start[1]],[start[0]], s=120, marker="*", c="#2ecc71",
                   edgecolors="k", linewidths=0.6, zorder=5)
    if goal:
        ax.scatter([goal[1]],[goal[0]], s=120, marker="*", c="#e74c3c",
                   edgecolors="k", linewidths=0.6, zorder=5)

    # ── 축/타이틀: 엣지 기준 노드/엣지 수 표기 ──
    ax.set_box_aspect(H/W)
    ax.set_xlim(-0.5, W-0.5); ax.set_ylim(-0.5, H-0.5)
    ax.invert_yaxis()
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title(f"Graph (Nodes={len(nodes_with_edges)}, Edges={len(seen)})", fontsize=11, pad=6)

    if created:
        plt.tight_layout(); plt.show()

def expand_vertices_with_modes_from_edges(
    v_groups, E, start_id, goal_id, *,
    labels=None, first_is_left=True,
    span_stride=0.35, max_per_group=None,
):
    n = len(v_groups)
    if n == 0:
        return [], [], [], []
    if n == 1:
        # 라벨 없어도 안전 처리 (start=goal=그룹0)
        v_ext = [v_groups[0].copy() for _ in range(3)]
        modes = ["D","SL","SR"]
        gmap  = [list(range(1,4))]
        return v_ext, modes, gmap, [3]

    E0 = np.asarray(E, int) - 1  # 0-based

    def _idx(x, side="first"):
        # 정수 처리: 0/1-based 모두 허용
        if isinstance(x, int):
            if 0 <= x < n:     return x  # 이미 0-based
            if 1 <= x <= n:    return x - 1  # 1-based
            raise ValueError(f"index out of range: {x}")

        # 라벨 처리: labels에서 위치 찾기
        if labels is None:
            raise ValueError("labels가 없어서 라벨→인덱스 매핑을 할 수 없습니다.")
        pos = [i for i, lab in enumerate(labels) if lab == x]
        if not pos:
            raise ValueError(f"라벨 {x} 이 labels에 없습니다.")
        return pos[0] if side == "first" else pos[-1]

    # 시작은 첫 발생, 목표는 마지막 발생을 사용
    s0 = _idx(start_id, side="first")
    t0 = _idx(goal_id, side="last")

    # --- 그룹 중심 ---
    centers = np.array([[0.25*r[0].sum(), 0.25*r[1].sum()] for r in v_groups], float)

    # --- 인접 리스트 ---
    outN = [[] for _ in range(n)]
    inN  = [[] for _ in range(n)]
    for u, w in E0:
        if 0 <= u < n and 0 <= w < n:
            outN[u].append(w); inN[w].append(u)

    # --- 전역 방향(백업) ---
    gdir = centers[t0] - centers[s0]
    gdir = gdir / (np.linalg.norm(gdir) + 1e-12)

    # --- 각 그룹 진행방향 = (out 이웃 or in 이웃) 평균벡터 ---
    dirs = np.zeros((n, 2), float)
    for g in range(n):
        neigh = outN[g] if outN[g] else inN[g]
        if neigh:
            d = (centers[neigh] - centers[g]).mean(axis=0)
            nrm = np.linalg.norm(d)
            dirs[g] = d / nrm if nrm > 1e-12 else gdir
        else:
            dirs[g] = gdir

    # --- copy_count 결정 ---
    copy_counts = []
    for g, rect in enumerate(v_groups):
        if span_stride is None:
            cnt = 3 if g in (s0, t0) else 2
        else:
            pts  = np.stack([rect[0], rect[1]], axis=1)  # (4,2)
            proj = pts @ dirs[g]
            span = float(proj.max() - proj.min())
            base = int(np.ceil(span / max(span_stride, 1e-9))) +4
            if g in (s0, t0):           # 시작/끝: 홀수 & 최소3
                cnt = max(7, base)
                if cnt % 2 == 0: cnt += 1
            else:                        # 중간: 짝수 & 최소2
                cnt = max(4, base)
                if cnt % 2 == 1: cnt += 1
        if max_per_group is not None:
            cnt = min(cnt, max_per_group)
        copy_counts.append(int(cnt))

    # --- 확장 + 모드 부여 ---
    v_ext, modes, gmap = [], [], []
    alt = ["SL", "SR"] if first_is_left else ["SR", "SL"]

    for g, (rect, k) in enumerate(zip(v_groups, copy_counts)):
        # 모드 시퀀스 만들기
        if g == s0 and g == t0:
            # (희귀) start==goal인 n==1 케이스는 위에서 이미 처리됨.
            mseq = ["D"] + [alt[i % 2] for i in range(max(0, k - 1))]
        elif g == s0:
            # 시작 그룹: D 1번만 맨 앞, 나머지는 SL/SR 번갈아
            mseq = ["D"] + [alt[i % 2] for i in range(k - 1)]
        elif g == t0:
            # 마지막 그룹: D 1번만 맨 뒤, 앞은 SL/SR 번갈아
            mseq = [alt[i % 2] for i in range(k - 1)] + ["D"]
        else:
            # 중간 그룹: 전부 SL/SR 번갈아 (짝수개)
            mseq = [alt[i % 2] for i in range(k)]

        # 실제 복제 + 기록
        idxs = []
        for i in range(k):
            v_ext.append(np.array(rect, copy=True))
            modes.append(mseq[i])
            idxs.append(len(v_ext))  # 1-based
        gmap.append(idxs)

    return v_ext, modes, gmap, copy_counts


def bfs_path(G: Dict[Coord, List[Coord]], s: Coord, t: Coord) -> List[Coord]:
    """무향 그래프 G에서 s→t 최단경로(BFS). 없으면 빈 리스트."""
    if s == t: return [s]
    q = deque([s])
    prev = {s: None}
    while q:
        u = q.popleft()
        for v in G.get(u, []):
            if v in prev:
                continue
            prev[v] = u
            if v == t:
                # 경로 복원
                path = [t]
                while path[-1] is not None:
                    path.append(prev[path[-1]])
                path.pop()               # 마지막 None 제거
                path.reverse()
                return path
            q.append(v)
    return []

def sample_start_goal_with_path_random(
    grid: List[List[str]],
    G: Dict[Coord, List[Coord]],
    *,
    floors_only: bool = True,
    floor_names: Tuple[str, str] = ("floor0", "floor1"),
    rng: Optional[random.Random] = None,
    max_trials: int = 10_000,
) -> Tuple[Coord, Coord, List[Coord]]:
    """
    - 후보 노드 집합에서 무작위로 (start, goal) 선택
    - BFS로 경로가 있으면 (start, goal, path) 반환, 없으면 재시도
    - floors_only=True면 floor0/floor1에서만 샘플링
    """
    rng = rng or random.Random()
    H, W = len(grid), (len(grid[0]) if grid else 0)
    F0, F1 = floor_names

    def is_floor(name: str) -> bool:
        return name == F0 or name == F1

    # 후보 노드 집합
    if floors_only:
        candidates = [(r, c) for r in range(H) for c in range(W) if is_floor(grid[r][c])]
    else:
        candidates = [(r, c) for r in range(H) for c in range(W)]
    # 그래프에 실제로 존재하는 노드만
    candidates = [p for p in candidates if p in G]
    if len(candidates) < 2:
        raise RuntimeError("샘플링 가능한 후보 노드가 2개 미만입니다.")

    for _ in range(max_trials):
        s = rng.choice(candidates)
        t = rng.choice(candidates)
        if s == t:
            continue
        path = bfs_path(G, s, t)
        if path:
            return s, t, path

    raise RuntimeError("연결 가능한 start/goal을 찾지 못했습니다. 맵/규칙을 확인하거나 max_trials를 늘리세요.")


# z=0 → white, z=h_step → gray
FLOOR_CMAP = LinearSegmentedColormap.from_list(
    "floors",
    [(0.0, (1.0, 1.0, 1.0)), (1.0, (0.70, 0.70, 0.70))],
)

# ─────────────────────────────────────────────────────────────────────────────
# 1) 타일/규칙 정의
# ─────────────────────────────────────────────────────────────────────────────
DIRS = ["N", "E", "S", "W"]
OPP  = {"N": "S", "S": "N", "E": "W", "W": "E"}

@dataclass(frozen=True)
class Tile:
    name: str
    weight: float = 1.0
    walkable: bool = True   # graph 구축시 통행 가능 여부

# 기본 타일 3종
F0 = Tile("floor0", weight=1.0, walkable=True)
F1 = Tile("floor1", weight=1.0, walkable=True)
# stair 변형 4종: 좌우/상하 이웃의 층 조합을 encode
SX_L0R1 = Tile("stair_x_L0R1", weight=1.4, walkable=True)
SX_L1R0 = Tile("stair_x_L1R0", weight=1.4, walkable=True)
SY_U0D1 = Tile("stair_y_U0D1", weight=1.4, walkable=True)
SY_U1D0 = Tile("stair_y_U1D0", weight=1.4, walkable=True)

STAIRS_X = {SX_L0R1.name, SX_L1R0.name}
STAIRS_Y = {SY_U0D1.name, SY_U1D0.name}


# 규칙 타입: rules[tile.name][dir] = {allowed_neighbor_tile_names, ...}
Rules = Dict[str, Dict[str, Set[str]]]

def make_default_rules() -> Rules:
    tiles = (F0, F1, SX_L0R1, SX_L1R0, SY_U0D1, SY_U1D0)
    names = {t.name for t in tiles}
    rules: Rules = {t.name: {d: set(names) for d in DIRS} for t in tiles}
    # ↑ 기본은 모든 방향에 모든 타일 허용(최소 제한)

    # ── 계단의 ‘필수 이웃’만 엄격히 설정 ──
    # stair_x_L0R1: W=F0, E=F1
    rules[SX_L0R1.name]["W"] = {F0.name}
    rules[SX_L0R1.name]["E"] = {F1.name}
    # stair_x_L1R0: W=F1, E=F0
    rules[SX_L1R0.name]["W"] = {F1.name}
    rules[SX_L1R0.name]["E"] = {F0.name}
    # stair_y_U0D1: N=F0, S=F1
    rules[SY_U0D1.name]["N"] = {F0.name}
    rules[SY_U0D1.name]["S"] = {F1.name}
    # stair_y_U1D0: N=F1, S=F0
    rules[SY_U1D0.name]["N"] = {F1.name}
    rules[SY_U1D0.name]["S"] = {F0.name}

    # ── 대칭성 보장(이웃 쪽에서도 되돌아보기 허용) ──
    # x: 왼쪽/오른쪽
    rules[F0.name]["E"].add(SX_L0R1.name)  # F0의 동쪽에 SX_L0R1 가능 (계단의 W=F0)
    rules[F1.name]["W"].add(SX_L0R1.name)
    rules[F1.name]["E"].add(SX_L1R0.name)  # 계단의 W=F1
    rules[F0.name]["W"].add(SX_L1R0.name)

    # y: 위/아래
    rules[F0.name]["S"].add(SY_U0D1.name)  # F0의 남쪽에 SY_U0D1 가능 (계단의 N=F0)
    rules[F1.name]["N"].add(SY_U0D1.name)
    rules[F1.name]["S"].add(SY_U1D0.name)  # 계단의 N=F1
    rules[F0.name]["N"].add(SY_U1D0.name)

    return rules


# ─────────────────────────────────────────────────────────────────────────────
# 2) Simple Tiled WFC (백트래킹 포함)
# ─────────────────────────────────────────────────────────────────────────────
# 상단에 같이 둬도 됨
FLOORS = {F0.name, F1.name}

class WFC:
    def __init__(self, width, height, tiles, rules, seed=None,
                 boundary="none",          # "none" | "floors_only" | "hints"
                 boundary_hints=None,
                 cluster_bias=None):      # {"left":F0.name, "right":F1.name, "top":None, "bottom":None} 등
        self.W, self.H = width, height
        self.tiles = {t.name: t for t in tiles}
        self.tile_list = tiles
        self.rules = rules
        self.rng = random.Random(seed)

        all_names = set(self.tiles.keys())
        self.domain = [[set(all_names) for _ in range(self.W)] for _ in range(self.H)]

        # ← 추가: 경계 설정 기억 & 한 번 적용
        self._boundary = boundary
        self._boundary_hints = boundary_hints or {}
        self._apply_boundary_once()
        self.cluster_bias = cluster_bias or {}
        self.weights = {t.name: t.weight for t in tiles}

    # ── 가장자리= floor0/1만 허용(또는 힌트) ──
    def _apply_boundary_once(self):
        if self._boundary == "none":
            return
        touched = []
        H, W = self.H, self.W

        # 어떤 모드인지 결정
        if self._boundary == "floors_only":
            def allowed_for(side): return FLOORS
        elif self._boundary == "hints":
            def allowed_for(side):
                val = self._boundary_hints.get(side, None)
                return FLOORS if val is None else {val}
        else:
            return

        # 위/아래 행
        top_allowed = allowed_for("top")
        bottom_allowed = allowed_for("bottom")
        for c in range(W):
            if not self.domain[0][c] <= top_allowed:
                self.domain[0][c].intersection_update(top_allowed); touched.append((0, c))
            if not self.domain[H-1][c] <= bottom_allowed:
                self.domain[H-1][c].intersection_update(bottom_allowed); touched.append((H-1, c))

        # 왼/오 열
        left_allowed = allowed_for("left")
        right_allowed = allowed_for("right")
        for r in range(H):
            if not self.domain[r][0] <= left_allowed:
                self.domain[r][0].intersection_update(left_allowed); touched.append((r, 0))
            if not self.domain[r][W-1] <= right_allowed:
                self.domain[r][W-1].intersection_update(right_allowed); touched.append((r, W-1))

        if touched:
            self.propagate(touched)

    # ── 리셋 시에도 자동 재적용 ──
    def reset_domains(self):
        all_names = set(self.tiles.keys())
        self.domain = [[set(all_names) for _ in range(self.W)] for _ in range(self.H)]
        self._apply_boundary_once()   # ← 한 줄로 끝

        r = self.rng.randrange(self.H); c = self.rng.randrange(self.W)
        pick = self.sample_tile(all_names)
        self.domain[r][c] = {pick}
        self.propagate([(r, c)])

    def neighbors(self, r: int, c: int) -> List[Tuple[int,int,str]]:
        out = []
        if r > 0:            out.append((r-1, c, "N"))
        if c < self.W-1:     out.append((r, c+1, "E"))
        if r < self.H-1:     out.append((r+1, c, "S"))
        if c > 0:            out.append((r, c-1, "W"))
        return out

    def entropy_key(self, r: int, c: int) -> Tuple[int, float]:
        # (후보 수, 무작위 tie-break) → 최소 엔트로피 셀 우선
        cand = self.domain[r][c]
        return (len(cand), self.rng.random())

    def observe(self) -> Optional[Tuple[int,int]]:
        # 아직 결정 안된 셀 중 최소 엔트로피 선택
        min_key = (10**9, 0.0)
        target = None
        for r in range(self.H):
            for c in range(self.W):
                if len(self.domain[r][c]) <= 1:
                    continue
                key = self.entropy_key(r, c)
                if key < min_key:
                    min_key, target = key, (r, c)
        return target

    def sample_tile(self, names: Set[str], r: int = None, c: int = None) -> str:
        arr = list(names)

        # 이웃 기반 보너스(클러스터링) 적용
        w = []
        for tname in arr:
            base = self.weights[tname]
            beta = self.cluster_bias.get(tname, 0.0)
            if r is None or c is None or beta == 0.0:
                w.append(base)
                continue

            # 이미 확정된 이웃 중 동일 타일 수를 센다
            same = 0
            for nr, nc, _ in self.neighbors(r, c):
                nbr_dom = self.domain[nr][nc]
                if len(nbr_dom) == 1 and next(iter(nbr_dom)) == tname:
                    same += 1

            # 보너스: exp(beta * same)
            w.append(base * float(np.exp(beta * same)))

        w = np.array(w, dtype=float)
        w = w / w.sum()
        return self.rng.choices(arr, weights=w, k=1)[0]

    def propagate(self, stack: List[Tuple[int,int]]):
        # AC-3 스타일 제약 전파
        while stack:
            r, c = stack.pop()
            cur = self.domain[r][c]
            for nr, nc, d in self.neighbors(r, c):
                allowed_neighbors = set()
                # 현재 셀의 후보 각각이 이 방향 d로 허용하는 이웃들의 합집합
                for t in cur:
                    allowed_neighbors.update(self.rules[t][d])
                # 이웃 셀 후보 중 허용되지 않는 것 제거
                before = len(self.domain[nr][nc])
                self.domain[nr][nc].intersection_update(allowed_neighbors)
                after = len(self.domain[nr][nc])

                if after == 0:
                    raise ValueError("Contradiction during propagation")

                if after < before:
                    stack.append((nr, nc))


    def collapse(self) -> List[List[str]]:
        # 백트래킹용 스냅샷 스택
        snapshots: List[Tuple[List[List[Set[str]]], Tuple[int,int], str]] = []

        while True:
            target = self.observe()
            if target is None:
                # 모두 확정
                break

            # collapse() 내부의 관측/선택 부분만 교체
            r, c = target
            cand = self.domain[r][c]
            snapshot = [[set(s) for s in row] for row in self.domain]

            # 1) 가중치 기반 첫 선택
            first = self.sample_tile(cand,r,c)  # ← weights 사용
            # 2) 실패 시를 위해 나머지 후보들 준비(순서는 셔플)
            rest = [x for x in cand if x != first]
            self.rng.shuffle(rest)
            try_choice_order = [first] + rest

            success = False
            for choice in try_choice_order:
                self.domain = [[set(s) for s in row] for row in snapshot]
                self.domain[r][c] = {choice}
                try:
                    self.propagate([(r, c)])
                    success = True
                    break
                except ValueError:
                    continue

            if not success:
                self.reset_domains()

        # 결과 맵
        return [[next(iter(self.domain[r][c])) for c in range(self.W)]
                for r in range(self.H)]

    def reset_domains(self):
        # 전역 초기화 후 랜덤 한두 곳을 강제 관측하여 다양성 확보
        all_names = set(self.tiles.keys())
        self.domain = [[set(all_names) for _ in range(self.W)] for _ in range(self.H)]
        # 한 점을 임의 타일로 고정해서 다시 시작(무한루프 방지에 도움)
        r = self.rng.randrange(self.H)
        c = self.rng.randrange(self.W)
        pick = self.sample_tile(all_names,r,c)
        self.domain[r][c] = {pick}
        self.propagate([(r, c)])


# ─────────────────────────────────────────────────────────────────────────────
# 4) 시각화 및 예시 실행
# ─────────────────────────────────────────────────────────────────────────────
# ── 색상 매핑(요청 색상) ──
COLOR = {
    "floor0": (1.0, 1.0, 1.0),      # 흰색
    "floor1": (0.75, 0.75, 0.75),      # 회색

    # 계단: 붉은 계열
    # stair_x = 초록빛 붉은색(붉은 계열에 green 성분 살짝)
    "stair_x_L0R1": (1.0, 0.4, 0.4),  # 약간 코럴 느낌
    "stair_x_L1R0": (1.0, 0.4, 0.4),

    # stair_y = 오렌지빛 붉은색
    "stair_y_U0D1": (1.0, 0.65, 0.0),  # 오렌지 레드
    "stair_y_U1D0": (1.0, 0.65, 0.0),
}

def color_of(tile_name: str):
    return COLOR.get(tile_name, (0.5, 0.5, 0.5))

def plot_wfc_terrain(
    grid,
    ax=None,
    *,
    legend_outside=True,
    legend_fontsize=8,
    legend_ncol=1,
    legend_alpha=0.8
):
    H, W = len(grid), len(grid[0])
    img = np.zeros((H, W, 3), float)
    for r in range(H):
        for c in range(W):
            img[r, c] = color_of(grid[r][c])

    created = False
    if ax is None:
        # 레이아웃 엔진 끔: subplots_adjust가 확실히 적용되게
        fig, ax = plt.subplots(figsize=(max(4, W*0.6), max(4, H*0.6)))
        created = True
    else:
        fig = ax.figure

    ax.imshow(img, interpolation="nearest", origin="upper")
    ax.set_xticks([]); ax.set_yticks([])

    legend_handles = [
        mpatches.Patch(color=COLOR["floor0"], label="ground floor"),
        mpatches.Patch(color=COLOR["floor1"], label="1st floor"),
        mpatches.Patch(color=COLOR["stair_x_L0R1"], label="stair x direction"),
        mpatches.Patch(color=COLOR["stair_y_U0D1"], label="stair y direction"),
    ]

    if legend_outside:
        # ★ 레이아웃 엔진을 none으로 바꾸고 → subplots_adjust 적용
        fig.set_layout_engine('none')      # ← 이 줄 추가
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True, framealpha=legend_alpha,
            fontsize=legend_fontsize, ncol=legend_ncol,
            borderaxespad=0.0
        )
    else:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True, framealpha=legend_alpha,
            fontsize=legend_fontsize, ncol=legend_ncol
        )

    ax.set_title(f"Wave Function Collapse({H}×{W})", fontsize=11, pad=6)

    if created:
        # plt.tight_layout()  # ← 호출하지 마세요(경고 원인)
        plt.show()




# === 1) heightmap 생성 (단일 모드: 셀 중심 샘플) ===

def build_heightmap(grid, *, tile_size=1.0, h_step=0.20, samples=12, stair_step_height=0.10):
    """
    grid: HxW 타일 문자열 2D 리스트
    tile_size: 타일 한 변 길이 [m]
    h_step: floor0↔floor1 전체 높이차 [m]  (예: 0.2)
    samples: 타일을 몇 등분해 샘플링할지(렌더 해상도)
    stair_step_height: 계단 한 '단' 높이 [m]  (예: 0.1 → 0, 0.1, 0.2처럼 계단 표시)
    """
    H, W = len(grid), len(grid[0])

    # 계단 단수 계산
    n_risers = max(1, int(round(h_step / max(1e-9, stair_step_height))))  # 예: 0.2/0.1=2
    n_levels = n_risers + 1  # 예: 3 (0, 0.1, 0.2)

    # 셀 중심 샘플
    HH, WW = H * samples, W * samples
    xs = np.linspace(0, W * tile_size, WW, endpoint=False) + (tile_size / (2 * samples))
    ys = np.linspace(0, H * tile_size, HH, endpoint=False) + (tile_size / (2 * samples))
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros((HH, WW), dtype=float)

    # 타일 내부 정규 좌표(0..1) - 셀 중심
    u = np.linspace(0.0, 1.0, samples, endpoint=False) + 0.5 / samples
    U_tile, V_tile = np.meshgrid(u, u)  # U:x(좌→우), V:y(위→아래)

    # 계단 양자화 도우미: U∈[0,1) → {0,1,...,n_levels-1} → [0..1]로 맵핑
    def quantize_01(U):
        k = np.floor(U * n_levels).astype(int)              # 0..n_levels-1
        k = np.clip(k, 0, n_levels - 1)
        denom = max(1, n_levels - 1)
        return k / denom                                    # 0, 1/(L-1), ..., 1

    for r in range(H):
        for c in range(W):
            t = grid[r][c]
            rr = slice(r * samples, (r + 1) * samples)
            cc = slice(c * samples, (c + 1) * samples)

            if t == "floor0":
                Z[rr, cc] = 0.0

            elif t == "floor1":
                Z[rr, cc] = h_step

            elif t == "stair_x_L0R1":
                frac = quantize_01(U_tile)                  # 0 → 1 (좌→우)
                Z[rr, cc] = h_step * frac

            elif t == "stair_x_L1R0":
                frac = quantize_01(U_tile)                  # 1 → 0 (좌→우)
                Z[rr, cc] = h_step * (1.0 - frac)

            elif t == "stair_y_U0D1":
                frac = quantize_01(V_tile)                  # 0 → 1 (위→아래)
                Z[rr, cc] = h_step * frac

            elif t == "stair_y_U1D0":
                frac = quantize_01(V_tile)                  # 1 → 0 (위→아래)
                Z[rr, cc] = h_step * (1.0 - frac)

            else:
                Z[rr, cc] = 0.0

    return X, Y, Z


def plot_compact_graph(
    grid,
    G,
    node_info,
    *,
    start_id=None,
    goal_id=None,
    draw_node_labels=False,
    ax=None,
):
    H = len(grid); W = len(grid[0]) if H else 0

    # ── 통일 팔레트 ──
    rect_face = {"floor0": "#FFFFFF", "floor1": "#C0C0C0"}   # 면색: floor0=흰, floor1=회
    rect_edge = {"floor0": "#333333", "floor1": "#333333"}   # 테두리: 동일하게 어두운 회색
    def stair_color_for(name: str) -> str:
        if name.startswith("stair_x"): return "#FF6666"  # 연한 빨강
        if name.startswith("stair_y"): return "#FFA500"  # 주황
        return "#999999"  # fallback

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, W*0.6), max(4, H*0.6)))
        created = True
    else:
        ax.cla()

    # ── (0) 그릴 노드 집합: 넘겨준 G 기준 ──
    nodes = set(G.keys())
    for u, nbrs in G.items():
        for v in nbrs:
            nodes.add(v)

    # ── (1) 노드 중심 좌표 ──
    pos = {}
    for nid in nodes:
        meta = node_info.get(nid)
        if not meta: continue
        if meta["type"] == "rect":
            r0, c0, h, w = meta["r0"], meta["c0"], meta["h"], meta["w"]
            pos[nid] = (c0 + (w - 1)/2.0, r0 + (h - 1)/2.0)
        elif meta["type"] == "stair":
            pos[nid] = (float(meta["c"]), float(meta["r"]))

    # ── (2) 엣지 ──
    seen = set()
    for u in nodes:
        for v in G.get(u, []):
            if u == v:
                continue
            a, b = (u, v) if u <= v else (v, u)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            if a in pos and b in pos:
                x1, y1 = pos[a]; x2, y2 = pos[b]
                ax.plot([x1, x2], [y1, y2], lw=2, alpha=0.7, color="#555", zorder=2)

    # ── (3) 사각형 노드(면/테두리 색 통일) ──
    for nid in nodes:
        meta = node_info.get(nid)
        if not meta or meta["type"] != "rect":
            continue
        r0, c0, h, w = meta["r0"], meta["c0"], meta["h"], meta["w"]
        floor = meta.get("floor", "floor0")
        fc = rect_face.get(floor, "#DDDDDD")
        ec = rect_edge.get(floor, "#333333")
        rect = patches.Rectangle((c0 - 0.5, r0 - 0.5), w, h,
                                 fill=True, facecolor=fc, edgecolor=ec,
                                 linewidth=2.0, alpha=0.8, zorder=3)
        ax.add_patch(rect)

    # ── (4) 계단 노드(타일명으로 색 결정) ──
    for nid in nodes:
        meta = node_info.get(nid)
        if not meta or meta["type"] != "stair":
            continue
        r, c = meta["r"], meta["c"]
        sc = stair_color_for(meta.get("name", ""))
        stair = patches.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0,
                                  fill=True, facecolor=sc,
                                  edgecolor="k", linewidth=1.0, alpha=0.8, zorder=4)
        ax.add_patch(stair)

    # ── 축/타이틀 ──
    ax.set_box_aspect(H/W)
    ax.set_xlim(-0.5, W-0.5); ax.set_ylim(-0.5, H-0.5)
    ax.invert_yaxis()
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title(f"Graph (Nodes={len(nodes)}, Edges={len(seen)})", fontsize=11, pad=6)

    if created:
        plt.tight_layout(); plt.show()


def compute_Gk_and_node_info_for_gcs(
    grid: List[List[str]],
    G: Dict[Tuple[int,int], List[Tuple[int,int]]],
    start: Tuple[int,int],
    goal: Tuple[int,int],
) -> Dict[str, Any]:
    """
    WFC 파이프라인에서 compact/pruned 그래프(Gk)와 node_info를 만들어
    GCS 브리지가 바로 쓸 수 있도록 반환.
    반환: {"grid": grid, "Gk": Gk, "node_info": node_info, "start": start, "goal": goal}
    """
    # --- 네 기존 코드 그대로 재사용 ---
    G_new = build_reachable_subgraph(G, start)
    node_info, cell_to_node = make_floor_rects_from_Gnew(
        grid, G_new, floor_names=("floor0", "floor1"),
        tile_size=1.0,  # 네가 쓰는 값
        floor_z={"floor0": 0.0, "floor1": 0.2},  # 네 파이프라인과 일치
        y_up=True,
    )
    G_compact = compact_graph_from_rects(grid, G_new, node_info, cell_to_node)
    start_id = map_point_to_compact_node(cell_to_node, start)
    goal_id  = map_point_to_compact_node(cell_to_node, goal)

    Gk = prune_to_st_kernel_undirected(G_compact, start_id, goal_id)
    Gk = trim_leaves_iterative_undirected(Gk, start_id, goal_id)

    return {
        "grid": grid,
        "Gk": Gk,
        "node_info": node_info,
        "start": start,
        "goal": goal,
    }



def overlay_start_goal_on_ax(ax, start, goal, *, use_tile_centers=False, tile_size=1.0, z=999):
    """어떤 Axes든 s,t(격자좌표)를 그대로 찍어준다."""
    if use_tile_centers:
        sx, sy = (start[1] + 0.5) * tile_size, (start[0] + 0.5) * tile_size
        gx, gy = (goal[1] + 0.5) * tile_size, (goal[0] + 0.5) * tile_size
    else:
        sx, sy = start[1], start[0]
        gx, gy = goal[1], goal[0]

    ax.scatter([sx], [sy], s=120, marker="*", c="#2ecc71",
               edgecolors="k", linewidths=0.6, zorder=z, label="start(orig)")
    ax.scatter([gx], [gy], s=140, marker="*", c="#e74c3c",
               edgecolors="k", linewidths=0.6, zorder=z, label="goal(orig)")

@dataclass
class WfcForGcs:
    stones: List[Dict[str, Any]]
    copy_counts: List[int]
    grid: List[List[str]]
    meta: Dict[str, Any]
    start: Optional[Tuple[int, int]] = None
    goal: Optional[Tuple[int, int]] = None

@dataclass
class Footsteps:
    # 각 프레임/스텝마다 하나씩 (1D np.array)
    x_L: np.ndarray; y_L: np.ndarray; z_L: np.ndarray; c_L: np.ndarray; s_L: np.ndarray
    x_R: np.ndarray; y_R: np.ndarray; z_R: np.ndarray; c_R: np.ndarray; s_R: np.ndarray
    first_is_left: bool = True

def _rect_corners_xy(x: float, y: float, c: float, s: float, h: float = 0.20, w: float = 0.10):
    """
    (x,y) 중심, yaw 벡터(c,s)에 따른 직사각형 4점(xy) 반환 (시계/반시계 상관 없음)
    """
    # 단위화(안정화)
    n = max(1e-12, float(np.hypot(c, s)))
    cc, ss = c / n, s / n
    # 발 길이방향을 [cc, ss], 너비방향을 [-ss, cc]로 둠
    f = 0.5 * h
    g = 0.5 * w
    ex = cc; ey = ss
    nx = -ss; ny = cc
    p0 = (x - f * ex - g * nx, y - f * ey - g * ny)
    p1 = (x + f * ex - g * nx, y + f * ey - g * ny)
    p2 = (x + f * ex + g * nx, y + f * ey + g * ny)
    p3 = (x - f * ex + g * nx, y - f * ey + g * ny)
    return [p0, p1, p2, p3]

def make_wfc_terrain_for_gcs(
    *,
    W: int = 20, H: int = 12,
    seed: Optional[int] = 0,
    tile_size: float = 1.0,
    floors_only: bool = True,
    floor_names: Tuple[str, str] = ("floor0", "floor1"),
    boundary: str = "floors_only",
    cluster_bias: Optional[Dict[str, float]] = None,
    max_trials: int = 10000,
) -> WfcForGcs:
    """
    1) WFC로 grid 생성 → 2) 탐색 그래프/압축 그래프/Gk → 3) GCS용 stones/copy_counts 생성
    반환: GCS 파이프라인에 바로 넣을 수 있는 stones/copy_counts와 플롯용 grid/meta
    """
    tiles = [F0, F1, SX_L0R1, SX_L1R0, SY_U0D1, SY_U1D0]  # 네 파일 상단의 타일 정의 사용
    rules = make_default_rules()
    wfc = WFC(W, H, tiles, rules, seed=seed, boundary=boundary,
              cluster_bias=(cluster_bias or {}))
    grid = wfc.collapse()

    # 그래프 구성
    G = build_nav_graph_simple(grid)
    s, t, p = sample_start_goal_with_path_random(
        grid, G,
        floors_only=floors_only,
        floor_names=floor_names,
        rng=random.Random(seed if seed is not None else 0),
        max_trials=max_trials,
    )

    G_new = build_reachable_subgraph(G, s)
    node_info, cell_to_node = make_floor_rects_from_Gnew(grid, G_new, floor_names=floor_names)
    G_compact = compact_graph_from_rects(grid, G_new, node_info, cell_to_node)
    start_id = map_point_to_compact_node(cell_to_node, s)
    goal_id  = map_point_to_compact_node(cell_to_node, t)
    Gk = prune_to_st_kernel_undirected(G_compact, start_id, goal_id)
    Gk = trim_leaves_iterative_undirected(Gk, start_id, goal_id)

    # ★ 핵심: GCS가 바로 먹을 수 있는 stones/copy_counts 생성
    # 1) 정확한 시그니처에 맞춰 호출 (전부 keyword-only)
    T = build_terrain_from_WFC_outputs(
        grid=grid,
        Gk=Gk,
        node_info=node_info,
        start=s,
        goal=t,
        tile_size=tile_size,
        # 필요시 아래 옵션만 네 상황에 맞게 켜기/끄기
        y_up=True,
        include_stairs=False,
        stair_z_mode="mid",
        add_initial_goal=True,
        start_is_grid=True,
        init_goal_size=0.6,
        name="WFC_terrain",
    )

    # 2) 반환 정규화 (dict(T형식) 기준)
    if not isinstance(T, dict):
        raise TypeError("[build_terrain_from_WFC_outputs] must return a dict-like terrain object.")

    stones = T.get("stones")
    copy_counts = T.get("copy_counts")
    meta = {
        "name": T.get("name", "WFC_terrain"),
        "tile_size": T.get("tile_size", tile_size),
    }

    if stones is None:
        raise KeyError("[build_terrain_from_WFC_outputs] returned dict missing key 'stones'.")

    # copy_counts 없으면 1로 채움 (build_graph가 요구)
    if copy_counts is None:
        copy_counts = [1] * len(stones)

    return WfcForGcs(
        stones=stones,
        copy_counts=copy_counts,
        grid=grid,
        meta=meta,
        start=s,
        goal=t,
    )

def plot_heightmap_3d_with_footsteps(
    grid,
    footsteps: Footsteps,
    *,
    tile_size: float = 1.0,
    start: Optional[Tuple[int,int]] = None,
    goal: Optional[Tuple[int,int]] = None,
    marker_size_ratio: float = 0.35,
    marker_lift: float = 0.10,
    foot_h: float = 0.20,    # 발 길이
    foot_w: float = 0.10,    # 발 너비
    elev: float = 50, azim: float = 240,
):
    """
    기존 plot_heightmap_3d 위에 GCS에서 나온 발 사각형을 3D로 얹어 보여줌.
    """
    # 3D 지형
    plot_heightmap_3d(None, None, None,
                      tile_grid=grid,
                      tile_size=tile_size,
                      h_step=0.2, stair_slices=4,
                      elev=elev, azim=azim,
                      start=start, goal=goal,
                      marker_size_ratio=marker_size_ratio, marker_lift=marker_lift)

    ax = plt.gca(projection='3d')

    def _add_foot_quads(x, y, z, c, s, color='C1'):
        n = int(np.asarray(x).size)
        for i in range(n):
            pts_xy = _rect_corners_xy(float(x[i]), float(y[i]),
                                      float(c[i]), float(s[i]),
                                      h=foot_h, w=foot_w)
            # z는 상수(해당 발 z) + 살짝 띄우기
            z0 = float(z[i]) + marker_lift
            verts = [[(px, py, z0) for (px, py) in pts_xy]]
            poly = Poly3DCollection(verts, alpha=0.85)
            poly.set_edgecolor('k'); poly.set_linewidth(0.5)
            poly.set_facecolor(color)
            ax.add_collection3d(poly)

    # 좌/우 순서대로 그리기 (시퀀스 길이가 다르면 min 길이 사용)
    _add_foot_quads(footsteps.x_L, footsteps.y_L, footsteps.z_L, footsteps.c_L, footsteps.s_L, color='C1')
    _add_foot_quads(footsteps.x_R, footsteps.y_R, footsteps.z_R, footsteps.c_R, footsteps.s_R, color='C0')

    plt.draw()


if __name__ == "__main__":
    # 예시: 32x20 맵 생성
    W, H = 5, 4
    tiles = [F0, F1, SX_L0R1, SX_L1R0, SY_U0D1,SY_U1D0]
    rules = make_default_rules()  # WFC 생성 규칙
    wfc = WFC(W, H, tiles, rules, seed=None, boundary="floors_only",cluster_bias={"floor1":0.7})
    grid = wfc.collapse()

    # 1) 그래프 생성
    G = build_nav_graph_simple(grid)

    # 2) start/goal/path 샘플링 (바닥에서만 뽑기)
    s, t, p = sample_start_goal_with_path_random(
        grid, G,
        floors_only=True,  # 계단까지 포함하려면 False
        floor_names=("floor0", "floor1"),
        rng=random.Random(42),
        max_trials=10000,
    )

    print("start:", s, "goal:", t, "path_len:", len(p))
    fig, axs = plt.subplots(3, 2, figsize=(8,12), constrained_layout=True)
    gs = axs[0, 0].get_gridspec()

    # 첫 행의 두 축 제거
    for ax in axs[0, :]:
        ax.remove()

    # 첫 행 전체를 차지하는 새 축 생성
    ax_top = fig.add_subplot(gs[0, :])

    plot_wfc_terrain(grid,ax=ax_top)


    # 3) 플롯으로 확인
    plot_nav_graph(grid, G, path=p, start=s, goal=t, ax=axs[1,0])

    G_new = build_reachable_subgraph(G,s)

    print("G_new nodes:", len(G_new))
    plot_nav_graph(grid, G_new, path=p, start=s, goal=t, ax=axs[1,1])

    # 1) floor를 직사각형으로 병합하고, 계단은 단일 노드로 추가
    node_info, cell_to_node = make_floor_rects_from_Gnew(grid, G_new, floor_names=("floor0", "floor1"))

    # 2) 원래 G_new의 엣지를 기준으로 압축 그래프 구성
    G_compact = compact_graph_from_rects(grid, G_new, node_info, cell_to_node)

    # 3) start/goal이 속한 새 노드 ID
    start_id = map_point_to_compact_node(cell_to_node, s)
    goal_id = map_point_to_compact_node(cell_to_node, t)

    print("compact nodes:", len(G_compact))
    print("start_id:", start_id, "goal_id:", goal_id)

    # G_compact, start_id, goal_id 준비돼 있다고 가정
    Gk = prune_to_st_kernel_undirected(G_compact, start_id, goal_id)
    Gk = trim_leaves_iterative_undirected(Gk, start_id, goal_id)
    print("nodes after pruning:", len(Gk))
    print(Gk)

    # G_new, node_info, cell_to_node, G_compact, start_id, goal_id 가 준비되었다고 가정
    plot_compact_graph(
        grid,
        G_compact,
        node_info,
        draw_node_labels=True,  # 원하면 노드 ID도 표시
        ax=axs[2,0]
    )
    overlay_start_goal_on_ax(axs[2,0],s,t, use_tile_centers=False)

    # G_new, node_info, cell_to_node, G_compact, start_id, goal_id 가 준비되었다고 가정
    plot_compact_graph(
        grid,
        Gk,
        node_info,
        draw_node_labels=True,  # 원하면 노드 ID도 표시
        ax=axs[2,1]
    )
    overlay_start_goal_on_ax(axs[2,1],s,t, use_tile_centers=False)


    # cnt = Counter(t for row in grid for t in row)
    # print(cnt)  # {'floor0': ..., 'floor1': ..., 'stair_x': ..., 'stair_y': ...}

    # 0) order 만들기
    order_raw = build_label_order_tree_from_start(Gk, start=start_id, goal=goal_id)
    label_order = normalize_label_order(order_raw)  # ★ 평탄화된 리스트
    # ★ 0.5) 여기서 재정렬!
    label_order = reorder_label_order_for_front_to_back(
        Gk, label_order, start=start_id, goal=goal_id
    )

    # 1) v 생성
    pack = build_G_GCS_vertices_strict_with_order(
        node_info, label_order, tile_size=1.0, floor_z={"floor0": 0.0, "floor1": 0.2}
    )

    # 2) E 생성: v와 동일한 label_order로!
    E = build_G_GCS_edges_from_Gk_directed(
        Gk, label_order, start_label=start_id, goal_label=goal_id
    )

    # 3) 확장 + 모드
    v, mode_list, group_map, copy_counts = expand_vertices_with_modes_from_edges(
        pack["v"], E,
        start_id=start_id,
        goal_id=goal_id,
        labels=pack["labels"],  # ★ pack이 준 labels 사용 (v와 1:1)
        first_is_left=True,
        span_stride=1.0,  # 영역 크기 기반 복제: 필요 없으면 None
    )

    # ===== 1) 그룹(확장 전) 덤프 =====
    labels = pack["labels"]  # v와 1:1
    v_groups = pack["v"]

    print(f"\n=== DUMP: vertex GROUPS (G={len(v_groups)}) ===")
    for gi, (lb, rect) in enumerate(zip(labels, v_groups), start=1):
        print(f"[g {gi:>4}] label={lb}")
        print(rect)  # 3x4
    print("=================================")

    # ===== 2) 그룹 간 엣지 덤프 =====
    E_arr = np.asarray(E, dtype=int)  # E는 1-based라고 가정
    print(f"\n=== DUMP: GROUP EDGES (E_g={E_arr.shape[0]}) ===")
    for ei, (u, w) in enumerate(E_arr, start=1):
        lu, lw = labels[u - 1], labels[w - 1]  # 1-based → 0-based
        print(f"[ge {ei:>4}] {u} -> {w}   {lu} -> {lw}")
    print("=================================")

    # ===== 3) 확장된 정점 + 모드 덤프 =====
    # expand 결과: v, mode_list, group_map, copy_counts 가 준비되어 있다고 가정
    V = len(v)

    # (편의) 확장 정점 → 원본 그룹 번호 역매핑 (1-based)
    group_of = np.zeros(V, dtype=int)
    for g, idxs in enumerate(group_map, start=1):
        for idx in idxs:  # idx는 1-based
            group_of[idx - 1] = g

    print(f"\n=== DUMP: expanded vertices (V={V}) ===")
    for i, vert in enumerate(v, start=1):
        g = group_of[i - 1]  # 1-based 그룹 인덱스
        lb = labels[g - 1]  # 해당 그룹의 라벨
        md = mode_list[i - 1]
        print(f"[v {i:>4}] group={g} label={lb} mode={md}")
        print(vert)  # 3x4
    print("copy_counts by group:", [len(ix) for ix in group_map])
    for g, idxs in enumerate(group_map, start=1):
        print(f"group {g:>2} -> {len(idxs)} copies: {idxs}")
    print("=================================\n")

    # s0/t0: 그룹 인덱스(0-based) 계산
    start_group = labels.index(start_id)  # 첫 발생
    goal_group = len(labels) - 1 - labels[::-1].index(goal_id)  # 마지막 발생

    # 확장 이후: v, mode_list, group_map (이미 있음)
    edge_expanded = build_expanded_edges_from_groups(
        group_map, mode_list, E,
        start_group=start_group,
        goal_group=goal_group,
    )

    print(f"\n=== DUMP: expanded edges (E={edge_expanded.shape[0]}) ===")
    for i, (u, w) in enumerate(edge_expanded, 1):
        print(f"[ee {i:>4}] {u} -> {w}")


    X, Y, Z = build_heightmap(grid, tile_size=1.0, h_step=0.2, samples=4)
    # 3D: 더 이상 X,Y,Z 필요 없음 (호환성 위해 None 넘겨도 됨)
    plot_heightmap_3d(None, None, None,
                      tile_grid=grid,
                      tile_size=1.0, h_step=0.2, stair_slices=4,
                      elev=50, azim=240,
                      start=s, goal=t,
                      marker_size_ratio=0.35, marker_lift=0.10)

    plt.ioff()
    plt.show()



