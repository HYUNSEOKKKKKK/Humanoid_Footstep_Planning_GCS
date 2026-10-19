from __future__ import annotations
from typing import Dict, List, Tuple, Any, Union, Set
import numpy as np


Label = Tuple[str, Any]  # 예: ('rect', 2) 또는 ('stair', (r,c))

def _ensure_list4_pts3(corners: Any) -> List[List[float]]:
    arr = np.asarray(corners, dtype=float)
    if arr.shape == (3, 4):
        arr = arr.T
    if arr.shape != (4, 3):
        raise ValueError(f"rect corners must be 4x3 or 3x4, got shape {arr.shape}")
    return arr.tolist()

def _to_3x4(corners_4x3: List[List[float]]) -> np.ndarray:
    arr = np.asarray(corners_4x3, dtype=float)  # (4,3)
    return arr.T  # (3,4)

def _corners_from_center_wh(center3: List[float], width: float, height: float) -> List[List[float]]:
    x, y, z = map(float, center3[:3])
    hx, hy = 0.5 * float(width), 0.5 * float(height)
    bl = [x - hx, y - hy, z]
    tl = [x - hx, y + hy, z]
    tr = [x + hx, y + hy, z]
    br = [x + hx, y - hy, z]
    return [bl, tl, tr, br]

def _extract_rect3d_for_rect(idx: int, node_info: Union[List[dict], Dict[Any, dict]],
                             *, default_z: float = 0.0) -> List[List[float]]:
    info = node_info[idx] if isinstance(node_info, list) else node_info[idx]

    if "rect_3d" in info:
        return _ensure_list4_pts3(info["rect_3d"])

    if "rect" in info:
        rect2 = np.asarray(info["rect"], dtype=float)
        if rect2.shape == (2, 4):
            rect2 = rect2.T
        if rect2.shape != (4, 2):
            raise ValueError(f"node_info[{idx}]['rect'] must be 4x2 or 2x4, got {rect2.shape}")
        z = float(info.get("z", default_z))
        rect3 = np.hstack([rect2, np.full((4, 1), z, float)])
        return rect3.tolist()

    if "center" in info and ("width" in info or "w" in info) and ("height" in info or "h" in info):
        c = info["center"]
        if len(c) == 2:
            c = [float(c[0]), float(c[1]), float(info.get("z", default_z))]
        else:
            c = [float(c[0]), float(c[1]), float(c[2])]
        w = float(info.get("width", info.get("w")))
        h = float(info.get("height", info.get("h")))
        return _corners_from_center_wh(c, w, h)

    raise KeyError(f"node_info[{idx}] must contain rect_3d or rect or center+width+height.")

def _rect3d_for_stair(rc: Tuple[int, int], *, tile_size: float, z_mid: float) -> List[List[float]]:
    r, c = int(rc[0]), int(rc[1])
    x = (c + 0.5) * float(tile_size)
    y = (r + 0.5) * float(tile_size)
    w = h = float(tile_size)
    return _corners_from_center_wh([x, y, z_mid], w, h)

def _as_label(x: Any) -> Label:
    """('rect',i)/('stair',(r,c)) 같은 라벨이 오면 그대로, 숫자면 ('rect',숫자)로 변환."""
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str):
        return x
    return ("rect", int(x))

def _label_key(lb):
    t, v = lb
    return (t, *(v if isinstance(v, tuple) else (int(v),)))

Label = Tuple[str, Any]  # ('rect', i) or ('stair', (r,c))




def build_label_order_tree_from_start(
    Gk: Dict[Label, List[Label]],
    *,
    start: Label,
    goal: Label | None = None,
) -> List[Label]:
    """
    규칙:
      - v1 = start
      - seeds = sorted(neigh(start)) 에서 goal 제외
      - 각 seed에 대해: seed를 append하고, goal을 완전히 무시한 DFS로 seed 서브트리를 끝까지 확장
      - 마지막에 goal만 append
      - start == goal 이면 [start, goal] (최소 2개 보장)
    """
    if start not in Gk:
        raise KeyError("start label not in Gk")

    # 최소 2개 보장
    if goal is not None and start == goal:
        return [start, goal]

    visited: Set[Label] = {start}
    ordered: List[Label] = [start]

    # 1) start의 이웃 정렬 (goal 제외)
    neigh0 = sorted(Gk.get(start, []), key=_label_key)
    seeds = [v for v in neigh0 if (goal is None or v != goal)]

    def expand(u: Label):
        # goal은 traversal에서 완전히 제외
        for w in sorted(Gk.get(u, []), key=_label_key):
            if (goal is not None) and (w == goal):
                continue
            if w in visited:
                continue
            ordered.append(w)
            visited.add(w)
            expand(w)

    # 2) seed를 하나씩 append → seed 서브트리를 DFS로 확장
    for seed in seeds:
        if seed not in visited:
            ordered.append(seed)
            visited.add(seed)
            expand(seed)

    # 3) goal은 맨 뒤
    if goal is not None:
        ordered.append(goal)

    return ordered



def build_G_GCS_vertices_strict_with_order(
    node_info: Dict[Label, dict],
    label_order: List[Label],
    *,
    tile_size: float = 1.0,
    floor_z: dict | None = None,
) -> dict:
    if floor_z is None:
        floor_z = {"floor0": 0.0, "floor1": 0.2}
    z_mid = 0.5 * (float(floor_z["floor0"]) + float(floor_z["floor1"]))

    v_list: List[np.ndarray] = []
    for lb in label_order:
        t, val = lb
        if t == "rect":
            rect3 = np.asarray(node_info[lb]["rect_3d"], float)
            if rect3.shape == (4, 3):
                rect3 = rect3.T
            v_list.append(rect3)
        elif t == "stair":
            r, c = val
            x = (c + 0.5) * tile_size; y = (r + 0.5) * tile_size
            w = h = tile_size; hx, hy = 0.5*w, 0.5*h
            rect4x3 = np.array([
                [x - hx, y - hy, z_mid],
                [x - hx, y + hy, z_mid],
                [x + hx, y + hy, z_mid],
                [x + hx, y - hy, z_mid],
            ], float)
            v_list.append(rect4x3.T)
        else:
            raise ValueError(f"Unknown label type: {t}")

    # 호환성 위해 둘 다 넣어줌
    return {"v": v_list, "labels": list(label_order), "label_order": list(label_order)}



def expand_vertices_with_modes(
    v_groups: List[np.ndarray],
    copy_counts: List[int] = None,
    *,
    first_is_left: bool = True,
) -> Dict[str, Any]:
    """
    입력:
      - v_groups: build_G_GCS_vertices_strict_with_order(...)가 만든 원본 v (길이 = 그룹 수)
                  각 원소는 (3x4) ndarray (사각형 좌표)
      - copy_counts: 그룹별 복제 개수. None이면 [3] + [2]*(n-2) + [3] 규칙 적용
      - first_is_left: 중간 그룹의 2개 모드 순서 기본값(SL, SR)을 바꿔야 하면 False로

    동작:
      - 첫 그룹과 마지막 그룹은 ['D', 'SL', 'SR'] 3개로 확장
      - 중간 그룹은 ['SL', 'SR'] 2개로 확장
      - copy_counts가 주어지면 해당 개수만큼 생성하되, 모드 패턴을 순환(cycle)하여 채움

    반환:
      {
        "v": v_ext (list of (3x4) ndarray),
        "mode_list": list[str]  # 각 v_ext와 1:1
        "group_map": List[List[int]]  # 원본 그룹 g -> 확장 후 인덱스(1-based) 목록
        "vertex_number": int
      }
    """
    n = len(v_groups)
    if n == 0:
        return {"v": [], "mode_list": [], "group_map": [], "vertex_number": 0}

    # 기본 복제 카운트: [3, 2, 2, ..., 2, 3]
    if copy_counts is None:
        if n == 1:
            copy_counts = [3]
        else:
            copy_counts = [3] + [2] * (n - 2) + [3]
    else:
        assert len(copy_counts) == n, f"copy_counts 길이({len(copy_counts)})가 그룹 수({n})와 다릅니다."

    v_ext: List[np.ndarray] = []
    mode_list: List[str] = []
    group_map: List[List[int]] = []  # 1-based index 매핑

    for g, (rect, k) in enumerate(zip(v_groups, copy_counts)):
        is_first = (g == 0)
        is_last  = (g == n - 1)

        if is_first or is_last:
            # 처음/마지막 그룹: D, SL, SR
            base_modes = ["D", "SL", "SR"]
        else:
            # 중간 그룹: SL, SR  (first_is_left에 따라 순서 반전 가능)
            base_modes = ["SL", "SR"] if first_is_left else ["SR", "SL"]

        # k개만큼 복제하면서 모드는 base_modes를 순환(cycle)
        idxs_this_group: List[int] = []
        for i in range(k):
            v_ext.append(rect.copy())  # 사각형은 동일한 좌표로 복제
            mode_list.append(base_modes[i % len(base_modes)])
            idxs_this_group.append(len(v_ext))  # 1-based index

        group_map.append(idxs_this_group)

        # 사용자가 “정확히 [3,2,2,...,3]로”를 원하면 아래 assert를 켜두면 실수 방지에 유용
        if is_first or is_last:
            assert k >= 3, f"그룹 {g}은 처음/마지막이라 최소 3개여야 합니다. (k={k})"
        else:
            assert k >= 2, f"그룹 {g}은 중간이라 최소 2개여야 합니다. (k={k})"

    return {
        "v": v_ext,
        "mode_list": mode_list,
        "group_map": group_map,
        "vertex_number": len(v_ext),
    }

def build_G_GCS_edges_from_Gk_directed(
    Gk: Dict[Label, List[Label]],
    label_order: List[Label],
    *,
    start_label: Label | None = None,
    goal_label: Label | None = None,
) -> np.ndarray:
    """
    단방향 E 생성:
      - label_order 순서(1-based)에 맞춰 u_idx < v_idx 인 경우에만 u->v
      - 중복 라벨(오직 start==goal 케이스)을 지원하기 위해 라벨→모든 인덱스 매핑 사용
      - start==goal이면 [첫 idx] -> [마지막 idx] 간선을 명시적으로 추가
    """
    # 1) 라벨 → 모든 등장 인덱스(1-based)
    label_to_idxs: Dict[Label, List[int]] = {}
    for i, lb in enumerate(label_order, start=1):
        label_to_idxs.setdefault(lb, []).append(i)

    def first_idx(lb: Label) -> int | None:
        lst = label_to_idxs.get(lb)
        return lst[0] if lst else None

    seen: set[Tuple[int, int]] = set()
    edges: List[Tuple[int, int]] = []

    # 2) Gk의 인접 정보를 기반으로, 앞→뒤 간선만 추가
    for u_lb, nbrs in Gk.items():
        u = first_idx(u_lb)
        if not u:
            continue
        for v_lb in nbrs:
            v = first_idx(v_lb)
            if not v or v == u:
                continue
            if u < v:  # 앞에서 뒤로만
                key = (u, v)
                if key not in seen:
                    seen.add(key)
                    edges.append(key)

    # 3) start==goal이면 (첫 idx) -> (마지막 idx) 간선을 강제로 1개 추가
    if start_label is not None and goal_label is not None and start_label == goal_label:
        idxs = label_to_idxs.get(start_label, [])
        if len(idxs) >= 2:
            key = (idxs[0], idxs[-1])
            if key not in seen:
                seen.add(key)
                edges.append(key)

    edges.sort(key=lambda e: (e[0], e[1]))
    return np.asarray(edges, dtype=int)



