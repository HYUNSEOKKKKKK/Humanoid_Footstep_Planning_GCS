from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np

from ..data.types import Stone, Vertex, Mode

class TerrainGCS:
    def __init__(self,
                 startpos_L: np.ndarray, startpos_R: np.ndarray,
                 targetpos_L: np.ndarray, targetpos_R: np.ndarray,
                 stones: List[Dict[str, Any]] | None = None):
        self.stepping_stones: List[Dict[str, Any]] = []

        if stones is None:
            # 기존 기본(default_flat) 구성
            initial_position = np.array([2.5, 0.5, 0.0], dtype=float)
            self.add_stone(initial_position, width=5.0, height=1.0, name="initial")

            # lateral 없음
            goal_position = np.array([2.5, 1.5, 0.0], dtype=float)
            self.add_stone(goal_position, width=5.0, height=1.0, name="goal")
        else:
            # 프리셋에서 입력된 스톤들로 초기화
            for st in stones:
                self.add_stone(np.asarray(st["center"], float),
                               float(st["width"]), float(st["height"]),
                               st.get("name", f"stone_{len(self.stepping_stones)+1}"))

    # ---------- 기본 유틸 ----------
    def add_stone(self, center: np.ndarray, width: float, height: float, name: str) -> Dict[str, Any]:
        center = np.asarray(center, dtype=float).reshape(-1)
        if center.size == 2:
            center = np.array([center[0], center[1], 0.0], dtype=float)
        stone = dict(center=center, width=float(width), height=float(height), name=name)
        self.stepping_stones.append(stone)
        return stone

    # ---------- Vertex 생성 ----------
    def _rectangle_vertices_3x4(self, center: np.ndarray, width: float, height: float) -> Vertex:
        cx, cy, cz = center.tolist()
        hx, hy = width / 2.0, height / 2.0
        pts = np.array([
            [cx - hx, cy - hy, cz],
            [cx - hx, cy + hy, cz],
            [cx + hx, cy + hy, cz],
            [cx + hx, cy - hy, cz],
        ], dtype=float).T  # (3,4)
        return pts

    def get_vertices(self, step_span: float, copy_counts: List[int]) -> List[Vertex]:
        n_stones = len(self.stepping_stones)
        if len(copy_counts) != n_stones:
            raise ValueError("Length of copy_counts must match the number of stepping stones.")
        vertices: List[Vertex] = []
        for i in range(n_stones):
            stone = self.stepping_stones[i]
            single_vertex = self._rectangle_vertices_3x4(stone["center"], stone["width"], stone["height"])
            for _ in range(int(copy_counts[i])):
                vertices.append(single_vertex.copy())
        return vertices

    # ---------- Edge 생성 (규칙 기반) ----------
    def generate_edges_from_rules(self, copy_counts: List[int]) -> np.ndarray:
        copy_counts = list(map(int, copy_counts))
        copy_cumsum = np.cumsum(copy_counts)
        group_start = np.concatenate(([1], copy_cumsum[:-1] + 1))
        n_groups = len(copy_counts)

        initial_ids = np.arange(group_start[0], copy_cumsum[0] + 1)
        num_lateral_groups = n_groups - 2
        lateral_ids = [np.arange(group_start[i+1], copy_cumsum[i+1] + 1) for i in range(num_lateral_groups)]
        goal_ids = np.arange(group_start[-1], copy_cumsum[-1] + 1)

        edges: List[Tuple[int, int]] = []

        # 1) Initial group 내부
        if initial_ids.size >= 3:
            edges.append((int(initial_ids[0]), int(initial_ids[1])))
            edges.append((int(initial_ids[0]), int(initial_ids[2])))
        for i in range(2, initial_ids.size):
            edges.append((int(initial_ids[i-1]), int(initial_ids[i])))

        # 2) Initial group -> 각 Lateral group
        for lat in lateral_ids:
            first_odd = next((int(x) for x in lat if x % 2 == 1), None)
            first_even = next((int(x) for x in lat if x % 2 == 0), None)
            for idx in initial_ids[1:]:
                idx = int(idx)
                if (idx % 2 == 0) and (first_odd is not None):
                    target = first_odd
                elif first_even is not None:
                    target = first_even
                else:
                    continue
                edges.append((idx, target))

        # 3) 각 Lateral 내부 인접 연결
        for lat in lateral_ids:
            for i in range(lat.size - 1):
                edges.append((int(lat[i]), int(lat[i+1])))

        # 4) 각 Lateral -> Goal 첫 odd/even
        first_goal_odd = next((int(x) for x in goal_ids if x % 2 == 1), None)
        first_goal_even = next((int(x) for x in goal_ids if x % 2 == 0), None)
        for lat in lateral_ids:
            for idx in lat:
                idx = int(idx)
                if (idx % 2 == 0) and (first_goal_odd is not None):
                    target = first_goal_odd
                elif first_goal_even is not None:
                    target = first_goal_even
                else:
                    continue
                edges.append((idx, target))

        # 5) Goal 내부: 체인 + 마지막 노드와 모두 연결
        if goal_ids.size >= 2:
            for i in range(goal_ids.size - 2):
                edges.append((int(goal_ids[i]), int(goal_ids[i+1])))
            last_goal = int(goal_ids[-1])
            for i in range(goal_ids.size - 1):
                edges.append((int(goal_ids[i]), last_goal))

        return np.array(edges, dtype=int).reshape(-1, 2)

    def create_edges_general(self, vertices: List[Vertex], group_sizes: List[int]) -> np.ndarray:
        edges: List[Tuple[int, int]] = []
        n_groups = len(group_sizes)

        # 그룹별 인덱스 범위 (1-based)
        group_indices = []
        cur = 1
        for sz in group_sizes:
            group_indices.append(np.arange(cur, cur + int(sz)))
            cur += int(sz)

        # group 1
        g1 = group_indices[0]
        if g1.size >= 3:
            edges += [(int(g1[0]), int(g1[1])), (int(g1[0]), int(g1[2]))]
        for j in range(1, g1.size - 1):
            edges.append((int(g1[j]), int(g1[j+1])))

        # group 1 -> group 2
        g2 = group_indices[1] if n_groups >= 2 else np.array([], dtype=int)
        for m in range(1, g1.size):
            src = int(g1[m])
            if (m + 1) % 2 == 0 and g2.size >= 2:
                edges.append((src, int(g2[1])))
            elif g2.size >= 1:
                edges.append((src, int(g2[0])))

        # group 2->3, 3->4, ...
        if n_groups > 2:
            for i in range(2, n_groups):
                g_pre = group_indices[i-1]
                g_curr = group_indices[i]

                # g_pre 내부 체인
                for j in range(g_pre.size - 1):
                    edges.append((int(g_pre[j]), int(g_pre[j+1])))

                # g_pre 각 노드 -> g_curr 첫 두 노드에 번갈아 연결
                for m in range(g_pre.size):
                    src = int(g_pre[m])
                    if (m + 1) % 2 == 0 and g_curr.size >= 1:
                        edges.append((src, int(g_curr[0])))
                    elif g_curr.size >= 2:
                        edges.append((src, int(g_curr[1])))

        # group last
        g_last = group_indices[-1]
        if g_last.size >= 2:
            for j in range(g_last.size - 2):
                edges.append((int(g_last[j]), int(g_last[j+1])))
            for m in range(g_last.size - 1):
                edges.append((int(g_last[m]), int(g_last[-1])))

        return np.array(edges, dtype=int).reshape(-1, 2)

    # ---------- Vertex + Mode ----------
    def get_vertices_with_modes(self, step_span: float, copy_counts: List[int]) -> Tuple[List[Vertex], List[Mode]]:
        n_stones = len(self.stepping_stones)
        if len(copy_counts) != n_stones:
            raise ValueError("Length of copy_counts must match the number of stepping stones.")

        vertices: List[Vertex] = []
        mode_list: List[Mode] = []

        for g in range(n_stones):
            stone = self.stepping_stones[g]
            num_copies = int(copy_counts[g])
            vtx = self._rectangle_vertices_3x4(stone["center"], stone["width"], stone["height"])

            for j in range(1, num_copies + 1):
                vertices.append(vtx.copy())

                if g == 0:  # 첫 스톤 그룹
                    if j == 1:
                        mode_list.append("D")
                    else:
                        mode_list.append("SL" if (j % 2 == 0) else "SR")
                elif g == n_stones - 1:  # 마지막 스톤 그룹
                    if j == num_copies:
                        mode_list.append("D")
                    else:
                        mode_list.append("SL" if (j % 2 == 1) else "SR")
                else:  # 중간 스톤 그룹
                    mode_list.append("SL" if (j % 2 == 1) else "SR")

        return vertices, mode_list