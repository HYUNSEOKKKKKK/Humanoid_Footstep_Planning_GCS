# src/terrain_ddr.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Stone:
    center: Tuple[float, float]
    width: float
    height: float
    label: str
    mode: str  # "Do" | "SL" | "SR"

class Terrain_GCS_DDR:
    """
    MATLAB 'Terrain_GCS_DDR' 포팅 (DDR 슬라이딩 윈도우용).
    - 시작 Do는 항상 'All' 영역(0.45,0.45, w=0.7, h=0.7)을 사용.
    - mid labels(First..Third)는 SL/SR 쌍으로 추가.
    - goal은 Do로 종료.
    - get_edges()는 1-based (MATLAB 호환).
    """

    def __init__(self, start: str, *mid_and_goal: str):
        assert len(mid_and_goal) >= 2, "At least First and goal labels are required."
        *mid_labels, goal = mid_and_goal

        self.stone_map: Dict[str, Dict] = {

            "DL": {"center": (-0.21, -0.21), "width": 0.2, "height": 0.2},
            "D": {"center": (0.0, -0.21), "width": 0.2, "height": 0.2},
            "DR": {"center": (0.21, -0.21), "width": 0.2, "height": 0.2},
            "L": {"center": (-0.21, 0.21), "width": 0.2, "height": 0.2},
            "C": {"center": (0.0, 0.0), "width": 0.2, "height": 0.06},
            "R": {"center": (0.21, 0.0), "width": 0.2, "height": 0.2},
            "UL": {"center": (-0.21, 0.21), "width": 0.2, "height": 0.2},
            "U": {"center": (0.0, 0.21), "width": 0.2, "height": 0.2},
            "UR": {"center": (0.21, 0.21), "width": 0.2, "height": 0.2},
            "All": {"center": (0.0, 0.0), "width": 1.0, "height": 1.0},  # 시작 Do
        }

        stones: List[Stone] = []

        # --- Start: 항상 'All'을 Do로 사용 ---
        g = self.stone_map["All"]
        stones.append(Stone(g["center"], g["width"], g["height"], "All", "Do"))

        # --- Middle: 각 라벨을 SL / SR / SL (3개) ---
        for label in mid_labels:
            g = self.stone_map[label]
            stones.extend([
                Stone(g["center"], g["width"], g["height"], label, "SL"),  # SL_a
                Stone(g["center"], g["width"], g["height"], label, "SR"),  # SR
                Stone(g["center"], g["width"], g["height"], label, "SL"),  # SL_b
            ])

        # --- Goal: Do (기존 코드 유지) ---
        g = self.stone_map[goal]
        stones.append(Stone(g["center"], g["width"], g["height"], "goal", "Do"))

        self.stepping_stones: List[Stone] = stones
        self.label_history: List[str] = list(mid_labels)

    # --- Helpers ---
    def _corners2d(self, x: float, y: float, w: float, h: float):
        # (2 x 4), column별 꼭짓점
        return [
            [x - w/2, y - h/2],  # BL
            [x - w/2, y + h/2],  # TL
            [x + w/2, y + h/2],  # TR
            [x + w/2, y - h/2],  # BR
        ]

    def make_stone(self, label: str, mode: str) -> Stone:
        g = self.stone_map[label]
        return Stone(g["center"], g["width"], g["height"], label, mode)

    def add_stone_by_label(self, label: str, mode: str) -> None:
        self.stepping_stones.append(self.make_stone(label, mode))

    def update(self, new_label: str) -> None:
        if new_label not in self.stone_map:
            raise ValueError(f"Invalid label: {new_label}")

        # 슬라이딩: 중간 라벨만 한 칸씩 이동 (시작/goal 고정)
        self.label_history = self.label_history[1:] + [new_label]

        new_stones = [self.stepping_stones[0]]  # Do(start)
        for label in self.label_history:
            g = self.stone_map[label]
            new_stones.extend([
                Stone(g["center"], g["width"], g["height"], label, "SL"),
                Stone(g["center"], g["width"], g["height"], label, "SR"),
                Stone(g["center"], g["width"], g["height"], label, "SL"),
            ])
        new_stones.append(self.stepping_stones[-1])  # Do(goal)
        self.stepping_stones = new_stones

    def get_vertices(self):
        """
        각 vertex를 (3 x 4) numpy 배열로 반환:
        [ [x1 x2 x3 x4],
          [y1 y2 y3 y4],
          [z1 z2 z3 z4] ]  with z=0
        """
        import numpy as np
        verts = []
        for s in self.stepping_stones:
            x, y = s.center
            w, h = s.width, s.height
            # 2D corners (2 x 4)
            corners2d = np.array([
                [x - w/2, x - w/2, x + w/2, x + w/2],
                [y - h/2, y + h/2, y + h/2, y - h/2],
            ], dtype=float)
            # z=0 추가 → (3 x 4)
            corners3d = np.vstack([corners2d, np.zeros((1, 4), dtype=float)])
            verts.append(corners3d)
        return verts

    # ############# all edges connected (same foot can be chosen in row)
    # def get_edges(self):
    #     """
    #     1-based indexing.
    #     Vertex 순서:
    #       1: Do(start='All')
    #       각 라벨 i = 0..m-1: (base=2+3*i)
    #         base:   SL_i(a)   # 상단 SL
    #         base+1: SR_i      # 중앙 SR
    #         base+2: SL_i(b)   # 하단 SL
    #       n: Do(goal)
    #
    #     엣지 규칙:
    #       - 시작: (1,2), (1,3)
    #       - 라벨 내부: (SL_a -> SR), (SR -> SL_b)
    #       - 라벨 i → i+1:
    #           # 기존 3개에서 → 각 노드가 SL_{i+1}(a), SR_{i+1} 둘 다로 (총 6개)
    #           SL_i(a) -> SL_{i+1}(a), SR_{i+1}
    #           SR_i    -> SL_{i+1}(a), SR_{i+1}
    #           SL_i(b) -> SL_{i+1}(a), SR_{i+1}
    #       - 마지막 라벨 → goal: 3개 모두 (SL_a, SR, SL_b) → Do(goal)
    #     """
    #     import numpy as np
    #     from typing import List, Tuple
    #
    #     n = len(self.stepping_stones)
    #     assert (n - 2) % 3 == 0, "Middle stepping stones must be triples (SL/SR/SL)."
    #     m = (n - 2) // 3  # 중간 라벨 개수
    #
    #     edges: List[Tuple[int, int]] = []
    #
    #     # 시작 Do -> 첫 라벨의 SL_a, SR
    #     if m >= 1:
    #         edges.append((1, 2))  # Do -> SL_1(a)
    #         edges.append((1, 3))  # Do -> SR_1
    #
    #     # 각 라벨 내부: SL_a -> SR, SR -> SL_b
    #     for i in range(m):
    #         base = 2 + 3 * i
    #         edges.append((base, base + 1))  # SL_i(a) -> SR_i
    #         edges.append((base + 1, base + 2))  # SR_i -> SL_i(b)
    #
    #     # 라벨 i -> i+1 연결 (각 노드가 다음 라벨의 SL_a와 SR 둘 다로 연결)
    #     for i in range(m - 1):
    #         base_i = 2 + 3 * i
    #         base_ip1 = 2 + 3 * (i + 1)
    #
    #         # SL_i(a) -> {SL_{i+1}(a), SR_{i+1}}
    #         edges.append((base_i, base_ip1))  # (추가) SL_i(a) -> SL_{i+1}(a)
    #         edges.append((base_i, base_ip1 + 1))  # (기존) SL_i(a) -> SR_{i+1}
    #
    #         # SR_i -> {SL_{i+1}(a), SR_{i+1}}
    #         edges.append((base_i + 1, base_ip1))  # (기존) SR_i    -> SL_{i+1}(a)
    #         edges.append((base_i + 1, base_ip1 + 1))  # (추가) SR_i    -> SR_{i+1}
    #
    #         # SL_i(b) -> {SL_{i+1}(a), SR_{i+1}}
    #         edges.append((base_i + 2, base_ip1))  # (추가) SL_i(b) -> SL_{i+1}(a)
    #         edges.append((base_i + 2, base_ip1 + 1))  # (기존) SL_i(b) -> SR_{i+1}
    #
    #     # 마지막 라벨의 3개 모두 -> goal
    #     if m >= 1:
    #         base_last = 2 + 3 * (m - 1)
    #         edges.append((base_last, n))  # SL_last(a) -> Do(goal)
    #         edges.append((base_last + 1, n))  # SR_last    -> Do(goal)
    #         edges.append((base_last + 2, n))  # SL_last(b) -> Do(goal)
    #
    #     return np.array(edges, dtype=int)

    ############  (Always left right left right)
    def get_edges(self):
        """
        1-based indexing.
        Vertex 순서:
          1: Do(start='All')
          각 라벨 i = 0..m-1: (base=2+3*i)
            base:   SL_i(a)
            base+1: SR_i
            base+2: SL_i(b)
          n: Do(goal)

        엣지 규칙:
          - 시작: (1,2), (1,3)
          - 라벨 내부: (SL_a -> SR), (SR -> SL_b)
          - 라벨 i → i+1:
              SL_i(a) -> SR_{i+1}
              SR_i    -> SL_{i+1}(a)
              SL_i(b) -> SR_{i+1}
            (항상 다음 라벨의 '상단 SL/중앙 SR'로만 도착)
          - 마지막 라벨 → goal: 3개 모두 (SL_a, SR, SL_b) → Do(goal)
        """
        import numpy as np

        n = len(self.stepping_stones)
        assert (n - 2) % 3 == 0, "Middle stepping stones must be triples (SL/SR/SL)."
        m = (n - 2) // 3  # 중간 라벨 개수

        edges: List[Tuple[int, int]] = []

        # 시작 Do -> 첫 라벨의 SL_a, SR
        if m >= 1:
            edges.append((1, 2))  # Do -> SL_1(a)
            edges.append((1, 3))  # Do -> SR_1

        # 각 라벨 내부: SL_a -> SR, SR -> SL_b
        for i in range(m):
            base = 2 + 3 * i
            edges.append((base, base + 1))     # SL_i(a) -> SR_i
            edges.append((base + 1, base + 2)) # SR_i -> SL_i(b)

        # 라벨 i -> i+1 연결
        for i in range(m - 1):
            base_i   = 2 + 3 * i
            base_ip1 = 2 + 3 * (i + 1)
            edges.append((base_i,     base_ip1 + 1))  # SL_i(a) -> SR_{i+1}
            edges.append((base_i + 1, base_ip1    ))  # SR_i    -> SL_{i+1}(a)
            edges.append((base_i + 2, base_ip1 + 1))  # SL_i(b) -> SR_{i+1}

        # 마지막 라벨의 3개 모두 -> goal
        if m >= 1:
            base_last = 2 + 3 * (m - 1)
            edges.append((base_last,     n))  # SL_last(a) -> Do(goal)
            edges.append((base_last + 1, n))  # SR_last    -> Do(goal)
            edges.append((base_last + 2, n))  # SL_last(b) -> Do(goal)

        return np.array(edges, dtype=int)

    def get_mode_list(self) -> List[str]:
        return [s.mode for s in self.stepping_stones]
