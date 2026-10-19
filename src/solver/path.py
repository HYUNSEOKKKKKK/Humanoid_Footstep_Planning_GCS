from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np


def pick_path_from_relaxation(y: np.ndarray, edge: np.ndarray, vertex_number: int) -> List[int]:
    """
    (간단 더미) relaxed y에서 경로 선택.
    TODO: 최대 유량/DFS 기반의 제대로 된 복원으로 대체.
    """
    if edge.shape[0] == 0:
        return []
    # 1-based 노드 라벨로 시작/끝만 담은 더미 경로
    return [int(edge[0, 0]), int(edge[0, 1])]


def relaxation_max(y: np.ndarray, edge: np.ndarray, vertex_number: int, max_backtracks: int = 20000) -> List[int]:
    """
    y: (n_edge,)  — relaxed 연속값
    edge: (n_edge,2) 1-based (u,v)
    vertex_number: 마지막 노드 번호 N
    반환: path (노드 번호 리스트) e.g., [1, 7, 12, ..., N]

    전략: y 가중치 내림차순으로 정렬한 인접리스트를 사용해 DFS.
    """
    n_edge = edge.shape[0]
    # 인접리스트: u -> [(v, weight, edge_idx), ...] 내림차순
    adj: Dict[int, List[Tuple[int, float, int]]] = {i: [] for i in range(1, vertex_number + 1)}
    for ei in range(n_edge):
        u, v = int(edge[ei, 0]), int(edge[ei, 1])
        w = float(y[ei])
        adj[u].append((v, w, ei))
    for u in adj:
        adj[u].sort(key=lambda t: t[1], reverse=True)

    start, goal = 1, vertex_number
    path_nodes: List[int] = [start]
    visited_edges = set()
    visited_nodes = {start}
    explored = 0

    def dfs(u: int) -> bool:
        nonlocal explored
        if u == goal:
            return True
        for (v, w, ei) in adj.get(u, []):
            if ei in visited_edges or v in visited_nodes:
                continue
            explored += 1
            if explored > max_backtracks:
                return False
            visited_edges.add(ei)
            visited_nodes.add(v)
            path_nodes.append(v)
            if dfs(v):
                return True
            # backtrack
            path_nodes.pop()
            visited_nodes.remove(v)
            visited_edges.remove(ei)
        return False

    if dfs(start):
        return path_nodes

    # fallback: 탐욕적으로 가장 큰 y를 따라가기
    node = start
    greedy_path = [start]
    used_edges = set()
    while node != goal and adj.get(node):
        v, w, ei = adj[node][0]
        if ei in used_edges or v in greedy_path:
            break
        used_edges.add(ei)
        greedy_path.append(v)
        node = v
    if node == goal:
        return greedy_path

    raise RuntimeError("DFS로 1→N 경로를 찾지 못했습니다. 그래프/모드/제약을 확인하세요.")

def relaxation_prob(y, edge, vertex_number, *, temperature: float = 1.0,
                    seed: int | None = None, max_backtracks: int = 20000):
    """
    확률 DFS (1 → N):
    - 각 노드 u에서 나가는 엣지(e)의 선택확률 ∝ (y_e + eps)^(1/temperature)
      · temperature < 1  → 더 탐욕적
      · temperature > 1  → 더 랜덤
    - 방문 노드/엣지 재사용 방지 + 백트랙
    - 실패 시 relaxation_max로 fallback
    입력:
      y: (n_edge,), edge: (n_edge,2)  1-based 노드 인덱스, vertex_number: N
    반환:
      path_nodes: [1, ..., N]
    """
    y = np.asarray(y, float).reshape(-1)
    n_edge = int(edge.shape[0])
    start, goal = 1, int(vertex_number)

    rng = np.random.default_rng(seed)
    eps = 1e-12
    temp = max(1e-6, float(temperature))

    # u -> [(v, ei, w)]
    adj = {i: [] for i in range(1, vertex_number + 1)}
    for ei in range(n_edge):
        u, v = int(edge[ei, 0]), int(edge[ei, 1])
        w = float(y[ei])
        adj[u].append((v, ei, w))

    def pick_next(u, banned_edges, banned_nodes):
        cands = [(v, ei, w) for (v, ei, w) in adj.get(u, []) if ei not in banned_edges and v not in banned_nodes]
        if not cands:
            return None
        ws = np.array([w for (_, _, w) in cands], float)
        ws = np.power(ws + eps, 1.0 / temp)
        S = float(np.sum(ws))
        if not np.isfinite(S) or S <= 0.0:
            probs = np.full(len(cands), 1.0 / len(cands))
        else:
            probs = ws / S
        idx = rng.choice(len(cands), p=probs)
        return cands[idx]  # (v, ei, w)

    path_nodes = [start]
    visited_nodes = {start}
    used_edges = set()
    tried_per_node = {i: set() for i in range(1, vertex_number + 1)}
    explored = 0

    def dfs(u):
        nonlocal explored
        if u == goal:
            return True
        while True:
            explored += 1
            if explored > max_backtracks:
                return False
            choice = pick_next(u, tried_per_node[u] | used_edges, visited_nodes)
            if choice is None:
                return False
            v, ei, _ = choice
            tried_per_node[u].add(ei)
            # advance
            path_nodes.append(v)
            visited_nodes.add(v)
            used_edges.add(ei)
            if dfs(v):
                return True
            # backtrack
            path_nodes.pop()
            visited_nodes.remove(v)
            used_edges.remove(ei)

    if dfs(start):
        return path_nodes

    # 실패 시: 탐욕 그리디로 복구
    return relaxation_max(y, edge, vertex_number)

def path_to_y(edge: np.ndarray, path: List[int], n_edge: int) -> np.ndarray:
    """
    path (노드열, 1-based 라벨 가정) → 엣지 0/1 벡터 (n_edge,1).
    """
    y_relax = np.zeros((n_edge, 1), dtype=float)
    # (u,v) → edge index 맵
    idx_map = {}
    for i in range(n_edge):
        u, v = int(edge[i, 0]), int(edge[i, 1])
        idx_map[(u, v)] = i
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        ei = idx_map.get((u, v), None)
        if ei is None:
            raise RuntimeError(f"경로에 없는 엣지 ({u}->{v})")
        y_relax[ei, 0] = 1.0
    return y_relax
