from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, TypeVar, Generic

N = TypeVar("N")

@dataclass
class DirectedGraph(Generic[N]):
    # Minimal directed graph that stores:
    # - best-known cost from start (g)
    # - predecessor edges (for path reconstruction)
    g_score: Dict[N, float] = field(default_factory=dict)
    came_from: Dict[N, N] = field(default_factory=dict)

    def initialize(self, start_node: N) -> None:
        self.g_score.clear()
        self.came_from.clear()
        self.g_score[start_node] = 0.0

    def get_cost_from_start(self, node: N) -> float:
        return self.g_score.get(node, float("inf"))

    def check_and_set_edge(self, parent: N, child: N, edge_cost: float) -> None:
        # Equivalent in spirit to IHMC's DirectedGraph.checkAndSetEdge:
        # update child's best g if edge_cost provides improvement.
        gp = self.get_cost_from_start(parent)
        if gp == float("inf") or edge_cost == float("inf"):
            return
        new_g = gp + edge_cost
        if new_g < self.get_cost_from_start(child):
            self.g_score[child] = new_g
            self.came_from[child] = parent

    def reconstruct_path(self, goal: N) -> list[N]:
        if goal not in self.g_score:
            return []
        path = [goal]
        cur = goal
        while cur in self.came_from:
            cur = self.came_from[cur]
            path.append(cur)
        path.reverse()
        return path
