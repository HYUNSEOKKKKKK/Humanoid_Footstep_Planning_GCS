from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Set, Tuple
import heapq
import itertools

from ..graph import FootstepGraphNode, DiscreteFootstep
from ..step_expansion.footstep_expansion import FootstepExpansion
from .a_star_iteration_data import AStarIterationData
from .directed_graph import DirectedGraph

class FootstepCheckerInterface(Protocol):
    def is_step_valid(self, candidate_step: DiscreteFootstep, stance_step: DiscreteFootstep, start_of_swing: DiscreteFootstep) -> bool:
        ...

class FootstepCostCalculatorInterface(Protocol):
    def compute_cost(self, candidate_step: DiscreteFootstep, stance_step: DiscreteFootstep, start_of_swing: DiscreteFootstep) -> float:
        ...

@dataclass
class AStarFootstepPlannerIterationConductor:
    # Port of AStarFootstepPlannerIterationConductor.java, but:
    # - we implement a small DirectedGraph locally
    # - we implement the PriorityQueue behavior with heapq (f = g + h)
    node_expansion: FootstepExpansion
    edge_checker: FootstepCheckerInterface
    edge_cost_calculator: FootstepCostCalculatorInterface
    heuristic: Callable[[FootstepGraphNode], float]

    expanded_node_set: Set[FootstepGraphNode] = None  # type: ignore
    graph: DirectedGraph[FootstepGraphNode] = None  # type: ignore
    iteration_data: AStarIterationData[FootstepGraphNode] = None  # type: ignore
    neighbors: List[FootstepGraphNode] = None  # type: ignore

    # heap entries: (f, tie, node)
    _heap: list = None  # type: ignore
    _tie: itertools.count = None  # type: ignore

    def __post_init__(self):
        self.expanded_node_set = set()
        self.graph = DirectedGraph()
        self.iteration_data = AStarIterationData()
        self.neighbors = []
        self._heap = []
        self._tie = itertools.count()

    def initialize(self, start_node: FootstepGraphNode) -> None:
        self._heap.clear()
        heapq.heappush(self._heap, (0.0 + self.heuristic(start_node), next(self._tie), start_node))
        self.graph.initialize(start_node)
        self.expanded_node_set.clear()

    def do_planning_iteration(self, node_to_expand: FootstepGraphNode, perform_iterative_expansion: bool) -> \
    AStarIterationData[FootstepGraphNode]:
        self.iteration_data.clear()
        self.iteration_data.set_parent_node(node_to_expand)

        if perform_iterative_expansion:
            partial = self.node_expansion.do_iterative_expansion(node_to_expand, self.neighbors)
        else:
            self.node_expansion.do_full_expansion(node_to_expand, self.neighbors)
            partial = False

        # ---- stats (added) ----
        generated = len(self.neighbors)
        valid = 0
        invalid = 0

        # cumulative stats: 없으면 생성
        if not hasattr(self, "total_generated"):
            self.total_generated = 0
            self.total_valid = 0
            self.total_invalid = 0
            self.last_generated = 0
            self.last_valid = 0
            self.last_invalid = 0

        for neighbor in list(self.neighbors):
            if self.edge_checker.is_step_valid(neighbor.get_second_step(), node_to_expand.get_second_step(),
                                               node_to_expand.get_first_step()):
                valid += 1
                edge_cost = self.edge_cost_calculator.compute_cost(neighbor.get_second_step(),
                                                                   node_to_expand.get_second_step(),
                                                                   node_to_expand.get_first_step())
                self.graph.check_and_set_edge(node_to_expand, neighbor, edge_cost)
                self.iteration_data.get_valid_child_nodes().append(neighbor)

                f = self.graph.get_cost_from_start(neighbor) + self.heuristic(neighbor)
                heapq.heappush(self._heap, (f, next(self._tie), neighbor))
            else:
                invalid += 1
                self.iteration_data.get_invalid_child_nodes().append(neighbor)

        # save last + accumulate
        self.last_generated = generated
        self.last_valid = valid
        self.last_invalid = invalid
        self.total_generated += generated
        self.total_valid += valid
        self.total_invalid += invalid
        # ----------------------

        if partial:
            f = self.graph.get_cost_from_start(node_to_expand) + self.heuristic(node_to_expand)
            heapq.heappush(self._heap, (f, next(self._tie), node_to_expand))
        else:
            self.expanded_node_set.add(node_to_expand)

        return self.iteration_data

    def get_next_node(self) -> Optional[FootstepGraphNode]:
        while self._heap:
            _, _, node = heapq.heappop(self._heap)
            if node not in self.expanded_node_set:
                return node
        return None

    def get_cost_from_start(self, node: FootstepGraphNode) -> float:
        return self.graph.get_cost_from_start(node)
