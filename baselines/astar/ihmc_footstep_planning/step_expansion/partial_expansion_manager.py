from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from ..graph import FootstepGraphNode
from ..parameters import DefaultFootstepPlannerParameters

@dataclass
class PartialExpansionManager:
    parameters: DefaultFootstepPlannerParameters
    all_child_nodes: List[FootstepGraphNode] = field(default_factory=list)
    expansion_count: int = 0

    def initialize(self, child_nodes: List[FootstepGraphNode]) -> None:
        self.all_child_nodes = list(child_nodes)
        self.expansion_count = 0

    def pack_partial_expansion(self, expansion_to_pack: List[FootstepGraphNode]) -> None:
        expansion_to_pack.clear()
        start_idx = self.expansion_count * self.parameters.getMaxBranchFactor()
        end_idx = min((self.expansion_count + 1) * self.parameters.getMaxBranchFactor(), len(self.all_child_nodes))
        for i in range(start_idx, end_idx):
            expansion_to_pack.append(self.all_child_nodes[i])
        self.expansion_count += 1

    def finished_expansion(self) -> bool:
        return self.expansion_count * self.parameters.getMaxBranchFactor() >= len(self.all_child_nodes)
