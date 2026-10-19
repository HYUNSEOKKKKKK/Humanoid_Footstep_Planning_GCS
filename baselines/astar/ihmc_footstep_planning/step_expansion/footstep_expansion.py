from __future__ import annotations
from typing import List, Protocol
from ..graph import FootstepGraphNode

class FootstepExpansion(Protocol):
    def do_full_expansion(self, stance_node: FootstepGraphNode, expansion_to_pack: List[FootstepGraphNode]) -> None:
        ...

    def do_iterative_expansion(self, stance_node: FootstepGraphNode, expansion_to_pack: List[FootstepGraphNode]) -> bool:
        # default: same as full expansion
        self.do_full_expansion(stance_node, expansion_to_pack)
        return False
