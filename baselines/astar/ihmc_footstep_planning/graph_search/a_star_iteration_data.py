from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generic, List, Optional, TypeVar

N = TypeVar("N")

@dataclass
class AStarIterationData(Generic[N]):
    parent_node: Optional[N] = None
    valid_child_nodes: List[N] = field(default_factory=list)
    invalid_child_nodes: List[N] = field(default_factory=list)

    def get_parent_node(self) -> Optional[N]:
        return self.parent_node

    def set_parent_node(self, parent: N) -> None:
        self.parent_node = parent

    def clear(self) -> None:
        self.parent_node = None
        self.valid_child_nodes.clear()
        self.invalid_child_nodes.clear()

    def get_valid_child_nodes(self) -> List[N]:
        return self.valid_child_nodes

    def get_invalid_child_nodes(self) -> List[N]:
        return self.invalid_child_nodes
