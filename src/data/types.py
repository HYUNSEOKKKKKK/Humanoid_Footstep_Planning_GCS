from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List
import numpy as np

@dataclass
class Stone:
    center: np.ndarray  # (3,)
    width: float
    height: float
    name: str

# Vertex is a 3x4 array (each column is a corner)
Vertex = np.ndarray  # shape: (3, 4)

# Mode labels for each vertex copy
Mode = str  # "D" | "SL" | "SR"

# Convenience container returned by build_graph
@dataclass
class GraphBuildResult:
    v: List[Vertex]
    edge: np.ndarray
    mode_list: List[Mode]
    vertex_number: int
    n_edge: int

    # optional: backward-compat for dict-style access
    def __getitem__(self, key: str):
        return getattr(self, key)
