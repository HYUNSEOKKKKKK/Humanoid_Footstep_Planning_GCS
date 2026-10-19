from __future__ import annotations
from typing import List
import numpy as np

from .terrain import TerrainGCS
from ..data.types import GraphBuildResult


def build_graph(startpos_L, startpos_R, targetpos_L, targetpos_R,
                step_span: float, copy_counts: List[int],
                edge_rule: str = "parallel",
                terrain_stones: List[Dict[str, Any]] | None = None):
    """
    terrain_stones: 프리셋에서 읽어온 stones 리스트. None이면 기본(initial/goal) 사용.
    """
    terrain = TerrainGCS(startpos_L, startpos_R, targetpos_L, targetpos_R,
                         stones=terrain_stones)
    v, mode_list = terrain.get_vertices_with_modes(step_span, copy_counts)

    if edge_rule == "parallel":
        edge = terrain.generate_edges_from_rules(copy_counts)
    elif edge_rule == "general":
        edge = terrain.create_edges_general(v, copy_counts)
    else:
        raise ValueError("edge_rule must be 'parallel' or 'general'.")

    return dict(
        v=v,
        edge=edge.astype(int).reshape(-1, 2),
        mode_list=mode_list,
        vertex_number=len(v),
        n_edge=edge.shape[0],
    )