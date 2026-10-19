from __future__ import annotations

# Graph
from .graph import (
    build_graph,
    compute_centroids_from_vertices,
    centroid_and_scaled_vertices,
    TerrainGCS,
)

# Model
from .model import (
    pack_relaxed_params_for_model,
    pack_fixedpath_params_for_model,
    prepare_problem_params,
    pack_base_param,
    build_gcs_model,
)

# Solver
from .solver import (
    set_parameters,
    solve_relaxed,
    solve_with_fixed_path,
    choose_best_solution,
    get_relaxed_y,
    extract_solution,
    recover_x_from_z_over_y,
    relaxation_max,
    path_to_y,
    pick_path_from_relaxation,
    # Fusion ops
    eq0, le0, ge0, cone2, repeat_scalar, sum_expr, sum_rows_col, sum_pick_1d,
)

# Viz
from .viz import compute_foot_rects, plot_feet_3d


__all__ = [
    # graph
    "build_graph", "compute_centroids_from_vertices",
    "centroid_and_scaled_vertices", "TerrainGCS",
    # model
    "pack_relaxed_params_for_model", "pack_fixedpath_params_for_model",
    "prepare_problem_params", "pack_base_param", "build_gcs_model",
    # solver
    "set_parameters", "solve_relaxed", "solve_with_fixed_path",
    "choose_best_solution", "get_relaxed_y", "extract_solution",
    "recover_x_from_z_over_y", "relaxation_max", "path_to_y",
    "pick_path_from_relaxation",
    "eq0", "le0", "ge0", "cone2", "repeat_scalar", "sum_expr", "sum_rows_col", "sum_pick_1d",
    # viz
    "compute_foot_rects", "plot_feet_3d",
]
