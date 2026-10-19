#!/usr/bin/env python3
"""Extended graph-construction sensitivity data for Reviewer Comment 10.3.

This executable reuses the repository's CVAE terrain, graph, OptAPI model,
relaxed solver, fixed-path solver, and postprocessing code.  It changes only
the duplicate vector or the graph edge candidate set.  No plot or table is
created; the outputs are raw and summary CSV files for later MATLAB analysis.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

# Importing repository modules also imports plotting helpers.  Nothing here
# draws a plot, but a writable Matplotlib cache avoids irrelevant warnings.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/footstep_planner_mpl_cache")

import mosek
import numpy as np

from experiments import graph_sensitivity as legacy
from main_VAE_optapi import make_entry_from_stones, rotate_vertices_with_yaw
from src.graph import build_graph, compute_centroids_from_vertices
from src.model import build_gcs_model_VAE_optapi, pack_relaxed_params_for_model
from src.solver.path import path_to_y, relaxation_max, relaxation_prob
from src.solver.stochastic import _solve_fixed_once_optapi
from src.solver.utils import set_parameters_optapi, solve_relaxed_optapi
from src.utils import expand_shorthand, postprocess_solution


DEFAULT_TERRAIN_DIR = Path("data/terrains/alpha")
DEFAULT_OUTPUT_DIR = Path("outputs/Com10_3_extended")
DEFAULT_EXISTING_DIR = Path("data/reference/gcs/Com10_3")

# These settings are deliberately constants, rather than graph-dependent CLI
# knobs.  That prevents an accidental planner-setting change between graph
# configurations.
ALPHA = 0.0
STEP_SPAN = 0.7
YAW_BOUND_DEG = 30.0
STANCE_WIDTH = 0.25
M_LARGE = 1000.0
M_SMALL = 1e-6
RECOVERY_CANDIDATE_BUDGET = 100
PLANNING_DEADLINE_SEC = 5.0
TEMPERATURE = 1.0
MAX_BACKTRACKS = 20000
DEFAULT_BASE_SEED = 123
SIMPLE_PATH_COUNT_CAP = 1_000_000_000

DUPLICATE_CONFIGS: dict[str, tuple[int, int, int, int]] = {
    "dup_reduced_1": (4, 3, 3, 4),
    "dup_reduced_2": (3, 2, 2, 3),
    "dup_reduced_3": (2, 1, 1, 2),
    "nominal": (5, 4, 4, 5),
    "dup_doubled": (10, 8, 8, 10),
    "dup_tripled": (15, 12, 12, 15),
    "dup_quadrupled": (20, 16, 16, 20),
}
EDGE_CONFIGS = ("edge_dense_mode_compatible", "edge_fully_connected_forward")
CONFIGURATIONS = tuple(DUPLICATE_CONFIGS) + EDGE_CONFIGS
# Nominal runs first so that equivalence is checked before alternatives.
EXECUTION_ORDER = ("nominal",) + tuple(x for x in CONFIGURATIONS if x != "nominal")
ALTERNATIVES = tuple(x for x in CONFIGURATIONS if x != "nominal")

EXPECTED_GRAPH_COUNTS: dict[str, tuple[int, int]] = {
    "dup_reduced_1": (14, 22),
    "dup_reduced_2": (10, 14),
    "dup_reduced_3": (6, 3),
    "nominal": (18, 30),
    "dup_doubled": (36, 66),
    "dup_tripled": (54, 102),
    "dup_quadrupled": (72, 138),
    "edge_dense_mode_compatible": (18, 72),
    "edge_fully_connected_forward": (18, 152),
}

RAW_FIELDS = [
    "terrain_id", "terrain_seed", "recovery_base_seed", "alpha", "number_of_physical_regions",
    "start_region_index", "goal_region_index", "experiment_group",
    "configuration", "copy_counts", "copy_count_region_1",
    "copy_count_region_2", "copy_count_region_3", "copy_count_region_4",
    "edge_rule", "preserves_mode_compatibility", "allows_same_mode_edges",
    "allows_backward_edges", "allows_cycles", "number_of_vertices",
    "number_of_edges", "number_of_SL_vertices", "number_of_SR_vertices",
    "number_of_D_vertices", "edge_to_vertex_ratio", "directed_graph_density",
    "source_to_target_path_exists", "is_directed_acyclic_graph",
    "topological_sort_succeeded", "source_indegree", "target_outdegree",
    "number_of_simple_source_to_target_paths", "success", "failure_stage",
    "failure_reason", "timeout_occurred", "solver_status_relaxed",
    "solver_status_fixed", "final_feasibility_validated",
    "graph_construction_time_ms", "model_build_time_ms", "relaxed_solve_time_ms",
    "path_recovery_search_time_ms", "fixed_path_solve_time_ms",
    "time_to_first_feasible_ms", "total_elapsed_time_ms", "C_relax", "C_round",
    "final_objective", "delta_relax", "delta_relax_percent",
    "selected_path_length", "actual_number_of_footsteps",
    "number_of_recovery_candidates_tested", "selected_region_sequence",
    "selected_support_mode_sequence", "number_of_edges_with_positive_relaxed_flow",
    "number_of_edges_with_fractional_flow", "maximum_fractional_edge_value",
    "relaxed_flow_entropy",
]

TERRAIN_FIELDS = [
    "terrain_id", "terrain_seed", "alpha", "number_of_physical_regions",
    "start_region_index", "goal_region_index", "terrain_file",
    "terrain_file_sha256", "start_region", "goal_region",
]

SUMMARY_FIELDS = [
    "experiment_group", "configuration", "number_of_trials", "vertices_mean",
    "edges_mean", "edge_to_vertex_ratio_mean", "graph_density_mean",
    "number_of_successes", "success_rate_percent", "wilson_95_ci_lower_percent",
    "wilson_95_ci_upper_percent", "graph_disconnected_count",
    "graph_insufficient_count", "solver_failure_count", "recovery_failure_count",
    "deadline_failure_count", "delta_relax_mean_percent",
    "delta_relax_std_percent", "delta_relax_median_percent",
    "delta_relax_q25_percent", "delta_relax_q75_percent",
    "time_to_first_feasible_mean_ms", "time_to_first_feasible_std_ms",
    "time_to_first_feasible_median_ms", "time_to_first_feasible_q25_ms",
    "time_to_first_feasible_q75_ms", "total_elapsed_time_mean_ms",
    "total_elapsed_time_median_ms", "final_objective_mean_success_only",
    "final_objective_median_success_only",
    "actual_number_of_footsteps_mean_success_only",
    "fractional_edges_mean_success_only", "relaxed_flow_entropy_mean_success_only",
]

PAIRED_FIELDS = [
    "terrain_id", "experiment_group", "alternative_configuration",
    "nominal_success", "alternative_success", "nominal_failure_stage",
    "alternative_failure_stage", "nominal_vertices", "alternative_vertices",
    "nominal_edges", "alternative_edges", "nominal_delta_relax_percent",
    "alternative_delta_relax_percent", "delta_relax_difference_percentage_points",
    "nominal_time_to_first_feasible_ms", "alternative_time_to_first_feasible_ms",
    "time_difference_ms", "time_ratio", "nominal_final_objective",
    "alternative_final_objective", "objective_difference",
    "nominal_actual_number_of_footsteps", "alternative_actual_number_of_footsteps",
]

FAILURE_FIELDS = [
    "experiment_group", "configuration", "failure_stage", "count",
    "percent_of_trials",
]


@dataclass
class GraphProblem:
    stones: list[dict[str, Any]]
    vertices: list[np.ndarray]
    edge: np.ndarray
    modes: list[str]
    region_of_vertex: list[int]
    startpos_L: np.ndarray
    startpos_R: np.ndarray
    targetpos_L: np.ndarray
    targetpos_R: np.ndarray
    start_L: np.ndarray
    start_R: np.ndarray
    target_L: np.ndarray
    target_R: np.ndarray
    centroid_mat: np.ndarray
    x_ref: np.ndarray
    y_ref: np.ndarray

    @property
    def vertex_number(self) -> int:
        return len(self.vertices)

    @property
    def n_edge(self) -> int:
        return int(self.edge.shape[0])


def _configuration_group(configuration: str) -> str:
    return "edge_density" if configuration in EDGE_CONFIGS else "duplicate_sensitivity"


def _copy_counts(configuration: str) -> tuple[int, int, int, int]:
    return DUPLICATE_CONFIGS.get(configuration, DUPLICATE_CONFIGS["nominal"])


def _edge_metadata(configuration: str) -> tuple[str, bool, bool]:
    if configuration == "edge_dense_mode_compatible":
        return "all_forward_mode_compatible", True, False
    if configuration == "edge_fully_connected_forward":
        return "all_forward_pairs_except_source_target", False, True
    return "TerrainGCS.create_edges_general", True, False


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def _atomic_write_csv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite_values(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values = [_as_float(row.get(field)) for row in rows]
    return [value for value in values if np.isfinite(value)]


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _std(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return 0.0 if len(values) == 1 else float(np.std(values, ddof=1))


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else math.nan


def _error_text(error: BaseException | str, limit: int = 1600) -> str:
    value = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return value.replace("\n", " | ")[:limit]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pose5(position: np.ndarray, yaw: float) -> np.ndarray:
    value = np.asarray(position, dtype=float).reshape(-1)
    return np.asarray(
        [value[0], value[1], value[2], math.cos(yaw), math.sin(yaw)], dtype=float
    )


def _reference_axes(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(left, dtype=float)[:2] - np.asarray(right, dtype=float)[:2]
    length = float(np.linalg.norm(direction))
    y_ref = np.asarray([0.0, 1.0]) if length < 1e-12 else direction / length
    return np.asarray([y_ref[1], -y_ref[0]], dtype=float), y_ref


def _fully_connected_forward_edges(vertex_number: int) -> np.ndarray:
    """All forward pairs in fixed order, excluding direct source-to-target."""
    edges = [
        (u, v)
        for u in range(1, int(vertex_number) + 1)
        for v in range(u + 1, int(vertex_number) + 1)
        if not (u == 1 and v == int(vertex_number))
    ]
    return np.asarray(edges, dtype=int).reshape(-1, 2)


def _prepare_problem(case: legacy.TerrainCase, configuration: str) -> GraphProblem:
    stones = copy.deepcopy(list(case.stones))
    copy_counts = list(_copy_counts(configuration))
    entry = make_entry_from_stones(stones, STANCE_WIDTH)
    expanded = expand_shorthand(entry, STANCE_WIDTH)
    startpos_L = np.asarray(expanded["startpos_L"], dtype=float)
    startpos_R = np.asarray(expanded["startpos_R"], dtype=float)
    targetpos_L = np.asarray(expanded["targetpos_L"], dtype=float)
    targetpos_R = np.asarray(expanded["targetpos_R"], dtype=float)

    graph = build_graph(
        startpos_L,
        startpos_R,
        targetpos_L,
        targetpos_R,
        step_span=STEP_SPAN,
        copy_counts=copy_counts,
        edge_rule="general",
        terrain_stones=stones,
    )
    vertices = rotate_vertices_with_yaw(graph["v"], stones, copy_counts)
    modes = list(graph["mode_list"])
    region_of_vertex = legacy._region_map(copy_counts)
    sparse_edges = np.asarray(graph["edge"], dtype=int).reshape(-1, 2)
    if configuration == "edge_dense_mode_compatible":
        edge = legacy._dense_mode_compatible_edges(modes, region_of_vertex)
    elif configuration == "edge_fully_connected_forward":
        edge = _fully_connected_forward_edges(len(vertices))
    else:
        edge = sparse_edges

    expected_vertices, expected_edges = EXPECTED_GRAPH_COUNTS[configuration]
    if len(vertices) != expected_vertices or len(edge) != expected_edges:
        raise AssertionError(
            f"{configuration}: expected |V|={expected_vertices}, |E|={expected_edges}; "
            f"got |V|={len(vertices)}, |E|={len(edge)}"
        )
    if configuration == "edge_fully_connected_forward" and len(edge) != 152:
        raise AssertionError("fully connected forward DAG must have exactly 152 edges")

    centroid_mat, _ = compute_centroids_from_vertices(vertices)
    x_ref, y_ref = _reference_axes(startpos_L, startpos_R)
    return GraphProblem(
        stones=stones,
        vertices=[np.asarray(item, dtype=float) for item in vertices],
        edge=edge,
        modes=modes,
        region_of_vertex=region_of_vertex,
        startpos_L=startpos_L,
        startpos_R=startpos_R,
        targetpos_L=targetpos_L,
        targetpos_R=targetpos_R,
        start_L=_pose5(startpos_L, float(expanded["startyaw_L"])),
        start_R=_pose5(startpos_R, float(expanded["startyaw_R"])),
        target_L=_pose5(targetpos_L, float(expanded["targetyaw_L"])),
        target_R=_pose5(targetpos_R, float(expanded["targetyaw_R"])),
        centroid_mat=np.asarray(centroid_mat, dtype=float),
        x_ref=x_ref,
        y_ref=y_ref,
    )


def _graph_audit(problem: GraphProblem) -> dict[str, Any]:
    n = problem.vertex_number
    edge_pairs = [(int(u), int(v)) for u, v in problem.edge]
    adjacency: dict[int, list[int]] = {node: [] for node in range(1, n + 1)}
    indegree = {node: 0 for node in range(1, n + 1)}
    valid_nodes = True
    for u, v in edge_pairs:
        if not (1 <= u <= n and 1 <= v <= n) or u == v:
            valid_nodes = False
            continue
        adjacency[u].append(v)
        indegree[v] += 1

    queue = sorted(node for node in range(1, n + 1) if indegree[node] == 0)
    topo: list[int] = []
    mutable_indegree = dict(indegree)
    while queue:
        node = queue.pop(0)
        topo.append(node)
        for successor in adjacency[node]:
            mutable_indegree[successor] -= 1
            if mutable_indegree[successor] == 0:
                queue.append(successor)
                queue.sort()
    is_dag = valid_nodes and len(topo) == n
    topological_succeeded = is_dag
    all_forward = all(u < v for u, v in edge_pairs)
    # Source-to-target reachability does not rely on forward numbering.
    stack, visited = [1], {1}
    while stack:
        current = stack.pop()
        for successor in adjacency.get(current, []):
            if successor not in visited:
                visited.add(successor)
                stack.append(successor)
    path_exists = n in visited

    return {
        "source_to_target_path_exists": path_exists,
        "is_directed_acyclic_graph": is_dag,
        "topological_sort_succeeded": topological_succeeded,
        "all_edges_follow_fixed_forward_order": all_forward,
        "source_indegree": int(indegree.get(1, 0)),
        "target_outdegree": len(adjacency.get(n, [])),
        "topological_order": topo,
    }


def _count_simple_paths(problem: GraphProblem, topological_order: Sequence[int]) -> float:
    """Count source-to-target DAG paths with a cap, outside planner timing."""
    if len(topological_order) != problem.vertex_number:
        return math.nan
    adjacency: dict[int, list[int]] = {
        node: [] for node in range(1, problem.vertex_number + 1)
    }
    for u, v in problem.edge:
        adjacency[int(u)].append(int(v))
    counts = {node: 0 for node in range(1, problem.vertex_number + 1)}
    counts[1] = 1
    for node in topological_order:
        for successor in adjacency[node]:
            counts[successor] = min(
                SIMPLE_PATH_COUNT_CAP, counts[successor] + counts[node]
            )
    return float(counts[problem.vertex_number])


def _acceptable_status(task: Any) -> bool:
    return legacy._acceptable_status(task)


def _status(task: Any) -> str:
    return legacy._status(task)


def _close_task(task: Any) -> None:
    legacy._close_task(task)


def _valid_path(path: Sequence[int], edge: np.ndarray, vertex_number: int) -> bool:
    return legacy._valid_path(path, edge, vertex_number)


def _recovery_seed(base_seed: int, terrain_id: int, probabilistic_index: int) -> int:
    sequence = np.random.SeedSequence(
        [int(base_seed), int(terrain_id), int(probabilistic_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _blank_row(case: legacy.TerrainCase, configuration: str) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in RAW_FIELDS}
    counts = _copy_counts(configuration)
    edge_rule, preserves_mode, allows_same = _edge_metadata(configuration)
    row.update(
        {
            "terrain_id": case.terrain_id,
            "terrain_seed": case.terrain_seed,
            "alpha": ALPHA,
            "number_of_physical_regions": 4,
            "start_region_index": 0,
            "goal_region_index": 3,
            "experiment_group": _configuration_group(configuration),
            "configuration": configuration,
            "copy_counts": json.dumps(list(counts), separators=(",", ":")),
            "copy_count_region_1": counts[0],
            "copy_count_region_2": counts[1],
            "copy_count_region_3": counts[2],
            "copy_count_region_4": counts[3],
            "edge_rule": edge_rule,
            "preserves_mode_compatibility": preserves_mode,
            "allows_same_mode_edges": allows_same,
            "allows_backward_edges": False,
            "allows_cycles": False,
            "number_of_vertices": math.nan,
            "number_of_edges": math.nan,
            "number_of_SL_vertices": math.nan,
            "number_of_SR_vertices": math.nan,
            "number_of_D_vertices": math.nan,
            "edge_to_vertex_ratio": math.nan,
            "directed_graph_density": math.nan,
            "source_to_target_path_exists": False,
            "is_directed_acyclic_graph": False,
            "topological_sort_succeeded": False,
            "source_indegree": math.nan,
            "target_outdegree": math.nan,
            "number_of_simple_source_to_target_paths": math.nan,
            "success": False,
            "failure_stage": "unknown",
            "failure_reason": "",
            "timeout_occurred": False,
            "solver_status_relaxed": "",
            "solver_status_fixed": "",
            "final_feasibility_validated": False,
            "graph_construction_time_ms": 0.0,
            "model_build_time_ms": 0.0,
            "relaxed_solve_time_ms": 0.0,
            "path_recovery_search_time_ms": 0.0,
            "fixed_path_solve_time_ms": 0.0,
            "time_to_first_feasible_ms": math.nan,
            "total_elapsed_time_ms": 0.0,
            "C_relax": math.nan,
            "C_round": math.nan,
            "final_objective": math.nan,
            "delta_relax": math.nan,
            "delta_relax_percent": math.nan,
            "selected_path_length": math.nan,
            "actual_number_of_footsteps": math.nan,
            "number_of_recovery_candidates_tested": 0,
            "selected_region_sequence": "",
            "selected_support_mode_sequence": "",
            "number_of_edges_with_positive_relaxed_flow": math.nan,
            "number_of_edges_with_fractional_flow": math.nan,
            "maximum_fractional_edge_value": math.nan,
            "relaxed_flow_entropy": math.nan,
        }
    )
    return row


def _classify_relaxed_failure(task: Any, deadline: float) -> str:
    if time.perf_counter() >= deadline:
        return "relaxed_solver_timeout"
    status = _status(task).lower()
    return "relaxed_solver_infeasible" if "infeas" in status else "relaxed_solver_unknown"


def _relaxed_flow_diagnostics(y_relaxed: np.ndarray) -> dict[str, Any]:
    values = np.asarray(y_relaxed, dtype=float).reshape(-1)
    eps = 1e-8
    positive = values > eps
    fractional = positive & (values < 1.0 - eps)
    fractional_values = values[fractional]
    positive_values = np.clip(values[positive], 0.0, None)
    if positive_values.size and float(np.sum(positive_values)) > 0.0:
        probabilities = positive_values / float(np.sum(positive_values))
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
    else:
        entropy = 0.0
    return {
        "number_of_edges_with_positive_relaxed_flow": int(np.sum(positive)),
        "number_of_edges_with_fractional_flow": int(np.sum(fractional)),
        "maximum_fractional_edge_value": (
            float(np.max(fractional_values)) if fractional_values.size else 0.0
        ),
        "relaxed_flow_entropy": entropy,
    }


def _run_trial(
    case: legacy.TerrainCase, configuration: str, base_seed: int, solver_log: int
) -> dict[str, Any]:
    """Run one graph/planner pair with a cooperative five-second deadline."""
    _seed_everything(case.terrain_seed)
    row = _blank_row(case, configuration)
    row["recovery_base_seed"] = int(base_seed)
    total_start = time.perf_counter()
    deadline = total_start + PLANNING_DEADLINE_SEC
    audit_time_excluded = 0.0
    task = None

    try:
        graph_start = time.perf_counter()
        try:
            problem = _prepare_problem(case, configuration)
        except Exception as error:
            row["failure_stage"] = "graph_construction"
            row["failure_reason"] = _error_text(error)
            return row
        finally:
            row["graph_construction_time_ms"] = (
                time.perf_counter() - graph_start
            ) * 1000.0

        row["number_of_vertices"] = problem.vertex_number
        row["number_of_edges"] = problem.n_edge
        row["number_of_SL_vertices"] = problem.modes.count("SL")
        row["number_of_SR_vertices"] = problem.modes.count("SR")
        row["number_of_D_vertices"] = problem.modes.count("D")
        row["edge_to_vertex_ratio"] = problem.n_edge / problem.vertex_number
        possible_directed = problem.vertex_number * (problem.vertex_number - 1)
        row["directed_graph_density"] = (
            problem.n_edge / possible_directed if possible_directed else math.nan
        )

        # Structural DAG checks are part of graph construction and therefore of
        # planner time. Simple-path counting is a separate audit diagnostic.
        audit_start = time.perf_counter()
        graph_audit = _graph_audit(problem)
        row["graph_construction_time_ms"] += (
            time.perf_counter() - audit_start
        ) * 1000.0
        for key in (
            "source_to_target_path_exists", "is_directed_acyclic_graph",
            "topological_sort_succeeded", "source_indegree", "target_outdegree",
        ):
            row[key] = graph_audit[key]

        path_count_start = time.perf_counter()
        row["number_of_simple_source_to_target_paths"] = _count_simple_paths(
            problem, graph_audit["topological_order"]
        )
        path_count_duration = time.perf_counter() - path_count_start
        audit_time_excluded += path_count_duration
        deadline += path_count_duration

        if not graph_audit["is_directed_acyclic_graph"]:
            row["failure_stage"] = "graph_not_acyclic"
            row["failure_reason"] = "constructed directed graph is not acyclic"
            return row
        if not graph_audit["all_edges_follow_fixed_forward_order"]:
            row["failure_stage"] = "graph_construction"
            row["failure_reason"] = "at least one edge violates the fixed forward ordering"
            return row
        if graph_audit["source_indegree"] != 0 or graph_audit["target_outdegree"] != 0:
            row["failure_stage"] = "graph_construction"
            row["failure_reason"] = "source indegree or target outdegree invariant failed"
            return row
        if not graph_audit["source_to_target_path_exists"]:
            row["failure_stage"] = "graph_disconnected"
            row["failure_reason"] = "no directed source-to-target path before optimization"
            return row
        if time.perf_counter() >= deadline:
            row["failure_stage"] = "planning_deadline"
            row["failure_reason"] = "deadline reached after graph construction"
            row["timeout_occurred"] = True
            return row

        model_start = time.perf_counter()
        try:
            task, variables, idx, _, _ = build_gcs_model_VAE_optapi(
                problem.vertices,
                problem.edge,
                STEP_SPAN,
                math.radians(YAW_BOUND_DEG),
                problem.modes,
                problem.x_ref,
                problem.y_ref,
            )
        except Exception as error:
            row["failure_stage"] = "model_build"
            row["failure_reason"] = _error_text(error)
            return row
        finally:
            row["model_build_time_ms"] = (time.perf_counter() - model_start) * 1000.0

        if time.perf_counter() >= deadline:
            row["failure_stage"] = "planning_deadline"
            row["failure_reason"] = "deadline reached after model construction"
            row["timeout_occurred"] = True
            return row

        relaxed_start = time.perf_counter()
        try:
            parameters = pack_relaxed_params_for_model(
                centroid_mat=problem.centroid_mat,
                n_edge=int(idx["n_edge"]),
                M_large=M_LARGE,
                start_L=problem.start_L,
                start_R=problem.start_R,
                target_L=problem.target_L,
                target_R=problem.target_R,
            )
            set_parameters_optapi(task, variables, parameters)
            relaxed_solution, relaxed_objective = solve_relaxed_optapi(
                task, variables, log_level=int(solver_log)
            )
        except Exception as error:
            row["solver_status_relaxed"] = _status(task)
            row["failure_stage"] = _classify_relaxed_failure(task, deadline)
            row["failure_reason"] = _error_text(error)
            row["timeout_occurred"] = row["failure_stage"] == "relaxed_solver_timeout"
            return row
        finally:
            row["relaxed_solve_time_ms"] = (
                time.perf_counter() - relaxed_start
            ) * 1000.0

        row["solver_status_relaxed"] = _status(task)
        if (
            not _acceptable_status(task)
            or relaxed_objective is None
            or not np.isfinite(float(relaxed_objective))
        ):
            row["failure_stage"] = _classify_relaxed_failure(task, deadline)
            row["failure_reason"] = (
                f"unacceptable relaxed status/objective: {_status(task)}, {relaxed_objective}"
            )
            row["timeout_occurred"] = row["failure_stage"] == "relaxed_solver_timeout"
            return row

        y_relaxed = np.asarray(relaxed_solution["y"], dtype=float).reshape(-1)
        if y_relaxed.size != problem.n_edge or not np.all(np.isfinite(y_relaxed)):
            row["failure_stage"] = "relaxed_solver_unknown"
            row["failure_reason"] = "relaxed edge vector is nonfinite or has the wrong size"
            return row
        row["C_relax"] = float(relaxed_objective)
        row.update(_relaxed_flow_diagnostics(y_relaxed))

        if time.perf_counter() >= deadline:
            row["failure_stage"] = "planning_deadline"
            row["failure_reason"] = "deadline reached before path recovery"
            row["timeout_occurred"] = True
            return row

        last_stage = "path_recovery"
        last_reason = "no recovery candidate attempted"
        for candidate_index in range(RECOVERY_CANDIDATE_BUDGET):
            if time.perf_counter() >= deadline:
                row["failure_stage"] = "planning_deadline"
                row["failure_reason"] = "deadline reached before first feasible fixed path"
                row["timeout_occurred"] = True
                return row
            row["number_of_recovery_candidates_tested"] = candidate_index + 1

            search_start = time.perf_counter()
            try:
                if candidate_index == 0:
                    candidate_path = relaxation_max(
                        y_relaxed,
                        problem.edge,
                        problem.vertex_number,
                        problem.vertex_number,
                        max_backtracks=MAX_BACKTRACKS,
                    )
                else:
                    candidate_path = relaxation_prob(
                        y_relaxed,
                        problem.edge,
                        problem.vertex_number,
                        temperature=TEMPERATURE,
                        seed=_recovery_seed(
                            base_seed, case.terrain_id, candidate_index - 1
                        ),
                        max_backtracks=MAX_BACKTRACKS,
                    )
                candidate_path = [int(node) for node in candidate_path]
                if not _valid_path(candidate_path, problem.edge, problem.vertex_number):
                    raise ValueError("recovery returned an invalid source-to-target path")
                fixed_y = path_to_y(problem.edge, candidate_path, problem.n_edge)
            except Exception as error:
                row["path_recovery_search_time_ms"] += (
                    time.perf_counter() - search_start
                ) * 1000.0
                last_stage = "path_recovery"
                last_reason = _error_text(error)
                continue
            row["path_recovery_search_time_ms"] += (
                time.perf_counter() - search_start
            ) * 1000.0

            fixed_start = time.perf_counter()
            try:
                _, objective, solution = _solve_fixed_once_optapi(
                    task,
                    variables,
                    idx,
                    centroid_mat=problem.centroid_mat,
                    y_relax=fixed_y,
                    M_small=M_SMALL,
                    start_L=problem.start_L,
                    start_R=problem.start_R,
                    target_L=problem.target_L,
                    target_R=problem.target_R,
                    log_level=int(solver_log),
                )
            except Exception as error:
                row["fixed_path_solve_time_ms"] += (
                    time.perf_counter() - fixed_start
                ) * 1000.0
                row["solver_status_fixed"] = _status(task)
                last_stage = "fixed_path_solver"
                last_reason = _error_text(error)
                continue
            row["fixed_path_solve_time_ms"] += (
                time.perf_counter() - fixed_start
            ) * 1000.0
            row["solver_status_fixed"] = _status(task)
            if (
                not _acceptable_status(task)
                or objective is None
                or not np.isfinite(float(objective))
            ):
                last_stage = "fixed_path_solver"
                last_reason = (
                    f"unacceptable fixed status/objective: {_status(task)}, {objective}"
                )
                continue

            try:
                processed = postprocess_solution(
                    solution,
                    problem.edge,
                    candidate_path,
                    problem.startpos_L,
                    problem.startpos_R,
                    problem.targetpos_L,
                    problem.targetpos_R,
                    problem.target_L,
                    problem.target_R,
                )
            except Exception as error:
                last_stage = "final_feasibility_check"
                last_reason = _error_text(error)
                continue

            # This is the existing CVAE OptAPI success criterion: acceptable,
            # finite fixed solve plus successful existing postprocessing.  No new
            # terminal tolerance is introduced here.
            row["success"] = True
            row["failure_stage"] = ""
            row["failure_reason"] = ""
            row["final_feasibility_validated"] = True
            row["C_round"] = float(objective)
            row["final_objective"] = float(objective)
            if abs(float(relaxed_objective)) > np.finfo(float).eps:
                gap = (float(objective) - float(relaxed_objective)) / float(
                    relaxed_objective
                )
            else:
                gap = math.nan
            row["delta_relax"] = gap
            row["delta_relax_percent"] = 100.0 * gap if np.isfinite(gap) else math.nan
            row["selected_path_length"] = len(candidate_path) - 1
            row["actual_number_of_footsteps"] = int(
                len(processed.L.x) + len(processed.R.x)
            )
            row["selected_region_sequence"] = "-".join(
                str(problem.region_of_vertex[node - 1]) for node in candidate_path
            )
            row["selected_support_mode_sequence"] = "-".join(
                problem.modes[node - 1] for node in candidate_path
            )
            row["time_to_first_feasible_ms"] = (
                time.perf_counter() - total_start - audit_time_excluded
            ) * 1000.0
            # Memory-only field used for nominal equivalence checking.  The CSV
            # writer ignores it by design.
            row["_selected_path_nodes"] = candidate_path
            return row

        row["failure_stage"] = last_stage
        row["failure_reason"] = last_reason
        return row
    finally:
        row["total_elapsed_time_ms"] = max(
            0.0, (time.perf_counter() - total_start - audit_time_excluded) * 1000.0
        )
        _close_task(task)


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2)) / denominator
    return 100.0 * (center - radius), 100.0 * (center + radius)


def _configuration_summary(
    rows: Sequence[Mapping[str, Any]], configuration: str, group: str
) -> Optional[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("configuration")) == configuration]
    if not selected:
        return None
    successful = [row for row in selected if _as_bool(row.get("success"))]
    failures = [str(row.get("failure_stage", "")) for row in selected if not _as_bool(row.get("success"))]
    lower, upper = _wilson_interval(len(successful), len(selected))
    gaps = _finite_values(successful, "delta_relax_percent")
    first_time = _finite_values(successful, "time_to_first_feasible_ms")
    elapsed = _finite_values(selected, "total_elapsed_time_ms")
    solver_stages = {
        "model_build", "relaxed_solver_infeasible", "relaxed_solver_unknown",
        "fixed_path_solver",
    }
    recovery_stages = {"path_recovery", "final_feasibility_check"}
    deadline_stages = {"planning_deadline", "relaxed_solver_timeout"}
    return {
        "experiment_group": group,
        "configuration": configuration,
        "number_of_trials": len(selected),
        "vertices_mean": _mean(_finite_values(selected, "number_of_vertices")),
        "edges_mean": _mean(_finite_values(selected, "number_of_edges")),
        "edge_to_vertex_ratio_mean": _mean(_finite_values(selected, "edge_to_vertex_ratio")),
        "graph_density_mean": _mean(_finite_values(selected, "directed_graph_density")),
        "number_of_successes": len(successful),
        "success_rate_percent": 100.0 * len(successful) / len(selected),
        "wilson_95_ci_lower_percent": lower,
        "wilson_95_ci_upper_percent": upper,
        "graph_disconnected_count": failures.count("graph_disconnected"),
        "graph_insufficient_count": failures.count("graph_insufficient"),
        "solver_failure_count": sum(stage in solver_stages for stage in failures),
        "recovery_failure_count": sum(stage in recovery_stages for stage in failures),
        "deadline_failure_count": sum(stage in deadline_stages for stage in failures),
        "delta_relax_mean_percent": _mean(gaps),
        "delta_relax_std_percent": _std(gaps),
        "delta_relax_median_percent": _quantile(gaps, 0.5),
        "delta_relax_q25_percent": _quantile(gaps, 0.25),
        "delta_relax_q75_percent": _quantile(gaps, 0.75),
        "time_to_first_feasible_mean_ms": _mean(first_time),
        "time_to_first_feasible_std_ms": _std(first_time),
        "time_to_first_feasible_median_ms": _quantile(first_time, 0.5),
        "time_to_first_feasible_q25_ms": _quantile(first_time, 0.25),
        "time_to_first_feasible_q75_ms": _quantile(first_time, 0.75),
        "total_elapsed_time_mean_ms": _mean(elapsed),
        "total_elapsed_time_median_ms": _quantile(elapsed, 0.5),
        "final_objective_mean_success_only": _mean(_finite_values(successful, "final_objective")),
        "final_objective_median_success_only": _quantile(_finite_values(successful, "final_objective"), 0.5),
        "actual_number_of_footsteps_mean_success_only": _mean(
            _finite_values(successful, "actual_number_of_footsteps")
        ),
        "fractional_edges_mean_success_only": _mean(
            _finite_values(successful, "number_of_edges_with_fractional_flow")
        ),
        "relaxed_flow_entropy_mean_success_only": _mean(
            _finite_values(successful, "relaxed_flow_entropy")
        ),
    }


def _summary_rows(
    raw_rows: Sequence[Mapping[str, Any]], configurations: Sequence[str], group: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for configuration in configurations:
        summary = _configuration_summary(raw_rows, configuration, group)
        if summary is not None:
            output.append(summary)
    return output


def _paired_rows(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (int(_as_float(row.get("terrain_id"))), str(row.get("configuration"))): row
        for row in raw_rows
        if np.isfinite(_as_float(row.get("terrain_id")))
    }
    terrain_ids = sorted(terrain_id for terrain_id, config in indexed if config == "nominal")
    paired: list[dict[str, Any]] = []
    for terrain_id in terrain_ids:
        nominal = indexed[(terrain_id, "nominal")]
        for alternative in ALTERNATIVES:
            other = indexed.get((terrain_id, alternative))
            if other is None:
                continue
            nominal_success = _as_bool(nominal.get("success"))
            alternative_success = _as_bool(other.get("success"))
            both = nominal_success and alternative_success
            nominal_gap = _as_float(nominal.get("delta_relax_percent"))
            other_gap = _as_float(other.get("delta_relax_percent"))
            nominal_time = _as_float(nominal.get("time_to_first_feasible_ms"))
            other_time = _as_float(other.get("time_to_first_feasible_ms"))
            nominal_objective = _as_float(nominal.get("final_objective"))
            other_objective = _as_float(other.get("final_objective"))
            paired.append(
                {
                    "terrain_id": terrain_id,
                    "experiment_group": _configuration_group(alternative),
                    "alternative_configuration": alternative,
                    "nominal_success": nominal_success,
                    "alternative_success": alternative_success,
                    "nominal_failure_stage": nominal.get("failure_stage", ""),
                    "alternative_failure_stage": other.get("failure_stage", ""),
                    "nominal_vertices": nominal.get("number_of_vertices", math.nan),
                    "alternative_vertices": other.get("number_of_vertices", math.nan),
                    "nominal_edges": nominal.get("number_of_edges", math.nan),
                    "alternative_edges": other.get("number_of_edges", math.nan),
                    "nominal_delta_relax_percent": nominal_gap if both else math.nan,
                    "alternative_delta_relax_percent": other_gap if both else math.nan,
                    "delta_relax_difference_percentage_points": (
                        other_gap - nominal_gap if both else math.nan
                    ),
                    "nominal_time_to_first_feasible_ms": nominal_time if both else math.nan,
                    "alternative_time_to_first_feasible_ms": other_time if both else math.nan,
                    "time_difference_ms": other_time - nominal_time if both else math.nan,
                    "time_ratio": (
                        other_time / nominal_time
                        if both and np.isfinite(nominal_time) and nominal_time != 0.0
                        else math.nan
                    ),
                    "nominal_final_objective": nominal_objective if both else math.nan,
                    "alternative_final_objective": other_objective if both else math.nan,
                    "objective_difference": other_objective - nominal_objective if both else math.nan,
                    "nominal_actual_number_of_footsteps": (
                        nominal.get("actual_number_of_footsteps", math.nan) if both else math.nan
                    ),
                    "alternative_actual_number_of_footsteps": (
                        other.get("actual_number_of_footsteps", math.nan) if both else math.nan
                    ),
                }
            )
    return paired


def _failure_rows(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        selected = [row for row in raw_rows if str(row.get("configuration")) == configuration]
        if not selected:
            continue
        counts: dict[str, int] = {}
        for row in selected:
            if _as_bool(row.get("success")):
                continue
            stage = str(row.get("failure_stage") or "unknown")
            counts[stage] = counts.get(stage, 0) + 1
        for stage in sorted(counts):
            output.append(
                {
                    "experiment_group": _configuration_group(configuration),
                    "configuration": configuration,
                    "failure_stage": stage,
                    "count": counts[stage],
                    "percent_of_trials": 100.0 * counts[stage] / len(selected),
                }
            )
    return output


def _write_analysis(output_dir: Path, raw_rows: Sequence[Mapping[str, Any]]) -> None:
    duplicate_rows = _summary_rows(
        raw_rows, tuple(DUPLICATE_CONFIGS), "duplicate_sensitivity"
    )
    # Nominal is the shared reference and is repeated only in this summary view,
    # not as a second raw trial.
    edge_rows = _summary_rows(
        raw_rows, ("nominal",) + EDGE_CONFIGS, "edge_density"
    )
    _atomic_write_csv(
        output_dir / "summary_duplicate_sensitivity.csv", SUMMARY_FIELDS, duplicate_rows
    )
    _atomic_write_csv(
        output_dir / "summary_edge_density.csv", SUMMARY_FIELDS, edge_rows
    )
    _atomic_write_csv(
        output_dir / "paired_with_nominal.csv", PAIRED_FIELDS, _paired_rows(raw_rows)
    )
    _atomic_write_csv(
        output_dir / "failure_breakdown.csv", FAILURE_FIELDS, _failure_rows(raw_rows)
    )


def _terrain_rows(cases: Sequence[legacy.TerrainCase]) -> list[dict[str, Any]]:
    output = []
    for case in cases:
        start = case.stones[0]
        goal = case.stones[3]
        output.append(
            {
                "terrain_id": case.terrain_id,
                "terrain_seed": case.terrain_seed,
                "alpha": ALPHA,
                "number_of_physical_regions": 4,
                "start_region_index": 0,
                "goal_region_index": 3,
                "terrain_file": str(case.path),
                "terrain_file_sha256": _sha256(case.path),
                "start_region": json.dumps(
                    {"name": start.get("name", "stone_1"), "center": start.get("center")},
                    separators=(",", ":"),
                ),
                "goal_region": json.dumps(
                    {"name": goal.get("name", "stone_4"), "center": goal.get("center")},
                    separators=(",", ":"),
                ),
            }
        )
    return output


def _graph_configuration_audit(
    case_zero: legacy.TerrainCase, reuse_note: str, recovery_base_seed: int
) -> str:
    sections = [
        "Extended graph-configuration audit",
        "==================================",
        "",
        "Dataset and fixed planner settings",
        "----------------------------------",
        f"- CVAE terrain directory: {case_zero.path.parent}",
        "- alpha=0.0; 100 JSON terrains; exactly four physical regions.",
        "- region 0 is the physical start and region 3 the physical goal; neither is extra.",
        "- terrain_seed=123+terrain_id, matching the existing CVAE generator convention.",
        f"- step_span={STEP_SPAN}, yaw_bound={YAW_BOUND_DEG} deg, stance_width={STANCE_WIDTH}.",
        f"- M_large={M_LARGE}, M_small={M_SMALL}; existing objective and SDP/RLT model unchanged.",
        f"- greedy first, probabilistic fallback, first feasible early stop, candidate budget={RECOVERY_CANDIDATE_BUDGET}.",
        f"- cooperative deadline={PLANNING_DEADLINE_SEC} sec; MOSEK calls already running are not interrupted.",
        f"- SeedSequence([{int(recovery_base_seed)}, terrain_id, probabilistic_index]) is shared by configurations.",
        "",
        "Verified repository duplicate and mode rule",
        "------------------------------------------",
        "- TerrainGCS copies the same convex region; it does not split region geometry.",
        "- Start group: first copy D, then the repository's SL/SR alternation.",
        "- Middle groups: repository SL/SR alternation.",
        "- Goal group: preceding copies alternate and the last copy is target D.",
        "",
        "Actual graph configurations on terrain 0",
        "----------------------------------------",
    ]
    for configuration in CONFIGURATIONS:
        problem = _prepare_problem(case_zero, configuration)
        audit = _graph_audit(problem)
        path_count = _count_simple_paths(problem, audit["topological_order"])
        sections.append(
            f"- {configuration}: copy_counts={list(_copy_counts(configuration))}, "
            f"|V|={problem.vertex_number}, |E|={problem.n_edge}, "
            f"SL={problem.modes.count('SL')}, SR={problem.modes.count('SR')}, "
            f"D={problem.modes.count('D')}, DAG={audit['is_directed_acyclic_graph']}, "
            f"forward={audit['all_edges_follow_fixed_forward_order']}, "
            f"source_to_target={audit['source_to_target_path_exists']}, "
            f"paths(capped at {SIMPLE_PATH_COUNT_CAP})={path_count}"
        )

    nominal = _prepare_problem(case_zero, "nominal")
    dense = _prepare_problem(case_zero, "edge_dense_mode_compatible")
    fully = _prepare_problem(case_zero, "edge_fully_connected_forward")
    vertex_order = "\n".join(
        f"  {index}: region={nominal.region_of_vertex[index - 1]}, mode={nominal.modes[index - 1]}"
        for index in range(1, nominal.vertex_number + 1)
    )
    sections.extend(
        [
            "",
            "Edge definitions",
            "----------------",
            "- nominal uses TerrainGCS.create_edges_general exactly (sparse, parity/topology based DAG).",
            "- No geometric-distance pruning occurs in create_edges_general; step_span is an optimization SOC constraint.",
            "- edge_dense_mode_compatible is the existing Com10_3.py edge_dense definition: all u<v SL<->SR pairs,",
            "  source D only to start-region single-support copies, goal-region single-support copies only to target D.",
            f"  Actual dense edge count={dense.n_edge}; exact edges={dense.edge.tolist()}",
            "- edge_fully_connected_forward adds every u<v pair except direct source->target.",
            f"  Calculation: C(18,2)-1=152; actual={fully.n_edge}; exact edges={fully.edge.tolist()}",
            "- Fully-forward intentionally permits SL->SL and SR->SR graph candidates. Existing optimization constraints",
            "  are not removed or relaxed. The repository's edge-membership branch has explicit cases only for",
            "  D->SL/SR, SL<->SR, and SL/SR->D; this limitation is preserved rather than patched here.",
            "- Every graph is checked with Kahn topological sort, all edges u<v, source indegree=0, target outdegree=0,",
            "  and source-to-target reachability before any solver call.",
            f"- Nominal exact edges={nominal.edge.tolist()}",
            "",
            "Fixed nominal vertex ordering",
            "-----------------------------",
            vertex_order,
            "",
            "Timing definitions",
            "------------------",
            "- time_to_first_feasible_ms ends immediately after the first acceptable fixed solve and successful existing postprocess.",
            "- total_elapsed_time_ms records actual termination; capped simple-path counting is excluded from planner timing.",
            "- The five-second deadline is cooperative: it is checked between stages/candidates, not inside MOSEK optimize calls.",
            "",
            "Existing-result reuse audit",
            "---------------------------",
            reuse_note,
            "",
        ]
    )
    return "\n".join(sections)


def _reuse_audit(requested: bool, existing_dir: Path) -> str:
    if not requested:
        return "- --reuse-existing was not requested; zero legacy rows reused."
    old_raw = existing_dir / "raw_results.csv"
    old_terrain = existing_dir / "terrain_list.csv"
    if not old_raw.exists() or not old_terrain.exists():
        return (
            f"- Reuse requested but legacy files are missing in {existing_dir}; zero rows reused."
        )
    terrain_fields = set((_read_csv(old_terrain) or [{}])[0])
    raw_fields = set((_read_csv(old_raw) or [{}])[0])
    reasons = []
    if "terrain_file_sha256" not in terrain_fields:
        reasons.append("legacy terrain_list.csv has no per-JSON SHA-256 identity")
    if "time_to_first_feasible_ms" not in raw_fields:
        reasons.append("legacy raw timing does not explicitly store time-to-first-feasible")
    if "number_of_edges_with_fractional_flow" not in raw_fields:
        reasons.append("legacy raw rows omit the required relaxed-flow diagnostics")
    if reasons:
        return (
            "- Reuse requested, but equivalence could not be fully proven; zero rows reused. Reasons: "
            + "; ".join(reasons)
            + ". Missing extended rows are rerun."
        )
    return (
        "- Legacy schemas appear compatible, but automatic row conversion is intentionally disabled because "
        "solver/recovery provenance is not cryptographically embedded in each row; zero rows reused."
    )


def _reference_namespace(base_seed: int, solver_log: int) -> argparse.Namespace:
    return argparse.Namespace(
        step_span=STEP_SPAN,
        alpha_deg=YAW_BOUND_DEG,
        M_large=M_LARGE,
        M_small=M_SMALL,
        stance_width=STANCE_WIDTH,
        trials=RECOVERY_CANDIDATE_BUDGET,
        timeout_sec=PLANNING_DEADLINE_SEC,
        temperature=TEMPERATURE,
        max_backtracks=MAX_BACKTRACKS,
        solver_log=solver_log,
        base_seed=base_seed,
    )


def _validate_nominal(
    case: legacy.TerrainCase,
    experimental: Mapping[str, Any],
    base_seed: int,
    solver_log: int,
) -> str:
    selected_path = experimental.get("_selected_path_nodes")
    if not selected_path:
        rerun = _run_trial(case, "nominal", base_seed, solver_log)
        selected_path = rerun.get("_selected_path_nodes")
        experimental = rerun
    if not selected_path:
        raise AssertionError("nominal equivalence validation has no successful path")
    args = _reference_namespace(base_seed, solver_log)
    reference = legacy._reference_default_result(case, selected_path, args)
    if not _as_bool(reference.get("success")) or not _as_bool(experimental.get("success")):
        raise AssertionError(f"nominal success mismatch: reference={reference}, extended={experimental}")
    for field in ("number_of_vertices", "number_of_edges", "actual_number_of_footsteps"):
        if _as_float(reference.get(field)) != _as_float(experimental.get(field)):
            raise AssertionError(f"nominal {field} mismatch: reference={reference}, extended={experimental}")
    for reference_field, experimental_field in (
        ("C_relax", "C_relax"),
        ("final_objective", "C_round"),
    ):
        if not np.isclose(
            _as_float(reference.get(reference_field)),
            _as_float(experimental.get(experimental_field)),
            rtol=1e-7,
            atol=1e-6,
        ):
            raise AssertionError(
                f"nominal {reference_field} mismatch: reference={reference}, extended={experimental}"
            )
    expected_delta = (
        _as_float(reference["final_objective"]) - _as_float(reference["C_relax"])
    ) / _as_float(reference["C_relax"])
    if not np.isclose(expected_delta, _as_float(experimental.get("delta_relax")), rtol=1e-7, atol=1e-8):
        raise AssertionError("nominal delta_relax mismatch")
    return (
        "Nominal equivalence validation: PASS for terrain 0 using the identical selected path.\n"
        f"success=True, |V|={experimental['number_of_vertices']}, |E|={experimental['number_of_edges']}, "
        f"C_relax={experimental['C_relax']}, C_round={experimental['C_round']}, "
        f"delta_relax={experimental['delta_relax']}, footsteps={experimental['actual_number_of_footsteps']}.\n"
    )


def _exception_row(
    case: legacy.TerrainCase,
    configuration: str,
    error: BaseException,
    recovery_base_seed: int,
) -> dict[str, Any]:
    row = _blank_row(case, configuration)
    row["recovery_base_seed"] = int(recovery_base_seed)
    row["failure_stage"] = "exception"
    row["failure_reason"] = _error_text(error)
    return row


def _append_exception_log(
    path: Path, case: legacy.TerrainCase, configuration: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"terrain_id={case.terrain_id} configuration={configuration}\n"
            + traceback.format_exc()
            + "\n"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extended graph-construction sensitivity analysis for Comment 10.3"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true")
    action.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--config", choices=("all",) + CONFIGURATIONS, default="all")
    parser.add_argument("--group", choices=("duplicate", "edge"))
    parser.add_argument("--terrain-start", type=int, default=0)
    parser.add_argument(
        "--terrain-end", type=int, default=100,
        help="exclusive; 0--3 executes terrains 0, 1, and 2",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--terrain-dir", type=Path, default=DEFAULT_TERRAIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--existing-results-dir", type=Path, default=DEFAULT_EXISTING_DIR)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--terrain-key", default="terrain")
    parser.add_argument("--solver-log", type=int, default=0)
    parser.add_argument(
        "--validate-nominal", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.group and args.config != "all":
        parser.error("use either --group or a specific --config, not both")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.reuse_existing and not args.resume:
        parser.error("--reuse-existing requires --resume")
    if not 0 <= args.terrain_start < args.terrain_end <= 100:
        parser.error("terrain range must satisfy 0 <= start < end <= 100")
    return args


def _selected_configurations(args: argparse.Namespace) -> tuple[str, ...]:
    if args.group == "duplicate":
        requested = set(DUPLICATE_CONFIGS)
    elif args.group == "edge":
        requested = {"nominal", *EDGE_CONFIGS}
    elif args.config == "all":
        requested = set(CONFIGURATIONS)
    else:
        requested = {args.config}
    return tuple(item for item in EXECUTION_ORDER if item in requested)


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    raw_path = output_dir / "raw_results.csv"
    output_files = (
        raw_path,
        output_dir / "terrain_list.csv",
        output_dir / "graph_configuration_audit.txt",
        output_dir / "summary_duplicate_sensitivity.csv",
        output_dir / "summary_edge_density.csv",
        output_dir / "paired_with_nominal.csv",
        output_dir / "failure_breakdown.csv",
    )

    if args.analyze_only:
        rows = _read_csv(raw_path)
        if not rows:
            raise FileNotFoundError(f"no raw data found in {raw_path}")
        _write_analysis(output_dir, rows)
        print(f"Analyzed {len(rows)} raw rows")
        return 0

    discovery_args = argparse.Namespace(
        terrain_dir=Path(args.terrain_dir),
        terrain_key=args.terrain_key,
        # The existing CVAE alpha=0 JSON set was generated with seed 123+i.
        # --base-seed controls recovery only; it must not rewrite terrain provenance.
        base_seed=DEFAULT_BASE_SEED,
    )
    cases = legacy._discover_terrains(discovery_args)
    terrain_rows = _terrain_rows(cases)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        raw_rows: list[dict[str, Any]] = []
    elif args.resume:
        raw_rows = _read_csv(raw_path)
    else:
        if any(path.exists() for path in output_files):
            raise FileExistsError(
                f"results already exist in {output_dir}; use --resume or --overwrite"
            )
        raw_rows = []

    existing_terrain_rows = _read_csv(output_dir / "terrain_list.csv")
    if args.resume and raw_rows:
        old_hashes = {
            int(_as_float(row.get("terrain_id"))): str(row.get("terrain_file_sha256", ""))
            for row in existing_terrain_rows
            if np.isfinite(_as_float(row.get("terrain_id")))
        }
        new_hashes = {
            int(row["terrain_id"]): str(row["terrain_file_sha256"])
            for row in terrain_rows
        }
        if not old_hashes or old_hashes != new_hashes:
            raise ValueError(
                "--resume terrain provenance mismatch: existing terrain_list SHA-256 values "
                "do not match the requested 100 JSON files"
            )
        saved_seeds = {
            int(_as_float(row.get("recovery_base_seed")))
            for row in raw_rows
            if np.isfinite(_as_float(row.get("recovery_base_seed")))
        }
        if saved_seeds != {int(args.base_seed)}:
            raise ValueError(
                "--resume recovery seed mismatch or missing recovery_base_seed in raw rows"
            )
    _atomic_write_csv(output_dir / "terrain_list.csv", TERRAIN_FIELDS, terrain_rows)
    reuse_note = _reuse_audit(bool(args.reuse_existing), Path(args.existing_results_dir))
    audit_text = _graph_configuration_audit(cases[0], reuse_note, args.base_seed)
    _atomic_write_text(output_dir / "graph_configuration_audit.txt", audit_text)

    completed = {
        (int(_as_float(row.get("terrain_id"))), str(row.get("configuration")))
        for row in raw_rows
        if np.isfinite(_as_float(row.get("terrain_id")))
    }
    configurations = _selected_configurations(args)
    selected_cases = cases[args.terrain_start : args.terrain_end]
    total = len(configurations) * len(selected_cases)
    ordinal = 0
    nominal_validation_done = False

    for configuration in configurations:
        for case in selected_cases:
            ordinal += 1
            key = (case.terrain_id, configuration)
            if key in completed:
                print(
                    f"[{ordinal}/{total}] SKIP terrain={case.terrain_id} config={configuration}"
                )
                continue
            print(f"[{ordinal}/{total}] RUN terrain={case.terrain_id} config={configuration}")
            try:
                row = _run_trial(case, configuration, args.base_seed, args.solver_log)
            except Exception as error:
                _append_exception_log(
                    output_dir / "exceptions.log", case, configuration
                )
                row = _exception_row(case, configuration, error, args.base_seed)
            raw_rows.append(row)
            completed.add(key)
            raw_rows.sort(
                key=lambda item: (
                    int(_as_float(item.get("terrain_id"))),
                    CONFIGURATIONS.index(str(item.get("configuration"))),
                )
            )
            _atomic_write_csv(raw_path, RAW_FIELDS, raw_rows)
            _write_analysis(output_dir, raw_rows)

            if (
                args.validate_nominal
                and case.terrain_id == 0
                and configuration == "nominal"
            ):
                try:
                    validation = _validate_nominal(
                        case, row, args.base_seed, args.solver_log
                    )
                except Exception as error:
                    _append_exception_log(
                        output_dir / "exceptions.log", case, "nominal_validation"
                    )
                    validation = "Nominal equivalence validation: FAIL: " + _error_text(error) + "\n"
                audit_text += "\nNominal equivalence validation\n------------------------------\n" + validation
                _atomic_write_text(
                    output_dir / "graph_configuration_audit.txt", audit_text
                )
                nominal_validation_done = True

    if (
        args.validate_nominal
        and "nominal" in configurations
        and args.terrain_start <= 0 < args.terrain_end
        and not nominal_validation_done
    ):
        existing = next(
            (
                row
                for row in raw_rows
                if int(_as_float(row.get("terrain_id"))) == 0
                and str(row.get("configuration")) == "nominal"
            ),
            None,
        )
        if existing is not None:
            try:
                validation = _validate_nominal(
                    cases[0], existing, args.base_seed, args.solver_log
                )
            except Exception as error:
                _append_exception_log(
                    output_dir / "exceptions.log", cases[0], "nominal_validation"
                )
                validation = "Nominal equivalence validation: FAIL: " + _error_text(error) + "\n"
            audit_text += "\nNominal equivalence validation\n------------------------------\n" + validation
            _atomic_write_text(
                output_dir / "graph_configuration_audit.txt", audit_text
            )

    _write_analysis(output_dir, raw_rows)
    print(f"Saved {len(raw_rows)} raw rows to {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
