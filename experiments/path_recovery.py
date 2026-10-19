#!/usr/bin/env python3
"""Reviewer Comment 1.5 path-recovery benchmark.

This executable is intentionally self-contained.  It imports the repository's
CVAE graph/model/fixed-solve/postprocessing functions without modifying them.
For each terrain it solves the relaxation exactly once and compares three
recovery settings against the same immutable relaxed edge vector.

Only these files are written below ``--output-dir``:

* per_candidate_results.csv
* per_terrain_results.csv
* recovery_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import mosek
import numpy as np

from src.graph import build_graph, compute_centroids_from_vertices
from src.model import build_gcs_model_VAE_optapi, pack_relaxed_params_for_model
from src.solver.path import path_to_y, relaxation_max
from src.solver.stochastic import _solve_fixed_once_optapi
from src.solver.utils import set_parameters_optapi, solve_relaxed_optapi
from src.terrain_preset import load_terrain_yaw
from src.utils import expand_shorthand, postprocess_solution


DEFAULT_TERRAIN_DIR = Path("data/terrains/alpha")
DEFAULT_OUTPUT_DIR = Path("outputs/Com_1_5")
DEFAULT_COPY_COUNTS_4_REGION = (5, 4, 4, 5)
CSV_FLOAT_FORMAT = ".17g"


PER_CANDIDATE_FIELDS = [
    "alpha",
    "alpha_index",
    "terrain_id",
    "terrain_index",
    "terrain_file",
    "terrain_seed",
    "strategy",
    "candidate_index",
    "recovery_seed",
    "temperature",
    "max_backtracks",
    "num_vertices",
    "num_edges",
    "relaxed_y_hash",
    "relax_solver_status",
    "relax_success",
    "C_relax",
    "relax_time_ms",
    "path_generation_success",
    "path_node_sequence",
    "path_hash",
    "path_length",
    "duplicate_path",
    "previously_seen_candidate_index",
    "probabilistic_internal_greedy_fallback_used",
    "path_generation_time_ms",
    "fixed_solver_status",
    "fixed_success",
    "fixed_objective",
    "fixed_solve_time_ms",
    "total_candidate_time_ms",
    "goal_reached",
    "failure_reason",
    "error_message",
]


PER_TERRAIN_FIELDS = [
    "alpha",
    "alpha_index",
    "terrain_id",
    "terrain_index",
    "terrain_file",
    "terrain_seed",
    "num_vertices",
    "num_edges",
    "relaxed_y_hash",
    "relax_solver_status",
    "relax_success",
    "C_relax",
    "relax_time_ms",
    "greedy_path_generation_success",
    "greedy_fixed_success",
    "greedy_fixed_objective",
    "greedy_path_generation_time_ms",
    "greedy_fixed_solve_time_ms",
    "greedy_total_recovery_time_ms",
    "probabilistic_trial_count",
    "probabilistic_success_count",
    "probabilistic_failure_count",
    "probabilistic_any_success",
    "probabilistic_first_success_index",
    "probabilistic_best_objective",
    "probabilistic_mean_fixed_objective",
    "probabilistic_total_recovery_time_ms",
    "probabilistic_unique_path_count",
    "probabilistic_duplicate_path_count",
    "combined_success",
    "combined_selected_strategy",
    "combined_candidates_attempted",
    "combined_probabilistic_candidates_attempted",
    "combined_candidates_until_success",
    "combined_fixed_objective",
    "combined_total_recovery_time_ms",
    "fallback_activated",
    "fallback_success",
    "all_candidates_failed",
    "final_failure_reason",
    "error_message",
]


SUMMARY_FIELDS = [
    "alpha",
    "num_terrains",
    "num_relax_success",
    "num_relax_failure",
    "greedy_success_count",
    "greedy_feasibility_rate_percent",
    "greedy_mean_recovery_time_ms",
    "greedy_std_recovery_time_ms",
    "greedy_mean_fixed_objective",
    "greedy_std_fixed_objective",
    "probabilistic_total_candidate_runs",
    "probabilistic_successful_candidate_runs",
    "probabilistic_single_trial_feasibility_rate_percent",
    "probabilistic_terrain_success_count",
    "probabilistic_per_terrain_feasibility_rate_percent",
    "probabilistic_mean_recovery_time_ms",
    "probabilistic_std_recovery_time_ms",
    "probabilistic_mean_fixed_objective",
    "probabilistic_std_fixed_objective",
    "combined_success_count",
    "combined_feasibility_rate_percent",
    "combined_mean_recovery_time_ms",
    "combined_std_recovery_time_ms",
    "combined_mean_candidates_attempted",
    "combined_std_candidates_attempted",
    "combined_mean_candidates_until_success",
    "combined_mean_fixed_objective",
    "combined_std_fixed_objective",
    "fallback_activation_count",
    "fallback_activation_rate_percent",
    "fallback_success_count",
    "fallback_success_rate_percent",
    "all_candidates_failed_count",
    "all_candidates_failed_rate_percent",
]


FAILURE_REASONS = {
    "none",
    "relax_solver_failure",
    "no_source_target_path",
    "max_backtracks_exceeded",
    "invalid_path_edge",
    "fixed_problem_infeasible",
    "fixed_solver_error",
    "nonfinite_objective",
    "postprocess_failure",
    "goal_not_reached",
    "candidate_budget_exhausted",
    "unknown",
}


class PureProbabilisticPathError(RuntimeError):
    """Path-generation failure with a CSV-compatible reason."""

    def __init__(self, reason: str, message: str):
        if reason not in FAILURE_REASONS:
            raise ValueError(f"unsupported failure reason: {reason}")
        super().__init__(message)
        self.reason = reason


def relaxation_prob_pure(
    y: np.ndarray,
    edge: np.ndarray,
    vertex_number: int,
    *,
    temperature: float = 1.0,
    seed: Optional[int] = None,
    max_backtracks: int = 20000,
) -> List[int]:
    """Pure version of ``src.solver.path.relaxation_prob``.

    Outgoing-edge construction, weighted sampling, DFS state, and backtracking
    match the repository function.  The only intentional difference is that a
    failed probabilistic DFS raises an explicit error instead of invoking
    ``relaxation_max``.  Consequently this helper never uses a greedy fallback.
    """

    y = np.asarray(y, dtype=float).reshape(-1)
    edge = np.asarray(edge, dtype=int).reshape(-1, 2)
    n_edge = int(edge.shape[0])
    start, goal = 1, int(vertex_number)
    if y.size != n_edge:
        raise ValueError(f"y/edge length mismatch: {y.size} != {n_edge}")
    if int(max_backtracks) < 1:
        raise ValueError("max_backtracks must be at least 1")

    rng = np.random.default_rng(seed)
    eps = 1e-12
    temp = max(1e-6, float(temperature))

    adjacency: Dict[int, List[Tuple[int, int, float]]] = {
        node: [] for node in range(1, int(vertex_number) + 1)
    }
    for edge_index in range(n_edge):
        u, v = int(edge[edge_index, 0]), int(edge[edge_index, 1])
        adjacency[u].append((v, edge_index, float(y[edge_index])))

    def pick_next(
        node: int,
        banned_edges: set[int],
        banned_nodes: set[int],
    ) -> Optional[Tuple[int, int, float]]:
        candidates = [
            (v, edge_index, weight)
            for v, edge_index, weight in adjacency.get(node, [])
            if edge_index not in banned_edges and v not in banned_nodes
        ]
        if not candidates:
            return None
        weights = np.asarray([weight for _, _, weight in candidates], dtype=float)
        weights = np.power(weights + eps, 1.0 / temp)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            probabilities = np.full(len(candidates), 1.0 / len(candidates))
        else:
            probabilities = weights / total
        chosen = int(rng.choice(len(candidates), p=probabilities))
        return candidates[chosen]

    path_nodes = [start]
    visited_nodes = {start}
    used_edges: set[int] = set()
    tried_per_node: Dict[int, set[int]] = {
        node: set() for node in range(1, int(vertex_number) + 1)
    }
    explored = 0
    limit_hit = False

    def dfs(node: int) -> bool:
        nonlocal explored, limit_hit
        if node == goal:
            return True
        while True:
            explored += 1
            if explored > int(max_backtracks):
                limit_hit = True
                return False
            choice = pick_next(
                node,
                tried_per_node[node] | used_edges,
                visited_nodes,
            )
            if choice is None:
                return False
            next_node, edge_index, _ = choice
            tried_per_node[node].add(edge_index)
            path_nodes.append(next_node)
            visited_nodes.add(next_node)
            used_edges.add(edge_index)
            if dfs(next_node):
                return True
            path_nodes.pop()
            visited_nodes.remove(next_node)
            used_edges.remove(edge_index)

    if dfs(start):
        return path_nodes

    if limit_hit:
        raise PureProbabilisticPathError(
            "max_backtracks_exceeded",
            f"probabilistic DFS exceeded max_backtracks={max_backtracks}",
        )
    raise PureProbabilisticPathError(
        "no_source_target_path",
        "probabilistic DFS found no source-to-target path",
    )


@dataclass(frozen=True)
class TerrainCase:
    alpha: float
    alpha_index: int
    terrain_index: int
    path: Path
    terrain_seed: Optional[int]

    @property
    def alpha_label(self) -> str:
        return str(float(self.alpha))

    @property
    def key(self) -> Tuple[str, int, str]:
        return (self.alpha_label, int(self.terrain_index), str(self.path.resolve()))


@dataclass
class TerrainProblem:
    stones: List[Dict[str, Any]]
    graph: Dict[str, Any]
    vertices: List[np.ndarray]
    centroid_mat: np.ndarray
    startpos_L: np.ndarray
    startpos_R: np.ndarray
    targetpos_L: np.ndarray
    targetpos_R: np.ndarray
    start_L: np.ndarray
    start_R: np.ndarray
    target_L: np.ndarray
    target_R: np.ndarray
    x_ref: np.ndarray
    y_ref: np.ndarray


def _pose5(position: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Pure equivalent of prepare_problem_params' pose packing.

    The repository helper also mutates a module-level reference-frame cache.
    This benchmark deliberately avoids that global-state mutation.
    """

    position = np.asarray(position, dtype=float).reshape(-1)
    cosine = float(np.cos(float(yaw_rad)))
    sine = float(np.sin(float(yaw_rad)))
    norm = float(np.hypot(cosine, sine)) or 1.0
    return np.asarray(
        [position[0], position[1], position[2], cosine / norm, sine / norm],
        dtype=float,
    )


def _reference_axes(startpos_L: np.ndarray, startpos_R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    left_xy = np.asarray(startpos_L, dtype=float).reshape(-1)[:2]
    right_xy = np.asarray(startpos_R, dtype=float).reshape(-1)[:2]
    right_to_left = left_xy - right_xy
    norm = float(np.linalg.norm(right_to_left))
    y_ref = np.asarray([0.0, 1.0]) if norm < 1e-12 else right_to_left / norm
    x_ref = np.asarray([y_ref[1], -y_ref[0]], dtype=float)
    return x_ref, y_ref


def _heading_deg(center_from: Sequence[float], center_to: Sequence[float]) -> float:
    dx = float(center_to[0]) - float(center_from[0])
    dy = float(center_to[1]) - float(center_from[1])
    return 0.0 if abs(dx) + abs(dy) < 1e-9 else math.degrees(math.atan2(dy, dx))


def _make_entry_from_stones(
    stones: Sequence[Mapping[str, Any]], stance_width: float
) -> Dict[str, Any]:
    """Local, side-effect-free copy of main_VAE_optapi's shorthand input."""

    start_heading = _heading_deg(stones[0]["center"], stones[1]["center"]) if len(stones) >= 2 else 0.0
    goal_heading = _heading_deg(stones[-2]["center"], stones[-1]["center"]) if len(stones) >= 2 else 0.0
    return {
        "start": {"center": stones[0]["center"], "heading_deg": start_heading},
        "goal": {"center": stones[-1]["center"], "heading_deg": goal_heading},
        "stance_width": float(stance_width),
        "foot_yaw_deg": {
            "start_L": 0.0,
            "start_R": 0.0,
            "goal_L": 0.0,
            "goal_R": 0.0,
        },
    }


def _rotate_vertices_with_yaw(
    vertices: Sequence[np.ndarray],
    stones: Sequence[Mapping[str, Any]],
    copy_counts: Sequence[int],
) -> List[np.ndarray]:
    """Local, side-effect-free copy of the existing CVAE yaw rotation."""

    rotated: List[np.ndarray] = []
    expanded_stone_indices = [
        stone_index
        for stone_index, count in enumerate(copy_counts)
        for _ in range(int(count))
    ]
    for vertex, stone_index in zip(vertices, expanded_stone_indices):
        stone = stones[stone_index]
        yaw_deg = float(stone.get("yaw_deg", 0.0))
        if abs(yaw_deg) < 1e-9:
            rotated.append(np.array(vertex, copy=True))
            continue
        center_x, center_y, _ = stone["center"]
        yaw = math.radians(yaw_deg)
        rotation = np.asarray(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        shifted = np.array(vertex, dtype=float, copy=True)
        shifted[0, :] -= float(center_x)
        shifted[1, :] -= float(center_y)
        transformed = rotation @ shifted
        transformed[0, :] += float(center_x)
        transformed[1, :] += float(center_y)
        rotated.append(transformed)
    if len(rotated) != len(vertices):
        raise ValueError("vertex/copy-count expansion length mismatch")
    return rotated


def _copy_counts_for_terrain(
    embedded: Optional[Sequence[int]],
    requested: Optional[Sequence[int]],
    num_regions: int,
) -> List[int]:
    if embedded is not None:
        counts = [int(value) for value in embedded]
    elif requested is not None:
        counts = [int(value) for value in requested]
    elif int(num_regions) == 4:
        # This is the existing four-region CVAE benchmark setting in
        # main_VAE_fusionapi.py.  No generic fallback is invented for other
        # region counts; those must provide JSON copy_counts or --copy-counts.
        counts = list(DEFAULT_COPY_COUNTS_4_REGION)
    else:
        raise ValueError(
            "terrain JSON has no copy_counts; pass --copy-counts for "
            f"{num_regions} regions"
        )
    if len(counts) != int(num_regions):
        raise ValueError(f"copy_counts length {len(counts)} != regions {num_regions}")
    if any(value < 1 for value in counts):
        raise ValueError(f"copy_counts must be positive: {counts}")
    return counts


def _load_problem(case: TerrainCase, args: argparse.Namespace) -> TerrainProblem:
    terrain = load_terrain_yaw(name=args.terrain_key, path=case.path)
    stones = list(terrain["stones"])
    if not stones:
        raise ValueError("terrain contains no regions")

    entry = _make_entry_from_stones(stones, float(args.stance_width))
    expanded = expand_shorthand(entry, float(args.stance_width))
    startpos_L = np.asarray(expanded["startpos_L"], dtype=float)
    startpos_R = np.asarray(expanded["startpos_R"], dtype=float)
    targetpos_L = np.asarray(expanded["targetpos_L"], dtype=float)
    targetpos_R = np.asarray(expanded["targetpos_R"], dtype=float)

    counts = _copy_counts_for_terrain(
        terrain.get("copy_counts"), args.copy_counts, len(stones)
    )
    graph = build_graph(
        startpos_L,
        startpos_R,
        targetpos_L,
        targetpos_R,
        step_span=float(args.step_span),
        copy_counts=counts,
        edge_rule="general",
        terrain_stones=stones,
    )
    vertices = _rotate_vertices_with_yaw(graph["v"], stones, counts)
    centroid_mat, _ = compute_centroids_from_vertices(vertices)
    x_ref, y_ref = _reference_axes(startpos_L, startpos_R)

    return TerrainProblem(
        stones=stones,
        graph=graph,
        vertices=vertices,
        centroid_mat=np.asarray(centroid_mat, dtype=float),
        startpos_L=startpos_L,
        startpos_R=startpos_R,
        targetpos_L=targetpos_L,
        targetpos_R=targetpos_R,
        start_L=_pose5(startpos_L, float(expanded["startyaw_L"])),
        start_R=_pose5(startpos_R, float(expanded["startyaw_R"])),
        target_L=_pose5(targetpos_L, float(expanded["targetyaw_L"])),
        target_R=_pose5(targetpos_R, float(expanded["targetyaw_R"])),
        x_ref=x_ref,
        y_ref=y_ref,
    )


def _build_model(problem: TerrainProblem, args: argparse.Namespace):
    return build_gcs_model_VAE_optapi(
        problem.vertices,
        problem.graph["edge"],
        float(args.step_span),
        math.radians(float(args.alpha_deg)),
        problem.graph["mode_list"],
        problem.x_ref,
        problem.y_ref,
    )


def _close_task(task: Any) -> None:
    if task is None:
        return
    try:
        task.__exit__(None, None, None)
        return
    except Exception:
        pass
    try:
        task.dispose()
    except Exception:
        pass


def _acceptable_solsta(solsta: Any) -> bool:
    acceptable = solsta == mosek.solsta.optimal
    if hasattr(mosek.solsta, "integer_optimal"):
        acceptable = acceptable or solsta == mosek.solsta.integer_optimal
    if hasattr(mosek.solsta, "near_optimal"):
        acceptable = acceptable or solsta == mosek.solsta.near_optimal
    return bool(acceptable)


def _status_strings(task: Any) -> Tuple[str, str]:
    if task is None:
        return "", ""
    prosta, solsta = "", ""
    try:
        prosta = str(task.getprosta(mosek.soltype.itr))
    except Exception:
        pass
    try:
        solsta = str(task.getsolsta(mosek.soltype.itr))
    except Exception:
        pass
    return prosta, solsta


def _solver_failure_reason(task: Any) -> str:
    prosta, solsta = _status_strings(task)
    status = f"{prosta} {solsta}".lower()
    return "fixed_problem_infeasible" if "infeas" in status else "fixed_solver_error"


def _float_hash(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype="<f8").reshape(-1))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _path_hash(path: Sequence[int]) -> str:
    encoded = ",".join(str(int(node)) for node in path).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _path_text(path: Sequence[int]) -> str:
    return json.dumps([int(node) for node in path], separators=(",", ":"))


def _validate_path(path: Sequence[int], edge: np.ndarray, vertex_number: int) -> None:
    nodes = [int(node) for node in path]
    if len(nodes) < 2 or nodes[0] != 1 or nodes[-1] != int(vertex_number):
        raise PureProbabilisticPathError(
            "no_source_target_path", f"invalid source/target path endpoints: {nodes}"
        )
    if any(node < 1 or node > int(vertex_number) for node in nodes):
        raise PureProbabilisticPathError(
            "no_source_target_path", f"path node outside [1,{vertex_number}]: {nodes}"
        )
    available = {(int(u), int(v)) for u, v in np.asarray(edge, dtype=int)}
    for u, v in zip(nodes[:-1], nodes[1:]):
        if (u, v) not in available:
            raise PureProbabilisticPathError(
                "invalid_path_edge", f"path contains missing edge {u}->{v}"
            )


def _stable_recovery_seed(
    base_seed: int,
    alpha_index: int,
    terrain_index: int,
    recovery_index: int,
) -> int:
    sequence = np.random.SeedSequence(
        [
            int(base_seed),
            int(alpha_index),
            int(terrain_index),
            int(recovery_index),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _deadline(start_time: float, limit_sec: float) -> Optional[float]:
    return None if float(limit_sec) <= 0.0 else start_time + float(limit_sec)


def _time_expired(deadline: Optional[float]) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _candidate_base(
    case: TerrainCase,
    common: Mapping[str, Any],
    *,
    strategy: str,
    candidate_index: int,
    recovery_seed: Optional[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    row = {field: "" for field in PER_CANDIDATE_FIELDS}
    row.update(
        {
            "alpha": case.alpha_label,
            "alpha_index": int(case.alpha_index),
            "terrain_id": case.path.stem,
            "terrain_index": int(case.terrain_index),
            "terrain_file": str(case.path.resolve()),
            "terrain_seed": "" if case.terrain_seed is None else int(case.terrain_seed),
            "strategy": strategy,
            "candidate_index": int(candidate_index),
            "recovery_seed": "" if recovery_seed is None else int(recovery_seed),
            "temperature": float(args.temperature),
            "max_backtracks": int(args.max_backtracks),
            "num_vertices": common.get("num_vertices", ""),
            "num_edges": common.get("num_edges", ""),
            "relaxed_y_hash": common.get("relaxed_y_hash", ""),
            "relax_solver_status": common.get("relax_solver_status", ""),
            "relax_success": bool(common.get("relax_success", False)),
            "C_relax": common.get("C_relax", math.nan),
            "relax_time_ms": common.get("relax_time_ms", math.nan),
            "path_generation_success": False,
            "path_length": math.nan,
            "duplicate_path": False,
            "probabilistic_internal_greedy_fallback_used": False,
            "path_generation_time_ms": 0.0,
            "fixed_success": False,
            "fixed_objective": math.nan,
            "fixed_solve_time_ms": 0.0,
            "total_candidate_time_ms": 0.0,
            "goal_reached": False,
            "failure_reason": "unknown",
            "error_message": "",
        }
    )
    return row


def _terminal_goal_reached(
    task: Any,
    ex: Mapping[str, Any],
    edge: np.ndarray,
    path: Sequence[int],
    target_L: np.ndarray,
    target_R: np.ndarray,
    M_small: float,
) -> bool:
    edge_map = {
        (int(u), int(v)): index
        for index, (u, v) in enumerate(np.asarray(edge, dtype=int))
    }
    terminal_index = edge_map.get((int(path[-2]), int(path[-1])))
    if terminal_index is None:
        return False
    y_value = float(np.asarray(ex["y"], dtype=float).reshape(-1)[terminal_index])
    if not np.isfinite(y_value) or abs(y_value) < 1e-12:
        return False

    terminal_L = np.asarray(ex["zp_L"], dtype=float).reshape(-1, 5)[terminal_index] / y_value
    terminal_R = np.asarray(ex["zp_R"], dtype=float).reshape(-1, 5)[terminal_index] / y_value
    solver_tolerance = 0.0
    try:
        solver_tolerance = float(task.getdouparam(mosek.dparam.intpnt_co_tol_pfeas))
    except Exception:
        pass
    tolerance = max(float(M_small), solver_tolerance, np.finfo(float).eps)
    return bool(
        np.allclose(terminal_L, np.asarray(target_L, dtype=float), rtol=0.0, atol=tolerance)
        and np.allclose(terminal_R, np.asarray(target_R, dtype=float), rtol=0.0, atol=tolerance)
    )


def _run_candidate(
    case: TerrainCase,
    problem: TerrainProblem,
    task: Any,
    variables: Optional[Dict[str, Any]],
    idx: Optional[Dict[str, Any]],
    common: Mapping[str, Any],
    *,
    strategy: str,
    candidate_index: int,
    recovery_seed: Optional[int],
    path_factory: Callable[[], List[int]],
    seen_paths: Dict[str, int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    row = _candidate_base(
        case,
        common,
        strategy=strategy,
        candidate_index=candidate_index,
        recovery_seed=recovery_seed,
        args=args,
    )
    total_start = time.perf_counter()
    path: Optional[List[int]] = None

    try:
        path_start = time.perf_counter()
        try:
            path = [int(node) for node in path_factory()]
            _validate_path(path, problem.graph["edge"], int(problem.graph["vertex_number"]))
            row["path_generation_success"] = True
        finally:
            row["path_generation_time_ms"] = (time.perf_counter() - path_start) * 1000.0

        row["path_node_sequence"] = _path_text(path)
        row["path_hash"] = _path_hash(path)
        row["path_length"] = len(path) - 1
        previous_index = seen_paths.get(str(row["path_hash"]))
        row["duplicate_path"] = previous_index is not None
        row["previously_seen_candidate_index"] = "" if previous_index is None else previous_index
        if previous_index is None:
            seen_paths[str(row["path_hash"])] = int(candidate_index)

        if previous_index is not None and not bool(args.resolve_duplicate_paths):
            row["fixed_solver_status"] = "not_run_duplicate"
            row["failure_reason"] = "unknown"
            row["error_message"] = "duplicate path was not re-solved by command-line option"
            return row

        try:
            y_path = path_to_y(
                problem.graph["edge"], path, int(problem.graph["n_edge"])
            )
        except Exception as exc:
            row["failure_reason"] = "invalid_path_edge"
            row["error_message"] = _error_text(exc)
            return row

        if task is None or variables is None or idx is None:
            row["failure_reason"] = "fixed_solver_error"
            row["error_message"] = "independent fixed model could not be built"
            return row

        fixed_start = time.perf_counter()
        try:
            # State-isolation policy:
            # * each recovery strategy owns an independently built fixed model;
            # * before every candidate, the imported fixed solver explicitly
            #   resets y bounds, y_relax, M_scalar, start/target parameters, and
            #   the MOSEK log setting;
            # * no explicit warm start is supplied for any strategy.
            _, objective, ex = _solve_fixed_once_optapi(
                task,
                variables,
                idx,
                centroid_mat=problem.centroid_mat,
                y_relax=y_path,
                M_small=float(args.M_small),
                start_L=problem.start_L,
                start_R=problem.start_R,
                target_L=problem.target_L,
                target_R=problem.target_R,
                log_level=int(args.solver_log),
            )
        except Exception as exc:
            row["fixed_solve_time_ms"] = (time.perf_counter() - fixed_start) * 1000.0
            _, solsta = _status_strings(task)
            row["fixed_solver_status"] = solsta
            row["failure_reason"] = _solver_failure_reason(task)
            row["error_message"] = _error_text(exc)
            return row
        row["fixed_solve_time_ms"] = (time.perf_counter() - fixed_start) * 1000.0

        _, solsta = _status_strings(task)
        row["fixed_solver_status"] = solsta
        if objective is None or not np.isfinite(float(objective)):
            row["failure_reason"] = "nonfinite_objective"
            row["error_message"] = f"non-finite fixed objective: {objective}"
            return row
        row["fixed_objective"] = float(objective)
        if ex is None:
            row["failure_reason"] = "fixed_solver_error"
            row["error_message"] = "fixed solver returned no extracted solution"
            return row

        try:
            postprocess_solution(
                ex,
                problem.graph["edge"],
                path,
                problem.startpos_L,
                problem.startpos_R,
                problem.targetpos_L,
                problem.targetpos_R,
                problem.target_L,
                problem.target_R,
            )
        except Exception as exc:
            row["failure_reason"] = "postprocess_failure"
            row["error_message"] = _error_text(exc)
            return row

        row["goal_reached"] = _terminal_goal_reached(
            task,
            ex,
            problem.graph["edge"],
            path,
            problem.target_L,
            problem.target_R,
            float(args.M_small),
        )
        if not bool(row["goal_reached"]):
            row["failure_reason"] = "goal_not_reached"
            row["error_message"] = "terminal sink state is outside the active solver tolerance"
            return row

        row["fixed_success"] = True
        row["failure_reason"] = "none"
        row["error_message"] = ""
        return row

    except PureProbabilisticPathError as exc:
        row["failure_reason"] = exc.reason
        row["error_message"] = _error_text(exc)
        return row
    except Exception as exc:
        row["failure_reason"] = "no_source_target_path" if path is None else "unknown"
        row["error_message"] = _error_text(exc)
        return row
    finally:
        row["total_candidate_time_ms"] = (time.perf_counter() - total_start) * 1000.0


def _error_text(exc: BaseException, limit: int = 1200) -> str:
    return f"{type(exc).__name__}: {exc}".replace("\n", " | ")[:limit]


def _make_fixed_model(problem: TerrainProblem, args: argparse.Namespace):
    try:
        task, variables, idx, _, _ = _build_model(problem, args)
        return task, variables, idx, ""
    except Exception as exc:
        return None, None, None, _error_text(exc)


def _run_greedy_only(
    case: TerrainCase,
    problem: TerrainProblem,
    y_relaxed: np.ndarray,
    common: Mapping[str, Any],
    args: argparse.Namespace,
    terrain_deadline: Optional[float],
) -> List[Dict[str, Any]]:
    task = variables = idx = None
    try:
        task, variables, idx, build_error = _make_fixed_model(problem, args)
        if _time_expired(terrain_deadline):
            build_error = build_error or "terrain time limit expired before greedy recovery"
            task = variables = idx = None
        seen: Dict[str, int] = {}
        row = _run_candidate(
            case,
            problem,
            task,
            variables,
            idx,
            common,
            strategy="greedy_only",
            candidate_index=0,
            recovery_seed=None,
            path_factory=lambda: relaxation_max(
                y_relaxed,
                problem.graph["edge"],
                int(problem.graph["vertex_number"]),
                max_backtracks=int(args.max_backtracks),
            ),
            seen_paths=seen,
            args=args,
        )
        if build_error and not row["error_message"]:
            row["error_message"] = build_error
        return [row]
    finally:
        _close_task(task)


def _run_probabilistic_only(
    case: TerrainCase,
    problem: TerrainProblem,
    y_relaxed: np.ndarray,
    common: Mapping[str, Any],
    args: argparse.Namespace,
    terrain_deadline: Optional[float],
) -> List[Dict[str, Any]]:
    task = variables = idx = None
    rows: List[Dict[str, Any]] = []
    try:
        task, variables, idx, build_error = _make_fixed_model(problem, args)
        recovery_start = time.perf_counter()
        recovery_deadline = _deadline(recovery_start, float(args.recovery_time_limit_sec))
        seen: Dict[str, int] = {}
        for recovery_index in range(int(args.num_prob_seeds)):
            if _time_expired(terrain_deadline) or _time_expired(recovery_deadline):
                break
            recovery_seed = _stable_recovery_seed(
                int(args.base_seed),
                int(case.alpha_index),
                int(case.terrain_index),
                recovery_index,
            )
            row = _run_candidate(
                case,
                problem,
                task,
                variables,
                idx,
                common,
                strategy="probabilistic_only",
                candidate_index=recovery_index,
                recovery_seed=recovery_seed,
                path_factory=lambda seed=recovery_seed: relaxation_prob_pure(
                    y_relaxed,
                    problem.graph["edge"],
                    int(problem.graph["vertex_number"]),
                    temperature=float(args.temperature),
                    seed=seed,
                    max_backtracks=int(args.max_backtracks),
                ),
                seen_paths=seen,
                args=args,
            )
            if build_error and not row["error_message"]:
                row["error_message"] = build_error
            rows.append(row)
        return rows
    finally:
        _close_task(task)


def _run_combined(
    case: TerrainCase,
    problem: TerrainProblem,
    y_relaxed: np.ndarray,
    common: Mapping[str, Any],
    args: argparse.Namespace,
    terrain_deadline: Optional[float],
) -> List[Dict[str, Any]]:
    task = variables = idx = None
    rows: List[Dict[str, Any]] = []
    try:
        task, variables, idx, build_error = _make_fixed_model(problem, args)
        recovery_start = time.perf_counter()
        recovery_deadline = _deadline(recovery_start, float(args.recovery_time_limit_sec))
        seen: Dict[str, int] = {}

        if not _time_expired(terrain_deadline) and not _time_expired(recovery_deadline):
            greedy_row = _run_candidate(
                case,
                problem,
                task,
                variables,
                idx,
                common,
                strategy="combined_greedy",
                candidate_index=0,
                recovery_seed=None,
                path_factory=lambda: relaxation_max(
                    y_relaxed,
                    problem.graph["edge"],
                    int(problem.graph["vertex_number"]),
                    max_backtracks=int(args.max_backtracks),
                ),
                seen_paths=seen,
                args=args,
            )
            if build_error and not greedy_row["error_message"]:
                greedy_row["error_message"] = build_error
            rows.append(greedy_row)
            if bool(greedy_row["fixed_success"]):
                return rows

        maximum_probabilistic = max(0, int(args.max_combined_candidates) - 1)
        for recovery_index in range(maximum_probabilistic):
            if len(rows) >= int(args.max_combined_candidates):
                break
            if _time_expired(terrain_deadline) or _time_expired(recovery_deadline):
                break
            recovery_seed = _stable_recovery_seed(
                int(args.base_seed),
                int(case.alpha_index),
                int(case.terrain_index),
                recovery_index,
            )
            candidate_index = len(rows)
            row = _run_candidate(
                case,
                problem,
                task,
                variables,
                idx,
                common,
                strategy="combined_probabilistic",
                candidate_index=candidate_index,
                recovery_seed=recovery_seed,
                path_factory=lambda seed=recovery_seed: relaxation_prob_pure(
                    y_relaxed,
                    problem.graph["edge"],
                    int(problem.graph["vertex_number"]),
                    temperature=float(args.temperature),
                    seed=seed,
                    max_backtracks=int(args.max_backtracks),
                ),
                seen_paths=seen,
                args=args,
            )
            if build_error and not row["error_message"]:
                row["error_message"] = build_error
            rows.append(row)
            if bool(row["fixed_success"]):
                break
        return rows
    finally:
        _close_task(task)


def _terrain_row_base(case: TerrainCase) -> Dict[str, Any]:
    row = {field: "" for field in PER_TERRAIN_FIELDS}
    row.update(
        {
            "alpha": case.alpha_label,
            "alpha_index": int(case.alpha_index),
            "terrain_id": case.path.stem,
            "terrain_index": int(case.terrain_index),
            "terrain_file": str(case.path.resolve()),
            "terrain_seed": "" if case.terrain_seed is None else int(case.terrain_seed),
            "num_vertices": math.nan,
            "num_edges": math.nan,
            "relaxed_y_hash": "",
            "relax_solver_status": "not_run",
            "relax_success": False,
            "C_relax": math.nan,
            "relax_time_ms": 0.0,
            "greedy_path_generation_success": False,
            "greedy_fixed_success": False,
            "greedy_fixed_objective": math.nan,
            "greedy_path_generation_time_ms": 0.0,
            "greedy_fixed_solve_time_ms": 0.0,
            "greedy_total_recovery_time_ms": 0.0,
            "probabilistic_trial_count": 0,
            "probabilistic_success_count": 0,
            "probabilistic_failure_count": 0,
            "probabilistic_any_success": False,
            "probabilistic_first_success_index": math.nan,
            "probabilistic_best_objective": math.nan,
            "probabilistic_mean_fixed_objective": math.nan,
            "probabilistic_total_recovery_time_ms": 0.0,
            "probabilistic_unique_path_count": 0,
            "probabilistic_duplicate_path_count": 0,
            "combined_success": False,
            "combined_selected_strategy": "none",
            "combined_candidates_attempted": 0,
            "combined_probabilistic_candidates_attempted": 0,
            "combined_candidates_until_success": math.nan,
            "combined_fixed_objective": math.nan,
            "combined_total_recovery_time_ms": 0.0,
            "fallback_activated": False,
            "fallback_success": False,
            "all_candidates_failed": False,
            "final_failure_reason": "unknown",
            "error_message": "",
        }
    )
    return row


def _aggregate_terrain_row(
    terrain_row: Dict[str, Any], candidate_rows: Sequence[Mapping[str, Any]]
) -> None:
    greedy = [row for row in candidate_rows if row["strategy"] == "greedy_only"]
    probabilistic = [
        row for row in candidate_rows if row["strategy"] == "probabilistic_only"
    ]
    combined = [
        row for row in candidate_rows if str(row["strategy"]).startswith("combined_")
    ]

    if greedy:
        row = greedy[0]
        terrain_row["greedy_path_generation_success"] = _as_bool(
            row["path_generation_success"]
        )
        terrain_row["greedy_fixed_success"] = _as_bool(row["fixed_success"])
        terrain_row["greedy_fixed_objective"] = row["fixed_objective"]
        terrain_row["greedy_path_generation_time_ms"] = row["path_generation_time_ms"]
        terrain_row["greedy_fixed_solve_time_ms"] = row["fixed_solve_time_ms"]
        terrain_row["greedy_total_recovery_time_ms"] = row["total_candidate_time_ms"]

    prob_successes = [row for row in probabilistic if _as_bool(row["fixed_success"])]
    terrain_row["probabilistic_trial_count"] = len(probabilistic)
    terrain_row["probabilistic_success_count"] = len(prob_successes)
    terrain_row["probabilistic_failure_count"] = len(probabilistic) - len(prob_successes)
    terrain_row["probabilistic_any_success"] = bool(prob_successes)
    if prob_successes:
        terrain_row["probabilistic_first_success_index"] = int(
            prob_successes[0]["candidate_index"]
        )
    prob_objectives = [_as_float(row["fixed_objective"]) for row in prob_successes]
    prob_objectives = [value for value in prob_objectives if np.isfinite(value)]
    if prob_objectives:
        terrain_row["probabilistic_best_objective"] = min(prob_objectives)
        terrain_row["probabilistic_mean_fixed_objective"] = float(np.mean(prob_objectives))
    terrain_row["probabilistic_total_recovery_time_ms"] = _sum_field(
        probabilistic, "total_candidate_time_ms"
    )
    path_hashes = [str(row["path_hash"]) for row in probabilistic if str(row["path_hash"])]
    terrain_row["probabilistic_unique_path_count"] = len(set(path_hashes))
    terrain_row["probabilistic_duplicate_path_count"] = sum(
        _as_bool(row["duplicate_path"]) for row in probabilistic
    )

    combined_success_rows = [row for row in combined if _as_bool(row["fixed_success"])]
    terrain_row["combined_success"] = bool(combined_success_rows)
    terrain_row["combined_candidates_attempted"] = len(combined)
    terrain_row["combined_probabilistic_candidates_attempted"] = sum(
        row["strategy"] == "combined_probabilistic" for row in combined
    )
    terrain_row["combined_total_recovery_time_ms"] = _sum_field(
        combined, "total_candidate_time_ms"
    )
    if combined_success_rows:
        selected = combined_success_rows[0]
        terrain_row["combined_selected_strategy"] = (
            "greedy" if selected["strategy"] == "combined_greedy" else "probabilistic"
        )
        terrain_row["combined_candidates_until_success"] = int(
            selected["candidate_index"]
        ) + 1
        terrain_row["combined_fixed_objective"] = selected["fixed_objective"]

    combined_greedy = [row for row in combined if row["strategy"] == "combined_greedy"]
    fallback_activated = bool(combined_greedy) and not _as_bool(
        combined_greedy[0]["fixed_success"]
    )
    terrain_row["fallback_activated"] = fallback_activated
    terrain_row["fallback_success"] = bool(
        fallback_activated
        and combined_success_rows
        and combined_success_rows[0]["strategy"] == "combined_probabilistic"
    )
    terrain_row["all_candidates_failed"] = not bool(combined_success_rows)
    terrain_row["final_failure_reason"] = (
        "none" if combined_success_rows else "candidate_budget_exhausted"
    )
    if not combined_success_rows:
        terrain_row["error_message"] = (
            f"combined recovery produced no feasible path after {len(combined)} candidates"
        )


def _relaxation_failure_candidate(
    case: TerrainCase,
    terrain_row: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    row = _candidate_base(
        case,
        terrain_row,
        strategy="relaxation",
        candidate_index=-1,
        recovery_seed=None,
        args=args,
    )
    row["failure_reason"] = "relax_solver_failure"
    row["error_message"] = terrain_row.get("error_message", "")
    return row


def _process_terrain(
    case: TerrainCase, args: argparse.Namespace
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    terrain_start = time.perf_counter()
    terrain_deadline = _deadline(terrain_start, float(args.terrain_time_limit_sec))
    terrain_row = _terrain_row_base(case)
    relaxation_task = None
    problem: Optional[TerrainProblem] = None
    candidate_rows: List[Dict[str, Any]] = []

    try:
        problem = _load_problem(case, args)
        terrain_row["num_vertices"] = int(problem.graph["vertex_number"])
        terrain_row["num_edges"] = int(problem.graph["n_edge"])
        if _time_expired(terrain_deadline):
            raise TimeoutError("terrain time limit expired before model construction")

        relaxation_task, variables, idx, _, _ = _build_model(problem, args)
        if _time_expired(terrain_deadline):
            raise TimeoutError("terrain time limit expired before relaxed solve")

        relax_start = time.perf_counter()
        try:
            relaxed_parameters = pack_relaxed_params_for_model(
                centroid_mat=problem.centroid_mat,
                n_edge=int(idx["n_edge"]),
                M_large=float(args.M_large),
                start_L=problem.start_L,
                start_R=problem.start_R,
                target_L=problem.target_L,
                target_R=problem.target_R,
            )
            set_parameters_optapi(relaxation_task, variables, relaxed_parameters)
            relaxed_solution, relaxed_objective = solve_relaxed_optapi(
                relaxation_task, variables, log_level=int(args.solver_log)
            )
        finally:
            terrain_row["relax_time_ms"] = (time.perf_counter() - relax_start) * 1000.0

        _, relaxed_solsta = _status_strings(relaxation_task)
        terrain_row["relax_solver_status"] = relaxed_solsta
        try:
            solsta_enum = relaxation_task.getsolsta(mosek.soltype.itr)
        except Exception as exc:
            raise RuntimeError("could not read relaxed solver status") from exc
        if not _acceptable_solsta(solsta_enum):
            prosta, solsta = _status_strings(relaxation_task)
            raise RuntimeError(f"relaxed solve failed: prosta={prosta}, solsta={solsta}")
        if relaxed_objective is None or not np.isfinite(float(relaxed_objective)):
            raise RuntimeError(f"non-finite relaxed objective: {relaxed_objective}")

        y_relaxed = np.asarray(relaxed_solution["y"], dtype=float).reshape(-1).copy()
        if y_relaxed.size != int(problem.graph["n_edge"]):
            raise RuntimeError(
                f"relaxed y length {y_relaxed.size} != edges {problem.graph['n_edge']}"
            )
        if not np.all(np.isfinite(y_relaxed)):
            raise RuntimeError("relaxed y contains non-finite values")
        y_relaxed.setflags(write=False)
        terrain_row["relaxed_y_hash"] = _float_hash(y_relaxed)
        terrain_row["relax_success"] = True
        terrain_row["C_relax"] = float(relaxed_objective)
        common = {
            "num_vertices": terrain_row["num_vertices"],
            "num_edges": terrain_row["num_edges"],
            "relaxed_y_hash": terrain_row["relaxed_y_hash"],
            "relax_solver_status": terrain_row["relax_solver_status"],
            "relax_success": True,
            "C_relax": terrain_row["C_relax"],
            "relax_time_ms": terrain_row["relax_time_ms"],
        }
    except Exception as exc:
        if relaxation_task is not None:
            _, status = _status_strings(relaxation_task)
            if status:
                terrain_row["relax_solver_status"] = status
        terrain_row["relax_success"] = False
        terrain_row["final_failure_reason"] = "relax_solver_failure"
        terrain_row["error_message"] = _error_text(exc)
        candidate_rows.append(_relaxation_failure_candidate(case, terrain_row, args))
        return terrain_row, candidate_rows
    finally:
        _close_task(relaxation_task)

    assert problem is not None
    candidate_rows.extend(
        _run_greedy_only(case, problem, y_relaxed, common, args, terrain_deadline)
    )
    candidate_rows.extend(
        _run_probabilistic_only(case, problem, y_relaxed, common, args, terrain_deadline)
    )
    candidate_rows.extend(
        _run_combined(case, problem, y_relaxed, common, args, terrain_deadline)
    )
    _aggregate_terrain_row(terrain_row, candidate_rows)
    return terrain_row, candidate_rows


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)


def _sum_field(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    values = [_as_float(row.get(field)) for row in rows]
    return float(sum(value for value in values if np.isfinite(value)))


def _mean_std(values: Iterable[Any]) -> Tuple[float, float]:
    finite = np.asarray(
        [value for value in (_as_float(item) for item in values) if np.isfinite(value)],
        dtype=float,
    )
    if finite.size == 0:
        return math.nan, math.nan
    return float(np.mean(finite)), float(np.std(finite, ddof=0))


def _rate_percent(numerator: int, denominator: int) -> float:
    return math.nan if int(denominator) == 0 else 100.0 * int(numerator) / int(denominator)


def _summary_row(
    alpha_label: str,
    terrain_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    relaxed = [row for row in terrain_rows if _as_bool(row.get("relax_success"))]
    num_terrains = len(terrain_rows)
    num_relax_success = len(relaxed)
    greedy_successes = [row for row in relaxed if _as_bool(row.get("greedy_fixed_success"))]
    prob_candidates = [
        row for row in candidate_rows if row.get("strategy") == "probabilistic_only"
    ]
    prob_candidate_successes = [
        row for row in prob_candidates if _as_bool(row.get("fixed_success"))
    ]
    prob_terrain_successes = [
        row for row in relaxed if _as_bool(row.get("probabilistic_any_success"))
    ]
    combined_successes = [row for row in relaxed if _as_bool(row.get("combined_success"))]
    fallback_activated = [row for row in relaxed if _as_bool(row.get("fallback_activated"))]
    fallback_successes = [row for row in fallback_activated if _as_bool(row.get("fallback_success"))]
    all_failed = [row for row in relaxed if _as_bool(row.get("all_candidates_failed"))]

    greedy_time_mean, greedy_time_std = _mean_std(
        row.get("greedy_total_recovery_time_ms") for row in relaxed
    )
    greedy_obj_mean, greedy_obj_std = _mean_std(
        row.get("greedy_fixed_objective") for row in greedy_successes
    )
    prob_time_mean, prob_time_std = _mean_std(
        row.get("probabilistic_total_recovery_time_ms") for row in relaxed
    )
    prob_obj_mean, prob_obj_std = _mean_std(
        row.get("fixed_objective") for row in prob_candidate_successes
    )
    combined_time_mean, combined_time_std = _mean_std(
        row.get("combined_total_recovery_time_ms") for row in relaxed
    )
    combined_attempt_mean, combined_attempt_std = _mean_std(
        row.get("combined_candidates_attempted") for row in relaxed
    )
    combined_until_mean, _ = _mean_std(
        row.get("combined_candidates_until_success") for row in combined_successes
    )
    combined_obj_mean, combined_obj_std = _mean_std(
        row.get("combined_fixed_objective") for row in combined_successes
    )

    return {
        "alpha": alpha_label,
        "num_terrains": num_terrains,
        "num_relax_success": num_relax_success,
        "num_relax_failure": num_terrains - num_relax_success,
        "greedy_success_count": len(greedy_successes),
        "greedy_feasibility_rate_percent": _rate_percent(
            len(greedy_successes), num_relax_success
        ),
        "greedy_mean_recovery_time_ms": greedy_time_mean,
        "greedy_std_recovery_time_ms": greedy_time_std,
        "greedy_mean_fixed_objective": greedy_obj_mean,
        "greedy_std_fixed_objective": greedy_obj_std,
        "probabilistic_total_candidate_runs": len(prob_candidates),
        "probabilistic_successful_candidate_runs": len(prob_candidate_successes),
        "probabilistic_single_trial_feasibility_rate_percent": _rate_percent(
            len(prob_candidate_successes), len(prob_candidates)
        ),
        "probabilistic_terrain_success_count": len(prob_terrain_successes),
        "probabilistic_per_terrain_feasibility_rate_percent": _rate_percent(
            len(prob_terrain_successes), num_relax_success
        ),
        "probabilistic_mean_recovery_time_ms": prob_time_mean,
        "probabilistic_std_recovery_time_ms": prob_time_std,
        "probabilistic_mean_fixed_objective": prob_obj_mean,
        "probabilistic_std_fixed_objective": prob_obj_std,
        "combined_success_count": len(combined_successes),
        "combined_feasibility_rate_percent": _rate_percent(
            len(combined_successes), num_relax_success
        ),
        "combined_mean_recovery_time_ms": combined_time_mean,
        "combined_std_recovery_time_ms": combined_time_std,
        "combined_mean_candidates_attempted": combined_attempt_mean,
        "combined_std_candidates_attempted": combined_attempt_std,
        "combined_mean_candidates_until_success": combined_until_mean,
        "combined_mean_fixed_objective": combined_obj_mean,
        "combined_std_fixed_objective": combined_obj_std,
        "fallback_activation_count": len(fallback_activated),
        "fallback_activation_rate_percent": _rate_percent(
            len(fallback_activated), num_relax_success
        ),
        "fallback_success_count": len(fallback_successes),
        "fallback_success_rate_percent": _rate_percent(
            len(fallback_successes), len(fallback_activated)
        ),
        "all_candidates_failed_count": len(all_failed),
        "all_candidates_failed_rate_percent": _rate_percent(
            len(all_failed), num_relax_success
        ),
    }


def _build_summary(
    terrain_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    alpha_labels = sorted(
        {str(row.get("alpha")) for row in terrain_rows}, key=lambda value: float(value)
    )
    summary: List[Dict[str, Any]] = []
    for label in alpha_labels:
        selected_terrains = [row for row in terrain_rows if str(row.get("alpha")) == label]
        selected_candidates = [
            row for row in candidate_rows if str(row.get("alpha")) == label
        ]
        summary.append(_summary_row(label, selected_terrains, selected_candidates))
    summary.append(_summary_row("overall", terrain_rows, candidate_rows))
    return summary


def _csv_value(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "NaN"
        if math.isinf(float(value)):
            return "Infinity" if float(value) > 0 else "-Infinity"
        return format(float(value), CSV_FLOAT_FORMAT)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _atomic_write_csv(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _row_key(row: Mapping[str, Any]) -> Tuple[str, int, str]:
    return (
        str(row.get("alpha")),
        _as_int(row.get("terrain_index"), -1),
        str(Path(str(row.get("terrain_file"))).resolve()),
    )


def _replace_keyed_rows(
    rows: Sequence[Mapping[str, Any]],
    key: Tuple[str, int, str],
    replacements: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    return [row for row in rows if _row_key(row) != key] + list(replacements)


def _completed_row_is_compatible(
    terrain_row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> bool:
    if not _as_bool(terrain_row.get("relax_success")):
        return any(row.get("strategy") == "relaxation" for row in candidates)
    greedy = [row for row in candidates if row.get("strategy") == "greedy_only"]
    probabilistic = [
        row for row in candidates if row.get("strategy") == "probabilistic_only"
    ]
    combined = [
        row for row in candidates if str(row.get("strategy", "")).startswith("combined_")
    ]
    if len(greedy) != 1 or len(probabilistic) != int(args.num_prob_seeds) or not combined:
        return False
    if not _as_bool(terrain_row.get("combined_success")):
        if len(combined) != int(args.max_combined_candidates):
            return False
    alpha_index_by_label = {
        str(float(alpha)): index for index, alpha in enumerate(args.alphas)
    }
    alpha_label = str(terrain_row.get("alpha"))
    if alpha_label not in alpha_index_by_label:
        return False
    expected_alpha_index = int(alpha_index_by_label[alpha_label])
    terrain_index = _as_int(terrain_row.get("terrain_index"), -1)
    if _as_int(terrain_row.get("alpha_index"), -1) != expected_alpha_index:
        return False

    for row in probabilistic + combined:
        if row.get("strategy") in {"probabilistic_only", "combined_probabilistic"}:
            if not math.isclose(
                _as_float(row.get("temperature")), float(args.temperature), rel_tol=0.0, abs_tol=0.0
            ):
                return False
            if _as_int(row.get("max_backtracks"), -1) != int(args.max_backtracks):
                return False
            if row.get("strategy") == "probabilistic_only":
                recovery_index = _as_int(row.get("candidate_index"), -1)
            else:
                recovery_index = _as_int(row.get("candidate_index"), -1) - 1
            expected_seed = _stable_recovery_seed(
                int(args.base_seed),
                expected_alpha_index,
                terrain_index,
                recovery_index,
            )
            if _as_int(row.get("recovery_seed"), -1) != expected_seed:
                return False
        if _as_bool(row.get("duplicate_path")):
            skipped = str(row.get("fixed_solver_status")) == "not_run_duplicate"
            if bool(args.resolve_duplicate_paths) == skipped:
                return False
    return True


def _load_resume_state(
    candidate_path: Path,
    terrain_path: Path,
    args: argparse.Namespace,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]], set[Tuple[str, int, str]]]:
    candidates: List[Mapping[str, Any]] = _read_csv(candidate_path)
    terrains: List[Mapping[str, Any]] = _read_csv(terrain_path)
    completed: set[Tuple[str, int, str]] = set()
    retained_terrains: List[Mapping[str, Any]] = []

    for terrain in terrains:
        key = _row_key(terrain)
        matching = [row for row in candidates if _row_key(row) == key]
        if _completed_row_is_compatible(terrain, matching, args):
            completed.add(key)
            retained_terrains.append(terrain)

    # Candidate rows without a compatible committed terrain row are partial
    # transactions.  Remove them before rerunning that terrain.
    retained_candidates = [row for row in candidates if _row_key(row) in completed]
    return retained_candidates, retained_terrains, completed


def _validate_consistency(
    candidate_rows: Sequence[Mapping[str, Any]],
    terrain_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> None:
    if any(
        _as_bool(row.get("probabilistic_internal_greedy_fallback_used"))
        for row in candidate_rows
    ):
        raise AssertionError("pure probabilistic candidate reports an internal greedy fallback")
    for reason in (str(row.get("failure_reason")) for row in candidate_rows):
        if reason not in FAILURE_REASONS:
            raise AssertionError(f"invalid failure_reason in candidate CSV: {reason}")

    for terrain in terrain_rows:
        key = _row_key(terrain)
        candidates = [row for row in candidate_rows if _row_key(row) == key]
        if _as_bool(terrain.get("relax_success")):
            hashes = {str(row.get("relaxed_y_hash")) for row in candidates}
            if hashes != {str(terrain.get("relaxed_y_hash"))}:
                raise AssertionError(f"relaxed y hash mismatch for {key}: {hashes}")
            probabilistic = [
                row for row in candidates if row.get("strategy") == "probabilistic_only"
            ]
            if len(probabilistic) != _as_int(terrain.get("probabilistic_trial_count")):
                raise AssertionError(f"probabilistic trial count mismatch for {key}")
            success_count = sum(_as_bool(row.get("fixed_success")) for row in probabilistic)
            if success_count != _as_int(terrain.get("probabilistic_success_count")):
                raise AssertionError(f"probabilistic success count mismatch for {key}")
            combined = [
                row
                for row in candidates
                if str(row.get("strategy", "")).startswith("combined_")
            ]
            if len(combined) != _as_int(terrain.get("combined_candidates_attempted")):
                raise AssertionError(f"combined candidate count mismatch for {key}")
            combined_success = any(_as_bool(row.get("fixed_success")) for row in combined)
            if combined_success != _as_bool(terrain.get("combined_success")):
                raise AssertionError(f"combined success mismatch for {key}")

            # Probabilistic-only and combined use the same SeedSequence formula.
            # Whenever combined reaches a seed also evaluated by the standalone
            # strategy, pure recovery must reproduce exactly the same outcome.
            standalone_by_seed = {
                str(row.get("recovery_seed")): row
                for row in probabilistic
                if str(row.get("recovery_seed", ""))
            }
            combined_by_seed = {
                str(row.get("recovery_seed")): row
                for row in combined
                if row.get("strategy") == "combined_probabilistic"
                and str(row.get("recovery_seed", ""))
            }
            for recovery_seed in standalone_by_seed.keys() & combined_by_seed.keys():
                standalone = standalone_by_seed[recovery_seed]
                fallback = combined_by_seed[recovery_seed]
                paired_fields = (
                    "path_generation_success",
                    "path_node_sequence",
                    "path_hash",
                    "path_length",
                )
                if any(str(standalone.get(field)) != str(fallback.get(field)) for field in paired_fields):
                    raise AssertionError(
                        f"paired probabilistic path mismatch for {key}, seed={recovery_seed}"
                    )
        elif not any(row.get("strategy") == "relaxation" for row in candidates):
            raise AssertionError(f"relaxation failure diagnostic missing for {key}")

    rebuilt = _build_summary(terrain_rows, candidate_rows)
    if len(rebuilt) != len(summary_rows):
        raise AssertionError("summary row count mismatch")
    for expected, actual in zip(rebuilt, summary_rows):
        for field in SUMMARY_FIELDS:
            left, right = expected.get(field), actual.get(field)
            if field == "alpha":
                if str(left) != str(right):
                    raise AssertionError(f"summary alpha mismatch: {left} != {right}")
                continue
            left_float, right_float = _as_float(left), _as_float(right)
            if np.isnan(left_float) and np.isnan(right_float):
                continue
            if not math.isclose(left_float, right_float, rel_tol=1e-12, abs_tol=1e-9):
                raise AssertionError(
                    f"summary mismatch alpha={expected['alpha']} field={field}: "
                    f"{left_float} != {right_float}"
                )


def _extract_json_seed(path: Path, terrain_key: str) -> Optional[int]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except Exception:
        return None
    containers = [data]
    if isinstance(data, dict) and isinstance(data.get(terrain_key), dict):
        containers.insert(0, data[terrain_key])
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in ("terrain_seed", "generation_seed", "seed"):
            if key in container:
                try:
                    return int(container[key])
                except (TypeError, ValueError):
                    pass
    return None


def _infer_terrain_seed(path: Path, terrain_key: str, seed_base: Optional[int]) -> Optional[int]:
    embedded = _extract_json_seed(path, terrain_key)
    if embedded is not None:
        return embedded
    if seed_base is None:
        return None
    match = re.search(r"run_(\d+)$", path.stem)
    return None if match is None else int(seed_base) + int(match.group(1))


def _alpha_directory(root: Path, alpha: float, number_of_alphas: int) -> Path:
    label = str(float(alpha))
    subdirectory = root / label
    if subdirectory.is_dir():
        return subdirectory
    if number_of_alphas == 1 and root.is_dir() and list(root.glob("*.json")):
        return root
    raise FileNotFoundError(f"no terrain directory for alpha={label}: expected {subdirectory}")


def _discover_cases(args: argparse.Namespace) -> List[TerrainCase]:
    root = Path(args.terrain_dir)
    if not root.is_dir():
        raise NotADirectoryError(root)
    cases: List[TerrainCase] = []
    for alpha_index, alpha in enumerate(args.alphas):
        directory = _alpha_directory(root, float(alpha), len(args.alphas))
        files = sorted(directory.glob("*.json"))
        if int(args.num_terrains) > 0:
            files = files[: int(args.num_terrains)]
        if not files:
            raise FileNotFoundError(f"no JSON terrain files in {directory}")
        if int(args.num_terrains) > 0 and len(files) < int(args.num_terrains):
            raise FileNotFoundError(
                f"requested {args.num_terrains} terrains for alpha={alpha}, found {len(files)}"
            )
        for terrain_index, path in enumerate(files):
            cases.append(
                TerrainCase(
                    alpha=float(alpha),
                    alpha_index=int(alpha_index),
                    terrain_index=int(terrain_index),
                    path=path,
                    terrain_seed=_infer_terrain_seed(
                        path, str(args.terrain_key), args.terrain_seed_base
                    ),
                )
            )
    return cases


def _run_smoke_self_checks() -> None:
    edge = np.asarray([[1, 2], [1, 3], [2, 4], [3, 4]], dtype=int)
    y = np.asarray([0.7, 0.3, 0.7, 0.3], dtype=float)
    seed = _stable_recovery_seed(123, 0, 0, 0)
    first = relaxation_prob_pure(y, edge, 4, temperature=1.0, seed=seed)
    second = relaxation_prob_pure(y, edge, 4, temperature=1.0, seed=seed)
    if first != second:
        raise AssertionError("pure probabilistic recovery is not reproducible for a fixed seed")
    if first[0] != 1 or first[-1] != 4:
        raise AssertionError("pure probabilistic recovery did not produce a source-to-target path")
    disconnected = np.asarray([[1, 2], [3, 4]], dtype=int)
    try:
        relaxation_prob_pure(y[:2], disconnected, 4, temperature=1.0, seed=seed)
    except PureProbabilisticPathError as exc:
        if exc.reason != "no_source_target_path":
            raise
    else:
        raise AssertionError("pure probabilistic recovery unexpectedly used a fallback")

    empty_summary = _summary_row("overall", [], [])
    rate_fields = [field for field in SUMMARY_FIELDS if field.endswith("rate_percent")]
    if any(not math.isnan(_as_float(empty_summary[field])) for field in rate_fields):
        raise AssertionError("zero-denominator summary rate was not stored as NaN")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible CSV data for Reviewer Comment 1.5"
    )
    parser.add_argument("--terrain-dir", type=Path, default=DEFAULT_TERRAIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--num-terrains", type=int, default=100, help="0 means all")
    parser.add_argument("--base-seed", type=int, default=123, help="recovery seed base")
    parser.add_argument(
        "--terrain-seed-base",
        type=int,
        default=123,
        help="infer generator seed as this value + run_NNN when JSON has no seed",
    )
    parser.add_argument("--num-prob-seeds", type=int, default=10)
    parser.add_argument("--max-combined-candidates", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-backtracks", type=int, default=20000)
    parser.add_argument(
        "--resolve-duplicate-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="re-solve duplicate discrete paths (default: true)",
    )
    parser.add_argument(
        "--terrain-time-limit-sec",
        "--time-limit-sec",
        dest="terrain_time_limit_sec",
        type=float,
        default=0.0,
        help=(
            "cooperative wall-clock limit for one complete terrain; --time-limit-sec is an "
            "alias; <=0 disables. An active MOSEK optimize call is not interrupted."
        ),
    )
    parser.add_argument(
        "--recovery-time-limit-sec",
        type=float,
        default=0.0,
        help=(
            "separate cooperative limit for each recovery setting on one terrain; "
            "<=0 disables. Independent fixed-model build time is excluded."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")

    # Existing CVAE OptAPI formulation settings.  They are exposed for exact
    # provenance but their defaults match the existing experiment code.
    parser.add_argument("--terrain-key", default="terrain")
    parser.add_argument("--copy-counts", type=int, nargs="+", default=None)
    parser.add_argument("--step-span", type=float, default=0.7)
    parser.add_argument("--alpha-deg", type=float, default=30.0)
    parser.add_argument("--M-large", type=float, default=1e3)
    parser.add_argument("--M-small", type=float, default=1e-6)
    parser.add_argument("--stance-width", type=float, default=0.25)
    parser.add_argument("--solver-log", type=int, default=0)
    args = parser.parse_args()

    if int(args.num_terrains) < 0:
        parser.error("--num-terrains must be >= 0")
    if int(args.num_prob_seeds) < 1:
        parser.error("--num-prob-seeds must be >= 1")
    if int(args.max_combined_candidates) < 1:
        parser.error("--max-combined-candidates must be >= 1")
    if float(args.temperature) <= 0.0:
        parser.error("--temperature must be > 0")
    if int(args.max_backtracks) < 1:
        parser.error("--max-backtracks must be >= 1")
    if float(args.M_small) < 0.0:
        parser.error("--M-small must be >= 0")
    if not args.alphas:
        parser.error("--alphas must not be empty")
    return args


def main() -> None:
    args = _parse_arguments()
    if bool(args.smoke_test):
        _run_smoke_self_checks()

    cases = _discover_cases(args)
    output_dir = Path(args.output_dir)
    candidate_path = output_dir / "per_candidate_results.csv"
    terrain_path = output_dir / "per_terrain_results.csv"
    summary_path = output_dir / "recovery_summary.csv"
    output_paths = (candidate_path, terrain_path, summary_path)

    if bool(args.resume):
        candidate_rows, terrain_rows, completed = _load_resume_state(
            candidate_path, terrain_path, args
        )
    else:
        if not bool(args.overwrite) and any(path.exists() for path in output_paths):
            raise FileExistsError(
                f"output CSV already exists in {output_dir}; use --resume or --overwrite"
            )
        candidate_rows = []
        terrain_rows = []
        completed = set()

    output_dir.mkdir(parents=True, exist_ok=True)
    # Clean partial resume rows atomically before continuing.
    summary_rows = _build_summary(terrain_rows, candidate_rows)
    _atomic_write_csv(candidate_path, PER_CANDIDATE_FIELDS, candidate_rows)
    _atomic_write_csv(terrain_path, PER_TERRAIN_FIELDS, terrain_rows)
    _atomic_write_csv(summary_path, SUMMARY_FIELDS, summary_rows)

    total = len(cases)
    for ordinal, case in enumerate(cases, start=1):
        if case.key in completed:
            print(f"[{ordinal}/{total}] SKIP completed alpha={case.alpha_label} {case.path.name}")
            continue

        print(f"[{ordinal}/{total}] RUN alpha={case.alpha_label} {case.path.name}")
        terrain_row, new_candidates = _process_terrain(case, args)
        candidate_rows = _replace_keyed_rows(candidate_rows, case.key, new_candidates)
        terrain_rows = _replace_keyed_rows(terrain_rows, case.key, [terrain_row])

        # Commit candidate rows first, then the terrain completion marker.  If a
        # process stops between these atomic replaces, --resume removes the
        # uncommitted candidate rows and reruns the terrain without duplicates.
        _atomic_write_csv(candidate_path, PER_CANDIDATE_FIELDS, candidate_rows)
        _atomic_write_csv(terrain_path, PER_TERRAIN_FIELDS, terrain_rows)

        # Build the aggregate strictly from the just-committed raw CSV files.
        raw_candidates = _read_csv(candidate_path)
        raw_terrains = _read_csv(terrain_path)
        summary_rows = _build_summary(raw_terrains, raw_candidates)
        _atomic_write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
        raw_summary = _read_csv(summary_path)
        _validate_consistency(raw_candidates, raw_terrains, raw_summary)

        print(
            f"[{ordinal}/{total}] DONE relax={_as_bool(terrain_row['relax_success'])} "
            f"greedy={_as_bool(terrain_row['greedy_fixed_success'])} "
            f"prob_any={_as_bool(terrain_row['probabilistic_any_success'])} "
            f"combined={_as_bool(terrain_row['combined_success'])}"
        )

    final_candidates = _read_csv(candidate_path)
    final_terrains = _read_csv(terrain_path)
    final_summary = _read_csv(summary_path)
    _validate_consistency(final_candidates, final_terrains, final_summary)


if __name__ == "__main__":
    main()
