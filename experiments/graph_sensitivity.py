#!/usr/bin/env python3
"""Graph-construction sensitivity data for Reviewer Comment 10.3.

The existing CVAE terrain loader, graph builder, OptAPI model, relaxation,
fixed-path solver, and postprocessor are reused without modifying repository
source.  Only the graph variants that the existing ``build_graph`` interface
cannot express are implemented locally.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

# Importing the repository package also imports its plotting helpers.  Redirect
# Matplotlib's cache to writable temporary storage; no plot or animation is run.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/footstep_planner_mpl_cache")

import mosek
import numpy as np

from main_VAE_optapi import make_entry_from_stones, rotate_vertices_with_yaw
from src.graph import build_graph, compute_centroids_from_vertices
from src.model import build_gcs_model_VAE_optapi, pack_relaxed_params_for_model
from src.solver.path import path_to_y, relaxation_max, relaxation_prob
from src.solver.stochastic import (
    _solve_fixed_once_optapi,
)
from src.solver.utils import set_parameters_optapi, solve_relaxed_optapi
from src.terrain_preset import load_terrain_yaw
from src.utils import expand_shorthand, postprocess_solution


DEFAULT_TERRAIN_ROOT = Path("data/terrains/alpha")
DEFAULT_OUTPUT_DIR = Path("outputs/Com10_3")
DEFAULT_DUPLICATES = (5, 4, 4, 5)
CONFIGURATIONS = (
    "dup_minimal",
    "default",
    "dup_plus_one",
    "dup_double",
    "edge_strict",
    "edge_dense",
)
ALTERNATIVE_CONFIGURATIONS = tuple(x for x in CONFIGURATIONS if x != "default")
EXECUTION_ORDER = ("default",) + ALTERNATIVE_CONFIGURATIONS

RAW_FIELDS = [
    "terrain_id",
    "terrain_seed",
    "alpha",
    "number_of_regions",
    "configuration",
    "duplicate_policy",
    "duplicate_count_region_1",
    "duplicate_count_region_2",
    "duplicate_count_region_3",
    "duplicate_count_region_4",
    "total_duplicate_count",
    "edge_policy",
    "edge_threshold_scale",
    "number_of_vertices",
    "number_of_edges",
    "number_of_mode_compatible_candidate_edges",
    "number_of_edges_removed_by_geometric_pruning",
    "graph_density",
    "graph_path_exists_before_optimization",
    "success",
    "failure_stage",
    "failure_reason",
    "timeout_occurred",
    "solver_status_relaxed",
    "solver_status_fixed",
    "graph_construction_time_ms",
    "model_build_time_ms",
    "relaxed_solve_time_ms",
    "path_recovery_time_ms",
    "fixed_path_solve_time_ms",
    "path_recovery_and_fixed_solve_time_ms",
    "total_planning_time_ms",
    "C_relax",
    "C_round",
    "final_objective",
    "delta_relax",
    "actual_number_of_footsteps",
    "selected_path_length",
    "number_of_path_recovery_trials",
    "selected_region_sequence",
    "selected_support_mode_sequence",
]

SUMMARY_FIELDS = [
    "configuration",
    "number_of_trials",
    "number_of_successes",
    "success_rate_percent",
    "vertices_mean",
    "vertices_std",
    "edges_mean",
    "edges_std",
    "graph_construction_time_mean_ms",
    "graph_construction_time_std_ms",
    "model_build_time_mean_ms",
    "model_build_time_std_ms",
    "relaxed_solve_time_mean_ms",
    "relaxed_solve_time_std_ms",
    "path_recovery_and_fixed_solve_time_mean_ms",
    "path_recovery_and_fixed_solve_time_std_ms",
    "total_planning_time_mean_ms",
    "total_planning_time_std_ms",
    "total_planning_time_median_ms",
    "C_relax_mean_success_only",
    "C_round_mean_success_only",
    "final_objective_mean_success_only",
    "delta_relax_mean_success_only",
    "actual_number_of_footsteps_mean_success_only",
]

PAIRED_FIELDS = [
    "terrain_id",
    "alternative_configuration",
    "default_success",
    "alternative_success",
    "default_vertices",
    "alternative_vertices",
    "default_edges",
    "alternative_edges",
    "default_total_time_ms",
    "alternative_total_time_ms",
    "total_time_difference_ms",
    "total_time_ratio",
    "default_final_objective",
    "alternative_final_objective",
    "objective_difference",
    "default_delta_relax",
    "alternative_delta_relax",
    "delta_relax_difference",
    "default_number_of_footsteps",
    "alternative_number_of_footsteps",
]

TERRAIN_FIELDS = [
    "terrain_id",
    "terrain_seed",
    "alpha",
    "number_of_regions",
    "start_region",
    "goal_region",
]


@dataclass(frozen=True)
class TerrainCase:
    terrain_id: int
    terrain_seed: int
    path: Path
    stones: tuple[dict[str, Any], ...]
    default_duplicates: tuple[int, int, int, int]


@dataclass
class ProblemData:
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
    candidate_edge_count: int
    geometric_removed_count: int
    threshold_scale: float

    @property
    def vertex_number(self) -> int:
        return len(self.vertices)

    @property
    def n_edge(self) -> int:
        return int(self.edge.shape[0])


@dataclass
class CandidateResult:
    success: bool
    path: Optional[list[int]] = None
    objective: float = math.nan
    solution: Optional[dict[str, Any]] = None
    processed: Any = None
    solver_status: str = ""
    failure_stage: str = "path_recovery"
    failure_reason: str = ""


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


def _atomic_write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
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


def _error_text(error: BaseException | str, limit: int = 1200) -> str:
    text = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return text.replace("\n", " | ")[:limit]


def _pose5(position: np.ndarray, yaw: float) -> np.ndarray:
    position = np.asarray(position, dtype=float).reshape(-1)
    return np.asarray(
        [position[0], position[1], position[2], math.cos(yaw), math.sin(yaw)],
        dtype=float,
    )


def _reference_axes(start_left: np.ndarray, start_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(start_left, dtype=float)[:2] - np.asarray(start_right, dtype=float)[:2]
    norm = float(np.linalg.norm(direction))
    y_ref = np.asarray([0.0, 1.0]) if norm < 1e-12 else direction / norm
    return np.asarray([y_ref[1], -y_ref[0]], dtype=float), y_ref


def _discover_terrains(args: argparse.Namespace) -> list[TerrainCase]:
    alpha_dir = Path(args.terrain_dir) / "0.0"
    if not alpha_dir.is_dir() and Path(args.terrain_dir).is_dir():
        direct = sorted(Path(args.terrain_dir).glob("*.json"))
        if direct:
            alpha_dir = Path(args.terrain_dir)
    paths = sorted(alpha_dir.glob("*.json"))
    if len(paths) < 100:
        raise FileNotFoundError(f"alpha=0 dataset needs 100 JSON files; found {len(paths)} in {alpha_dir}")
    paths = paths[:100]

    cases: list[TerrainCase] = []
    for ordinal, path in enumerate(paths):
        match = re.fullmatch(r"run_(\d+)", path.stem)
        terrain_id = int(match.group(1)) if match else ordinal
        terrain = load_terrain_yaw(name=args.terrain_key, path=path)
        stones = terrain["stones"]
        if len(stones) != 4:
            raise ValueError(f"{path}: expected exactly four terrain regions, got {len(stones)}")
        embedded_counts = terrain.get("copy_counts")
        counts = tuple(int(x) for x in (embedded_counts or DEFAULT_DUPLICATES))
        if len(counts) != 4 or any(x < 1 for x in counts):
            raise ValueError(f"{path}: invalid default copy counts {counts}")
        cases.append(
            TerrainCase(
                terrain_id=terrain_id,
                terrain_seed=int(args.base_seed) + terrain_id,
                path=path.resolve(),
                stones=tuple(copy.deepcopy(stones)),
                default_duplicates=counts,  # type: ignore[arg-type]
            )
        )
    ids = [case.terrain_id for case in cases]
    if ids != list(range(100)):
        raise ValueError(f"terrain IDs must be exactly 0..99; got {ids[:5]}...{ids[-5:]}")
    return cases


def _terrain_rows(cases: Sequence[TerrainCase]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        start = case.stones[0]
        goal = case.stones[-1]
        rows.append(
            {
                "terrain_id": case.terrain_id,
                "terrain_seed": case.terrain_seed,
                "alpha": 0.0,
                "number_of_regions": 4,
                "start_region": json.dumps(
                    {"index": 0, "name": start.get("name", "stone_1"), "center": start["center"]},
                    separators=(",", ":"),
                ),
                "goal_region": json.dumps(
                    {"index": 3, "name": goal.get("name", "stone_4"), "center": goal["center"]},
                    separators=(",", ":"),
                ),
            }
        )
    return rows


def _duplicates_for(configuration: str, default: Sequence[int]) -> list[int]:
    default = [int(x) for x in default]
    if configuration == "dup_minimal":
        return [1, 1, 1, 1]
    if configuration == "dup_plus_one":
        return [x + 1 for x in default]
    if configuration == "dup_double":
        return [2 * x for x in default]
    return default


def _region_map(copy_counts: Sequence[int]) -> list[int]:
    return [region for region, count in enumerate(copy_counts) for _ in range(int(count))]


def _opposite_single_support(first: str, second: str) -> bool:
    return (first == "SL" and second == "SR") or (first == "SR" and second == "SL")


def _dense_mode_compatible_edges(
    modes: Sequence[str], region_of_vertex: Sequence[int]
) -> np.ndarray:
    """All forward support-compatible pairs with existing source/sink scope."""
    vertex_number = len(modes)
    source, sink = 1, vertex_number
    edges: list[tuple[int, int]] = []
    for u in range(1, vertex_number):
        for v in range(u + 1, vertex_number + 1):
            mode_u, mode_v = modes[u - 1], modes[v - 1]
            if u == source:
                compatible = region_of_vertex[v - 1] == 0 and mode_v in {"SL", "SR"}
            elif v == sink:
                compatible = region_of_vertex[u - 1] == 3 and mode_u in {"SL", "SR"}
            else:
                compatible = _opposite_single_support(mode_u, mode_v)
            if compatible:
                edges.append((u, v))
    return np.asarray(edges, dtype=int).reshape(-1, 2)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = b - a
    second = c - a
    return float(first[0] * second[1] - first[1] * second[0])


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    eps = 1e-12
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
        )

    return bool(
        (abs(o1) <= eps and on_segment(a, c, b))
        or (abs(o2) <= eps and on_segment(a, d, b))
        or (abs(o3) <= eps and on_segment(c, a, d))
        or (abs(o4) <= eps and on_segment(c, b, d))
    )


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    delta = b - a
    denominator = float(delta @ delta)
    if denominator <= 1e-24:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(((point - a) @ delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * delta)))


def _polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_xy = np.asarray(first, dtype=float)[:2, :].T
    second_xy = np.asarray(second, dtype=float)[:2, :].T
    first_segments = list(zip(first_xy, np.roll(first_xy, -1, axis=0)))
    second_segments = list(zip(second_xy, np.roll(second_xy, -1, axis=0)))
    if any(_segments_intersect(a, b, c, d) for a, b in first_segments for c, d in second_segments):
        return 0.0
    distances = []
    for a, b in first_segments:
        for c, d in second_segments:
            distances.extend(
                [
                    _point_segment_distance(a, c, d),
                    _point_segment_distance(b, c, d),
                    _point_segment_distance(c, a, b),
                    _point_segment_distance(d, a, b),
                ]
            )
    return min(distances) if distances else math.inf


def _strict_filter(
    edges: np.ndarray, vertices: Sequence[np.ndarray], threshold: float
) -> tuple[np.ndarray, int]:
    kept = []
    for u, v in np.asarray(edges, dtype=int):
        distance = _polygon_distance(vertices[int(u) - 1], vertices[int(v) - 1])
        if distance <= float(threshold) + 1e-12:
            kept.append((int(u), int(v)))
    result = np.asarray(kept, dtype=int).reshape(-1, 2)
    return result, int(len(edges) - len(result))


def _graph_path_exists(edges: np.ndarray, vertex_number: int) -> bool:
    adjacency: dict[int, list[int]] = {node: [] for node in range(1, vertex_number + 1)}
    for u, v in np.asarray(edges, dtype=int):
        adjacency[int(u)].append(int(v))
    stack, visited = [1], {1}
    while stack:
        node = stack.pop()
        if node == vertex_number:
            return True
        for successor in adjacency.get(node, []):
            if successor not in visited:
                visited.add(successor)
                stack.append(successor)
    return False


def _prepare_problem(
    case: TerrainCase, configuration: str, args: argparse.Namespace
) -> ProblemData:
    stones = copy.deepcopy(list(case.stones))
    duplicates = _duplicates_for(configuration, case.default_duplicates)
    entry = make_entry_from_stones(stones, float(args.stance_width))
    expanded = expand_shorthand(entry, float(args.stance_width))
    startpos_L = np.asarray(expanded["startpos_L"], dtype=float)
    startpos_R = np.asarray(expanded["startpos_R"], dtype=float)
    targetpos_L = np.asarray(expanded["targetpos_L"], dtype=float)
    targetpos_R = np.asarray(expanded["targetpos_R"], dtype=float)

    existing_graph = build_graph(
        startpos_L,
        startpos_R,
        targetpos_L,
        targetpos_R,
        step_span=float(args.step_span),
        copy_counts=duplicates,
        edge_rule="general",
        terrain_stones=stones,
    )
    vertices = rotate_vertices_with_yaw(existing_graph["v"], stones, duplicates)
    modes = list(existing_graph["mode_list"])
    region_of_vertex = _region_map(duplicates)
    default_edges = np.asarray(existing_graph["edge"], dtype=int).reshape(-1, 2)
    mode_compatible_candidates = _dense_mode_compatible_edges(modes, region_of_vertex)
    candidate_count = len(mode_compatible_candidates)

    if configuration == "edge_dense":
        edge = mode_compatible_candidates
        removed = 0
        threshold_scale = math.nan
    elif configuration == "edge_strict":
        edge, removed = _strict_filter(
            default_edges, vertices, 0.8 * float(args.step_span)
        )
        threshold_scale = 0.8
    else:
        edge = default_edges
        removed = 0
        threshold_scale = 1.0

    centroid_mat, _ = compute_centroids_from_vertices(vertices)
    x_ref, y_ref = _reference_axes(startpos_L, startpos_R)
    return ProblemData(
        stones=stones,
        vertices=[np.asarray(v, dtype=float) for v in vertices],
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
        candidate_edge_count=int(candidate_count),
        geometric_removed_count=int(removed),
        threshold_scale=float(threshold_scale),
    )


def _acceptable_status(task: Any) -> bool:
    try:
        status = task.getsolsta(mosek.soltype.itr)
    except Exception:
        return False
    acceptable = status == mosek.solsta.optimal
    for name in ("integer_optimal", "near_optimal"):
        if hasattr(mosek.solsta, name):
            acceptable = acceptable or status == getattr(mosek.solsta, name)
    return bool(acceptable)


def _status(task: Any) -> str:
    if task is None:
        return ""
    try:
        return str(task.getsolsta(mosek.soltype.itr))
    except Exception:
        return ""


def _close_task(task: Any) -> None:
    if task is None:
        return
    try:
        task.dispose()
    except Exception:
        try:
            task.__exit__(None, None, None)
        except Exception:
            pass


def _valid_path(path: Sequence[int], edge: np.ndarray, vertex_number: int) -> bool:
    if len(path) < 2 or int(path[0]) != 1 or int(path[-1]) != int(vertex_number):
        return False
    available = {(int(u), int(v)) for u, v in np.asarray(edge, dtype=int)}
    return all((int(u), int(v)) in available for u, v in zip(path[:-1], path[1:]))


def _recovery_seed(base_seed: int, terrain_id: int, candidate_index: int) -> int:
    sequence = np.random.SeedSequence(
        [int(base_seed), int(terrain_id), int(candidate_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _recover_and_solve(
    task: Any,
    variables: dict[str, Any],
    idx: dict[str, Any],
    problem: ProblemData,
    y_relaxed: np.ndarray,
    case: TerrainCase,
    args: argparse.Namespace,
    deadline: Optional[float],
) -> tuple[CandidateResult, float, float, int, bool]:
    recovery_ms = 0.0
    fixed_ms = 0.0
    attempts = 0
    timeout = False
    last = CandidateResult(False, failure_reason="no candidate attempted")

    for candidate_index in range(int(args.trials)):
        if deadline is not None and time.perf_counter() >= deadline:
            timeout = True
            break
        attempts += 1
        recovery_start = time.perf_counter()
        try:
            if candidate_index == 0:
                path = relaxation_max(
                    y_relaxed,
                    problem.edge,
                    problem.vertex_number,
                    max_backtracks=int(args.max_backtracks),
                )
            else:
                path = relaxation_prob(
                    y_relaxed,
                    problem.edge,
                    problem.vertex_number,
                    temperature=float(args.temperature),
                    seed=_recovery_seed(args.base_seed, case.terrain_id, candidate_index - 1),
                    max_backtracks=int(args.max_backtracks),
                )
            path = [int(node) for node in path]
            if not _valid_path(path, problem.edge, problem.vertex_number):
                raise RuntimeError("recovery returned an invalid source-to-target path")
            fixed_y = path_to_y(problem.edge, path, problem.n_edge)
        except Exception as error:
            recovery_ms += (time.perf_counter() - recovery_start) * 1000.0
            last = CandidateResult(
                False,
                failure_stage="path_recovery",
                failure_reason=_error_text(error),
            )
            continue
        recovery_ms += (time.perf_counter() - recovery_start) * 1000.0

        fixed_start = time.perf_counter()
        try:
            _, objective, solution = _solve_fixed_once_optapi(
                task,
                variables,
                idx,
                centroid_mat=problem.centroid_mat,
                y_relax=fixed_y,
                M_small=float(args.M_small),
                start_L=problem.start_L,
                start_R=problem.start_R,
                target_L=problem.target_L,
                target_R=problem.target_R,
                log_level=int(args.solver_log),
            )
        except Exception as error:
            fixed_ms += (time.perf_counter() - fixed_start) * 1000.0
            last = CandidateResult(
                False,
                path=path,
                solver_status=_status(task),
                failure_stage="fixed_path_solver",
                failure_reason=_error_text(error),
            )
            continue
        fixed_ms += (time.perf_counter() - fixed_start) * 1000.0

        if not _acceptable_status(task) or objective is None or not np.isfinite(float(objective)):
            last = CandidateResult(
                False,
                path=path,
                solver_status=_status(task),
                failure_stage="fixed_path_solver",
                failure_reason=f"unacceptable status or objective: {_status(task)}, {objective}",
            )
            continue
        try:
            processed = postprocess_solution(
                solution,
                problem.edge,
                path,
                problem.startpos_L,
                problem.startpos_R,
                problem.targetpos_L,
                problem.targetpos_R,
                problem.target_L,
                problem.target_R,
            )
        except Exception as error:
            last = CandidateResult(
                False,
                path=path,
                objective=float(objective),
                solution=solution,
                solver_status=_status(task),
                failure_stage="final_feasibility_check",
                failure_reason=_error_text(error),
            )
            continue

        return (
            CandidateResult(
                True,
                path=path,
                objective=float(objective),
                solution=solution,
                processed=processed,
                solver_status=_status(task),
                failure_stage="",
                failure_reason="",
            ),
            recovery_ms,
            fixed_ms,
            attempts,
            timeout,
        )

    if timeout:
        last.failure_stage = "timeout"
        last.failure_reason = "total planning deadline reached before a feasible fixed path"
    return last, recovery_ms, fixed_ms, attempts, timeout


def _base_row(case: TerrainCase, configuration: str) -> dict[str, Any]:
    row = {field: "" for field in RAW_FIELDS}
    duplicates = _duplicates_for(configuration, case.default_duplicates)
    duplicate_policy = configuration if configuration.startswith("dup_") else "default"
    edge_policy = configuration if configuration.startswith("edge_") else "default"
    row.update(
        {
            "terrain_id": case.terrain_id,
            "terrain_seed": case.terrain_seed,
            "alpha": 0.0,
            "number_of_regions": 4,
            "configuration": configuration,
            "duplicate_policy": duplicate_policy,
            "duplicate_count_region_1": duplicates[0],
            "duplicate_count_region_2": duplicates[1],
            "duplicate_count_region_3": duplicates[2],
            "duplicate_count_region_4": duplicates[3],
            "total_duplicate_count": sum(duplicates),
            "edge_policy": edge_policy,
            "edge_threshold_scale": math.nan,
            "number_of_vertices": math.nan,
            "number_of_edges": math.nan,
            "number_of_mode_compatible_candidate_edges": math.nan,
            "number_of_edges_removed_by_geometric_pruning": math.nan,
            "graph_density": math.nan,
            "graph_path_exists_before_optimization": False,
            "success": False,
            "failure_stage": "unknown",
            "failure_reason": "",
            "timeout_occurred": False,
            "solver_status_relaxed": "",
            "solver_status_fixed": "",
            "graph_construction_time_ms": 0.0,
            "model_build_time_ms": 0.0,
            "relaxed_solve_time_ms": 0.0,
            "path_recovery_time_ms": 0.0,
            "fixed_path_solve_time_ms": 0.0,
            "path_recovery_and_fixed_solve_time_ms": 0.0,
            "total_planning_time_ms": 0.0,
            "C_relax": math.nan,
            "C_round": math.nan,
            "final_objective": math.nan,
            "delta_relax": math.nan,
            "actual_number_of_footsteps": math.nan,
            "selected_path_length": math.nan,
            "number_of_path_recovery_trials": 0,
            "selected_region_sequence": "",
            "selected_support_mode_sequence": "",
        }
    )
    return row


def _run_trial(
    case: TerrainCase, configuration: str, args: argparse.Namespace
) -> dict[str, Any]:
    _seed_everything(case.terrain_seed)
    row = _base_row(case, configuration)
    total_start = time.perf_counter()
    deadline = (
        None
        if float(args.timeout_sec) <= 0.0
        else total_start + float(args.timeout_sec)
    )
    task = None
    try:
        graph_start = time.perf_counter()
        try:
            problem = _prepare_problem(case, configuration, args)
        finally:
            row["graph_construction_time_ms"] = (time.perf_counter() - graph_start) * 1000.0

        row["edge_threshold_scale"] = problem.threshold_scale
        row["number_of_vertices"] = problem.vertex_number
        row["number_of_edges"] = problem.n_edge
        row["number_of_mode_compatible_candidate_edges"] = problem.candidate_edge_count
        row["number_of_edges_removed_by_geometric_pruning"] = problem.geometric_removed_count
        possible_directed = problem.vertex_number * (problem.vertex_number - 1)
        row["graph_density"] = (
            problem.n_edge / possible_directed if possible_directed else math.nan
        )
        path_exists = _graph_path_exists(problem.edge, problem.vertex_number)
        row["graph_path_exists_before_optimization"] = path_exists
        if not path_exists:
            row["failure_stage"] = "graph_disconnected"
            row["failure_reason"] = "no directed source-to-sink path in constructed graph"
            return row
        if deadline is not None and time.perf_counter() >= deadline:
            row["failure_stage"] = "timeout"
            row["failure_reason"] = "deadline reached after graph construction"
            row["timeout_occurred"] = True
            return row

        model_start = time.perf_counter()
        try:
            task, variables, idx, _, _ = build_gcs_model_VAE_optapi(
                problem.vertices,
                problem.edge,
                float(args.step_span),
                math.radians(float(args.alpha_deg)),
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

        relaxed_start = time.perf_counter()
        try:
            parameters = pack_relaxed_params_for_model(
                centroid_mat=problem.centroid_mat,
                n_edge=int(idx["n_edge"]),
                M_large=float(args.M_large),
                start_L=problem.start_L,
                start_R=problem.start_R,
                target_L=problem.target_L,
                target_R=problem.target_R,
            )
            set_parameters_optapi(task, variables, parameters)
            relaxed_solution, relaxed_objective = solve_relaxed_optapi(
                task, variables, log_level=int(args.solver_log)
            )
        except Exception as error:
            row["solver_status_relaxed"] = _status(task)
            row["failure_stage"] = "relaxed_solver"
            row["failure_reason"] = _error_text(error)
            return row
        finally:
            row["relaxed_solve_time_ms"] = (time.perf_counter() - relaxed_start) * 1000.0

        row["solver_status_relaxed"] = _status(task)
        if (
            not _acceptable_status(task)
            or relaxed_objective is None
            or not np.isfinite(float(relaxed_objective))
        ):
            row["failure_stage"] = "relaxed_solver"
            row["failure_reason"] = (
                f"unacceptable relaxed status/objective: {_status(task)}, {relaxed_objective}"
            )
            return row
        y_relaxed = np.asarray(relaxed_solution["y"], dtype=float).reshape(-1)
        if y_relaxed.size != problem.n_edge or not np.all(np.isfinite(y_relaxed)):
            row["failure_stage"] = "relaxed_solver"
            row["failure_reason"] = "invalid relaxed edge vector"
            return row
        row["C_relax"] = float(relaxed_objective)

        if deadline is not None and time.perf_counter() >= deadline:
            row["failure_stage"] = "timeout"
            row["failure_reason"] = "deadline reached before path recovery"
            row["timeout_occurred"] = True
            return row

        result, recovery_ms, fixed_ms, attempts, timed_out = _recover_and_solve(
            task,
            variables,
            idx,
            problem,
            y_relaxed,
            case,
            args,
            deadline,
        )
        row["path_recovery_time_ms"] = recovery_ms
        row["fixed_path_solve_time_ms"] = fixed_ms
        row["path_recovery_and_fixed_solve_time_ms"] = recovery_ms + fixed_ms
        row["number_of_path_recovery_trials"] = attempts
        row["timeout_occurred"] = timed_out
        row["solver_status_fixed"] = result.solver_status
        if not result.success:
            row["failure_stage"] = result.failure_stage or "unknown"
            row["failure_reason"] = result.failure_reason
            return row

        assert result.path is not None and result.processed is not None
        row["success"] = True
        row["failure_stage"] = ""
        row["failure_reason"] = ""
        row["C_round"] = result.objective
        row["final_objective"] = result.objective
        row["delta_relax"] = (
            (result.objective - float(relaxed_objective)) / float(relaxed_objective)
            if abs(float(relaxed_objective)) > np.finfo(float).eps
            else math.nan
        )
        row["actual_number_of_footsteps"] = int(
            len(result.processed.L.x) + len(result.processed.R.x)
        )
        row["selected_path_length"] = len(result.path) - 1
        row["selected_region_sequence"] = "-".join(
            str(problem.region_of_vertex[node - 1]) for node in result.path
        )
        row["selected_support_mode_sequence"] = "-".join(
            str(problem.modes[node - 1]) for node in result.path
        )
        # Kept only in memory for paired-path default validation. DictWriter
        # ignores it, so the requested raw CSV schema remains unchanged.
        row["_selected_path_nodes"] = list(result.path)
        return row
    except Exception as error:
        if row["failure_stage"] == "unknown":
            row["failure_stage"] = "unknown"
        row["failure_reason"] = _error_text(error)
        return row
    finally:
        row["total_planning_time_ms"] = (time.perf_counter() - total_start) * 1000.0
        _close_task(task)


def _finite_values(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    return [value for value in (_as_float(row.get(field)) for row in rows) if np.isfinite(value)]


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _std(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return 0.0 if len(values) == 1 else float(np.std(values, ddof=1))


def _summary_rows(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for configuration in CONFIGURATIONS:
        rows = [row for row in raw_rows if row.get("configuration") == configuration]
        if not rows:
            continue
        successful = [row for row in rows if _as_bool(row.get("success"))]
        vertices = _finite_values(rows, "number_of_vertices")
        edges = _finite_values(rows, "number_of_edges")
        graph_time = _finite_values(rows, "graph_construction_time_ms")
        model_time = _finite_values(rows, "model_build_time_ms")
        relax_time = _finite_values(rows, "relaxed_solve_time_ms")
        recovery_fixed = _finite_values(rows, "path_recovery_and_fixed_solve_time_ms")
        total_time = _finite_values(rows, "total_planning_time_ms")
        summaries.append(
            {
                "configuration": configuration,
                "number_of_trials": len(rows),
                "number_of_successes": len(successful),
                "success_rate_percent": 100.0 * len(successful) / len(rows),
                "vertices_mean": _mean(vertices),
                "vertices_std": _std(vertices),
                "edges_mean": _mean(edges),
                "edges_std": _std(edges),
                "graph_construction_time_mean_ms": _mean(graph_time),
                "graph_construction_time_std_ms": _std(graph_time),
                "model_build_time_mean_ms": _mean(model_time),
                "model_build_time_std_ms": _std(model_time),
                "relaxed_solve_time_mean_ms": _mean(relax_time),
                "relaxed_solve_time_std_ms": _std(relax_time),
                "path_recovery_and_fixed_solve_time_mean_ms": _mean(recovery_fixed),
                "path_recovery_and_fixed_solve_time_std_ms": _std(recovery_fixed),
                "total_planning_time_mean_ms": _mean(total_time),
                "total_planning_time_std_ms": _std(total_time),
                "total_planning_time_median_ms": (
                    float(np.median(total_time)) if total_time else math.nan
                ),
                "C_relax_mean_success_only": _mean(_finite_values(successful, "C_relax")),
                "C_round_mean_success_only": _mean(_finite_values(successful, "C_round")),
                "final_objective_mean_success_only": _mean(
                    _finite_values(successful, "final_objective")
                ),
                "delta_relax_mean_success_only": _mean(
                    _finite_values(successful, "delta_relax")
                ),
                "actual_number_of_footsteps_mean_success_only": _mean(
                    _finite_values(successful, "actual_number_of_footsteps")
                ),
            }
        )
    return summaries


def _paired_rows(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (int(_as_float(row["terrain_id"])), str(row["configuration"])): row
        for row in raw_rows
        if np.isfinite(_as_float(row.get("terrain_id")))
    }
    paired = []
    terrain_ids = sorted({terrain_id for terrain_id, config in indexed if config == "default"})
    for terrain_id in terrain_ids:
        default = indexed[(terrain_id, "default")]
        for alternative in ALTERNATIVE_CONFIGURATIONS:
            other = indexed.get((terrain_id, alternative))
            if other is None:
                continue
            default_success = _as_bool(default.get("success"))
            other_success = _as_bool(other.get("success"))
            default_time = _as_float(default.get("total_planning_time_ms"))
            other_time = _as_float(other.get("total_planning_time_ms"))
            both_success = default_success and other_success
            default_objective = _as_float(default.get("final_objective"))
            other_objective = _as_float(other.get("final_objective"))
            default_delta = _as_float(default.get("delta_relax"))
            other_delta = _as_float(other.get("delta_relax"))
            paired.append(
                {
                    "terrain_id": terrain_id,
                    "alternative_configuration": alternative,
                    "default_success": default_success,
                    "alternative_success": other_success,
                    "default_vertices": default.get("number_of_vertices", math.nan),
                    "alternative_vertices": other.get("number_of_vertices", math.nan),
                    "default_edges": default.get("number_of_edges", math.nan),
                    "alternative_edges": other.get("number_of_edges", math.nan),
                    "default_total_time_ms": default_time,
                    "alternative_total_time_ms": other_time,
                    "total_time_difference_ms": (
                        other_time - default_time
                        if np.isfinite(default_time) and np.isfinite(other_time)
                        else math.nan
                    ),
                    "total_time_ratio": (
                        other_time / default_time
                        if np.isfinite(default_time) and np.isfinite(other_time) and default_time != 0.0
                        else math.nan
                    ),
                    "default_final_objective": default_objective if both_success else math.nan,
                    "alternative_final_objective": other_objective if both_success else math.nan,
                    "objective_difference": (
                        other_objective - default_objective if both_success else math.nan
                    ),
                    "default_delta_relax": default_delta if both_success else math.nan,
                    "alternative_delta_relax": other_delta if both_success else math.nan,
                    "delta_relax_difference": (
                        other_delta - default_delta if both_success else math.nan
                    ),
                    "default_number_of_footsteps": default.get(
                        "actual_number_of_footsteps", math.nan
                    ),
                    "alternative_number_of_footsteps": other.get(
                        "actual_number_of_footsteps", math.nan
                    ),
                }
            )
    return paired


def _write_analysis(output_dir: Path, raw_rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_csv(
        output_dir / "summary_by_configuration.csv",
        SUMMARY_FIELDS,
        _summary_rows(raw_rows),
    )
    _atomic_write_csv(
        output_dir / "paired_with_default.csv",
        PAIRED_FIELDS,
        _paired_rows(raw_rows),
    )


def _graph_rule_text(cases: Sequence[TerrainCase], args: argparse.Namespace) -> str:
    terrain_counts = "\n".join(
        f"  terrain_id={case.terrain_id}: {list(case.default_duplicates)}"
        for case in cases
    )
    return f"""Current CVAE graph-construction rule audit
==========================================

Dataset
-------
- Existing CVAE JSON files are reused from: {Path(args.terrain_dir) / '0.0'}
- difficulty alpha = 0.0; exactly four physical regions per JSON.
- terrain_seed is recorded as base_seed + terrain_id, matching the CVAE generator's
  seed_all(SEED + i) convention. The JSON itself does not embed a seed field.
- Region 0 is the physical start region and region 3 is the physical goal region;
  start/goal are not separate additional terrain regions.

Default duplicate rule
----------------------
- Four-region paper/evaluation default used here: [5, 4, 4, 5]. It is the
  valid four-region setting in main_VAE_fusionapi.py and Com_1_5.py.
- The current main_VAE_optapi.py CLI literal is [5, 2, 2, 2, 2, 5], which is
  a six-region vector and raises a length mismatch on these four-region JSONs;
  it is therefore not treated as a valid default for this dataset.
- JSON copy_counts takes precedence if present. These 100 JSON files do not embed it.
- TerrainGCS.get_vertices_with_modes creates identical convex-region copies; it does
  not geometrically split a region.
- In the start group, copy 1 is D, then SL/SR alternate. In the goal group, the last
  copy is D; preceding copies alternate SL/SR. Middle groups alternate SL/SR.

Per-terrain actual default duplicate counts
-------------------------------------------
{terrain_counts}

Default edge rule
-----------------
- main_VAE_optapi.py calls build_graph(..., edge_rule='general').
- TerrainGCS.create_edges_general connects a directed acyclic sparse topology:
  source D to the first SL/SR copies, alternating SL<->SR chains within groups,
  parity-compatible transitions to the next physical region, then goal-group
  single-support copies to the final D sink.
- Mode compatibility is implicit in duplicate parity/order, not a separate geometric test.

Geometric pruning and physical reachability
-------------------------------------------
- The current default graph builder performs NO geometric-distance pruning.
- step_span={float(args.step_span):.6g} is enforced inside build_gcs_model_VAE_optapi
  as the physical SOC step-reachability constraint; it is not used by create_edges_general.
- edge_strict is therefore a new graph-preprocessing sensitivity variant: it filters
  only the existing default edge set using minimum planar polygon distance <=
  0.8 * step_span = {0.8 * float(args.step_span):.6g}. The optimization constraint
  remains step_span={float(args.step_span):.6g}.
- edge_dense removes graph-level geometric screening and includes every forward
  SL<->SR-compatible pair. Source D only connects within the physical start group,
  and only physical-goal-group single-support vertices connect to sink D.
- number_of_mode_compatible_candidate_edges counts this full support-compatible
  forward-pair universe for every configuration. The difference between that count
  and default edges is topology sparsification, not geometric pruning.
- number_of_edges_removed_by_geometric_pruning counts only edges removed from the
  existing default topology by edge_strict; default and edge_dense record zero.

Start/goal handling
-------------------
- make_entry_from_stones and expand_shorthand form initial/final double-support foot
  poses from the first and last physical region centers.
- Graph source is vertex 1 (D); graph sink is the last vertex (D).

Planner/recovery settings held fixed
------------------------------------
- OptAPI formulation: build_gcs_model_VAE_optapi; step_span={float(args.step_span):.6g},
  yaw bound={float(args.alpha_deg):.6g} deg, M_large={float(args.M_large):.6g},
  M_small={float(args.M_small):.6g}, stance_width={float(args.stance_width):.6g}.
- Recovery: greedy first, then relaxation_prob fallback, early stop at the first
  acceptable fixed solution, total candidate budget={int(args.trials)}.
- Existing main_VAE_optapi.py uses seed=None inside probabilistic recovery. This
  experiment supplies SeedSequence-derived seeds shared by every configuration
  to meet the paired reproducibility requirement.
- Cooperative per-trial planning deadline={float(args.timeout_sec):.6g} seconds;
  an already-running MOSEK optimize call is not interrupted, matching the existing wrapper.
"""


def _reference_default_result(
    case: TerrainCase,
    selected_path: Sequence[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Re-solve the same path through the existing default evaluator calls.

    The repository's timed recovery wrapper calls ``default_rng(None)`` and is
    intentionally nondeterministic when greedy recovery is infeasible.  Using
    the experimental path here isolates graph/model/fixed-solver equivalence
    from that unrelated random-path variation.
    """
    _seed_everything(case.terrain_seed)
    start = time.perf_counter()
    deadline = None if args.timeout_sec <= 0 else start + float(args.timeout_sec)
    problem = _prepare_problem(case, "default", args)
    task = None
    try:
        task, variables, idx, _, _ = build_gcs_model_VAE_optapi(
            problem.vertices,
            problem.edge,
            float(args.step_span),
            math.radians(float(args.alpha_deg)),
            problem.modes,
            problem.x_ref,
            problem.y_ref,
        )
        parameters = pack_relaxed_params_for_model(
            centroid_mat=problem.centroid_mat,
            n_edge=int(idx["n_edge"]),
            M_large=float(args.M_large),
            start_L=problem.start_L,
            start_R=problem.start_R,
            target_L=problem.target_L,
            target_R=problem.target_R,
        )
        set_parameters_optapi(task, variables, parameters)
        relaxed_solution, relaxed_objective = solve_relaxed_optapi(
            task, variables, log_level=int(args.solver_log)
        )
        remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())
        if remaining is not None and remaining <= 0.0:
            raise TimeoutError("reference default exhausted deadline before recovery")
        path = [int(node) for node in selected_path]
        if not _valid_path(path, problem.edge, problem.vertex_number):
            raise ValueError("selected experimental path is invalid in reference default graph")
        fixed_y = path_to_y(problem.edge, path, problem.n_edge)
        _, final_objective, extracted = _solve_fixed_once_optapi(
            task,
            variables,
            idx,
            centroid_mat=problem.centroid_mat,
            y_relax=fixed_y,
            M_small=float(args.M_small),
            start_L=problem.start_L,
            start_R=problem.start_R,
            target_L=problem.target_L,
            target_R=problem.target_R,
            log_level=int(args.solver_log),
        )
        processed = postprocess_solution(
            extracted,
            problem.edge,
            path,
            problem.startpos_L,
            problem.startpos_R,
            problem.targetpos_L,
            problem.targetpos_R,
            problem.target_L,
            problem.target_R,
        )
        return {
            "success": True,
            "number_of_vertices": problem.vertex_number,
            "number_of_edges": problem.n_edge,
            "C_relax": float(relaxed_objective),
            "final_objective": float(final_objective),
            "actual_number_of_footsteps": len(processed.L.x) + len(processed.R.x),
        }
    except Exception as error:
        return {"success": False, "error": _error_text(error)}
    finally:
        _close_task(task)


def _validate_default(
    case: TerrainCase,
    experimental: Mapping[str, Any],
    args: argparse.Namespace,
) -> str:
    selected_path = experimental.get("_selected_path_nodes")
    if not selected_path:
        rerun = _run_trial(case, "default", args)
        selected_path = rerun.get("_selected_path_nodes")
    if not selected_path:
        raise AssertionError("default validation has no successful selected path")
    reference = _reference_default_result(case, selected_path, args)
    if _as_bool(reference.get("success")) != _as_bool(experimental.get("success")):
        raise AssertionError(f"default success mismatch: {reference} vs {experimental}")
    for field in ("number_of_vertices", "number_of_edges", "actual_number_of_footsteps"):
        if _as_bool(reference.get("success")) or field in {"number_of_vertices", "number_of_edges"}:
            if _as_float(reference.get(field)) != _as_float(experimental.get(field)):
                raise AssertionError(f"default {field} mismatch: {reference} vs {experimental}")
    if _as_bool(reference.get("success")):
        for field in ("C_relax", "final_objective"):
            if not np.isclose(
                _as_float(reference[field]),
                _as_float(experimental[field]),
                rtol=1e-7,
                atol=1e-6,
            ):
                raise AssertionError(f"default {field} mismatch: {reference} vs {experimental}")
    return (
        "Default equivalence validation: PASS for terrain 0 using the same selected "
        "discrete path (existing graph/model/fixed-solver calls vs Com10_3 default).\n"
        "The existing timed recovery wrapper is not used for this comparison because "
        "it calls probabilistic recovery with seed=None.\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reviewer Comment 10.3 graph-construction sensitivity analysis"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true")
    action.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--config", choices=("all",) + CONFIGURATIONS, default="all")
    parser.add_argument("--terrain-start", type=int, default=0)
    parser.add_argument(
        "--terrain-end",
        type=int,
        default=100,
        help="exclusive end index; --terrain-start 0 --terrain-end 3 runs terrains 0,1,2",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--terrain-dir", type=Path, default=DEFAULT_TERRAIN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--terrain-key", default="terrain")

    # Existing CVAE OptAPI experiment settings.
    parser.add_argument("--step-span", type=float, default=0.7)
    parser.add_argument("--alpha-deg", type=float, default=30.0)
    parser.add_argument("--M-large", type=float, default=1e3)
    parser.add_argument("--M-small", type=float, default=1e-6)
    parser.add_argument("--stance-width", type=float, default=0.25)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-backtracks", type=int, default=20000)
    parser.add_argument("--solver-log", type=int, default=0)
    parser.add_argument(
        "--validate-default",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compare terrain 0 default against the existing evaluation call chain",
    )
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if not 0 <= args.terrain_start < args.terrain_end <= 100:
        parser.error("terrain range must satisfy 0 <= start < end <= 100")
    if args.trials < 1 or args.max_backtracks < 1:
        parser.error("--trials and --max-backtracks must be positive")
    if args.temperature <= 0 or args.step_span <= 0 or args.M_small < 0:
        parser.error("temperature/step-span must be positive and M-small nonnegative")
    return args


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    raw_path = output_dir / "raw_results.csv"
    summary_path = output_dir / "summary_by_configuration.csv"
    paired_path = output_dir / "paired_with_default.csv"

    if args.analyze_only:
        raw_rows = _read_csv(raw_path)
        if not raw_rows:
            raise FileNotFoundError(f"no raw data in {raw_path}")
        _write_analysis(output_dir, raw_rows)
        print(f"Analyzed {len(raw_rows)} raw rows")
        return 0

    cases = _discover_terrains(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(output_dir / "terrain_list.csv", TERRAIN_FIELDS, _terrain_rows(cases))
    rule_path = output_dir / "current_graph_rule.txt"
    rule_text = _graph_rule_text(cases, args)
    _atomic_write_text(rule_path, rule_text)

    if args.overwrite:
        raw_rows: list[dict[str, Any]] = []
    elif args.resume:
        raw_rows = _read_csv(raw_path)
    else:
        if any(path.exists() for path in (raw_path, summary_path, paired_path)):
            raise FileExistsError(
                f"results already exist in {output_dir}; use --resume or --overwrite"
            )
        raw_rows = []

    completed = {
        (int(_as_float(row["terrain_id"])), str(row["configuration"]))
        for row in raw_rows
        if np.isfinite(_as_float(row.get("terrain_id")))
    }
    # Validate the existing default on terrain 0 before spending time on any
    # alternative graph configuration.
    configurations = EXECUTION_ORDER if args.config == "all" else (args.config,)
    selected_cases = cases[int(args.terrain_start) : int(args.terrain_end)]
    total = len(configurations) * len(selected_cases)
    ordinal = 0
    default_validation_done = False

    for configuration in configurations:
        for case in selected_cases:
            ordinal += 1
            key = (case.terrain_id, configuration)
            if key in completed:
                print(f"[{ordinal}/{total}] SKIP terrain={case.terrain_id} config={configuration}")
                continue
            print(f"[{ordinal}/{total}] RUN terrain={case.terrain_id} config={configuration}")
            row = _run_trial(case, configuration, args)
            raw_rows.append(row)
            raw_rows.sort(key=lambda item: (int(_as_float(item["terrain_id"])), CONFIGURATIONS.index(str(item["configuration"]))))
            _atomic_write_csv(raw_path, RAW_FIELDS, raw_rows)
            _write_analysis(output_dir, raw_rows)

            if (
                bool(args.validate_default)
                and case.terrain_id == 0
                and configuration == "default"
            ):
                validation = _validate_default(case, row, args)
                rule_text += "\nValidation\n----------\n" + validation
                _atomic_write_text(rule_path, rule_text)
                default_validation_done = True

    if (
        bool(args.validate_default)
        and 0 in [case.terrain_id for case in selected_cases]
        and "default" in configurations
        and not default_validation_done
    ):
        existing_default = next(
            (
                row
                for row in raw_rows
                if int(_as_float(row["terrain_id"])) == 0
                and row["configuration"] == "default"
            ),
            None,
        )
        if existing_default is not None:
            validation = _validate_default(cases[0], existing_default, args)
            rule_text += "\nValidation\n----------\n" + validation
            _atomic_write_text(rule_path, rule_text)

    _atomic_write_csv(raw_path, RAW_FIELDS, raw_rows)
    _write_analysis(output_dir, raw_rows)
    print(f"Wrote {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
