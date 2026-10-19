"""Reviewer Comment 1.2: CVAE horizon-scaling benchmark.

This file is intentionally self-contained.  It reuses the existing CVAE GCS
model and solver parameter helpers, but it does not modify any existing source
file.  For every terrain, it constructs a time-expanded graph with exactly H
single-foot placement layers and records one raw CSV row per (terrain, H).

Horizon convention
------------------
H is the exact number of intermediate single-foot placement states (SL/SR)
that every source-to-sink path must execute.  The initial and terminal
double-support states (D) are not included in H.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Some transitive project imports load Matplotlib even though this benchmark
# has no GUI.  Keep its cache out of the read-only home configuration.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/footstep_planner_matplotlib")

import mosek
import numpy as np

from src.graph import compute_centroids_from_vertices
from src.model import (
    build_gcs_model_VAE_optapi,
    pack_fixedpath_params_for_model,
    pack_relaxed_params_for_model,
    prepare_problem_params,
)
from src.solver import (
    path_to_y,
    relaxation_max,
    relaxation_prob,
    set_parameters_optapi,
    solve_relaxed_optapi,
)
from src.terrain_preset import load_terrain_yaw
from src.utils import expand_shorthand, postprocess_solution


DEFAULT_TERRAIN_DIR = "data/terrains/alpha/0.5"
DEFAULT_HORIZONS = [8, 10, 12, 14, 16]
STEP_PENALTY_WEIGHT = 1.9  # Must match gcs_model_VAE_optapi.py.

# Raw output only.  The user can aggregate/plot these columns later.
CSV_FIELDS = [
    "run_timestamp",
    "case_index",
    "terrain_index",
    "horizon_order_index",
    "terrain_id",
    "terrain_file",
    "terrain_sample_index",
    "difficulty",
    "num_regions",
    "horizon",
    "fixed_horizon",
    "horizon_definition",
    "edge_policy",
    "step_span",
    "alpha_deg",
    "stance_width",
    "M_large",
    "M_small",
    "trials_requested",
    "greedy_first",
    "structured_candidates",
    "early_stop",
    "fixed_budget_sec",
    "seed",
    "mosek_threads",
    "step_penalty_weight",
    "explicit_warm_start",
    "terminal_edges",
    "terminal_constraints_verified",
    "relaxed_success",
    "recovery_success",
    "final_success",
    "success",
    "status",
    "failure_stage",
    "error",
    "traceback_tail",
    "fixed_budget_hit",
    "attempted_trials",
    "num_recovery_attempts",
    "feasible_trials",
    "failed_trials",
    "fixed_trial_objectives",
    "trial_strategies",
    "best_rounding_strategy",
    "trial_errors",
    "num_vertices",
    "num_edges",
    "num_scalar_vars",
    "num_constraints",
    "num_cones",
    "num_psd_vars",
    "num_psd_blocks",
    "C_relax",
    "C_fixed",
    "C_round",
    "gap_abs",
    "gap_rel_fixed",
    "gap_rel_relaxed",
    "delta_relax",
    "gap_negative_beyond_tolerance",
    "path_horizon",
    "n_step",
    "horizon_slack",
    "path_num_vertices",
    "path_num_edges",
    "path_nodes",
    "path_region_sequence",
    "path_mode_sequence",
    "first_step_mode",
    "first_is_left",
    "postprocess_ok",
    "postprocess_error",
    "num_left_footsteps",
    "num_right_footsteps",
    "time_graph_ms",
    "time_setup_ms",
    "time_model_build_ms",
    "time_model_build_reported_ms",
    "time_relaxed_param_ms",
    "time_relaxed_solve_ms",
    "time_rounding_ms",
    "time_fixed_param_ms",
    "time_fixed_solve_ms",
    "time_fixed_extract_ms",
    "time_fixed_ms",
    "time_recovery_ms",
    "time_postprocess_ms",
    "time_solver_ms",
    "time_planning_ms",
    "time_total_ms",
    "time_wall_ms",
    "time_planning_per_horizon_stage_ms",
    "time_solver_per_horizon_stage_ms",
    "time_planning_per_actual_step_ms",
    "time_solver_per_actual_step_ms",
    "time_fixed_solve_per_attempt_ms",
]

HORIZON_DEFINITION = (
    "exact number of intermediate single-foot SL/SR stages; "
    "every path has n_step = H; excludes initial/final D"
)


@dataclass(frozen=True)
class TerrainContext:
    path: Path
    stones: List[Dict[str, Any]]
    startpos_L: np.ndarray
    startpos_R: np.ndarray
    targetpos_L: np.ndarray
    targetpos_R: np.ndarray
    startyaw_L: float
    startyaw_R: float
    targetyaw_L: float
    targetyaw_R: float


class TimedFixedSolveError(RuntimeError):
    """Fixed-path failure that preserves already spent stage times."""

    def __init__(self, message: str, timing: Dict[str, float]):
        super().__init__(message)
        self.timing = timing


def _elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _empty_row() -> Dict[str, Any]:
    return {name: "" for name in CSV_FIELDS}


def _safe_error(exc: BaseException, limit: int = 1000) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\n", " | ")
    return text[:limit]


def _traceback_tail(limit: int = 1800) -> str:
    text = traceback.format_exc().replace("\n", " | ")
    return text[-limit:]


def _terrain_sample_index(path: Path) -> Optional[int]:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def _heading_deg(c0: Sequence[float], c1: Sequence[float]) -> float:
    dx = float(c1[0]) - float(c0[0])
    dy = float(c1[1]) - float(c0[1])
    if abs(dx) + abs(dy) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def _make_entry_from_stones(stones: Sequence[Dict[str, Any]], stance_width: float) -> Dict[str, Any]:
    if not stones:
        raise ValueError("terrain has no regions")
    start_heading = _heading_deg(stones[0]["center"], stones[1]["center"]) if len(stones) >= 2 else 0.0
    goal_heading = _heading_deg(stones[-2]["center"], stones[-1]["center"]) if len(stones) >= 2 else 0.0
    return {
        "start": {"center": stones[0]["center"], "heading_deg": start_heading},
        "goal": {"center": stones[-1]["center"], "heading_deg": goal_heading},
        "stance_width": float(stance_width),
        # Preserve the convention used by main_VAE_optapi.py.
        "foot_yaw_deg": {
            "start_L": 0.0,
            "start_R": 0.0,
            "goal_L": 0.0,
            "goal_R": 0.0,
        },
    }


def load_terrain_context(path: Path, terrain_key: str, stance_width: float) -> TerrainContext:
    terrain = load_terrain_yaw(name=terrain_key, path=path)
    stones = terrain["stones"]
    entry = _make_entry_from_stones(stones, stance_width)
    exp = expand_shorthand(entry, stance_width)
    return TerrainContext(
        path=path,
        stones=stones,
        startpos_L=np.asarray(exp["startpos_L"], dtype=float),
        startpos_R=np.asarray(exp["startpos_R"], dtype=float),
        targetpos_L=np.asarray(exp["targetpos_L"], dtype=float),
        targetpos_R=np.asarray(exp["targetpos_R"], dtype=float),
        startyaw_L=float(exp["startyaw_L"]),
        startyaw_R=float(exp["startyaw_R"]),
        targetyaw_L=float(exp["targetyaw_L"]),
        targetyaw_R=float(exp["targetyaw_R"]),
    )


def stone_vertices(stone: Dict[str, Any]) -> np.ndarray:
    """Return one yaw-rotated rectangular region as a (3, 4) array."""
    center = np.asarray(stone["center"], dtype=float).reshape(-1)
    if center.size == 2:
        center = np.r_[center, 0.0]
    if center.size != 3:
        raise ValueError(f"invalid stone center: {stone['center']}")

    half_w = 0.5 * float(stone["width"])
    half_h = 0.5 * float(stone["height"])
    local_xy = np.array(
        [
            [-half_w, -half_h],
            [-half_w, +half_h],
            [+half_w, +half_h],
            [+half_w, -half_h],
        ],
        dtype=float,
    )
    yaw = math.radians(float(stone.get("yaw_deg", 0.0)))
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=float,
    )
    world_xy = (rotation @ local_xy.T).T + center[:2]
    return np.vstack([world_xy.T, np.full((1, 4), float(center[2]))])


def _region_successors(region: int, num_regions: int, edge_policy: str) -> Iterable[int]:
    if edge_policy == "all":
        return range(num_regions)
    if edge_policy == "forward":
        return range(region, num_regions)
    if edge_policy == "adjacent":
        return range(region, min(region + 2, num_regions))
    raise ValueError(f"unknown edge policy: {edge_policy}")


def build_time_expanded_graph(
    stones: Sequence[Dict[str, Any]],
    horizon: int,
    *,
    edge_policy: str = "all",
) -> Dict[str, Any]:
    """Build a layered graph with exact single-foot horizon H.

    Each of the H layers contains two copies (SL and SR) of every physical
    terrain region.  Edges only connect opposite foot modes in consecutive
    layers.  Only goal-region vertices in the final layer may enter terminal D,
    so every source-sink path contains exactly H single-foot stages.  The source
    and sink are the first and last physical regions in mode D.
    Node and edge labels follow the existing 1-based GCS convention.
    """
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    num_regions = len(stones)
    if num_regions < 2:
        raise ValueError("at least two terrain regions are required")

    base_vertices = [stone_vertices(stone) for stone in stones]
    vertices: List[np.ndarray] = [base_vertices[0].copy()]
    mode_list: List[str] = ["D"]
    vertex_region_ids: List[int] = [0]
    vertex_stages: List[int] = [-1]
    node_id: Dict[Tuple[int, int, str], int] = {}

    # Mode order is deliberate: the very first source edge is an SL edge,
    # preserving compatibility with existing diagnostics while path-based
    # first-foot detection below remains the authoritative method.
    for stage in range(horizon):
        for region in range(num_regions):
            for mode in ("SL", "SR"):
                vertices.append(base_vertices[region].copy())
                mode_list.append(mode)
                vertex_region_ids.append(region)
                vertex_stages.append(stage)
                node_id[(stage, region, mode)] = len(vertices)  # 1-based

    vertices.append(base_vertices[-1].copy())
    mode_list.append("D")
    vertex_region_ids.append(num_regions - 1)
    vertex_stages.append(horizon)
    sink_id = len(vertices)

    edges: List[Tuple[int, int]] = []

    # Initial D -> first SL/SR layer on the initial physical region only.
    # Likewise, only goal-region vertices in the final layer may enter terminal
    # D below.  This preserves the existing CVAE source/sink semantics while
    # enforcing an exact stage count.
    edges.append((1, node_id[(0, 0, "SL")]))
    edges.append((1, node_id[(0, 0, "SR")]))

    # Consecutive layers, always alternating the moving foot.
    for stage in range(horizon - 1):
        for region_from in range(num_regions):
            successors = tuple(_region_successors(region_from, num_regions, edge_policy))
            for mode_from, mode_to in (("SL", "SR"), ("SR", "SL")):
                source = node_id[(stage, region_from, mode_from)]
                for region_to in successors:
                    target = node_id[(stage + 1, region_to, mode_to)]
                    edges.append((source, target))

    # Only the final layer may enter terminal D, which enforces n_step = H.
    goal_region = num_regions - 1
    final_stage = horizon - 1
    edges.append((node_id[(final_stage, goal_region, "SL")], sink_id))
    edges.append((node_id[(final_stage, goal_region, "SR")], sink_id))

    edge_array = np.asarray(edges, dtype=int).reshape(-1, 2)
    return {
        "v": vertices,
        "edge": edge_array,
        "mode_list": mode_list,
        "vertex_region_ids": vertex_region_ids,
        "vertex_stages": vertex_stages,
        "node_id": node_id,
        "sink_id": sink_id,
        "vertex_number": len(vertices),
        "n_edge": int(edge_array.shape[0]),
    }


def audit_terminal_constraints(graph: Dict[str, Any], variables: Dict[str, Any]) -> int:
    """Verify that every terminal edge is registered as a full model sink.

    The existing model builder applies target pose, flow continuity, step-span,
    and D-mode constraints based on the common sink vertex/edge sets.  This
    audit guards against accidentally adding a graph-only shortcut that is not
    included in those model sink constraints.
    """
    edge = np.asarray(graph["edge"], dtype=int)
    sink_id = int(graph["sink_id"])
    goal_region = max(graph["vertex_region_ids"])
    final_stage = int(graph["vertex_stages"][sink_id - 1]) - 1
    expected = np.where(edge[:, 1] == sink_id)[0].astype(int).tolist()
    meta = variables["meta"]
    registered = [int(value) for value in np.asarray(meta["sink_edges"]).reshape(-1)]
    if expected != registered:
        raise RuntimeError(f"terminal edge registration mismatch: graph={expected}, model={registered}")

    rows_left = meta["B_rows_sink_zpL"]
    rows_right = meta["B_rows_sink_zpR"]
    if len(rows_left) != len(expected) or len(rows_right) != len(expected):
        raise RuntimeError("not every terminal edge has left/right target-pose rows")

    for local_index, edge_index in enumerate(expected):
        from_node = int(edge[edge_index, 0])
        from_zero = from_node - 1
        if graph["vertex_region_ids"][from_zero] != goal_region:
            raise RuntimeError(f"terminal edge {edge_index} does not originate in the goal region")
        if int(graph["vertex_stages"][from_zero]) != final_stage:
            raise RuntimeError(f"terminal edge {edge_index} does not originate in the final layer")
        if graph["mode_list"][from_zero] not in ("SL", "SR"):
            raise RuntimeError(f"terminal edge {edge_index} has invalid support mode")
        if len(rows_left[local_index]) != 5 or len(rows_right[local_index]) != 5:
            raise RuntimeError(f"terminal edge {edge_index} does not fix a full 5D pose for both feet")

    if graph["mode_list"][sink_id - 1] != "D":
        raise RuntimeError("terminal vertex is not a double-support D mode")
    return len(expected)


def _reference_axes(startpos_L: np.ndarray, startpos_R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p_left = np.asarray(startpos_L, dtype=float).reshape(-1)[:2]
    p_right = np.asarray(startpos_R, dtype=float).reshape(-1)[:2]
    right_to_left = p_left - p_right
    norm = float(np.linalg.norm(right_to_left))
    y_ref = np.array([0.0, 1.0]) if norm < 1e-12 else right_to_left / norm
    x_ref = np.array([y_ref[1], -y_ref[0]])
    return x_ref, y_ref


def _is_acceptable_solsta(solsta: Any) -> bool:
    acceptable = solsta == mosek.solsta.optimal
    if hasattr(mosek.solsta, "integer_optimal"):
        acceptable = acceptable or solsta == mosek.solsta.integer_optimal
    if hasattr(mosek.solsta, "near_optimal"):
        acceptable = acceptable or solsta == mosek.solsta.near_optimal
    return bool(acceptable)


def solve_fixed_path_timed(
    task: mosek.Task,
    variables: Dict[str, Any],
    idx: Dict[str, Any],
    *,
    centroid_mat: np.ndarray,
    y_path: np.ndarray,
    M_small: float,
    start_L: np.ndarray,
    start_R: np.ndarray,
    target_L: np.ndarray,
    target_R: np.ndarray,
    log_level: int,
) -> Tuple[float, Dict[str, np.ndarray], Dict[str, float]]:
    """Solve one fixed path without changing the existing solver module.

    Timings separate parameter/bound updates, MOSEK optimize, and extraction.
    Only arrays needed by the benchmark/postprocessor are extracted; PSD
    matrices are not read because the planner output does not use them.
    """
    timing = {"param_ms": 0.0, "solve_ms": 0.0, "extract_ms": 0.0}
    try:
        t0 = time.perf_counter()
        params = pack_fixedpath_params_for_model(
            centroid_mat=centroid_mat,
            y_relax=y_path,
            M_small=M_small,
            start_L=start_L,
            start_R=start_R,
            target_L=target_L,
            target_R=target_R,
        )
        set_parameters_optapi(task, variables, params)

        y_fixed = np.asarray(y_path, dtype=float).reshape(-1)
        y_slice = variables["y"]
        y_indices = list(range(y_slice.start, y_slice.stop))
        if len(y_indices) != y_fixed.size:
            raise ValueError(f"fixed y size mismatch: {y_fixed.size} != {len(y_indices)}")
        lower = np.maximum(0.0, y_fixed - float(M_small)).astype(float).tolist()
        upper = np.minimum(1.0, y_fixed + float(M_small)).astype(float).tolist()
        task.putvarboundlist(
            y_indices,
            [mosek.boundkey.ra] * len(y_indices),
            lower,
            upper,
        )
        timing["param_ms"] = _elapsed_ms(t0)

        task.putintparam(mosek.iparam.log, int(log_level))
        t0 = time.perf_counter()
        task.optimize()
        timing["solve_ms"] = _elapsed_ms(t0)

        solsta = task.getsolsta(mosek.soltype.itr)
        prosta = task.getprosta(mosek.soltype.itr)
        if not _is_acceptable_solsta(solsta):
            raise RuntimeError(f"MOSEK fixed solve failed: prosta={prosta}, solsta={solsta}")

        t0 = time.perf_counter()

        def read_slice(name: str) -> np.ndarray:
            slc = variables[name]
            values = [0.0] * (slc.stop - slc.start)
            task.getxxslice(mosek.soltype.itr, slc.start, slc.stop, values)
            return np.asarray(values, dtype=float)

        num_edges = int(idx["n_edge"])
        ex = {
            "y": read_slice("y").reshape(num_edges),
            "z_L": read_slice("z_L").reshape(num_edges, 5),
            "z_R": read_slice("z_R").reshape(num_edges, 5),
            "zp_L": read_slice("zp_L").reshape(num_edges, 5),
            "zp_R": read_slice("zp_R").reshape(num_edges, 5),
            "rho_L": read_slice("rho_L").reshape(num_edges),
            "rho_R": read_slice("rho_R").reshape(num_edges),
            "rhoz_L": read_slice("rhoz_L").reshape(num_edges),
            "rhoz_R": read_slice("rhoz_R").reshape(num_edges),
        }
        objective = float(task.getprimalobj(mosek.soltype.itr))
        timing["extract_ms"] = _elapsed_ms(t0)
        return objective, ex, timing
    except Exception as exc:
        raise TimedFixedSolveError(str(exc), timing) from exc


def _stable_trial_seed(base_seed: int, terrain_index: int, trial: int) -> int:
    # Deliberately omit H so the same terrain/trial uses the same random stream
    # at every fixed horizon.  This keeps the recovery condition paired.
    sequence = np.random.SeedSequence([int(base_seed), int(terrain_index), int(trial)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _select_path(
    y_relaxed: np.ndarray,
    graph: Dict[str, Any],
    *,
    trial: int,
    greedy_first: bool,
    structured_candidates: bool,
    temperature: float,
    seed: int,
) -> Tuple[List[int], str]:
    strategy_index = int(trial)
    if greedy_first:
        if strategy_index == 0:
            return (
                relaxation_max(y_relaxed, graph["edge"], int(graph["vertex_number"])),
                "relaxation_greedy",
            )
        strategy_index -= 1

    if structured_candidates and strategy_index in (0, 1, 2, 3):
        first_mode = "SL" if strategy_index % 2 == 0 else "SR"
        allocation = "front" if strategy_index < 2 else "back"
        fixed_horizon = max(graph["vertex_stages"])
        num_regions = max(graph["vertex_region_ids"]) + 1
        horizon = fixed_horizon
        # Existing CVAE source/sink semantics generally need one single-foot
        # state on the initial region and two on the goal region.  Allocate the
        # remaining stages monotonically across intermediate regions.  Two
        # deterministic front/back allocations and both starting feet are
        # included before stochastic trials.
        minimum = num_regions + 1
        if horizon >= minimum and num_regions >= 3:
            middle_count = num_regions - 2
            middle_budget = horizon - 3
            base, remainder = divmod(middle_budget, middle_count)
            middle_counts = [base] * middle_count
            remainder_order = range(middle_count) if allocation == "front" else range(middle_count - 1, -1, -1)
            for index in list(remainder_order)[:remainder]:
                middle_counts[index] += 1
            counts = [1, *middle_counts, 2]
            region_sequence = [region for region, count in enumerate(counts) for _ in range(count)]
        else:
            # Short-H fallback.  Such cases may legitimately be infeasible and
            # will be retained in the CSV as failures/success-rate evidence.
            region_sequence = [
                min(num_regions - 1, (stage * num_regions) // max(1, horizon))
                for stage in range(horizon)
            ]
            region_sequence[-1] = num_regions - 1
        path = [1]
        mode = first_mode
        for stage, region in enumerate(region_sequence):
            path.append(int(graph["node_id"][(stage, region, mode)]))
            mode = "SR" if mode == "SL" else "SL"
        path.append(int(graph["sink_id"]))
        return path, f"structured_exact_n{horizon}_{allocation}_{first_mode}"

    return (
        relaxation_prob(
        y_relaxed,
        graph["edge"],
        int(graph["vertex_number"]),
        temperature=float(temperature),
        seed=int(seed),
        ),
        "relaxation_probabilistic",
    )


def _join_numbers(values: Iterable[Any], fmt: str = "{}") -> str:
    return ";".join(fmt.format(value) for value in values)


def _sum_numeric(row: Dict[str, Any], names: Iterable[str]) -> float:
    total = 0.0
    for name in names:
        value = row.get(name, "")
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            total += float(value)
    return total


def _finish_timing_columns(row: Dict[str, Any], horizon: int) -> None:
    row["time_fixed_ms"] = _sum_numeric(
        row,
        ("time_fixed_param_ms", "time_fixed_solve_ms", "time_fixed_extract_ms"),
    )
    row["time_recovery_ms"] = float(row["time_rounding_ms"]) + float(row["time_fixed_ms"])
    row["time_solver_ms"] = _sum_numeric(
        row,
        ("time_relaxed_solve_ms", "time_fixed_solve_ms"),
    )
    planning_names = (
        "time_graph_ms",
        "time_setup_ms",
        "time_model_build_ms",
        "time_relaxed_param_ms",
        "time_relaxed_solve_ms",
        "time_rounding_ms",
        "time_fixed_param_ms",
        "time_fixed_solve_ms",
        "time_fixed_extract_ms",
    )
    row["time_planning_ms"] = _sum_numeric(row, planning_names)
    row["time_total_ms"] = float(row["time_planning_ms"]) + _sum_numeric(row, ("time_postprocess_ms",))
    row["time_planning_per_horizon_stage_ms"] = float(row["time_planning_ms"]) / max(1, int(horizon))
    row["time_solver_per_horizon_stage_ms"] = float(row["time_solver_ms"]) / max(1, int(horizon))
    actual_steps = int(row["n_step"]) if row.get("n_step", "") != "" else 0
    row["time_planning_per_actual_step_ms"] = (
        float(row["time_planning_ms"]) / actual_steps if actual_steps > 0 else ""
    )
    row["time_solver_per_actual_step_ms"] = (
        float(row["time_solver_ms"]) / actual_steps if actual_steps > 0 else ""
    )
    attempted = int(row["attempted_trials"]) if row.get("attempted_trials", "") != "" else 0
    row["time_fixed_solve_per_attempt_ms"] = (
        float(row["time_fixed_solve_ms"]) / attempted if attempted > 0 else ""
    )


def run_case(
    context: TerrainContext,
    horizon: int,
    args: argparse.Namespace,
    *,
    run_timestamp: str,
    case_index: int,
    terrain_index: int,
    horizon_order_index: int,
) -> Dict[str, Any]:
    row = _empty_row()
    row.update(
        {
            "run_timestamp": run_timestamp,
            "case_index": case_index,
            "terrain_index": terrain_index,
            "horizon_order_index": horizon_order_index,
            "terrain_id": context.path.stem,
            "terrain_file": str(context.path.resolve()),
            "terrain_sample_index": _terrain_sample_index(context.path),
            "difficulty": float(args.difficulty),
            "num_regions": len(context.stones),
            "horizon": int(horizon),
            "fixed_horizon": int(horizon),
            "horizon_definition": HORIZON_DEFINITION,
            "edge_policy": args.edge_policy,
            "step_span": float(args.step_span),
            "alpha_deg": float(args.alpha_deg),
            "stance_width": float(args.stance_width),
            "M_large": float(args.M_large),
            "M_small": float(args.M_small),
            "trials_requested": int(args.trials),
            "greedy_first": int(bool(args.greedy_first)),
            "structured_candidates": int(bool(args.structured_candidates)),
            "early_stop": int(bool(args.early_stop)),
            "fixed_budget_sec": float(args.fixed_budget_sec),
            "seed": int(args.seed),
            "mosek_threads": int(args.mosek_threads),
            "step_penalty_weight": STEP_PENALTY_WEIGHT,
            # No explicit primal/dual solution is supplied to MOSEK.  Reusing
            # the same task across recovery attempts is kept identical for all H.
            "explicit_warm_start": 0,
            "terminal_constraints_verified": 0,
            "relaxed_success": 0,
            "recovery_success": 0,
            "final_success": 0,
            "success": 0,
            "status": "started",
            "fixed_budget_hit": 0,
            "attempted_trials": 0,
            "num_recovery_attempts": 0,
            "feasible_trials": 0,
            "failed_trials": 0,
            "postprocess_ok": 0,
            "time_rounding_ms": 0.0,
            "time_fixed_param_ms": 0.0,
            "time_fixed_solve_ms": 0.0,
            "time_fixed_extract_ms": 0.0,
        }
    )

    task: Optional[mosek.Task] = None
    current_stage = "graph"
    wall_t0 = time.perf_counter()

    try:
        if args.expected_regions > 0 and len(context.stones) != int(args.expected_regions):
            raise ValueError(
                f"expected {args.expected_regions} regions, found {len(context.stones)} in {context.path.name}"
            )

        t0 = time.perf_counter()
        graph = build_time_expanded_graph(
            context.stones,
            int(horizon),
            edge_policy=args.edge_policy,
        )
        row["time_graph_ms"] = _elapsed_ms(t0)
        row["num_vertices"] = int(graph["vertex_number"])
        row["num_edges"] = int(graph["n_edge"])

        current_stage = "setup"
        t0 = time.perf_counter()
        pose_params = prepare_problem_params(
            context.startpos_L,
            context.startpos_R,
            context.targetpos_L,
            context.targetpos_R,
            context.startyaw_L,
            context.startyaw_R,
            context.targetyaw_L,
            context.targetyaw_R,
        )
        start_L = pose_params["start_L"]
        start_R = pose_params["start_R"]
        target_L = pose_params["target_L"]
        target_R = pose_params["target_R"]
        centroid_mat, _ = compute_centroids_from_vertices(graph["v"])
        x_ref, y_ref = _reference_axes(context.startpos_L, context.startpos_R)
        row["time_setup_ms"] = _elapsed_ms(t0)

        current_stage = "model_build"
        t0 = time.perf_counter()
        task, variables, idx, _par, reported_build_time = build_gcs_model_VAE_optapi(
            graph["v"],
            graph["edge"],
            float(args.step_span),
            math.radians(float(args.alpha_deg)),
            graph["mode_list"],
            x_ref,
            y_ref,
        )
        row["time_model_build_ms"] = _elapsed_ms(t0)
        row["time_model_build_reported_ms"] = float(reported_build_time) * 1000.0
        if int(args.mosek_threads) > 0:
            task.putintparam(mosek.iparam.num_threads, int(args.mosek_threads))

        row["num_scalar_vars"] = int(task.getnumvar())
        row["num_constraints"] = int(task.getnumcon())
        row["num_cones"] = int(task.getnumcone())
        row["num_psd_vars"] = int(task.getnumbarvar())
        row["num_psd_blocks"] = int(task.getnumbarvar())
        row["terminal_edges"] = audit_terminal_constraints(graph, variables)
        row["terminal_constraints_verified"] = 1

        current_stage = "relaxed_param"
        t0 = time.perf_counter()
        relaxed_params = pack_relaxed_params_for_model(
            centroid_mat=centroid_mat,
            n_edge=int(idx["n_edge"]),
            M_large=float(args.M_large),
            start_L=start_L,
            start_R=start_R,
            target_L=target_L,
            target_R=target_R,
        )
        set_parameters_optapi(task, variables, relaxed_params)
        row["time_relaxed_param_ms"] = _elapsed_ms(t0)

        current_stage = "relaxed_solve"
        t0 = time.perf_counter()
        relaxed_solution, relaxed_objective = solve_relaxed_optapi(
            task,
            variables,
            log_level=int(args.solver_log),
        )
        row["time_relaxed_solve_ms"] = _elapsed_ms(t0)
        relaxed_solsta = task.getsolsta(mosek.soltype.itr)
        relaxed_prosta = task.getprosta(mosek.soltype.itr)
        if not _is_acceptable_solsta(relaxed_solsta):
            raise RuntimeError(
                f"MOSEK relaxed solve failed: prosta={relaxed_prosta}, solsta={relaxed_solsta}"
            )
        C_relax = float(relaxed_objective)
        if not np.isfinite(C_relax):
            raise RuntimeError(f"non-finite relaxed objective: {C_relax}")
        y_relaxed = np.asarray(relaxed_solution["y"], dtype=float).reshape(-1)
        row["C_relax"] = C_relax
        row["relaxed_success"] = 1

        current_stage = "round_and_fixed"
        fixed_stage_t0 = time.perf_counter()
        best_objective = np.inf
        best_path: Optional[List[int]] = None
        best_ex: Optional[Dict[str, np.ndarray]] = None
        best_strategy = ""
        trial_objectives: List[float] = []
        trial_strategies: List[str] = []
        trial_errors: List[str] = []

        for trial in range(int(args.trials)):
            if float(args.fixed_budget_sec) > 0.0:
                fixed_elapsed = time.perf_counter() - fixed_stage_t0
                if fixed_elapsed >= float(args.fixed_budget_sec):
                    row["fixed_budget_hit"] = 1
                    break

            row["attempted_trials"] = int(row["attempted_trials"]) + 1
            row["num_recovery_attempts"] = int(row["attempted_trials"])
            selected_path: Optional[List[int]] = None
            strategy = "selection_failed"
            try:
                t0 = time.perf_counter()
                selected_path, strategy = _select_path(
                    y_relaxed,
                    graph,
                    trial=trial,
                    greedy_first=bool(args.greedy_first),
                    structured_candidates=bool(args.structured_candidates),
                    temperature=float(args.temp),
                    seed=_stable_trial_seed(args.seed, terrain_index, trial),
                )
                row["time_rounding_ms"] = float(row["time_rounding_ms"]) + _elapsed_ms(t0)
                trial_strategies.append(strategy)

                # The final-layer-only terminal edges enforce n_step = H.
                selected_horizon = len(selected_path) - 2
                if selected_horizon != int(horizon):
                    raise RuntimeError(
                        f"exact-horizon invariant failed: n_step={selected_horizon}, H={horizon}"
                    )

                y_path = path_to_y(graph["edge"], selected_path, int(graph["n_edge"]))
                fixed_objective, fixed_ex, fixed_timing = solve_fixed_path_timed(
                    task,
                    variables,
                    idx,
                    centroid_mat=centroid_mat,
                    y_path=y_path,
                    M_small=float(args.M_small),
                    start_L=start_L,
                    start_R=start_R,
                    target_L=target_L,
                    target_R=target_R,
                    log_level=int(args.solver_log),
                )
                row["time_fixed_param_ms"] = float(row["time_fixed_param_ms"]) + fixed_timing["param_ms"]
                row["time_fixed_solve_ms"] = float(row["time_fixed_solve_ms"]) + fixed_timing["solve_ms"]
                row["time_fixed_extract_ms"] = float(row["time_fixed_extract_ms"]) + fixed_timing["extract_ms"]

                row["feasible_trials"] = int(row["feasible_trials"]) + 1
                trial_objectives.append(float(fixed_objective))
                if float(fixed_objective) < float(best_objective):
                    best_objective = float(fixed_objective)
                    best_path = list(selected_path)
                    best_ex = fixed_ex
                    best_strategy = strategy
                if bool(args.early_stop):
                    break

            except TimedFixedSolveError as exc:
                row["time_fixed_param_ms"] = float(row["time_fixed_param_ms"]) + exc.timing["param_ms"]
                row["time_fixed_solve_ms"] = float(row["time_fixed_solve_ms"]) + exc.timing["solve_ms"]
                row["time_fixed_extract_ms"] = float(row["time_fixed_extract_ms"]) + exc.timing["extract_ms"]
                row["failed_trials"] = int(row["failed_trials"]) + 1
                path_hint = ""
                if selected_path is not None:
                    regions = [graph["vertex_region_ids"][node - 1] + 1 for node in selected_path[1:-1]]
                    path_hint = f" regions={_join_numbers(regions)}"
                trial_errors.append(f"trial {trial} ({strategy}): {_safe_error(exc, 400)}{path_hint}")
            except Exception as exc:
                row["failed_trials"] = int(row["failed_trials"]) + 1
                trial_errors.append(f"trial {trial} ({strategy}): {_safe_error(exc, 400)}")

        row["fixed_trial_objectives"] = _join_numbers(trial_objectives, "{:.12g}")
        row["trial_strategies"] = _join_numbers(trial_strategies)
        row["best_rounding_strategy"] = best_strategy
        row["trial_errors"] = " || ".join(trial_errors)[:3000]

        if best_path is None or best_ex is None or not np.isfinite(best_objective):
            if int(row["fixed_budget_hit"]):
                row["status"] = "fixed_budget_no_feasible_solution"
            else:
                row["status"] = "no_feasible_fixed_solution"
            raise RuntimeError(row["status"])

        C_fixed = float(best_objective)
        row["recovery_success"] = 1
        gap_abs = C_fixed - C_relax
        eps = 1e-12
        gap_rel_fixed = gap_abs / max(abs(C_fixed), eps)
        gap_rel_relaxed = gap_abs / max(abs(C_relax), eps)
        row.update(
            {
                "C_fixed": C_fixed,
                "C_round": C_fixed,
                "gap_abs": gap_abs,
                "gap_rel_fixed": gap_rel_fixed,
                "gap_rel_relaxed": gap_rel_relaxed,
                # Paper definition:
                # delta_relax = (C_fixed - C_relax) / abs(C_relax).
                # The fixed-cost denominator variant remains available in
                # gap_rel_fixed for reference.
                "delta_relax": gap_rel_relaxed,
                "gap_negative_beyond_tolerance": int(gap_abs < -1e-6),
            }
        )

        path_regions = [graph["vertex_region_ids"][node - 1] for node in best_path[1:-1]]
        path_modes = [graph["mode_list"][node - 1] for node in best_path[1:-1]]
        first_mode = path_modes[0]
        first_is_left = first_mode == "SL"
        row.update(
            {
                "path_horizon": len(best_path) - 2,
                "n_step": len(best_path) - 2,
                # Retained for CSV compatibility; exact horizon makes it zero.
                "horizon_slack": 0,
                "path_num_vertices": len(best_path),
                "path_num_edges": len(best_path) - 1,
                "path_nodes": _join_numbers(best_path),
                # Human-readable region IDs are 1-based in the CSV.
                "path_region_sequence": _join_numbers(region + 1 for region in path_regions),
                "path_mode_sequence": _join_numbers(path_modes),
                "first_step_mode": first_mode,
                "first_is_left": int(first_is_left),
            }
        )

        # Existing postprocess_solution assumes y[0] indicates the first foot.
        # A layered graph has several source edges, so use a private array copy
        # to provide the correct first-foot hint without changing that function.
        current_stage = "postprocess"
        t0 = time.perf_counter()
        try:
            ex_for_post = {
                key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
                for key, value in best_ex.items()
            }
            if np.asarray(ex_for_post["y"]).size:
                ex_for_post["y"][0] = 1.0 if first_is_left else 0.0
            processed = postprocess_solution(
                ex_for_post,
                graph["edge"],
                best_path,
                context.startpos_L,
                context.startpos_R,
                context.targetpos_L,
                context.targetpos_R,
                target_L,
                target_R,
            )
            row["postprocess_ok"] = 1
            row["num_left_footsteps"] = len(processed.L.x)
            row["num_right_footsteps"] = len(processed.R.x)
        except Exception as exc:
            # The relaxed/fixed costs remain available, but complete planning
            # is reported separately as failed to avoid hiding selection bias.
            row["postprocess_ok"] = 0
            row["postprocess_error"] = _safe_error(exc)
        row["time_postprocess_ms"] = _elapsed_ms(t0)

        if int(row["postprocess_ok"]):
            row["final_success"] = 1
            row["success"] = 1
            row["status"] = "ok"
            row["failure_stage"] = ""
            row["error"] = ""
        else:
            row["final_success"] = 0
            row["success"] = 0
            row["status"] = "postprocess_failed"
            row["failure_stage"] = "postprocess"
            row["error"] = row["postprocess_error"]

    except Exception as exc:
        if row.get("status") in ("", "started"):
            row["status"] = "failed"
        row["failure_stage"] = current_stage
        row["error"] = _safe_error(exc)
        row["traceback_tail"] = _traceback_tail()
        row["success"] = 0
    finally:
        row["time_wall_ms"] = _elapsed_ms(wall_t0)
        _finish_timing_columns(row, int(horizon))
        if task is not None:
            try:
                task.__exit__(None, None, None)
            except Exception:
                pass

    return row


def _failure_rows_for_terrain(
    path: Path,
    horizons: Sequence[int],
    args: argparse.Namespace,
    *,
    run_timestamp: str,
    case_index_start: int,
    terrain_index: int,
    exc: BaseException,
) -> List[Dict[str, Any]]:
    rows = []
    for order_index, horizon in enumerate(horizons):
        row = _empty_row()
        row.update(
            {
                "run_timestamp": run_timestamp,
                "case_index": case_index_start + order_index,
                "terrain_index": terrain_index,
                "horizon_order_index": order_index,
                "terrain_id": path.stem,
                "terrain_file": str(path.resolve()),
                "terrain_sample_index": _terrain_sample_index(path),
                "difficulty": float(args.difficulty),
                "horizon": int(horizon),
                "fixed_horizon": int(horizon),
                "horizon_definition": HORIZON_DEFINITION,
                "edge_policy": args.edge_policy,
                "step_span": float(args.step_span),
                "alpha_deg": float(args.alpha_deg),
                "stance_width": float(args.stance_width),
                "M_large": float(args.M_large),
                "M_small": float(args.M_small),
                "trials_requested": int(args.trials),
                "greedy_first": int(bool(args.greedy_first)),
                "structured_candidates": int(bool(args.structured_candidates)),
                "early_stop": int(bool(args.early_stop)),
                "fixed_budget_sec": float(args.fixed_budget_sec),
                "seed": int(args.seed),
                "mosek_threads": int(args.mosek_threads),
                "step_penalty_weight": STEP_PENALTY_WEIGHT,
                "explicit_warm_start": 0,
                "relaxed_success": 0,
                "recovery_success": 0,
                "final_success": 0,
                "success": 0,
                "num_recovery_attempts": 0,
                "status": "terrain_load_failed",
                "failure_stage": "terrain_load",
                "error": _safe_error(exc),
                "traceback_tail": _traceback_tail(),
            }
        )
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CVAE GCS benchmark for Reviewer Comment 1.2 (fixed footstep-horizon sweep)"
    )
    parser.add_argument("--terrain-dir", type=Path, default=Path(DEFAULT_TERRAIN_DIR))
    parser.add_argument("--terrain-key", default="terrain")
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument(
        "--expected-regions",
        type=int,
        default=4,
        help="fail a case when region count differs; <=0 disables the check",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
        help="exact number H of required single-foot stages, excluding initial/final D",
    )
    parser.add_argument("--max-terrains", type=int, default=10, help="0 means all JSON files")
    parser.add_argument(
        "--edge-policy",
        choices=("all", "forward", "adjacent"),
        default="adjacent",
        help="inter-layer connectivity; 'adjacent' matches the ordered CVAE corridor topology",
    )

    parser.add_argument("--step-span", type=float, default=0.7)
    parser.add_argument("--alpha-deg", type=float, default=30.0)
    parser.add_argument("--stance-width", type=float, default=0.25)
    parser.add_argument("--M-large", type=float, default=1e3)
    parser.add_argument("--M-small", type=float, default=1e-6)

    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="maximum number of seeded path-recovery trials per case",
    )
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--greedy-first",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--structured-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include deterministic balanced monotone SL/SR paths before probabilistic trials",
    )
    parser.add_argument(
        "--early-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "stop at the first feasible fixed-path solution (default); "
            "use --no-early-stop to run all trials and select the lowest-cost solution"
        ),
    )
    parser.add_argument(
        "--fixed-budget-sec",
        type=float,
        default=0.0,
        help="optional budget for rounding+fixed trials only; <=0 disables it",
    )
    parser.add_argument("--mosek-threads", type=int, default=1)
    parser.add_argument("--solver-log", type=int, default=0)

    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run one unrecorded one-trial case before timing",
    )
    parser.add_argument(
        "--shuffle-horizons",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="shuffle H order deterministically per terrain to reduce order/thermal bias",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.terrain_dir.is_dir():
        raise NotADirectoryError(args.terrain_dir)
    if not args.horizons:
        raise ValueError("at least one horizon is required")
    if any(int(horizon) < 1 for horizon in args.horizons):
        raise ValueError(f"all horizons must be >= 1: {args.horizons}")
    if len(set(map(int, args.horizons))) != len(args.horizons):
        raise ValueError(f"duplicate horizons are not allowed: {args.horizons}")
    if int(args.trials) < 1:
        raise ValueError("--trials must be >= 1")
    if float(args.M_small) < 0.0:
        raise ValueError("--M-small must be nonnegative")
    if int(args.mosek_threads) < 0:
        raise ValueError("--mosek-threads must be >= 0")


def _default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parents[1] / "outputs" / f"vae_horizon_Com1_2_{stamp}.csv"


def _write_manifest(path: Path, args: argparse.Namespace, terrain_files: Sequence[Path]) -> None:
    manifest_path = path.with_suffix(".json")
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "script": str(Path(__file__).resolve()),
        "command": sys.argv,
        "horizon_type": "fixed_exact",
        "horizon_definition": HORIZON_DEFINITION,
        "delta_relax_definition": "(C_fixed - C_relax) / abs(C_relax)",
        "gap_rel_relaxed_definition": "(C_fixed - C_relax) / abs(C_relax)",
        "success_definitions": {
            "relaxed_success": "MOSEK relaxed solve returned an accepted finite solution",
            "recovery_success": "at least one fixed-path recovery solve was feasible",
            "final_success": "recovery succeeded and planner postprocessing completed",
        },
        "step_penalty_weight": STEP_PENALTY_WEIGHT,
        "explicit_warm_start": False,
        "solver_tolerance": "MOSEK defaults; M_small is recorded per CSV row",
        "solver_time_limit": "none by default; fixed_budget_sec only limits the recovery loop",
        "timing_unit": "milliseconds",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "terrain_files": [str(path.resolve()) for path in terrain_files],
    }
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    _validate_args(args)

    terrain_files = sorted(args.terrain_dir.glob("*.json"))
    if int(args.max_terrains) > 0:
        terrain_files = terrain_files[: int(args.max_terrains)]
    if not terrain_files:
        raise FileNotFoundError(f"no JSON terrain files found in {args.terrain_dir}")

    output_path = args.output if args.output is not None else _default_output_path()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {output_path}")

    run_timestamp = datetime.now().astimezone().isoformat()
    _write_manifest(output_path, args, terrain_files)

    print("=== CVAE horizon benchmark: Reviewer Comment 1.2 ===")
    print(f"terrains       : {len(terrain_files)}")
    print(f"difficulty     : {args.difficulty}")
    print(f"expected R     : {args.expected_regions}")
    print(f"horizons       : {list(map(int, args.horizons))}")
    print(f"edge policy    : {args.edge_policy}")
    print(
        f"trials/case    : {args.trials} "
        f"(structured={args.structured_candidates}, early_stop={args.early_stop})"
    )
    print(f"MOSEK threads  : {args.mosek_threads}")
    print(f"output         : {output_path}")

    # Warm up imports, model construction, and MOSEK once; discard the row.
    if args.warmup:
        print("[warmup] starting one unrecorded case...")
        try:
            warm_context = load_terrain_context(
                terrain_files[0], args.terrain_key, float(args.stance_width)
            )
            warm_args = copy.copy(args)
            warm_args.trials = 1
            warm_args.early_stop = True
            warm_args.fixed_budget_sec = 0.0
            warm_row = run_case(
                warm_context,
                int(args.horizons[0]),
                warm_args,
                run_timestamp=run_timestamp,
                case_index=-1,
                terrain_index=0,
                horizon_order_index=-1,
            )
            print(
                f"[warmup] status={warm_row['status']} "
                f"wall={float(warm_row['time_wall_ms']):.1f} ms"
            )
        except Exception as exc:
            print(f"[warmup] warning: {_safe_error(exc)}")

    total_cases = len(terrain_files) * len(args.horizons)
    completed = 0
    successful = 0
    case_index = 0

    with open(output_path, "w", newline="", encoding="utf-8", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        stream.flush()

        for terrain_index, terrain_path in enumerate(terrain_files):
            horizons = list(map(int, args.horizons))
            if args.shuffle_horizons:
                rng = np.random.default_rng(
                    np.random.SeedSequence([int(args.seed), int(terrain_index), 9127])
                )
                rng.shuffle(horizons)

            try:
                context = load_terrain_context(
                    terrain_path,
                    args.terrain_key,
                    float(args.stance_width),
                )
            except Exception as exc:
                failure_rows = _failure_rows_for_terrain(
                    terrain_path,
                    horizons,
                    args,
                    run_timestamp=run_timestamp,
                    case_index_start=case_index,
                    terrain_index=terrain_index,
                    exc=exc,
                )
                for row in failure_rows:
                    writer.writerow(row)
                    stream.flush()
                    completed += 1
                    case_index += 1
                print(f"[{completed}/{total_cases}] {terrain_path.name}: terrain load failed")
                continue

            for order_index, horizon in enumerate(horizons):
                row = run_case(
                    context,
                    horizon,
                    args,
                    run_timestamp=run_timestamp,
                    case_index=case_index,
                    terrain_index=terrain_index,
                    horizon_order_index=order_index,
                )
                writer.writerow(row)
                stream.flush()

                completed += 1
                case_index += 1
                successful += int(row["success"])
                if row["success"]:
                    print(
                        f"[{completed}/{total_cases}] {terrain_path.name} H={horizon} OK | "
                        f"Crelax={float(row['C_relax']):.4f} "
                        f"Cfixed={float(row['C_fixed']):.4f} "
                        f"delta={float(row['delta_relax']):.4%} "
                        f"nstep={int(row['n_step'])}/{horizon} "
                        f"planning={float(row['time_planning_ms']):.1f} ms"
                    )
                else:
                    print(
                        f"[{completed}/{total_cases}] {terrain_path.name} H={horizon} FAIL | "
                        f"stage={row['failure_stage']} error={row['error']}"
                    )

    print("=== completed ===")
    print(f"successful cases: {successful}/{total_cases}")
    print(f"raw CSV         : {output_path}")
    print(f"run manifest    : {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
