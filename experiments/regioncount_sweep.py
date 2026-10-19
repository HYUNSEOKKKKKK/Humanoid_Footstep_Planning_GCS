"""CVAE region-count (raw stone count) scalability benchmark, extending Comment 1.2.

This is a separate, self-contained sibling to ``Com_1_2.py`` (same spirit as
``Com10_3_extended.py`` relative to ``Com10_3.py``): it reuses the shared
``src.graph`` / ``src.model`` / ``src.solver`` modules unmodified, and adds only
the graph-construction path Com_1_2.py's controlled partition sweep cannot
express -- namely genuinely different CVAE-sampled terrains with N distinct
physical stepping stones (N in {3,4,5,6}), built via the repository's original
``copy_counts``-based graph (``src.graph.build_graph``), instead of Com_1_2.py's
partition-preserving max-horizon graph.

Unlike Com_1_2.py's R-sweep (same physical geometry, only region *granularity*
changes), here each N is a genuinely different, independently CVAE-sampled
terrain (see terrain_generation/generate_regioncount_terrains.py,
which trains the CVAE exactly once and reuses that single checkpoint to sample
every N so the trained model is not a confound). Total horizon is therefore
expected to -- and is allowed to -- grow with N.

copy_counts rule (start=5, goal=5, every middle region=4):
    N=3: [5,4,5]        (sum=14)
    N=4: [5,4,4,5]      (sum=18, matches Com_1_5.py's DEFAULT_COPY_COUNTS_4_REGION)
    N=5: [5,4,4,4,5]    (sum=22)
    N=6: [5,4,4,4,4,5]  (sum=26)

An earlier version of this script used "every middle region=2" (matching
main_VAE_optapi.py's 6-region default). Direct testing showed that was too
tight: on a pre-existing, previously-well-behaved 4-stone terrain, copy_counts
[5,2,2,5] made the relaxed solve itself primal-infeasible (prosta.prim_infeas)
given step_span=0.7 -- there wasn't enough footstep slack to physically cross
some region-to-region gaps -- while [5,3,3,5]/[5,4,4,5]/[5,5,5,5] on the same
terrain all solved fine. The full 30-terrain x N in {3,4,5,6} run with
middle=2 confirmed this at scale: relaxed_success_rate collapsed from 60% (N=3)
to 20% (N=6), with every single failure at the relaxed_solve stage (not the
recovery loop) -- i.e. the 20s recovery budget was never the bottleneck.

The fixed-path recovery/trial loop budget (``--fixed-budget-sec``, default 20s)
is deliberately scoped to *only* the trial loop, timed from immediately before
it starts -- it does not share a deadline with graph construction, model
formulation, or the relaxed solve (those get their own generous MOSEK
``optimizer_max_time`` via ``--solver-time-limit-sec``). This mirrors
Com_1_2.py's separated-budget design; main_VAE_optapi.py's single shared
per-JSON deadline was the reason this had to be a fresh implementation rather
than a thin CLI wrapper.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/footstep_planner_matplotlib")

import mosek
import numpy as np

from src.graph import build_graph, compute_centroids_from_vertices
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
from src.utils import expand_shorthand


DEFAULT_TERRAIN_ROOT = Path(
    "data/terrains/region_count"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "Com_1_2_regioncount"
REGION_COUNTS = (3, 4, 5)
NUM_TERRAINS = 100
CVAE_BASE_SEED = 123
STEP_PENALTY_WEIGHT = 1.9  # gcs_model_VAE_optapi.py
EDGE_RULE = "general"
METHODOLOGY_VERSION = "Com_1_2_regioncount_copycounts_v2"
COPY_COUNTS_RULE = "start=5, goal=5, every middle region=4 (total grows with N)"


def copy_counts_for(num_regions: int) -> List[int]:
    if num_regions < 3:
        raise ValueError(f"num_regions must be >=3 (need >=1 middle region), got {num_regions}")
    return [5] + [4] * (num_regions - 2) + [5]


RAW_FIELDS = [
    # Identity
    "run_timestamp",
    "num_regions",
    "case_index",
    "condition_order",
    "terrain_index",
    "terrain_id",
    "terrain_sample_index",
    "terrain_seed",
    "terrain_file",
    "terrain_hash",
    # Controlled setting
    "copy_counts",
    "total_layers",
    "graph_edge_rule",
    "step_span",
    "alpha_deg",
    "stance_width",
    "M_large",
    "M_small",
    "recovery_trials_requested",
    "recovery_seed",
    "recovery_temperature",
    "greedy_first",
    "early_stop",
    "fixed_budget_sec",
    "mosek_threads",
    "solver_log",
    "solver_time_limit_sec",
    "step_penalty_weight",
    "solver_config_hash",
    "experiment_config_hash",
    # Geometry
    "start_x", "start_y", "start_z",
    "goal_x", "goal_y", "goal_z",
    "start_goal_distance_xy", "start_goal_distance_3d",
    "centerline_length_xy", "centerline_length_3d",
    "center_bbox_span_x", "center_bbox_span_y", "center_bbox_span_z",
    "terrain_envelope_span_x", "terrain_envelope_span_y", "terrain_envelope_span_z",
    "support_area_total",
    # Graph/model size
    "num_vertices",
    "num_edges",
    "num_scalar_vars",
    "num_constraints",
    "num_cones",
    "num_psd_vars",
    "num_psd_blocks",
    # Required outcomes
    "C_relax",
    "C_round",
    "delta_relax",
    "relaxed_success",
    "recovery_feasible",
    "num_recovery_trials",
    # Recovery/path diagnostics
    "feasible_trials",
    "failed_trials",
    "fixed_trial_objectives",
    "trial_strategies",
    "trial_path_horizons",
    "trial_errors",
    "best_rounding_strategy",
    "path_nodes",
    "path_region_sequence",
    "path_mode_sequence",
    "gap_abs",
    "gap_negative_beyond_tolerance",
    "fixed_budget_hit",
    # Phase timings (seconds)
    "graph_construction_time_sec",
    "model_formulation_time_sec",
    "relaxed_solve_time_sec",
    "path_recovery_fixed_solve_time_sec",
    "total_time_sec",
    # Extra timing diagnostics
    "relaxed_parameter_time_sec",
    "rounding_time_sec",
    "fixed_parameter_time_sec",
    "fixed_optimizer_time_sec",
    "fixed_extract_time_sec",
    "wall_time_sec",
    # Failure retention
    "status",
    "failure_stage",
    "error",
    "traceback_tail",
]

SUMMARY_METRICS = [
    "num_vertices",
    "num_edges",
    "graph_construction_time_sec",
    "model_formulation_time_sec",
    "relaxed_solve_time_sec",
    "path_recovery_fixed_solve_time_sec",
    "total_time_sec",
    "C_relax",
    "C_round",
    "delta_relax",
    "num_recovery_trials",
    "start_goal_distance_xy",
    "centerline_length_xy",
    "support_area_total",
]
SUMMARY_STATS = ("n", "mean", "std", "median", "p25", "p75", "p95", "min", "max")
SUMMARY_BASE_FIELDS = [
    "num_regions",
    "copy_counts",
    "total_layers",
    "n_expected",
    "n_rows",
    "n_unique_terrains",
    "n_relaxed_success",
    "relaxed_success_rate",
    "n_recovery_feasible",
    "recovery_feasibility_rate",
    "status_counts",
]
SUMMARY_FIELDS = SUMMARY_BASE_FIELDS + [
    f"{metric}_{stat}" for metric in SUMMARY_METRICS for stat in SUMMARY_STATS
]


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
    """Fixed-path failure that retains time already spent."""

    def __init__(self, message: str, timing: Dict[str, float]):
        super().__init__(message)
        self.timing = timing


def _seconds_since(t0: float) -> float:
    return float(time.perf_counter() - t0)


def _empty_row() -> Dict[str, Any]:
    return {name: "" for name in RAW_FIELDS}


def _safe_error(exc: BaseException, limit: int = 1200) -> str:
    return f"{type(exc).__name__}: {exc}".replace("\n", " | ")[:limit]


def _traceback_tail(limit: int = 2500) -> str:
    return traceback.format_exc().replace("\n", " | ")[-limit:]


def _canonical_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _terrain_sample_index(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch for ch in reversed(stem) if ch.isdigit())
    if not digits:
        raise ValueError(f"terrain file has no numeric sample suffix: {path.name}")
    return int(digits[::-1])


def _heading_deg(c0: Sequence[float], c1: Sequence[float]) -> float:
    dx = float(c1[0]) - float(c0[0])
    dy = float(c1[1]) - float(c0[1])
    return 0.0 if abs(dx) + abs(dy) < 1e-15 else math.degrees(math.atan2(dy, dx))


def _make_entry_from_stones(
    stones: Sequence[Dict[str, Any]], stance_width: float
) -> Dict[str, Any]:
    if len(stones) < 2:
        raise ValueError("terrain must contain at least start and goal regions")
    return {
        "start": {
            "center": stones[0]["center"],
            "heading_deg": _heading_deg(stones[0]["center"], stones[1]["center"]),
        },
        "goal": {
            "center": stones[-1]["center"],
            "heading_deg": _heading_deg(stones[-2]["center"], stones[-1]["center"]),
        },
        "stance_width": float(stance_width),
        "foot_yaw_deg": {
            "start_L": 0.0,
            "start_R": 0.0,
            "goal_L": 0.0,
            "goal_R": 0.0,
        },
    }


def load_terrain_context(
    path: Path, terrain_key: str, stance_width: float
) -> TerrainContext:
    terrain = load_terrain_yaw(name=terrain_key, path=path)
    stones = [copy.deepcopy(stone) for stone in terrain["stones"]]
    entry = _make_entry_from_stones(stones, stance_width)
    expanded = expand_shorthand(entry, stance_width)
    return TerrainContext(
        path=path,
        stones=stones,
        startpos_L=np.asarray(expanded["startpos_L"], dtype=float),
        startpos_R=np.asarray(expanded["startpos_R"], dtype=float),
        targetpos_L=np.asarray(expanded["targetpos_L"], dtype=float),
        targetpos_R=np.asarray(expanded["targetpos_R"], dtype=float),
        startyaw_L=float(expanded["startyaw_L"]),
        startyaw_R=float(expanded["startyaw_R"]),
        targetyaw_L=float(expanded["targetyaw_L"]),
        targetyaw_R=float(expanded["targetyaw_R"]),
    )


def stone_vertices(stone: Dict[str, Any]) -> np.ndarray:
    """Return a yawed rectangular region as a (3, 4) vertex matrix."""
    center = np.asarray(stone["center"], dtype=float).reshape(-1)
    if center.size == 2:
        center = np.r_[center, 0.0]
    if center.size != 3:
        raise ValueError(f"invalid stone center: {stone['center']}")
    half_width = 0.5 * float(stone["width"])
    half_height = 0.5 * float(stone["height"])
    local_xy = np.asarray(
        [
            [-half_width, -half_height],
            [-half_width, +half_height],
            [+half_width, +half_height],
            [+half_width, -half_height],
        ],
        dtype=float,
    )
    yaw = math.radians(float(stone.get("yaw_deg", 0.0)))
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=float,
    )
    world_xy = (rotation @ local_xy.T).T + center[:2]
    return np.vstack((world_xy.T, np.full((1, 4), center[2], dtype=float)))


def rotate_vertices_with_yaw(
    v_list: List[np.ndarray],
    terrain_stones: Sequence[Dict[str, Any]],
    copy_counts: Sequence[int],
) -> List[np.ndarray]:
    """Rotate each axis-aligned TerrainGCS vertex block by its stone's yaw_deg.

    ``TerrainGCS._rectangle_vertices_3x4`` (src/graph/terrain.py) ignores yaw
    entirely; this mirrors main_VAE_optapi.py's post-hoc rotation step so the
    CVAE-sampled yaw of every stone is actually reflected in the graph.
    """
    rotated: List[np.ndarray] = []
    stone_idx_expanded = [i for i, c in enumerate(copy_counts) for _ in range(int(c))]
    for vtx, stone_idx in zip(v_list, stone_idx_expanded):
        stone = terrain_stones[stone_idx]
        yaw_deg = float(stone.get("yaw_deg", 0.0))
        if abs(yaw_deg) < 1e-9:
            rotated.append(vtx)
            continue
        cx, cy, _cz = stone["center"]
        th = math.radians(yaw_deg)
        c, s = math.cos(th), math.sin(th)
        R = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        shifted = vtx.copy()
        shifted[0, :] -= cx
        shifted[1, :] -= cy
        rotated_block = R @ shifted
        rotated_block[0, :] += cx
        rotated_block[1, :] += cy
        rotated.append(rotated_block)
    return rotated


def vertex_region_ids_for(copy_counts: Sequence[int]) -> List[int]:
    return [i for i, c in enumerate(copy_counts) for _ in range(int(c))]


def terrain_metrics(stones: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    centers = np.asarray([stone["center"] for stone in stones], dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"invalid stone centers shape: {centers.shape}")
    differences = np.diff(centers, axis=0)
    endpoint = centers[-1] - centers[0]
    center_span = np.max(centers, axis=0) - np.min(centers, axis=0)
    all_vertices = np.concatenate(
        [stone_vertices(stone).T for stone in stones], axis=0
    )
    envelope_span = np.max(all_vertices, axis=0) - np.min(all_vertices, axis=0)
    return {
        "start_x": float(centers[0, 0]),
        "start_y": float(centers[0, 1]),
        "start_z": float(centers[0, 2]),
        "goal_x": float(centers[-1, 0]),
        "goal_y": float(centers[-1, 1]),
        "goal_z": float(centers[-1, 2]),
        "start_goal_distance_xy": float(np.linalg.norm(endpoint[:2])),
        "start_goal_distance_3d": float(np.linalg.norm(endpoint)),
        "centerline_length_xy": float(
            np.sum(np.linalg.norm(differences[:, :2], axis=1))
        ),
        "centerline_length_3d": float(
            np.sum(np.linalg.norm(differences, axis=1))
        ),
        "center_bbox_span_x": float(center_span[0]),
        "center_bbox_span_y": float(center_span[1]),
        "center_bbox_span_z": float(center_span[2]),
        "terrain_envelope_span_x": float(envelope_span[0]),
        "terrain_envelope_span_y": float(envelope_span[1]),
        "terrain_envelope_span_z": float(envelope_span[2]),
        "support_area_total": float(
            sum(float(stone["width"]) * float(stone["height"]) for stone in stones)
        ),
    }


def _reference_axes(
    startpos_L: np.ndarray, startpos_R: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    p_left = np.asarray(startpos_L, dtype=float).reshape(-1)[:2]
    p_right = np.asarray(startpos_R, dtype=float).reshape(-1)[:2]
    right_to_left = p_left - p_right
    norm = float(np.linalg.norm(right_to_left))
    y_ref = np.asarray([0.0, 1.0]) if norm < 1e-12 else right_to_left / norm
    x_ref = np.asarray([y_ref[1], -y_ref[0]])
    return x_ref, y_ref


def _is_acceptable_solsta(solsta: Any) -> bool:
    acceptable = solsta == mosek.solsta.optimal
    for name in ("integer_optimal", "near_optimal"):
        if hasattr(mosek.solsta, name):
            acceptable = acceptable or solsta == getattr(mosek.solsta, name)
    return bool(acceptable)


def solve_fixed_path_timed(
    task: mosek.Task,
    variables: Dict[str, Any],
    *,
    centroid_mat: np.ndarray,
    y_path: np.ndarray,
    M_small: float,
    start_L: np.ndarray,
    start_R: np.ndarray,
    target_L: np.ndarray,
    target_R: np.ndarray,
    log_level: int,
) -> Tuple[float, Dict[str, float]]:
    timing = {"parameter": 0.0, "optimizer": 0.0, "extract": 0.0}
    try:
        t0 = time.perf_counter()
        try:
            params = pack_fixedpath_params_for_model(
                centroid_mat=centroid_mat,
                y_relax=y_path,
                M_small=float(M_small),
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
                raise ValueError(
                    f"fixed y size mismatch: {y_fixed.size} != {len(y_indices)}"
                )
            task.putvarboundlist(
                y_indices,
                [mosek.boundkey.ra] * len(y_indices),
                np.maximum(0.0, y_fixed - float(M_small)).tolist(),
                np.minimum(1.0, y_fixed + float(M_small)).tolist(),
            )
        finally:
            timing["parameter"] = _seconds_since(t0)

        task.putintparam(mosek.iparam.log, int(log_level))
        t0 = time.perf_counter()
        try:
            task.optimize()
        finally:
            timing["optimizer"] = _seconds_since(t0)

        t0 = time.perf_counter()
        solsta = task.getsolsta(mosek.soltype.itr)
        prosta = task.getprosta(mosek.soltype.itr)
        if not _is_acceptable_solsta(solsta):
            raise RuntimeError(
                f"MOSEK fixed solve failed: prosta={prosta}, solsta={solsta}"
            )
        objective = float(task.getprimalobj(mosek.soltype.itr))
        if not np.isfinite(objective):
            raise RuntimeError(f"non-finite fixed objective: {objective}")
        timing["extract"] = _seconds_since(t0)
        return objective, timing
    except Exception as exc:
        raise TimedFixedSolveError(str(exc), timing) from exc


def _stable_trial_seed(base_seed: int, terrain_sample_index: int, num_regions: int, trial: int) -> int:
    sequence = np.random.SeedSequence(
        [int(base_seed), int(terrain_sample_index), int(num_regions), int(trial)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _select_path(
    y_relaxed: np.ndarray,
    edge: np.ndarray,
    vertex_number: int,
    *,
    trial: int,
    greedy_first: bool,
    temperature: float,
    seed: int,
) -> Tuple[List[int], str]:
    if bool(greedy_first) and int(trial) == 0:
        return (
            relaxation_max(y_relaxed, edge, int(vertex_number)),
            "relaxation_greedy",
        )
    return (
        relaxation_prob(
            y_relaxed, edge, int(vertex_number),
            temperature=float(temperature), seed=int(seed),
        ),
        "relaxation_probabilistic",
    )


def _join(values: Iterable[Any], fmt: str = "{}") -> str:
    return ";".join(fmt.format(value) for value in values)


def _sum_present(row: Dict[str, Any], names: Iterable[str]) -> float:
    total = 0.0
    for name in names:
        value = row.get(name, "")
        if value != "" and value is not None:
            number = float(value)
            if np.isfinite(number):
                total += number
    return total


def _solver_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "copy_counts_rule": COPY_COUNTS_RULE,
        "graph_edge_rule": EDGE_RULE,
        "step_span": float(args.step_span),
        "alpha_deg": float(args.alpha_deg),
        "stance_width": float(args.stance_width),
        "M_large": float(args.M_large),
        "M_small": float(args.M_small),
        "trials": int(args.trials),
        "temperature": float(args.temperature),
        "recovery_seed": int(args.recovery_seed),
        "greedy_first": bool(args.greedy_first),
        "early_stop": bool(args.early_stop),
        "fixed_budget_sec": float(args.fixed_budget_sec),
        "mosek_threads": int(args.mosek_threads),
        "solver_time_limit_sec": float(args.solver_time_limit_sec),
        "step_penalty_weight": STEP_PENALTY_WEIGHT,
    }


def run_case(
    context: TerrainContext,
    num_regions: int,
    args: argparse.Namespace,
    *,
    run_timestamp: str,
    case_index: int,
    condition_order: int,
    terrain_index: int,
    experiment_config_hash: str,
) -> Dict[str, Any]:
    row = _empty_row()
    sample_index = _terrain_sample_index(context.path)
    solver_hash = _canonical_hash(_solver_config(args))
    geometry = terrain_metrics(context.stones)
    terrain_hash = _canonical_hash(context.stones)
    copy_counts = copy_counts_for(num_regions)
    total_layers = int(sum(copy_counts))

    row.update(
        {
            "run_timestamp": run_timestamp,
            "num_regions": int(num_regions),
            "case_index": int(case_index),
            "condition_order": int(condition_order),
            "terrain_index": int(terrain_index),
            "terrain_id": context.path.stem,
            "terrain_sample_index": sample_index,
            "terrain_seed": int(args.cvae_base_seed) + sample_index,
            "terrain_file": str(context.path.resolve()),
            "terrain_hash": terrain_hash,
            "copy_counts": _join(copy_counts),
            "total_layers": total_layers,
            "graph_edge_rule": EDGE_RULE,
            "step_span": float(args.step_span),
            "alpha_deg": float(args.alpha_deg),
            "stance_width": float(args.stance_width),
            "M_large": float(args.M_large),
            "M_small": float(args.M_small),
            "recovery_trials_requested": int(args.trials),
            "recovery_seed": int(args.recovery_seed),
            "recovery_temperature": float(args.temperature),
            "greedy_first": int(bool(args.greedy_first)),
            "early_stop": int(bool(args.early_stop)),
            "fixed_budget_sec": float(args.fixed_budget_sec),
            "mosek_threads": int(args.mosek_threads),
            "solver_log": int(args.solver_log),
            "solver_time_limit_sec": float(args.solver_time_limit_sec),
            "step_penalty_weight": STEP_PENALTY_WEIGHT,
            "solver_config_hash": solver_hash,
            "experiment_config_hash": experiment_config_hash,
            "relaxed_success": 0,
            "recovery_feasible": 0,
            "num_recovery_trials": 0,
            "feasible_trials": 0,
            "failed_trials": 0,
            "fixed_budget_hit": 0,
            "status": "started",
        }
    )
    row.update(geometry)

    task: Optional[mosek.Task] = None
    current_stage = "graph_construction"
    wall_t0 = time.perf_counter()
    phase_t0 = wall_t0
    try:
        t0 = time.perf_counter()
        phase_t0 = t0
        G = build_graph(
            context.startpos_L, context.startpos_R,
            context.targetpos_L, context.targetpos_R,
            step_span=float(args.step_span),
            copy_counts=copy_counts,
            edge_rule=EDGE_RULE,
            terrain_stones=context.stones,
        )
        v = rotate_vertices_with_yaw(G["v"], context.stones, copy_counts)
        edge = G["edge"]
        mode_list = G["mode_list"]
        vertex_number = int(G["vertex_number"])
        n_edge = int(G["n_edge"])
        if vertex_number != total_layers:
            raise RuntimeError(
                f"vertex count invariant failed: {vertex_number} != {total_layers}"
            )
        row["graph_construction_time_sec"] = _seconds_since(t0)
        row["num_vertices"] = vertex_number
        row["num_edges"] = n_edge

        current_stage = "model_formulation"
        formulation_t0 = time.perf_counter()
        phase_t0 = formulation_t0
        pose_params = prepare_problem_params(
            context.startpos_L, context.startpos_R,
            context.targetpos_L, context.targetpos_R,
            context.startyaw_L, context.startyaw_R,
            context.targetyaw_L, context.targetyaw_R,
        )
        start_L = pose_params["start_L"]
        start_R = pose_params["start_R"]
        target_L = pose_params["target_L"]
        target_R = pose_params["target_R"]
        centroid_mat, _ = compute_centroids_from_vertices(v)
        x_ref, y_ref = _reference_axes(context.startpos_L, context.startpos_R)

        task, variables, idx, _par, reported_build_time = (
            build_gcs_model_VAE_optapi(
                v, edge, float(args.step_span),
                math.radians(float(args.alpha_deg)),
                mode_list, x_ref, y_ref,
            )
        )
        if int(args.mosek_threads) > 0:
            task.putintparam(mosek.iparam.num_threads, int(args.mosek_threads))
        if float(args.solver_time_limit_sec) > 0.0:
            task.putdouparam(
                mosek.dparam.optimizer_max_time, float(args.solver_time_limit_sec)
            )
        row["num_scalar_vars"] = int(task.getnumvar())
        row["num_constraints"] = int(task.getnumcon())
        row["num_cones"] = int(task.getnumcone())
        num_psd_blocks = int(task.getnumbarvar())
        row["num_psd_blocks"] = num_psd_blocks
        if num_psd_blocks != 4 * int(idx["n_edge"]):
            raise RuntimeError(
                f"unexpected PSD block count: {num_psd_blocks} "
                f"!= {4 * int(idx['n_edge'])}"
            )
        row["num_psd_vars"] = 86 * int(idx["n_edge"])

        parameter_t0 = time.perf_counter()
        relaxed_params = pack_relaxed_params_for_model(
            centroid_mat=centroid_mat,
            n_edge=int(idx["n_edge"]),
            M_large=float(args.M_large),
            start_L=start_L, start_R=start_R,
            target_L=target_L, target_R=target_R,
        )
        set_parameters_optapi(task, variables, relaxed_params)
        row["relaxed_parameter_time_sec"] = _seconds_since(parameter_t0)
        row["model_formulation_time_sec"] = _seconds_since(formulation_t0)

        current_stage = "relaxed_solve"
        relaxed_t0 = time.perf_counter()
        phase_t0 = relaxed_t0
        relaxed_solution, relaxed_objective = solve_relaxed_optapi(
            task, variables, log_level=int(args.solver_log),
        )
        row["relaxed_solve_time_sec"] = _seconds_since(relaxed_t0)
        solsta = task.getsolsta(mosek.soltype.itr)
        prosta = task.getprosta(mosek.soltype.itr)
        if not _is_acceptable_solsta(solsta):
            raise RuntimeError(
                f"MOSEK relaxed solve failed: prosta={prosta}, solsta={solsta}"
            )
        C_relax = float(relaxed_objective)
        y_relaxed = np.asarray(relaxed_solution["y"], dtype=float).reshape(-1)
        if not np.isfinite(C_relax):
            raise RuntimeError(f"non-finite relaxed objective: {C_relax}")
        if y_relaxed.size != n_edge or not np.all(np.isfinite(y_relaxed)):
            raise RuntimeError("relaxed y is missing, non-finite, or wrong-sized")
        row["C_relax"] = C_relax
        row["relaxed_success"] = 1

        # --- Fixed-path recovery loop: budget is scoped to ONLY this loop. ---
        current_stage = "path_recovery_fixed_solve"
        recovery_t0 = time.perf_counter()
        phase_t0 = recovery_t0
        fixed_loop_t0 = time.perf_counter()
        best_objective = math.inf
        best_path: Optional[List[int]] = None
        best_strategy = ""
        trial_objectives: List[float] = []
        trial_strategies: List[str] = []
        trial_path_horizons: List[int] = []
        trial_errors: List[str] = []
        rounding_time = 0.0
        fixed_parameter_time = 0.0
        fixed_optimizer_time = 0.0
        fixed_extract_time = 0.0

        for trial in range(int(args.trials)):
            if (
                float(args.fixed_budget_sec) > 0.0
                and _seconds_since(fixed_loop_t0) >= float(args.fixed_budget_sec)
            ):
                row["fixed_budget_hit"] = 1
                break
            row["num_recovery_trials"] = int(row["num_recovery_trials"]) + 1
            strategy = "selection_failed"
            try:
                selection_t0 = time.perf_counter()
                selected_path, strategy = _select_path(
                    y_relaxed, edge, vertex_number,
                    trial=trial,
                    greedy_first=bool(args.greedy_first),
                    temperature=float(args.temperature),
                    seed=_stable_trial_seed(
                        int(args.recovery_seed), sample_index, num_regions, trial
                    ),
                )
                rounding_time += _seconds_since(selection_t0)
                trial_strategies.append(strategy)
                trial_path_horizons.append(len(selected_path) - 2)
                y_path = path_to_y(edge, selected_path, n_edge)
                fixed_objective, fixed_timing = solve_fixed_path_timed(
                    task, variables,
                    centroid_mat=centroid_mat,
                    y_path=y_path,
                    M_small=float(args.M_small),
                    start_L=start_L, start_R=start_R,
                    target_L=target_L, target_R=target_R,
                    log_level=int(args.solver_log),
                )
                fixed_parameter_time += fixed_timing["parameter"]
                fixed_optimizer_time += fixed_timing["optimizer"]
                fixed_extract_time += fixed_timing["extract"]
                row["feasible_trials"] = int(row["feasible_trials"]) + 1
                trial_objectives.append(float(fixed_objective))
                if float(fixed_objective) < best_objective:
                    best_objective = float(fixed_objective)
                    best_path = list(selected_path)
                    best_strategy = strategy
                if bool(args.early_stop):
                    break
            except TimedFixedSolveError as exc:
                fixed_parameter_time += exc.timing["parameter"]
                fixed_optimizer_time += exc.timing["optimizer"]
                fixed_extract_time += exc.timing["extract"]
                row["failed_trials"] = int(row["failed_trials"]) + 1
                trial_errors.append(f"trial {trial} ({strategy}): {_safe_error(exc, 400)}")
            except Exception as exc:
                row["failed_trials"] = int(row["failed_trials"]) + 1
                trial_errors.append(f"trial {trial} ({strategy}): {_safe_error(exc, 400)}")

        row["rounding_time_sec"] = rounding_time
        row["fixed_parameter_time_sec"] = fixed_parameter_time
        row["fixed_optimizer_time_sec"] = fixed_optimizer_time
        row["fixed_extract_time_sec"] = fixed_extract_time
        row["path_recovery_fixed_solve_time_sec"] = _seconds_since(recovery_t0)
        row["fixed_trial_objectives"] = _join(trial_objectives, "{:.12g}")
        row["trial_strategies"] = _join(trial_strategies)
        row["trial_path_horizons"] = _join(trial_path_horizons)
        row["trial_errors"] = " || ".join(trial_errors)[:3500]
        row["best_rounding_strategy"] = best_strategy

        if best_path is None or not np.isfinite(best_objective):
            row["status"] = (
                "fixed_budget_no_feasible_solution"
                if int(row["fixed_budget_hit"])
                else "no_feasible_fixed_solution"
            )
            raise RuntimeError(str(row["status"]))

        C_round = float(best_objective)
        gap_abs = C_round - C_relax
        delta = gap_abs / max(abs(C_relax), 1e-12)
        region_ids = vertex_region_ids_for(copy_counts)
        path_regions = [region_ids[node - 1] for node in best_path[1:-1]]
        path_modes = [str(mode_list[node - 1]) for node in best_path[1:-1]]
        row.update(
            {
                "C_round": C_round,
                "delta_relax": delta,
                "recovery_feasible": 1,
                "path_nodes": _join(best_path),
                "path_region_sequence": _join(region + 1 for region in path_regions),
                "path_mode_sequence": _join(path_modes),
                "gap_abs": gap_abs,
                "gap_negative_beyond_tolerance": int(gap_abs < -1e-6),
                "status": "ok",
                "failure_stage": "",
                "error": "",
            }
        )
    except Exception as exc:
        timing_field = {
            "graph_construction": "graph_construction_time_sec",
            "model_formulation": "model_formulation_time_sec",
            "relaxed_solve": "relaxed_solve_time_sec",
            "path_recovery_fixed_solve": "path_recovery_fixed_solve_time_sec",
        }.get(current_stage)
        if timing_field is not None and row.get(timing_field, "") in ("", None):
            row[timing_field] = _seconds_since(phase_t0)
        if row.get("status") in ("", "started"):
            row["status"] = "failed"
        row["failure_stage"] = current_stage
        row["error"] = _safe_error(exc)
        row["traceback_tail"] = _traceback_tail()
    finally:
        phase_names = (
            "graph_construction_time_sec",
            "model_formulation_time_sec",
            "relaxed_solve_time_sec",
            "path_recovery_fixed_solve_time_sec",
        )
        row["total_time_sec"] = _sum_present(row, phase_names)
        row["wall_time_sec"] = _seconds_since(wall_t0)
        if task is not None:
            try:
                task.__exit__(None, None, None)
            except Exception:
                pass
    return row


def _numeric_values(rows: Sequence[Dict[str, Any]], field: str) -> np.ndarray:
    values: List[float] = []
    for row in rows:
        value = row.get(field, "")
        if value in ("", None):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            values.append(number)
    return np.asarray(values, dtype=float)


def _metric_stats(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {name: "" for name in SUMMARY_STATS}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size >= 2 else 0.0,
        "median": float(np.median(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_summary(
    rows: Sequence[Dict[str, Any]], expected_terrains: int
) -> List[Dict[str, Any]]:
    settings = sorted({int(row["num_regions"]) for row in rows})
    output: List[Dict[str, Any]] = []
    for num_regions in settings:
        group = [row for row in rows if int(row["num_regions"]) == num_regions]
        relaxed_count = sum(int(float(row.get("relaxed_success", 0) or 0)) for row in group)
        recovery_count = sum(int(float(row.get("recovery_feasible", 0) or 0)) for row in group)
        status_counts: Dict[str, int] = {}
        for row in group:
            status = str(row.get("status", ""))
            status_counts[status] = status_counts.get(status, 0) + 1
        terrain_ids = {str(row.get("terrain_id", "")) for row in group}
        copy_counts = copy_counts_for(num_regions)
        summary: Dict[str, Any] = {
            "num_regions": num_regions,
            "copy_counts": _join(copy_counts),
            "total_layers": int(sum(copy_counts)),
            "n_expected": int(expected_terrains),
            "n_rows": len(group),
            "n_unique_terrains": len(terrain_ids),
            "n_relaxed_success": relaxed_count,
            "relaxed_success_rate": relaxed_count / len(group) if group else "",
            "n_recovery_feasible": recovery_count,
            "recovery_feasibility_rate": recovery_count / len(group) if group else "",
            "status_counts": json.dumps(status_counts, sort_keys=True, separators=(",", ":")),
        }
        for metric in SUMMARY_METRICS:
            stats = _metric_stats(_numeric_values(group, metric))
            for stat, value in stats.items():
                summary[f"{metric}_{stat}"] = value
        output.append(summary)
    return output


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _raw_path(output_dir: Path) -> Path:
    return output_dir / "Com_1_2_regioncount_raw.csv"


def _summary_path(output_dir: Path) -> Path:
    return output_dir / "Com_1_2_regioncount_summary.csv"


def _terrain_files(terrain_root: Path, num_regions: int, num_terrains: int) -> List[Path]:
    directory = terrain_root / f"N{num_regions}"
    files = sorted(directory.glob("run_*.json"))
    if len(files) < num_terrains:
        raise FileNotFoundError(
            f"{directory}: found {len(files)} terrain files, need {num_terrains}"
        )
    return files[:num_terrains]


def _case_key(row: Dict[str, Any]) -> Tuple[int, str]:
    return (int(float(row["num_regions"])), str(row["terrain_id"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CVAE region-count (copy_counts) scalability benchmark, extending Com_1_2"
    )
    parser.add_argument("--terrain-root", type=Path, default=DEFAULT_TERRAIN_ROOT)
    parser.add_argument("--terrain-key", default="terrain")
    parser.add_argument("--cvae-base-seed", type=int, default=CVAE_BASE_SEED)
    parser.add_argument("--num-terrains", type=int, default=NUM_TERRAINS)
    parser.add_argument(
        "--region-counts", type=int, nargs="+", default=list(REGION_COUNTS)
    )

    parser.add_argument("--step-span", type=float, default=0.7)
    parser.add_argument("--alpha-deg", type=float, default=30.0)
    parser.add_argument("--stance-width", type=float, default=0.25)
    parser.add_argument("--M-large", type=float, default=1e3)
    parser.add_argument("--M-small", type=float, default=1e-6)
    parser.add_argument(
        "--trials", type=int, default=500,
        help=(
            "hard cap on recovery attempts; kept well above what "
            "--fixed-budget-sec can exhaust (~0.2s/trial observed) so wall-clock "
            "time, not this count, is the binding stop condition"
        ),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--recovery-seed", type=int, default=123)
    parser.add_argument(
        "--greedy-first", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--early-stop", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fixed-budget-sec", type=float, default=60.0,
        help="wall-clock budget (sec) for ONLY the fixed-path recovery/trial loop",
    )
    parser.add_argument("--mosek-threads", type=int, default=1)
    parser.add_argument("--solver-log", type=int, default=0)
    parser.add_argument(
        "--solver-time-limit-sec", type=float, default=300.0,
        help="per-MOSEK-optimize() cap applied to every relaxed/fixed solve; <=0 disables",
    )
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--shuffle-conditions", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.terrain_root.is_dir():
        raise NotADirectoryError(args.terrain_root)
    if args.smoke_test:
        args.num_terrains = 2
        args.region_counts = [3, 5]
        args.warmup = False
    if int(args.num_terrains) < 1:
        raise ValueError("--num-terrains must be >=1")
    for n in args.region_counts:
        if int(n) < 3:
            raise ValueError(f"--region-counts entries must be >=3, got {n}")
    if int(args.cvae_base_seed) != CVAE_BASE_SEED:
        raise ValueError(
            f"terrain files were generated with base seed {CVAE_BASE_SEED}; "
            "changing --cvae-base-seed would mislabel them"
        )
    if int(args.trials) < 1:
        raise ValueError("--trials must be >=1")
    if int(args.mosek_threads) < 0:
        raise ValueError("--mosek-threads must be >=0")
    if float(args.M_small) < 0.0:
        raise ValueError("--M-small must be nonnegative")
    if float(args.fixed_budget_sec) < 0.0:
        raise ValueError("--fixed-budget-sec must be nonnegative")
    if float(args.solver_time_limit_sec) < 0.0:
        raise ValueError("--solver-time-limit-sec must be nonnegative")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")


def main() -> None:
    args = parse_args()
    _validate_args(args)

    output_dir = Path(args.output_dir)
    raw_path = _raw_path(output_dir)
    if raw_path.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(
            f"{raw_path} exists; pass --overwrite or --resume"
        )

    run_timestamp = datetime.now().isoformat(timespec="seconds")
    experiment_config_hash = _canonical_hash(
        {
            "solver_config": _solver_config(args),
            "region_counts": list(map(int, args.region_counts)),
            "num_terrains": int(args.num_terrains),
            "copy_counts_rule": COPY_COUNTS_RULE,
        }
    )

    rows: List[Dict[str, Any]] = []
    completed_keys: set = set()
    if args.resume and raw_path.exists():
        rows = _read_csv(raw_path)
        existing_hashes = {str(r.get("experiment_config_hash", "")) for r in rows}
        if existing_hashes and existing_hashes != {experiment_config_hash}:
            raise RuntimeError("cannot resume: experiment_config_hash differs from prior run")
        completed_keys = {_case_key(r) for r in rows if str(r.get("status", "")) == "ok"}

    print(f"region_counts   : {list(map(int, args.region_counts))}")
    print(f"num_terrains    : {int(args.num_terrains)}")
    print(f"fixed_budget_sec: {float(args.fixed_budget_sec)}")
    print(f"copy_counts_rule: {COPY_COUNTS_RULE}")
    for n in args.region_counts:
        print(f"  N={n}: copy_counts={copy_counts_for(int(n))}")

    if args.warmup:
        warm_files = _terrain_files(args.terrain_root, int(args.region_counts[0]), 1)
        warm_context = load_terrain_context(warm_files[0], args.terrain_key, float(args.stance_width))
        warm_args = copy.copy(args)
        warm_args.trials = 1
        warm_args.early_stop = True
        print("[warmup] one unrecorded case")
        warm_row = run_case(
            warm_context, int(args.region_counts[0]), warm_args,
            run_timestamp=run_timestamp, case_index=-1, condition_order=-1,
            terrain_index=0, experiment_config_hash=experiment_config_hash,
        )
        print(f"[warmup] status={warm_row['status']} wall={float(warm_row['wall_time_sec']):.3f}s")

    case_index = len(rows)
    try:
        for num_regions in args.region_counts:
            num_regions = int(num_regions)
            terrain_files = _terrain_files(args.terrain_root, num_regions, int(args.num_terrains))
            ordered = list(enumerate(terrain_files))
            if args.shuffle_conditions:
                rng = np.random.default_rng(
                    np.random.SeedSequence([int(args.recovery_seed), num_regions, 4242])
                )
                rng.shuffle(ordered)
            for condition_order, (terrain_index, terrain_path) in enumerate(ordered):
                terrain_id = terrain_path.stem
                if (num_regions, terrain_id) in completed_keys:
                    continue
                context = load_terrain_context(terrain_path, args.terrain_key, float(args.stance_width))
                if len(context.stones) != num_regions:
                    raise ValueError(
                        f"{terrain_path.name}: expected {num_regions} stones, "
                        f"found {len(context.stones)}"
                    )
                row = run_case(
                    context, num_regions, args,
                    run_timestamp=run_timestamp,
                    case_index=case_index,
                    condition_order=condition_order,
                    terrain_index=terrain_index,
                    experiment_config_hash=experiment_config_hash,
                )
                rows.append(row)
                case_index += 1
                print(
                    f"[N={num_regions} {condition_order + 1}/{len(ordered)}] "
                    f"{terrain_id}: status={row['status']} "
                    f"C_relax={row.get('C_relax', '')} C_round={row.get('C_round', '')} "
                    f"wall={float(row['wall_time_sec']):.2f}s"
                )
                _write_csv(raw_path, rows, RAW_FIELDS)
                _write_csv(_summary_path(output_dir), build_summary(rows, int(args.num_terrains)), SUMMARY_FIELDS)
    finally:
        manifest = {
            "run_timestamp": run_timestamp,
            "experiment_config_hash": experiment_config_hash,
            "region_counts": list(map(int, args.region_counts)),
            "num_terrains": int(args.num_terrains),
            "n_rows_written": len(rows),
            "python": sys.version,
            "platform": platform.platform(),
            "terrain_root": str(args.terrain_root),
            "output_dir": str(output_dir),
        }
        with open(output_dir / "Com_1_2_regioncount_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    n_ok = sum(1 for r in rows if str(r.get("status")) == "ok")
    print(f"\nDone: {n_ok}/{len(rows)} cases ok. Output: {output_dir}")


if __name__ == "__main__":
    main()
