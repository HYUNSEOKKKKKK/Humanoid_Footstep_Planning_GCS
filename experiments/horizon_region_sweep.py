"""CVAE scalability benchmark for Reviewer Comment 1.2.

This is a new, self-contained benchmark driver.  It intentionally does not
modify or import the previous ``main_VAE_optapi_Com1_2.py`` experiment.

Two paired sweeps are produced:

* maximum horizon H in {6, 8, 10, 12, 14, 16}, with the original four CVAE
  regions fixed;
* number of regions R in {4, 8, 12, 16, 20}, with maximum horizon fixed at 12.

H is a *maximum*: every goal-region vertex at every layer has a terminal edge,
so a path may terminate as soon as the goal is reached.  For the region sweep,
the two original intermediate rectangles are partitioned in their local frames
into disjoint strips.  Their union, area, outer envelope, and the start/goal
rectangles therefore remain exactly the same.  The graph uses original
physical-region groups, so increasing R does not force a longer logical route.

The normal/default run uses 20 paired CVAE terrain seeds.  ``--smoke-test`` is
the only supported shortcut below 20 and writes to a caller-selected directory.
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
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Transitive project imports load Matplotlib even though this is headless.
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
from src.utils import expand_shorthand


DEFAULT_TERRAIN_DIR = Path("data/terrains/alpha/0.5")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "Com_1_2"
DEFAULT_HORIZONS = (6, 8, 10, 12, 14, 16)
DEFAULT_REGIONS = (4, 8, 12, 16, 20)
MIN_TERRAINS = 20
CVAE_BASE_SEED = 123
STEP_PENALTY_WEIGHT = 1.9  # gcs_model_VAE_optapi.py
GRAPH_RULE = "parent_same_or_next_group_complete"
METHODOLOGY_VERSION = "Com_1_2_partition_union_v2"
HORIZON_DEFINITION = (
    "maximum number of intermediate single-foot SL/SR stages; "
    "goal may terminate at any stage; initial/final D excluded"
)
DELTA_DEFINITION = "(C_round - C_relax) / abs(C_relax)"

RAW_FIELDS = [
    # Identity and pairing
    "run_timestamp",
    "sweep",
    "case_index",
    "condition_order",
    "setting_value",
    "terrain_index",
    "terrain_id",
    "terrain_sample_index",
    "terrain_seed",
    "terrain_file",
    "terrain_family_hash",
    "terrain_variant_hash",
    "difficulty",
    # Controlled setting
    "max_horizon",
    "actual_horizon",
    "horizon_slack",
    "num_regions",
    "base_num_regions",
    "region_derivation",
    "graph_rule",
    "horizon_definition",
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
    # Geometry controls
    "start_x",
    "start_y",
    "start_z",
    "goal_x",
    "goal_y",
    "goal_z",
    "start_goal_distance_xy",
    "start_goal_distance_3d",
    "centerline_length_xy",
    "centerline_length_3d",
    "center_bbox_span_x",
    "center_bbox_span_y",
    "center_bbox_span_z",
    "terrain_envelope_span_x",
    "terrain_envelope_span_y",
    "terrain_envelope_span_z",
    "support_area_total",
    "geometry_control_error",
    # Graph/model size
    "num_vertices",
    "num_edges",
    "terminal_edges",
    "terminal_constraints_verified",
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
    # Required phase timings (seconds)
    "graph_construction_time_sec",
    "model_formulation_time_sec",
    "relaxed_solve_time_sec",
    "path_recovery_fixed_solve_time_sec",
    "total_time_sec",
    # Extra timing diagnostics
    "terrain_derivation_time_sec",
    "model_build_reported_time_sec",
    "relaxed_parameter_time_sec",
    "relaxed_optimizer_extract_time_sec",
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
    "actual_horizon",
    "horizon_slack",
    "start_goal_distance_xy",
    "centerline_length_xy",
    "center_bbox_span_x",
    "center_bbox_span_y",
    "terrain_envelope_span_x",
    "terrain_envelope_span_y",
    "support_area_total",
]
SUMMARY_STATS = ("n", "mean", "std", "median", "p25", "p75", "p95", "min", "max")
SUMMARY_BASE_FIELDS = [
    "sweep",
    "setting_value",
    "max_horizon",
    "num_regions",
    "n_expected",
    "n_rows",
    "n_unique_terrains",
    "paired_complete",
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _terrain_sample_index(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"terrain file has no numeric sample suffix: {path.name}")
    return int(match.group(1))


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


def _wrap_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def partition_intermediate_regions(
    base_stones: Sequence[Dict[str, Any]], target_regions: int
) -> List[Dict[str, Any]]:
    """Partition the two intermediate rectangles without changing their union.

    For R=2+2k, each original intermediate yawed rectangle is divided into k
    equal closed strips along its local width axis.  Boundaries may touch but
    interiors do not overlap.  Start and goal rectangles remain unchanged.
    Metadata retains the original physical-region group for graph construction.
    """
    if len(base_stones) != 4:
        raise ValueError(
            f"the controlled CVAE region sweep requires base R=4, got {len(base_stones)}"
        )
    target_regions = int(target_regions)
    if target_regions < 4 or (target_regions - 2) % 2 != 0:
        raise ValueError(
            f"target R must have form 2+2k with k>=1, got {target_regions}"
        )
    partition_count = (target_regions - 2) // 2
    output: List[Dict[str, Any]] = []

    def annotated_single(stone: Dict[str, Any], parent: int) -> Dict[str, Any]:
        result = copy.deepcopy(stone)
        result.update(
            {
                "_parent_region": int(parent),
                "_partition_index": 0,
                "_partition_count": 1,
            }
        )
        return result

    output.append(annotated_single(base_stones[0], 0))
    for parent in (1, 2):
        original = base_stones[parent]
        center = np.asarray(original["center"], dtype=float)
        width = float(original["width"])
        strip_width = width / partition_count
        yaw = math.radians(float(original.get("yaw_deg", 0.0)))
        local_width_axis = np.asarray([math.cos(yaw), math.sin(yaw)])
        for partition_index in range(partition_count):
            local_offset = (
                -0.5 * width + (partition_index + 0.5) * strip_width
            )
            child = copy.deepcopy(original)
            child["name"] = (
                f"{original.get('name', f'stone_{parent + 1}')}"
                f"_partition_{partition_index + 1:02d}_of_{partition_count:02d}"
            )
            child["center"] = (
                center
                + np.asarray(
                    [
                        local_offset * local_width_axis[0],
                        local_offset * local_width_axis[1],
                        0.0,
                    ]
                )
            ).tolist()
            child["width"] = strip_width
            child.update(
                {
                    "_parent_region": int(parent),
                    "_partition_index": int(partition_index),
                    "_partition_count": int(partition_count),
                }
            )
            output.append(child)
    output.append(annotated_single(base_stones[-1], 3))
    if len(output) != target_regions:
        raise RuntimeError(
            f"partition count invariant failed: {len(output)} != {target_regions}"
        )
    return output


def terrain_metrics(stones: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    grouped_centers: Dict[int, List[np.ndarray]] = {}
    for fallback_parent, stone in enumerate(stones):
        parent = int(stone.get("_parent_region", fallback_parent))
        grouped_centers.setdefault(parent, []).append(
            np.asarray(stone["center"], dtype=float)
        )
    parents = sorted(grouped_centers)
    representative_centers = np.asarray(
        [
            np.mean(np.asarray(grouped_centers[parent]), axis=0)
            for parent in parents
        ],
        dtype=float,
    )
    if representative_centers.ndim != 2 or representative_centers.shape[1] != 3:
        raise ValueError(
            f"invalid representative center shape: {representative_centers.shape}"
        )
    differences = np.diff(representative_centers, axis=0)
    endpoint = representative_centers[-1] - representative_centers[0]
    center_span = (
        np.max(representative_centers, axis=0)
        - np.min(representative_centers, axis=0)
    )
    all_vertices = np.concatenate(
        [stone_vertices(stone).T for stone in stones], axis=0
    )
    envelope_span = np.max(all_vertices, axis=0) - np.min(all_vertices, axis=0)
    return {
        "start_x": float(representative_centers[0, 0]),
        "start_y": float(representative_centers[0, 1]),
        "start_z": float(representative_centers[0, 2]),
        "goal_x": float(representative_centers[-1, 0]),
        "goal_y": float(representative_centers[-1, 1]),
        "goal_z": float(representative_centers[-1, 2]),
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


def assert_geometry_controls(
    base_stones: Sequence[Dict[str, Any]],
    variant_stones: Sequence[Dict[str, Any]],
    *,
    atol: float = 1e-10,
) -> float:
    base = terrain_metrics(base_stones)
    variant = terrain_metrics(variant_stones)
    controlled = (
        "start_x",
        "start_y",
        "start_z",
        "goal_x",
        "goal_y",
        "goal_z",
        "start_goal_distance_xy",
        "start_goal_distance_3d",
        "centerline_length_xy",
        "centerline_length_3d",
        "center_bbox_span_x",
        "center_bbox_span_y",
        "center_bbox_span_z",
        "terrain_envelope_span_x",
        "terrain_envelope_span_y",
        "terrain_envelope_span_z",
        "support_area_total",
    )
    max_error = max(abs(base[name] - variant[name]) for name in controlled)
    if max_error > float(atol):
        raise RuntimeError(
            f"region derivation changed controlled geometry by {max_error:.3e} "
            f"(tolerance {atol:.3e})"
        )

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for stone in variant_stones:
        groups.setdefault(int(stone["_parent_region"]), []).append(stone)
    if sorted(groups) != [0, 1, 2, 3]:
        raise RuntimeError(f"invalid parent-region groups: {sorted(groups)}")
    if len(groups[0]) != 1 or len(groups[3]) != 1:
        raise RuntimeError("start and goal rectangles must remain single regions")
    if len(groups[1]) != len(groups[2]):
        raise RuntimeError("the two intermediate rectangles must use equal partitions")

    # Verify each strip family algebraically, not only via aggregate metrics.
    structural_errors: List[float] = []
    for parent in (1, 2):
        original = base_stones[parent]
        children = sorted(
            groups[parent], key=lambda stone: int(stone["_partition_index"])
        )
        count = len(children)
        expected_width = float(original["width"]) / count
        yaw = math.radians(float(original.get("yaw_deg", 0.0)))
        axis = np.asarray([math.cos(yaw), math.sin(yaw)])
        original_center = np.asarray(original["center"], dtype=float)
        for index, child in enumerate(children):
            expected_offset = (
                -0.5 * float(original["width"])
                + (index + 0.5) * expected_width
            )
            expected_center = original_center.copy()
            expected_center[:2] += expected_offset * axis
            structural_errors.extend(
                [
                    abs(float(child["width"]) - expected_width),
                    abs(float(child["height"]) - float(original["height"])),
                    abs(
                        _wrap_deg(
                            float(child.get("yaw_deg", 0.0))
                            - float(original.get("yaw_deg", 0.0))
                        )
                    ),
                    float(
                        np.max(
                            np.abs(
                                np.asarray(child["center"], dtype=float)
                                - expected_center
                            )
                        )
                    ),
                ]
            )
        structural_errors.append(
            abs(
                sum(float(child["width"]) for child in children)
                - float(original["width"])
            )
        )
    structural_error = max(structural_errors, default=0.0)
    if structural_error > float(atol):
        raise RuntimeError(
            f"partition structure changed rectangle union by {structural_error:.3e}"
        )

    # Partitioning must leave source/goal headings unchanged.
    base_start_heading = _heading_deg(
        base_stones[0]["center"], base_stones[1]["center"]
    )
    variant_parent_centers = {
        parent: np.mean(
            np.asarray([stone["center"] for stone in children], dtype=float),
            axis=0,
        )
        for parent, children in groups.items()
    }
    variant_start_heading = _heading_deg(
        variant_parent_centers[0], variant_parent_centers[1]
    )
    base_goal_heading = _heading_deg(
        base_stones[-2]["center"], base_stones[-1]["center"]
    )
    variant_goal_heading = _heading_deg(
        variant_parent_centers[2], variant_parent_centers[3]
    )
    heading_error = max(
        abs(_wrap_deg(base_start_heading - variant_start_heading)),
        abs(_wrap_deg(base_goal_heading - variant_goal_heading)),
    )
    if heading_error > 1e-9:
        raise RuntimeError(f"region derivation changed endpoint heading by {heading_error}")
    return float(max(max_error, structural_error, heading_error))


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


def _parent_successors(
    region: int,
    stones: Sequence[Dict[str, Any]],
) -> Tuple[int, ...]:
    """Stay in the same parent union or advance to the next parent union.

    At base R=4 this is exactly the legacy ordered-adjacent rule ``i -> i,i+1``.
    At larger R this is the complete lift of that physical-parent rule: every
    child strip connects to every child strip of the same or next parent.  Thus
    partitioning does not remove any transition that existed at R=4.
    """
    source = stones[int(region)]
    parent = int(source.get("_parent_region", region))
    return tuple(
        index
        for index, stone in enumerate(stones)
        if int(stone.get("_parent_region", index)) in (parent, parent + 1)
    )


def expected_graph_counts(
    max_horizon: int,
    stones: Sequence[Dict[str, Any]],
) -> Tuple[int, int]:
    num_regions = len(stones)
    successor_count = sum(
        len(_parent_successors(region, stones))
        for region in range(num_regions)
    )
    vertices = 2 * int(max_horizon) * int(num_regions) + 2
    edges = (
        2
        + (int(max_horizon) - 1) * 2 * successor_count
        + 2 * int(max_horizon)
    )
    return vertices, edges


def build_max_horizon_graph(
    stones: Sequence[Dict[str, Any]],
    max_horizon: int,
) -> Dict[str, Any]:
    """Build a layered, alternating-foot graph with early goal termination."""
    max_horizon = int(max_horizon)
    num_regions = len(stones)
    if max_horizon < 1:
        raise ValueError(f"maximum horizon must be >=1, got {max_horizon}")
    if num_regions < 2:
        raise ValueError("at least two regions are required")
    base_vertices = [stone_vertices(stone) for stone in stones]
    vertices: List[np.ndarray] = [base_vertices[0].copy()]
    mode_list: List[str] = ["D"]
    vertex_region_ids: List[int] = [0]
    vertex_stages: List[int] = [-1]
    node_id: Dict[Tuple[int, int, str], int] = {}

    for stage in range(max_horizon):
        for region in range(num_regions):
            for mode in ("SL", "SR"):
                vertices.append(base_vertices[region].copy())
                mode_list.append(mode)
                vertex_region_ids.append(region)
                vertex_stages.append(stage)
                node_id[(stage, region, mode)] = len(vertices)

    vertices.append(base_vertices[-1].copy())
    mode_list.append("D")
    vertex_region_ids.append(num_regions - 1)
    vertex_stages.append(max_horizon)
    sink_id = len(vertices)

    edges: List[Tuple[int, int]] = [
        (1, node_id[(0, 0, "SL")]),
        (1, node_id[(0, 0, "SR")]),
    ]
    for stage in range(max_horizon - 1):
        for region_from in range(num_regions):
            successors = _parent_successors(region_from, stones)
            if not successors:
                raise RuntimeError(f"region {region_from} has no graph successor")
            for mode_from, mode_to in (("SL", "SR"), ("SR", "SL")):
                source = node_id[(stage, region_from, mode_from)]
                for region_to in successors:
                    edges.append(
                        (source, node_id[(stage + 1, region_to, mode_to)])
                    )

    # Maximum-horizon semantics: every goal layer may terminate.
    goal_region = num_regions - 1
    for stage in range(max_horizon):
        edges.append((node_id[(stage, goal_region, "SL")], sink_id))
        edges.append((node_id[(stage, goal_region, "SR")], sink_id))

    edge_array = np.asarray(edges, dtype=int).reshape(-1, 2)
    expected_vertices, expected_edges = expected_graph_counts(
        max_horizon, stones
    )
    if len(vertices) != expected_vertices or edge_array.shape[0] != expected_edges:
        raise RuntimeError(
            "graph count invariant failed: "
            f"V={len(vertices)}/{expected_vertices}, "
            f"E={edge_array.shape[0]}/{expected_edges}"
        )
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
        "max_horizon": max_horizon,
        "num_regions": num_regions,
        "region_parent_ids": [
            int(stone.get("_parent_region", index))
            for index, stone in enumerate(stones)
        ],
    }


def audit_terminal_constraints(
    graph: Dict[str, Any], variables: Dict[str, Any]
) -> int:
    """Verify that all 2H early terminal edges have full target constraints."""
    edge = np.asarray(graph["edge"], dtype=int)
    sink_id = int(graph["sink_id"])
    max_horizon = int(graph["max_horizon"])
    goal_region = int(graph["num_regions"]) - 1
    expected = np.where(edge[:, 1] == sink_id)[0].astype(int).tolist()
    registered = [
        int(value)
        for value in np.asarray(variables["meta"]["sink_edges"]).reshape(-1)
    ]
    if expected != registered:
        raise RuntimeError(
            f"terminal edge registration mismatch: graph={expected}, model={registered}"
        )
    if len(expected) != 2 * max_horizon:
        raise RuntimeError(
            f"expected {2 * max_horizon} terminal edges, found {len(expected)}"
        )

    rows_left = variables["meta"]["B_rows_sink_zpL"]
    rows_right = variables["meta"]["B_rows_sink_zpR"]
    if len(rows_left) != len(expected) or len(rows_right) != len(expected):
        raise RuntimeError("not every terminal edge has target-pose rows")

    seen: set[Tuple[int, str]] = set()
    for local_index, edge_index in enumerate(expected):
        from_zero = int(edge[edge_index, 0]) - 1
        region = int(graph["vertex_region_ids"][from_zero])
        stage = int(graph["vertex_stages"][from_zero])
        mode = str(graph["mode_list"][from_zero])
        if region != goal_region or mode not in ("SL", "SR"):
            raise RuntimeError(f"invalid terminal origin on edge {edge_index}")
        if not 0 <= stage < max_horizon:
            raise RuntimeError(f"invalid terminal stage {stage} on edge {edge_index}")
        if len(rows_left[local_index]) != 5 or len(rows_right[local_index]) != 5:
            raise RuntimeError(f"terminal edge {edge_index} lacks a full 5D target pose")
        seen.add((stage, mode))
    required = {
        (stage, mode)
        for stage in range(max_horizon)
        for mode in ("SL", "SR")
    }
    if seen != required:
        raise RuntimeError(f"terminal stage/mode coverage mismatch: {seen ^ required}")
    if graph["mode_list"][sink_id - 1] != "D":
        raise RuntimeError("terminal vertex is not double support")
    return len(expected)


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
    """Set one fixed path, solve it, and return objective plus phase timings."""
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


def _stable_trial_seed(base_seed: int, terrain_sample_index: int, trial: int) -> int:
    # H/R are deliberately omitted: paired settings use the same random stream.
    sequence = np.random.SeedSequence(
        [int(base_seed), int(terrain_sample_index), int(trial)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _select_path(
    y_relaxed: np.ndarray,
    graph: Dict[str, Any],
    *,
    trial: int,
    greedy_first: bool,
    temperature: float,
    seed: int,
) -> Tuple[List[int], str]:
    if bool(greedy_first) and int(trial) == 0:
        return (
            relaxation_max(
                y_relaxed,
                graph["edge"],
                int(graph["vertex_number"]),
            ),
            "relaxation_greedy",
        )
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
        "graph_rule": GRAPH_RULE,
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
    stones: Sequence[Dict[str, Any]],
    max_horizon: int,
    args: argparse.Namespace,
    *,
    sweep: str,
    setting_value: int,
    run_timestamp: str,
    case_index: int,
    condition_order: int,
    terrain_index: int,
    derivation_time_sec: float,
    experiment_config_hash: str,
) -> Dict[str, Any]:
    row = _empty_row()
    sample_index = _terrain_sample_index(context.path)
    solver_hash = _canonical_hash(_solver_config(args))
    geometry = terrain_metrics(stones)
    family_hash = _canonical_hash(context.stones)
    variant_hash = _canonical_hash(stones)
    row.update(
        {
            "run_timestamp": run_timestamp,
            "sweep": sweep,
            "case_index": int(case_index),
            "condition_order": int(condition_order),
            "setting_value": int(setting_value),
            "terrain_index": int(terrain_index),
            "terrain_id": context.path.stem,
            "terrain_sample_index": sample_index,
            "terrain_seed": int(args.cvae_base_seed) + sample_index,
            "terrain_file": str(context.path.resolve()),
            "terrain_family_hash": family_hash,
            "terrain_variant_hash": variant_hash,
            "difficulty": float(args.difficulty),
            "max_horizon": int(max_horizon),
            "num_regions": len(stones),
            "base_num_regions": len(context.stones),
            "region_derivation": "exact_union_intermediate_rectangle_partition",
            "graph_rule": GRAPH_RULE,
            "horizon_definition": HORIZON_DEFINITION,
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
            "terrain_derivation_time_sec": float(derivation_time_sec),
            "geometry_control_error": assert_geometry_controls(
                context.stones, stones
            ),
            "relaxed_success": 0,
            "recovery_feasible": 0,
            "num_recovery_trials": 0,
            "feasible_trials": 0,
            "failed_trials": 0,
            "terminal_constraints_verified": 0,
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
        graph = build_max_horizon_graph(
            stones,
            int(max_horizon),
        )
        row["graph_construction_time_sec"] = _seconds_since(t0)
        row["num_vertices"] = int(graph["vertex_number"])
        row["num_edges"] = int(graph["n_edge"])

        current_stage = "model_formulation"
        formulation_t0 = time.perf_counter()
        phase_t0 = formulation_t0
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

        task, variables, idx, _par, reported_build_time = (
            build_gcs_model_VAE_optapi(
                graph["v"],
                graph["edge"],
                float(args.step_span),
                math.radians(float(args.alpha_deg)),
                graph["mode_list"],
                x_ref,
                y_ref,
            )
        )
        row["model_build_reported_time_sec"] = float(reported_build_time)
        if int(args.mosek_threads) > 0:
            task.putintparam(mosek.iparam.num_threads, int(args.mosek_threads))
        if float(args.solver_time_limit_sec) > 0.0:
            task.putdouparam(
                mosek.dparam.optimizer_max_time,
                float(args.solver_time_limit_sec),
            )
        row["num_scalar_vars"] = int(task.getnumvar())
        row["num_constraints"] = int(task.getnumcon())
        row["num_cones"] = int(task.getnumcone())
        num_psd_blocks = int(task.getnumbarvar())
        row["num_psd_blocks"] = num_psd_blocks
        # The current model creates two 5x5 and two 7x7 PSD blocks per edge:
        # 2*15 + 2*28 = 86 unique lower-triangular scalar entries per edge.
        if num_psd_blocks != 4 * int(idx["n_edge"]):
            raise RuntimeError(
                f"unexpected PSD block count: {num_psd_blocks} "
                f"!= {4 * int(idx['n_edge'])}"
            )
        row["num_psd_vars"] = 86 * int(idx["n_edge"])
        row["terminal_edges"] = audit_terminal_constraints(graph, variables)
        row["terminal_constraints_verified"] = 1

        parameter_t0 = time.perf_counter()
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
        row["relaxed_parameter_time_sec"] = _seconds_since(parameter_t0)
        row["model_formulation_time_sec"] = _seconds_since(formulation_t0)

        current_stage = "relaxed_solve"
        relaxed_t0 = time.perf_counter()
        phase_t0 = relaxed_t0
        relaxed_solution, relaxed_objective = solve_relaxed_optapi(
            task,
            variables,
            log_level=int(args.solver_log),
        )
        row["relaxed_solve_time_sec"] = _seconds_since(relaxed_t0)
        row["relaxed_optimizer_extract_time_sec"] = row["relaxed_solve_time_sec"]
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
        if y_relaxed.size != int(graph["n_edge"]) or not np.all(
            np.isfinite(y_relaxed)
        ):
            raise RuntimeError("relaxed y is missing, non-finite, or wrong-sized")
        row["C_relax"] = C_relax
        row["relaxed_success"] = 1

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
            selected_path: Optional[List[int]] = None
            strategy = "selection_failed"
            try:
                selection_t0 = time.perf_counter()
                selected_path, strategy = _select_path(
                    y_relaxed,
                    graph,
                    trial=trial,
                    greedy_first=bool(args.greedy_first),
                    temperature=float(args.temperature),
                    seed=_stable_trial_seed(
                        int(args.recovery_seed), sample_index, trial
                    ),
                )
                rounding_time += _seconds_since(selection_t0)
                actual_horizon = len(selected_path) - 2
                if not 1 <= actual_horizon <= int(max_horizon):
                    raise RuntimeError(
                        f"maximum-horizon invariant failed: "
                        f"n_step={actual_horizon}, H={max_horizon}"
                    )
                trial_strategies.append(strategy)
                trial_path_horizons.append(actual_horizon)
                y_path = path_to_y(
                    graph["edge"], selected_path, int(graph["n_edge"])
                )
                fixed_objective, fixed_timing = solve_fixed_path_timed(
                    task,
                    variables,
                    centroid_mat=centroid_mat,
                    y_path=y_path,
                    M_small=float(args.M_small),
                    start_L=start_L,
                    start_R=start_R,
                    target_L=target_L,
                    target_R=target_R,
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
                trial_errors.append(
                    f"trial {trial} ({strategy}): {_safe_error(exc, 400)}"
                )
            except Exception as exc:
                row["failed_trials"] = int(row["failed_trials"]) + 1
                trial_errors.append(
                    f"trial {trial} ({strategy}): {_safe_error(exc, 400)}"
                )

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

        actual_horizon = len(best_path) - 2
        C_round = float(best_objective)
        gap_abs = C_round - C_relax
        delta = gap_abs / max(abs(C_relax), 1e-12)
        path_regions = [
            int(graph["vertex_region_ids"][node - 1])
            for node in best_path[1:-1]
        ]
        path_modes = [
            str(graph["mode_list"][node - 1]) for node in best_path[1:-1]
        ]
        row.update(
            {
                "C_round": C_round,
                "delta_relax": delta,
                "recovery_feasible": 1,
                "actual_horizon": actual_horizon,
                "horizon_slack": int(max_horizon) - actual_horizon,
                "path_nodes": _join(best_path),
                "path_region_sequence": _join(
                    region + 1 for region in path_regions
                ),
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
            if current_stage == "relaxed_solve":
                row["relaxed_optimizer_extract_time_sec"] = row[timing_field]
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
    rows: Sequence[Dict[str, Any]], sweep: str, expected_terrains: int
) -> List[Dict[str, Any]]:
    sweep_rows = [row for row in rows if row.get("sweep") == sweep]
    settings = sorted({int(row["setting_value"]) for row in sweep_rows})
    output: List[Dict[str, Any]] = []
    for setting in settings:
        group = [
            row for row in sweep_rows if int(row["setting_value"]) == setting
        ]
        relaxed_count = sum(int(float(row.get("relaxed_success", 0) or 0)) for row in group)
        recovery_count = sum(
            int(float(row.get("recovery_feasible", 0) or 0)) for row in group
        )
        status_counts: Dict[str, int] = {}
        for row in group:
            status = str(row.get("status", ""))
            status_counts[status] = status_counts.get(status, 0) + 1
        terrain_ids = {str(row.get("terrain_id", "")) for row in group}
        summary: Dict[str, Any] = {
            "sweep": sweep,
            "setting_value": setting,
            "max_horizon": int(float(group[0]["max_horizon"])) if group else "",
            "num_regions": int(float(group[0]["num_regions"])) if group else "",
            "n_expected": int(expected_terrains),
            "n_rows": len(group),
            "n_unique_terrains": len(terrain_ids),
            "paired_complete": int(
                len(group) == int(expected_terrains)
                and len(terrain_ids) == int(expected_terrains)
            ),
            "n_relaxed_success": relaxed_count,
            "relaxed_success_rate": relaxed_count / len(group) if group else "",
            "n_recovery_feasible": recovery_count,
            "recovery_feasibility_rate": recovery_count / len(group) if group else "",
            "status_counts": json.dumps(
                status_counts, sort_keys=True, separators=(",", ":")
            ),
        }
        for metric in SUMMARY_METRICS:
            stats = _metric_stats(_numeric_values(group, metric))
            for stat, value in stats.items():
                summary[f"{metric}_{stat}"] = value
        output.append(summary)
    return output


def _write_csv(
    path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]
) -> None:
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


def _raw_path(output_dir: Path, sweep: str) -> Path:
    return output_dir / f"Com_1_2_{sweep}_sweep_raw.csv"


def _summary_path(output_dir: Path, sweep: str) -> Path:
    return output_dir / f"Com_1_2_{sweep}_sweep_summary.csv"


def _write_all_outputs(
    output_dir: Path,
    all_rows: Dict[str, List[Dict[str, Any]]],
    expected_terrains: int,
    active_sweeps: Sequence[str],
) -> None:
    for sweep in active_sweeps:
        rows = all_rows[sweep]
        _write_csv(_raw_path(output_dir, sweep), rows, RAW_FIELDS)
        _write_csv(
            _summary_path(output_dir, sweep),
            build_summary(rows, sweep, expected_terrains),
            SUMMARY_FIELDS,
        )


def _case_key(row: Dict[str, Any]) -> Tuple[str, int, str]:
    return (
        str(row["sweep"]),
        int(float(row["setting_value"])),
        str(row["terrain_id"]),
    )


def _validate_completed_matrix(
    all_rows: Dict[str, List[Dict[str, Any]]],
    terrain_files: Sequence[Path],
    horizons: Sequence[int],
    regions: Sequence[int],
    sweeps: Sequence[str],
) -> None:
    terrain_ids = {path.stem for path in terrain_files}
    for sweep in sweeps:
        settings = horizons if sweep == "horizon" else regions
        rows = all_rows[sweep]
        keys = [_case_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"duplicate raw rows in {sweep} sweep")
        expected = {
            (sweep, int(setting), terrain_id)
            for setting in settings
            for terrain_id in terrain_ids
        }
        missing = expected - set(keys)
        extra = set(keys) - expected
        if missing or extra:
            raise RuntimeError(
                f"{sweep} paired matrix mismatch: missing={len(missing)}, extra={len(extra)}"
            )

        solver_hashes = {str(row["solver_config_hash"]) for row in rows}
        if len(solver_hashes) != 1:
            raise RuntimeError(f"{sweep} did not keep solver settings fixed")
        by_terrain: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_terrain.setdefault(str(row["terrain_id"]), []).append(row)
        for terrain_id, group in by_terrain.items():
            family_hashes = {str(row["terrain_family_hash"]) for row in group}
            if len(family_hashes) != 1:
                raise RuntimeError(f"{terrain_id}: terrain family changed across {sweep}")
            if sweep == "horizon":
                variants = {str(row["terrain_variant_hash"]) for row in group}
                region_counts = {int(float(row["num_regions"])) for row in group}
                if len(variants) != 1 or region_counts != {4}:
                    raise RuntimeError(
                        f"{terrain_id}: horizon sweep terrain/region count changed"
                    )
            else:
                if {
                    int(float(row["max_horizon"])) for row in group
                } != {12}:
                    raise RuntimeError(
                        f"{terrain_id}: region sweep did not keep maximum H=12"
                    )
                controlled_metrics = (
                    "start_goal_distance_xy",
                    "start_goal_distance_3d",
                    "centerline_length_xy",
                    "centerline_length_3d",
                    "center_bbox_span_x",
                    "center_bbox_span_y",
                    "center_bbox_span_z",
                    "terrain_envelope_span_x",
                    "terrain_envelope_span_y",
                    "terrain_envelope_span_z",
                    "support_area_total",
                )
                for metric in controlled_metrics:
                    values = np.asarray(
                        [float(row[metric]) for row in group], dtype=float
                    )
                    if float(np.max(values) - np.min(values)) > 1e-9:
                        raise RuntimeError(
                            f"{terrain_id}: controlled metric {metric} changed across R"
                        )
            for row in group:
                if int(float(row.get("recovery_feasible", 0) or 0)) > int(
                    float(row.get("relaxed_success", 0) or 0)
                ):
                    raise RuntimeError(
                        f"{terrain_id}: recovery succeeded without relaxed success"
                    )
                actual = row.get("actual_horizon", "")
                if actual not in ("", None) and int(float(actual)) > int(
                    float(row["max_horizon"])
                ):
                    raise RuntimeError(f"{terrain_id}: actual horizon exceeds maximum")
                phases = (
                    "graph_construction_time_sec",
                    "model_formulation_time_sec",
                    "relaxed_solve_time_sec",
                    "path_recovery_fixed_solve_time_sec",
                )
                phase_sum = _sum_present(row, phases)
                if not math.isclose(
                    float(row["total_time_sec"]),
                    phase_sum,
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ):
                    raise RuntimeError(f"{terrain_id}: total timing sum mismatch")
                if int(float(row.get("recovery_feasible", 0) or 0)):
                    C_relax = float(row["C_relax"])
                    C_round = float(row["C_round"])
                    expected_delta = (C_round - C_relax) / max(
                        abs(C_relax), 1e-12
                    )
                    if not math.isclose(
                        float(row["delta_relax"]),
                        expected_delta,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        raise RuntimeError(f"{terrain_id}: delta_relax mismatch")

    if set(sweeps) == {"horizon", "region"}:
        horizon_families = {
            str(row["terrain_id"]): str(row["terrain_family_hash"])
            for row in all_rows["horizon"]
        }
        region_families = {
            str(row["terrain_id"]): str(row["terrain_family_hash"])
            for row in all_rows["region"]
        }
        if horizon_families != region_families:
            raise RuntimeError("terrain families differ between horizon and region sweeps")


def _terrain_files(args: argparse.Namespace) -> List[Path]:
    files = sorted(
        args.terrain_dir.glob("*.json"),
        key=lambda path: (_terrain_sample_index(path), path.name),
    )
    files = [
        path
        for path in files
        if _terrain_sample_index(path) >= int(args.terrain_start_index)
    ]
    selected = files[: int(args.num_terrains)]
    if len(selected) != int(args.num_terrains):
        raise FileNotFoundError(
            f"requested {args.num_terrains} terrain files, found {len(selected)} "
            f"at/after index {args.terrain_start_index} in {args.terrain_dir}"
        )
    return selected


def _experiment_config(
    args: argparse.Namespace, terrain_files: Sequence[Path]
) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "implementation_hashes": {
            "Com_1_2.py": _file_sha256(Path(__file__).resolve()),
            "gcs_model_VAE_optapi.py": _file_sha256(
                project_root / "src" / "model" / "gcs_model_VAE_optapi.py"
            ),
            "solver_path.py": _file_sha256(
                project_root / "src" / "solver" / "path.py"
            ),
            "solver_utils.py": _file_sha256(
                project_root / "src" / "solver" / "utils.py"
            ),
        },
        "horizons": list(map(int, args.horizons)),
        "regions": list(map(int, args.regions)),
        "region_horizon": int(args.region_horizon),
        "horizon_regions": 4,
        "num_terrains": int(args.num_terrains),
        "terrain_dir": str(args.terrain_dir.resolve()),
        "terrain_files": [
            {
                "name": path.name,
                "resolved_path": str(path.resolve()),
                "sha256": _file_sha256(path),
            }
            for path in terrain_files
        ],
        "difficulty": float(args.difficulty),
        "terrain_key": args.terrain_key,
        "cvae_base_seed": int(args.cvae_base_seed),
        "region_derivation": "exact_union_intermediate_rectangle_partition",
        "horizon_definition": HORIZON_DEFINITION,
        "execution_protocol": {
            "shuffle_conditions": bool(args.shuffle_conditions),
            "warmup": bool(args.warmup),
        },
        "solver": _solver_config(args),
    }


def _write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    terrain_files: Sequence[Path],
    experiment_config_hash: str,
    active_sweeps: Sequence[str],
) -> None:
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "script": str(Path(__file__).resolve()),
        "command": sys.argv,
        "methodology_version": METHODOLOGY_VERSION,
        "active_sweeps": list(active_sweeps),
        "experiment_config_hash": experiment_config_hash,
        "horizon_definition": HORIZON_DEFINITION,
        "delta_relax_definition": DELTA_DEFINITION,
        "timing_unit": "seconds",
        "timing_boundaries": {
            "graph_construction_time_sec": "time-expanded vertices and edges only",
            "model_formulation_time_sec": (
                "pose/centroid preparation, OptAPI task construction, terminal audit, "
                "and relaxed parameter packing/setting"
            ),
            "relaxed_solve_time_sec": "MOSEK relaxed optimize plus solution extraction",
            "path_recovery_fixed_solve_time_sec": (
                "all path selections and fixed-path parameter/solve/extraction attempts"
            ),
            "total_time_sec": (
                "sum of graph, model formulation, relaxed solve, and recovery/fixed phases; "
                "terrain derivation, warmup, CSV, and summary aggregation excluded"
            ),
        },
        "success_definitions": {
            "relaxed_success": "accepted finite MOSEK relaxed solution",
            "recovery_feasible": "at least one accepted finite fixed-path solve",
        },
        "timeout_policy": {
            "optimizer_time_limit_sec": float(args.solver_time_limit_sec),
            "recovery_loop_budget_sec": float(args.fixed_budget_sec),
            "recovery_budget_check": (
                "between trials only; an in-progress optimizer call is governed "
                "by optimizer_time_limit_sec"
            ),
        },
        "paired_seed_definition": "terrain_seed = cvae_base_seed + run_XXX index",
        "region_control": {
            "method": (
                "local-width partition of both original intermediate yawed "
                "rectangles into disjoint strips"
            ),
            "preserved": [
                "physical support-region union",
                "total support area",
                "terrain outer envelope",
                "start and goal rectangles",
                "start and goal headings",
                "endpoint XY/3D distance",
                "centre-line XY/3D length",
                "centre-coordinate bounding box",
            ],
            "graph_semantics": (
                "complete child-strip lift of the base parent adjacency: "
                "any strip in the same or next original physical-region group"
            ),
        },
        "summary": {
            "std_ddof": 1,
            "quantiles": "NumPy linear default",
            "missing_values": "excluded per metric; each metric has its own n",
            "failure_rows": "retained",
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "terrain_files": [str(path.resolve()) for path in terrain_files],
        "outputs": {
            sweep: {
                "raw": str(_raw_path(output_dir, sweep)),
                "summary": str(_summary_path(output_dir, sweep)),
            }
            for sweep in ("horizon", "region")
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "Com_1_2_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _self_test(args: argparse.Namespace) -> None:
    # Pure graph tests catch exact-H regression without invoking MOSEK.
    dummy = [
        {
            "name": f"stone_{index}",
            "center": [float(index), 0.0, 0.0],
            "width": 0.4,
            "height": 0.4,
            "yaw_deg": 0.0,
        }
        for index in range(4)
    ]
    base_variant = partition_intermediate_regions(dummy, 4)
    for horizon in args.horizons:
        graph = build_max_horizon_graph(
            base_variant,
            int(horizon),
        )
        sink_edges = np.where(
            np.asarray(graph["edge"])[:, 1] == graph["sink_id"]
        )[0]
        if sink_edges.size != 2 * int(horizon):
            raise RuntimeError("maximum-horizon self-test failed")
    for region_count in sorted(map(int, args.regions)):
        variant = partition_intermediate_regions(dummy, region_count)
        assert_geometry_controls(dummy, variant)
        graph = build_max_horizon_graph(
            variant,
            int(args.region_horizon),
        )
        adjacency: Dict[int, List[int]] = {}
        for source, target in np.asarray(graph["edge"], dtype=int):
            adjacency.setdefault(int(source), []).append(int(target))
        frontier: List[Tuple[int, int]] = [(1, 0)]
        visited = {1}
        shortest_edges: Optional[int] = None
        while frontier:
            node, distance = frontier.pop(0)
            if node == int(graph["sink_id"]):
                shortest_edges = distance
                break
            for target in adjacency.get(node, []):
                if target not in visited:
                    visited.add(target)
                    frontier.append((target, distance + 1))
        # source->four physical parent stages->sink is five graph edges.
        if shortest_edges != 5:
            raise RuntimeError(
                f"R={region_count} changed minimum logical path depth: "
                f"{shortest_edges} graph edges"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CVAE maximum-horizon and region scalability benchmark (Com_1_2)"
    )
    parser.add_argument("--terrain-dir", type=Path, default=DEFAULT_TERRAIN_DIR)
    parser.add_argument("--terrain-key", default="terrain")
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--cvae-base-seed", type=int, default=CVAE_BASE_SEED)
    parser.add_argument("--terrain-start-index", type=int, default=0)
    parser.add_argument("--num-terrains", type=int, default=MIN_TERRAINS)
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS)
    )
    parser.add_argument(
        "--regions", type=int, nargs="+", default=list(DEFAULT_REGIONS)
    )
    parser.add_argument("--region-horizon", type=int, default=12)
    parser.add_argument(
        "--sweeps",
        choices=("both", "horizon", "region"),
        default="both",
    )

    parser.add_argument("--step-span", type=float, default=0.7)
    parser.add_argument("--alpha-deg", type=float, default=30.0)
    parser.add_argument("--stance-width", type=float, default=0.25)
    parser.add_argument("--M-large", type=float, default=1e3)
    parser.add_argument("--M-small", type=float, default=1e-6)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--recovery-seed", type=int, default=123)
    parser.add_argument(
        "--greedy-first", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--early-stop", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fixed-budget-sec",
        type=float,
        default=30.0,
        help=(
            "recovery-loop budget checked between fixed-path trials; "
            "a running MOSEK attempt is not interrupted"
        ),
    )
    parser.add_argument("--mosek-threads", type=int, default=1)
    parser.add_argument("--solver-log", type=int, default=0)
    parser.add_argument(
        "--solver-time-limit-sec",
        type=float,
        default=300.0,
        help="fixed MOSEK optimizer limit applied to every relaxed/fixed solve; <=0 disables",
    )
    parser.add_argument(
        "--warmup", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--shuffle-conditions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="explicitly permit a one-terrain, one-setting-per-sweep test",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate terrain pairing, transforms, and graph invariants without solving",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.terrain_dir.is_dir():
        raise NotADirectoryError(args.terrain_dir)
    if args.smoke_test:
        args.num_terrains = 1
        args.horizons = [8]
        args.regions = [4]
        args.warmup = False
    elif int(args.num_terrains) < MIN_TERRAINS:
        raise ValueError(
            f"normal Com_1_2 runs require at least {MIN_TERRAINS} terrains; "
            "use --smoke-test only for explicit validation"
        )
    if int(args.num_terrains) < 1:
        raise ValueError("--num-terrains must be >=1")
    if list(map(int, args.horizons)) != list(DEFAULT_HORIZONS) and not args.smoke_test:
        raise ValueError(
            f"Com_1_2 horizon sweep is fixed to {list(DEFAULT_HORIZONS)}"
        )
    if list(map(int, args.regions)) != list(DEFAULT_REGIONS) and not args.smoke_test:
        raise ValueError(f"Com_1_2 region sweep is fixed to {list(DEFAULT_REGIONS)}")
    if int(args.region_horizon) != 12 and not args.smoke_test:
        raise ValueError("Com_1_2 region sweep fixes maximum H=12")
    if int(args.cvae_base_seed) != CVAE_BASE_SEED:
        raise ValueError(
            f"the stored CVAE JSON files were generated with base seed "
            f"{CVAE_BASE_SEED}; changing --cvae-base-seed would mislabel them"
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
    _self_test(args)
    terrain_files = _terrain_files(args)
    sweeps = (
        ["horizon", "region"]
        if args.sweeps == "both"
        else [str(args.sweeps)]
    )
    config = _experiment_config(args, terrain_files)
    experiment_config_hash = _canonical_hash(config)
    output_dir = args.output_dir.resolve()

    existing_paths = [
        _raw_path(output_dir, sweep)
        for sweep in sweeps
        if _raw_path(output_dir, sweep).exists()
    ]
    if existing_paths and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"output exists; use --resume or --overwrite: {existing_paths[0]}"
        )

    all_rows: Dict[str, List[Dict[str, Any]]] = {
        "horizon": [],
        "region": [],
    }
    if args.resume:
        for sweep in sweeps:
            all_rows[sweep] = _read_csv(_raw_path(output_dir, sweep))
            hashes = {
                str(row.get("experiment_config_hash", ""))
                for row in all_rows[sweep]
            }
            if hashes and hashes != {experiment_config_hash}:
                raise RuntimeError(
                    f"cannot resume {sweep}: experiment config hash differs"
                )

    for inactive_sweep in ({"horizon", "region"} - set(sweeps)):
        inactive_path = _raw_path(output_dir, inactive_sweep)
        if not inactive_path.exists():
            continue
        inactive_rows = _read_csv(inactive_path)
        inactive_hashes = {
            str(row.get("experiment_config_hash", "")) for row in inactive_rows
        }
        if inactive_hashes and inactive_hashes != {experiment_config_hash}:
            raise RuntimeError(
                f"inactive {inactive_sweep} output uses a different experiment "
                f"configuration; use a separate --output-dir or rerun both sweeps"
            )

    _write_manifest(
        output_dir,
        args,
        terrain_files,
        experiment_config_hash,
        sweeps,
    )
    if args.validate_only:
        for path in terrain_files:
            context = load_terrain_context(
                path, args.terrain_key, float(args.stance_width)
            )
            if len(context.stones) != 4:
                raise ValueError(
                    f"{path.name}: expected base R=4, found {len(context.stones)}"
                )
            for region_count in args.regions:
                variant = partition_intermediate_regions(
                    context.stones, int(region_count)
                )
                assert_geometry_controls(context.stones, variant)
                build_max_horizon_graph(
                    variant,
                    int(args.region_horizon),
                )
        print(
            f"Validated {len(terrain_files)} paired terrains, transforms, and graph invariants."
        )
        return

    run_timestamp = datetime.now().astimezone().isoformat()
    total_cases = len(terrain_files) * sum(
        len(args.horizons) if sweep == "horizon" else len(args.regions)
        for sweep in sweeps
    )
    completed_keys = {
        _case_key(row)
        for sweep in sweeps
        for row in all_rows[sweep]
    }
    completed = len(completed_keys)
    case_index = completed

    print("=== CVAE scalability benchmark: Com_1_2 ===")
    print(f"paired terrains : {len(terrain_files)}")
    print(f"horizon sweep   : {list(map(int, args.horizons))}")
    print(f"region sweep    : {list(map(int, args.regions))}, H={args.region_horizon}")
    print(f"graph rule      : {GRAPH_RULE}")
    print(f"solver threads  : {args.mosek_threads}")
    print(f"output dir      : {output_dir}")

    if args.warmup:
        print("[warmup] one unrecorded R=4, H=8 case")
        warm_context = load_terrain_context(
            terrain_files[0], args.terrain_key, float(args.stance_width)
        )
        warm_args = copy.copy(args)
        warm_args.trials = 1
        warm_args.early_stop = True
        warm_row = run_case(
            warm_context,
            partition_intermediate_regions(warm_context.stones, 4),
            8,
            warm_args,
            sweep="horizon",
            setting_value=8,
            run_timestamp=run_timestamp,
            case_index=-1,
            condition_order=-1,
            terrain_index=0,
            derivation_time_sec=0.0,
            experiment_config_hash=experiment_config_hash,
        )
        print(
            f"[warmup] status={warm_row['status']} "
            f"wall={float(warm_row['wall_time_sec']):.3f}s"
        )

    try:
        for sweep in sweeps:
            settings = list(
                map(
                    int,
                    args.horizons if sweep == "horizon" else args.regions,
                )
            )
            for terrain_index, terrain_path in enumerate(terrain_files):
                context = load_terrain_context(
                    terrain_path, args.terrain_key, float(args.stance_width)
                )
                if len(context.stones) != 4:
                    raise ValueError(
                        f"{terrain_path.name}: expected base R=4, "
                        f"found {len(context.stones)}"
                    )
                ordered_settings = list(settings)
                if args.shuffle_conditions:
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                int(args.recovery_seed),
                                _terrain_sample_index(terrain_path),
                                1701 if sweep == "horizon" else 1702,
                            ]
                        )
                    )
                    rng.shuffle(ordered_settings)

                for condition_order, setting in enumerate(ordered_settings):
                    key = (sweep, int(setting), terrain_path.stem)
                    if key in completed_keys:
                        continue
                    derivation_t0 = time.perf_counter()
                    if sweep == "horizon":
                        stones = partition_intermediate_regions(context.stones, 4)
                        max_horizon = int(setting)
                    else:
                        stones = partition_intermediate_regions(
                            context.stones, int(setting)
                        )
                        max_horizon = int(args.region_horizon)
                    derivation_time = _seconds_since(derivation_t0)

                    row = run_case(
                        context,
                        stones,
                        max_horizon,
                        args,
                        sweep=sweep,
                        setting_value=int(setting),
                        run_timestamp=run_timestamp,
                        case_index=case_index,
                        condition_order=condition_order,
                        terrain_index=terrain_index,
                        derivation_time_sec=derivation_time,
                        experiment_config_hash=experiment_config_hash,
                    )
                    all_rows[sweep].append(row)
                    completed_keys.add(key)
                    case_index += 1
                    completed += 1
                    _write_all_outputs(
                        output_dir,
                        all_rows,
                        int(args.num_terrains),
                        sweeps,
                    )

                    success = int(row.get("recovery_feasible", 0) or 0)
                    result = "OK" if success else "FAIL"
                    actual = row.get("actual_horizon", "")
                    actual_text = (
                        f"{int(actual)}/{max_horizon}" if actual != "" else f"-/{max_horizon}"
                    )
                    print(
                        f"[{completed}/{total_cases}] {sweep} "
                        f"{terrain_path.name} setting={setting} {result} | "
                        f"nstep={actual_text} "
                        f"V={row.get('num_vertices', '')} "
                        f"E={row.get('num_edges', '')} "
                        f"total={float(row['total_time_sec']):.3f}s "
                        f"status={row['status']}",
                        flush=True,
                    )
    finally:
        _write_all_outputs(
            output_dir,
            all_rows,
            int(args.num_terrains),
            sweeps,
        )

    _validate_completed_matrix(
        all_rows,
        terrain_files,
        list(map(int, args.horizons)),
        list(map(int, args.regions)),
        sweeps,
    )
    print("=== completed and validated ===")
    for sweep in sweeps:
        print(f"{sweep} raw     : {_raw_path(output_dir, sweep)}")
        print(f"{sweep} summary : {_summary_path(output_dir, sweep)}")
    print(f"manifest        : {output_dir / 'Com_1_2_manifest.json'}")


if __name__ == "__main__":
    main()
