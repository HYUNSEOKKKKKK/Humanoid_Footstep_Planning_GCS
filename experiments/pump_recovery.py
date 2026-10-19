#!/usr/bin/env python3
"""Offline TORO Pump path-recovery feasibility experiment for Comment 1.5.

This executable deliberately imports neither the TORO runtime entry point nor
its UDP/visualization helpers.  It rebuilds the same DDR terrain and OptAPI
model directly, solves one relaxation per replay cycle, and evaluates every
recovery setting from that one relaxed edge vector.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.graph.centroid import compute_centroids_from_vertices
from src.model.gcs_model_ddr import build_gcs_model_ddr_TORO_optapi
from src.solver.path import path_to_y, relaxation_max
from src.solver.stochastic import _is_feasible_opt, _solve_fixed_once_ddr_optapi
from src.solver.utils import (
    get_relaxed_y_ddr_optapi,
    set_parameters_ddr_optapi,
    solve_relaxed_ddr_optapi,
)
from src.terrain_ddr import Terrain_GCS_DDR


DEFAULT_INITIAL_WINDOW = ["All", "UR", "UL", "DR", "C", "UR"]
DEFAULT_WINDOW_LEN = 10
KEY_TO_LABEL = {
    "0": "All", "1": "DL", "2": "D", "3": "DR", "4": "L",
    "5": "C", "6": "R", "7": "UL", "8": "U", "9": "UR",
}
VALID_LABELS = set(KEY_TO_LABEL.values())

# The immediately-after-real-demo source snapshot uses trials=5.  The wrapper
# counts the greedy candidate in trials, so this is greedy + at most 4
# probabilistic calls, not 5 probabilistic retries.
DEPLOYED_TRIALS = 5
DEPLOYED_PROBABILISTIC_BUDGET = DEPLOYED_TRIALS - 1

PER_CYCLE_FIELDS = [
    "run_id", "cycle_index", "label_window", "relax_success",
    "relax_solver_status", "relaxed_y_hash", "greedy_success",
    "probabilistic_candidate_count", "probabilistic_success_count",
    "probabilistic_single_trial_feasibility_rate_percent",
    "probabilistic_first_success_index", "ours_p5_success",
    "ours_p10_success", "ours_p20_success", "ours_p99_success",
    "deployed_trials", "deployed_probabilistic_budget",
    "deployed_setting_success", "error_message",
]

SUMMARY_FIELDS = [
    "num_total_cycles", "num_relax_success", "num_relax_failure",
    "greedy_success_count", "greedy_feasibility_rate_percent",
    "probabilistic_total_candidate_runs",
    "probabilistic_successful_candidate_runs",
    "probabilistic_single_trial_feasibility_rate_percent",
    "ours_p5_success_count", "ours_p5_feasibility_rate_percent",
    "ours_p10_success_count", "ours_p10_feasibility_rate_percent",
    "ours_p20_success_count", "ours_p20_feasibility_rate_percent",
    "ours_p99_success_count", "ours_p99_feasibility_rate_percent",
    "deployed_trials", "deployed_probabilistic_budget",
    "deployed_success_count", "deployed_feasibility_rate_percent",
]

AUDIT_FIELDS = ["setting", "current_code_value", "deployed_value", "evidence", "note"]


class PureProbabilisticPathFailure(RuntimeError):
    """A pure probabilistic DFS exhausted its search without a path."""


@dataclass
class ReplayRun:
    run_id: str
    windows: list[list[str]]
    source: str


@dataclass
class FixedAttempt:
    success: bool
    path: list[int] | None = None
    objective: float | None = None
    solution: dict[str, Any] | None = None
    status: str = ""
    error: str = ""


def relaxation_prob_pure(
    y: np.ndarray,
    edge: np.ndarray,
    vertex_number: int,
    *,
    temperature: float = 1.0,
    seed: int | None = None,
    max_backtracks: int = 20000,
) -> list[int]:
    """Copy of relaxation_prob's probabilistic DFS, without greedy fallback."""
    y = np.asarray(y, dtype=float).reshape(-1)
    edge = np.asarray(edge, dtype=int)
    n_edge = int(edge.shape[0])
    start, goal = 1, int(vertex_number)
    rng = np.random.default_rng(seed)
    eps = 1e-12
    temp = max(1e-6, float(temperature))

    adjacency: dict[int, list[tuple[int, int, float]]] = {
        i: [] for i in range(1, int(vertex_number) + 1)
    }
    for edge_index in range(n_edge):
        u, v = int(edge[edge_index, 0]), int(edge[edge_index, 1])
        adjacency[u].append((v, edge_index, float(y[edge_index])))

    def pick_next(
        u: int, banned_edges: set[int], banned_nodes: set[int]
    ) -> tuple[int, int, float] | None:
        candidates = [
            (v, edge_index, weight)
            for v, edge_index, weight in adjacency.get(u, [])
            if edge_index not in banned_edges and v not in banned_nodes
        ]
        if not candidates:
            return None
        weights = np.asarray([item[2] for item in candidates], dtype=float)
        # This is intentionally identical to src/solver/path.py:
        # weight_e = (y_e + 1e-12) ** (1 / temperature).
        weights = np.power(weights + eps, 1.0 / temp)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            probabilities = np.full(len(candidates), 1.0 / len(candidates))
        else:
            probabilities = weights / total
        return candidates[int(rng.choice(len(candidates), p=probabilities))]

    path_nodes = [start]
    visited_nodes = {start}
    used_edges: set[int] = set()
    tried_per_node: dict[int, set[int]] = {
        i: set() for i in range(1, int(vertex_number) + 1)
    }
    explored = 0

    def dfs(u: int) -> bool:
        nonlocal explored
        if u == goal:
            return True
        while True:
            explored += 1
            if explored > int(max_backtracks):
                return False
            choice = pick_next(
                u, tried_per_node[u] | used_edges, visited_nodes
            )
            if choice is None:
                return False
            v, edge_index, _ = choice
            tried_per_node[u].add(edge_index)
            path_nodes.append(v)
            visited_nodes.add(v)
            used_edges.add(edge_index)
            if dfs(v):
                return True
            path_nodes.pop()
            visited_nodes.remove(v)
            used_edges.remove(edge_index)

    if dfs(start):
        return path_nodes
    raise PureProbabilisticPathFailure(
        f"pure probabilistic DFS found no 1->{goal} path "
        f"within max_backtracks={max_backtracks}"
    )


def _config_audit_rows() -> list[dict[str, Any]]:
    opt_call = (
        "src/run_main_DDR_sliding_TORO.py:"
        "run_main_DDR_sliding_TORO_optapi (run_trials_select_best_ddr_optapi call)"
    )
    wrapper = "src/solver/stochastic.py:run_trials_select_best_ddr_optapi"
    path_code = "src/solver/path.py:relaxation_prob"
    demo_commit = (
        "git commit e80a6e7 (2025-11-07, 'Right after TORO pump real demo') "
        "and commit 01b110f (initial OptAPI version)"
    )
    log_note = (
        "Existing logs contain label windows and costs but do not serialize "
        "trials, greedy_first, early_stop, temperature, max_backtracks, or seed."
    )
    return [
        {
            "setting": "pump_recovery_api",
            "current_code_value": "Optimizer API",
            "deployed_value": "Fusion snapshot; initial OptAPI snapshot also present",
            "evidence": f"main_pump_TORO_optapi.py; {demo_commit}",
            "note": "The exact API used by every archived text log cannot be proven from the log itself.",
        },
        {
            "setting": "trials",
            "current_code_value": 100,
            "deployed_value": 5,
            "evidence": f"{opt_call}; {demo_commit}",
            "note": (
                "trials=5 is explicit in the immediately-after-real-demo source snapshot. "
                "Archived runtime logs do not independently record it; later OptAPI history changed 5->10->100."
            ),
        },
        {
            "setting": "trials_semantics",
            "current_code_value": "total candidate budget including greedy",
            "deployed_value": "total candidate budget including greedy",
            "evidence": f"{wrapper}: rest=trials; greedy branch decrements rest by one",
            "note": "Thus trials=5 means 1 greedy plus at most 4 probabilistic candidates.",
        },
        {
            "setting": "probabilistic_budget",
            "current_code_value": 99,
            "deployed_value": 4,
            "evidence": f"Derived from {wrapper} and trials values",
            "note": "This is the maximum after greedy when early_stop is enabled.",
        },
        {
            "setting": "greedy_first",
            "current_code_value": True,
            "deployed_value": True,
            "evidence": f"{opt_call}; {demo_commit}",
            "note": "Explicit argument at the Pump call site.",
        },
        {
            "setting": "early_stop",
            "current_code_value": True,
            "deployed_value": True,
            "evidence": f"{wrapper}: function default; Pump call omits override",
            "note": "A feasible greedy/fallback candidate is returned immediately.",
        },
        {
            "setting": "temperature",
            "current_code_value": 1.0,
            "deployed_value": 1.0,
            "evidence": f"{opt_call}; {demo_commit}",
            "note": "Explicit temp=1.0 at the Pump recovery call.",
        },
        {
            "setting": "max_backtracks",
            "current_code_value": 20000,
            "deployed_value": 20000,
            "evidence": f"{path_code}: default; {wrapper} does not override it",
            "note": "The new experiment exposes this as a CLI argument.",
        },
        {
            "setting": "probabilistic_seed",
            "current_code_value": "None (non-explicit RNG seed)",
            "deployed_value": "not logged",
            "evidence": f"{wrapper} calls {path_code} without seed; {log_note}",
            "note": "This experiment instead uses explicit SeedSequence-derived seeds for reproducibility.",
        },
        {
            "setting": "probabilistic_internal_greedy_fallback",
            "current_code_value": True,
            "deployed_value": True,
            "evidence": f"{path_code}: failed probabilistic DFS returns relaxation_max()",
            "note": "The comparison helper relaxation_prob_pure intentionally removes this fallback.",
        },
        {
            "setting": "archived_log_configuration",
            "current_code_value": "not applicable",
            "deployed_value": "not recoverable from text logs",
            "evidence": log_note,
            "note": "Only source/Git evidence supports the deployed-value audit.",
        },
        {
            "setting": "offline_replay_state",
            "current_code_value": "planner state reconstructed from selected fixed solution",
            "deployed_value": "robot-measured cycle poses not present in archived text logs",
            "evidence": "logs/*.txt and src/run_main_DDR_sliding_TORO.py:_pairs_from_path",
            "note": "Exact label windows can be replayed; exact measured foot-pose snapshots cannot be verified from these logs.",
        },
    ]


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp_path, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _is_label_list(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and all(isinstance(item, str) and item in VALID_LABELS for item in value)
    )


def _windows_from_keys(
    keys: Sequence[str],
    *,
    initial_window: Sequence[str] = DEFAULT_INITIAL_WINDOW,
    window_len: int = DEFAULT_WINDOW_LEN,
) -> list[list[str]]:
    window = deque([str(x) for x in initial_window], maxlen=int(window_len))
    windows = [list(window)]
    for raw_key in keys:
        key = str(raw_key)
        if key not in KEY_TO_LABEL:
            raise ValueError(f"unknown Pump key {key!r}")
        window.append(KEY_TO_LABEL[key])
        windows.append(list(window))
    return windows


def _run_from_mapping(data: dict[str, Any], fallback_id: str) -> ReplayRun:
    run_id = str(data.get("run_id", fallback_id))
    if "label_windows" in data:
        windows = data["label_windows"]
        if not isinstance(windows, list) or not all(_is_label_list(x) for x in windows):
            raise ValueError(f"{run_id}: label_windows must be a list of valid label lists")
        return ReplayRun(run_id, [list(x) for x in windows], "JSON label_windows")
    keys = data.get("sequence_keys", data.get("keys"))
    if keys is not None:
        initial = data.get("initial_window", DEFAULT_INITIAL_WINDOW)
        window_len = int(data.get("window_len", DEFAULT_WINDOW_LEN))
        return ReplayRun(
            run_id,
            _windows_from_keys(keys, initial_window=initial, window_len=window_len),
            "JSON sequence_keys offline replay",
        )
    raise ValueError(f"{run_id}: expected label_windows or sequence_keys")


def _parse_json_sequence(path: Path) -> list[ReplayRun]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, dict) and "runs" in data:
        return [
            _run_from_mapping(item, f"{path.stem}_{index:03d}")
            for index, item in enumerate(data["runs"])
        ]
    if isinstance(data, dict):
        return [_run_from_mapping(data, path.stem)]
    if isinstance(data, list) and data and all(_is_label_list(x) for x in data):
        return [ReplayRun(path.stem, [list(x) for x in data], "JSON label_windows")]
    if isinstance(data, list) and all(str(x) in KEY_TO_LABEL for x in data):
        return [ReplayRun(path.stem, _windows_from_keys(data), "JSON sequence_keys offline replay")]
    raise ValueError(f"unsupported sequence JSON structure in {path}")


def _parse_text_sequence(path: Path) -> list[ReplayRun]:
    text = path.read_text(encoding="utf-8", errors="replace")
    windows: list[list[str]] = []
    literal_values: list[Any] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("["):
            continue
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            continue
        literal_values.append(value)
        if _is_label_list(value):
            windows.append(list(value))
    if windows:
        return [ReplayRun(path.stem, windows, "archived log label-window replay")]

    for value in literal_values:
        if isinstance(value, list) and all(str(x) in KEY_TO_LABEL for x in value):
            return [
                ReplayRun(
                    path.stem,
                    _windows_from_keys([str(x) for x in value]),
                    "text sequence_keys offline replay",
                )
            ]

    tokens = [token for token in text.replace(",", " ").split() if token]
    if tokens and all(token in KEY_TO_LABEL for token in tokens):
        return [ReplayRun(path.stem, _windows_from_keys(tokens), "text sequence_keys offline replay")]
    raise ValueError(f"no Pump label windows or key sequence found in {path}")


def _load_replay_runs(sequence_file: Path | None, log_dir: Path | None) -> list[ReplayRun]:
    paths: list[Path]
    if sequence_file is not None:
        paths = [sequence_file]
    elif log_dir is not None:
        paths = sorted(path for path in log_dir.glob("*.txt") if path.is_file())
        if not paths:
            raise ValueError(f"no .txt logs found in {log_dir}")
    else:
        raise ValueError("--sequence-file or --log-dir is required unless --audit-only is used")

    runs: list[ReplayRun] = []
    used_ids: set[str] = set()
    for path in paths:
        parsed = _parse_json_sequence(path) if path.suffix.lower() == ".json" else _parse_text_sequence(path)
        for run in parsed:
            original = run.run_id
            suffix = 1
            while run.run_id in used_ids:
                run.run_id = f"{original}_{suffix}"
                suffix += 1
            used_ids.add(run.run_id)
            runs.append(run)
    if not runs:
        raise ValueError("no replay runs were loaded")
    return runs


def _build_ddr_terrain(label_window: Sequence[str], goal: str = "All") -> Terrain_GCS_DDR:
    labels = list(label_window)
    if not labels:
        raise ValueError("label_window must not be empty")
    # Mirrors src/run_main_DDR_sliding_TORO.py:_build_ddr_terrain without
    # importing the runtime module (which also imports UDP/animation code).
    if hasattr(Terrain_GCS_DDR, "from_window"):
        return Terrain_GCS_DDR.from_window(labels, goal=goal)
    return Terrain_GCS_DDR(*(labels + [goal]))


def _pack_pose5(position: Sequence[float], yaw: float, z: float = 0.0) -> np.ndarray:
    position = np.asarray(position, dtype=float).reshape(-1)
    return np.asarray([position[0], position[1], z, math.cos(yaw), math.sin(yaw)], dtype=float)


def _initial_poses() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.pi / 2.0
    start_l = _pack_pose5([-0.115, 0.0], yaw)
    start_r = _pack_pose5([0.115, 0.0], yaw)
    return start_l.copy(), start_r.copy(), start_l.copy(), start_r.copy()


def _normalize_replayed_pose5(pose: Sequence[float]) -> np.ndarray:
    """Mirror _pose_from_prev()+_pack_pose5() in the TORO runtime wrapper."""
    value = np.asarray(pose, dtype=float).reshape(-1)
    if value.size < 5:
        raise ValueError("replayed foot pose must have five components")
    yaw = math.atan2(float(value[4]), float(value[3]))
    return _pack_pose5(value[:2], yaw, z=0.0)


def _valid_path(path: Sequence[int] | None, edge: np.ndarray, vertex_number: int) -> bool:
    if not path or int(path[0]) != 1 or int(path[-1]) != int(vertex_number):
        return False
    edge_set = {(int(u), int(v)) for u, v in np.asarray(edge, dtype=int)}
    return all((int(path[i]), int(path[i + 1])) in edge_set for i in range(len(path) - 1))


def _fixed_attempt(
    task: Any,
    variables: dict[str, Any],
    indices: dict[str, Any],
    *,
    centroid_mat: np.ndarray,
    edge: np.ndarray,
    path: list[int],
    start_l: np.ndarray,
    start_r: np.ndarray,
    target_l: np.ndarray,
    target_r: np.ndarray,
    m_small: float = 1e-6,
) -> FixedAttempt:
    if not _valid_path(path, edge, int(indices["vertex_number"])):
        return FixedAttempt(False, path=path, error="invalid source-to-target path")
    try:
        fixed_y = path_to_y(edge, path, int(indices["n_edge"]))
        # _solve_fixed_once_ddr_optapi resets y_relax, M_scalar, geometry, and
        # all boundary poses before every solve.  Therefore a previous method's
        # fixed solve cannot leave the next candidate with stale fixed-path
        # parameters.  Warm-start behavior is identical for every method.
        info, objective, solution = _solve_fixed_once_ddr_optapi(
            task,
            variables,
            indices,
            centroid_mat=centroid_mat,
            y_relax=fixed_y,
            M_small=m_small,
            start_L=start_l,
            start_R=start_r,
            target_L=target_l,
            target_R=target_r,
            log_level=0,
        )
        status = f"{info.get('pstatus', '')}|{info.get('sstatus', '')}"
        success = (
            _is_feasible_opt(info.get("pstatus"), info.get("sstatus"))
            and objective is not None
            and np.isfinite(float(objective))
            and isinstance(solution, dict)
            and bool(solution)
        )
        return FixedAttempt(
            success,
            path=path,
            objective=float(objective) if objective is not None else None,
            solution=solution,
            status=status,
            error="" if success else "fixed solve not feasible or objective/solution invalid",
        )
    except Exception as exc:
        return FixedAttempt(False, path=path, error=f"fixed solve error: {type(exc).__name__}: {exc}")


def _relaxed_y_hash(y: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(y, dtype="<f8").reshape(-1))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _candidate_seed(base_seed: int, run_index: int, cycle_index: int, candidate_index: int) -> int:
    sequence = np.random.SeedSequence(
        [int(base_seed), int(run_index), int(cycle_index), int(candidate_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _pairs_from_path(
    solution: dict[str, Any], edge: np.ndarray, path: Sequence[int]
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(solution["y"], dtype=float).reshape(-1)
    zp_l = np.asarray(solution["zp_L"], dtype=float)
    zp_r = np.asarray(solution["zp_R"], dtype=float)
    edge = np.asarray(edge, dtype=int)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for u, v in zip(path[:-1], path[1:]):
        matches = np.where((edge[:, 0] == int(u)) & (edge[:, 1] == int(v)))[0]
        if not matches.size:
            continue
        edge_index = int(matches[0])
        denominator = max(1e-9, float(y[edge_index]))
        pairs.append((zp_l[edge_index] / denominator, zp_r[edge_index] / denominator))
    return pairs


def _as_success_with_budget(
    greedy_success: bool, probabilistic_results: Sequence[FixedAttempt], budget: int
) -> bool | float:
    # With early_stop=True, a feasible greedy candidate makes every larger
    # fallback budget successful even when a smoke test did not generate the
    # unused suffix of probabilistic candidates.
    if greedy_success:
        return True
    if len(probabilistic_results) < int(budget):
        return float("nan")
    return bool(any(item.success for item in probabilistic_results[:budget]))


def _failed_cycle_row(run_id: str, cycle_index: int, window: Sequence[str], error: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "cycle_index": cycle_index,
        "label_window": json.dumps(list(window), separators=(",", ":")),
        "relax_success": False,
        "relax_solver_status": "",
        "relaxed_y_hash": "",
        "greedy_success": False,
        "probabilistic_candidate_count": 0,
        "probabilistic_success_count": 0,
        "probabilistic_single_trial_feasibility_rate_percent": float("nan"),
        "probabilistic_first_success_index": float("nan"),
        "ours_p5_success": False,
        "ours_p10_success": False,
        "ours_p20_success": False,
        "ours_p99_success": False,
        "deployed_trials": DEPLOYED_TRIALS,
        "deployed_probabilistic_budget": DEPLOYED_PROBABILISTIC_BUDGET,
        "deployed_setting_success": False,
        "error_message": error,
    }


def _solve_cycle(
    *,
    run_id: str,
    run_index: int,
    cycle_index: int,
    label_window: Sequence[str],
    start_l: np.ndarray,
    start_r: np.ndarray,
    target_l: np.ndarray,
    target_r: np.ndarray,
    base_seed: int,
    temperature: float,
    max_backtracks: int,
    num_prob_candidates: int,
) -> tuple[dict[str, Any], FixedAttempt | None, np.ndarray | None, np.ndarray | None]:
    row = _failed_cycle_row(run_id, cycle_index, label_window, "")
    task = None
    try:
        terrain = _build_ddr_terrain(label_window, goal="All")
        vertices = terrain.get_vertices()
        edge = np.asarray(terrain.get_edges(), dtype=int)
        modes = terrain.get_mode_list()
        vertex_names = [stone.label for stone in terrain.stepping_stones]
        centroid_mat, _ = compute_centroids_from_vertices(vertices)

        task, variables, indices, parameters, _ = build_gcs_model_ddr_TORO_optapi(
            vertices,
            edge,
            math.pi / 2.0,
            modes,
            start_l,
            start_r,
            target_l,
            target_r,
            v_names=vertex_names,
        )
        vertex_count = centroid_mat.shape[1]
        geometry = np.empty(2 * vertex_count, dtype=float)
        geometry[0::2] = centroid_mat[0, :]
        geometry[1::2] = centroid_mat[1, :]
        relaxed_parameters = {
            "geometry": geometry,
            "start_L": start_l,
            "start_R": start_r,
            "target_L": target_l,
            "target_R": target_r,
            "M_scalar": 1.0,
            "y_relax": np.zeros(int(indices["n_edge"]), dtype=float),
        }
        set_parameters_ddr_optapi(task, variables, indices, parameters, relaxed_parameters)
        relax_info, relax_objective = solve_relaxed_ddr_optapi(
            task, variables, indices, parameters, M_large=1e3, log_level=0
        )
        relax_status = f"{relax_info.get('pstatus', '')}|{relax_info.get('sstatus', '')}"
        row["relax_solver_status"] = relax_status
        relax_success = (
            _is_feasible_opt(relax_info.get("pstatus"), relax_info.get("sstatus"))
            and relax_objective is not None
            and np.isfinite(float(relax_objective))
        )
        if not relax_success:
            row["error_message"] = "relaxed solve failed or returned a nonfinite objective"
            return row, None, None, None

        relaxed_y = get_relaxed_y_ddr_optapi(task, variables)
        if relaxed_y.size != int(indices["n_edge"]) or not np.all(np.isfinite(relaxed_y)):
            row["error_message"] = "relaxed edge vector is missing or nonfinite"
            return row, None, None, None
        row["relax_success"] = True
        row["relaxed_y_hash"] = _relaxed_y_hash(relaxed_y)

        greedy_attempt: FixedAttempt
        try:
            greedy_path = relaxation_max(
                relaxed_y,
                edge,
                int(indices["vertex_number"]),
                max_backtracks=int(max_backtracks),
            )
            greedy_attempt = _fixed_attempt(
                task,
                variables,
                indices,
                centroid_mat=centroid_mat,
                edge=edge,
                path=greedy_path,
                start_l=start_l,
                start_r=start_r,
                target_l=target_l,
                target_r=target_r,
            )
        except Exception as exc:
            greedy_attempt = FixedAttempt(
                False, error=f"greedy path error: {type(exc).__name__}: {exc}"
            )
        row["greedy_success"] = greedy_attempt.success

        # The paths and fixed results are generated once in this order.  Every
        # Ours budget below is derived from a prefix of this exact same list.
        probabilistic_results: list[FixedAttempt] = []
        candidate_errors: list[str] = []
        for candidate_index in range(int(num_prob_candidates)):
            seed = _candidate_seed(base_seed, run_index, cycle_index, candidate_index)
            try:
                path = relaxation_prob_pure(
                    relaxed_y,
                    edge,
                    int(indices["vertex_number"]),
                    temperature=float(temperature),
                    seed=seed,
                    max_backtracks=int(max_backtracks),
                )
                attempt = _fixed_attempt(
                    task,
                    variables,
                    indices,
                    centroid_mat=centroid_mat,
                    edge=edge,
                    path=path,
                    start_l=start_l,
                    start_r=start_r,
                    target_l=target_l,
                    target_r=target_r,
                )
            except Exception as exc:
                attempt = FixedAttempt(
                    False,
                    error=f"probabilistic path error: {type(exc).__name__}: {exc}",
                )
            probabilistic_results.append(attempt)
            if attempt.error:
                candidate_errors.append(f"P{candidate_index + 1}: {attempt.error}")

        success_indices = [
            index + 1 for index, attempt in enumerate(probabilistic_results) if attempt.success
        ]
        success_count = len(success_indices)
        candidate_count = len(probabilistic_results)
        row["probabilistic_candidate_count"] = candidate_count
        row["probabilistic_success_count"] = success_count
        row["probabilistic_single_trial_feasibility_rate_percent"] = (
            100.0 * success_count / candidate_count if candidate_count else float("nan")
        )
        row["probabilistic_first_success_index"] = (
            success_indices[0] if success_indices else float("nan")
        )
        for budget in (5, 10, 20, 99):
            row[f"ours_p{budget}_success"] = _as_success_with_budget(
                greedy_attempt.success, probabilistic_results, budget
            )

        deployed_success = _as_success_with_budget(
            greedy_attempt.success,
            probabilistic_results,
            DEPLOYED_PROBABILISTIC_BUDGET,
        )
        row["deployed_setting_success"] = deployed_success

        selected: FixedAttempt | None = None
        if greedy_attempt.success:
            selected = greedy_attempt
        else:
            selected = next(
                (
                    item
                    for item in probabilistic_results[:DEPLOYED_PROBABILISTIC_BUDGET]
                    if item.success
                ),
                None,
            )
        if selected is None:
            details = "; ".join(candidate_errors[:3])
            row["error_message"] = (
                "deployed trials=5 recovery failed; subsequent offline cycle pose is unavailable"
                + (f"; {details}" if details else "")
            )
            return row, None, None, None

        pairs = _pairs_from_path(selected.solution or {}, edge, selected.path or [])
        if not pairs:
            row["error_message"] = "selected fixed solution has no next-state pair for offline replay"
            return row, None, None, None
        next_l, next_r = pairs[0]
        if candidate_errors:
            row["error_message"] = "; ".join(candidate_errors[:3])
        return row, selected, np.asarray(next_l, dtype=float), np.asarray(next_r, dtype=float)
    except Exception as exc:
        row["error_message"] = f"cycle error: {type(exc).__name__}: {exc}"
        return row, None, None, None
    finally:
        if task is not None:
            try:
                task.dispose()
            except Exception:
                pass


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _percent(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else float("nan")


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    relax_rows = [row for row in rows if _parse_bool(row.get("relax_success"))]
    denominator = len(relax_rows)
    summary: dict[str, Any] = {
        "num_total_cycles": total,
        "num_relax_success": denominator,
        "num_relax_failure": total - denominator,
        "deployed_trials": DEPLOYED_TRIALS,
        "deployed_probabilistic_budget": DEPLOYED_PROBABILISTIC_BUDGET,
    }

    greedy_count = sum(_parse_bool(row.get("greedy_success")) for row in relax_rows)
    summary["greedy_success_count"] = greedy_count
    summary["greedy_feasibility_rate_percent"] = _percent(greedy_count, denominator)

    prob_total = sum(int(_parse_float(row.get("probabilistic_candidate_count"))) for row in relax_rows)
    prob_success = sum(int(_parse_float(row.get("probabilistic_success_count"))) for row in relax_rows)
    summary["probabilistic_total_candidate_runs"] = prob_total
    summary["probabilistic_successful_candidate_runs"] = prob_success
    summary["probabilistic_single_trial_feasibility_rate_percent"] = _percent(
        prob_success, prob_total
    )

    for budget in (5, 10, 20, 99):
        values = [_parse_float(row.get(f"ours_p{budget}_success")) for row in relax_rows]
        fully_available = len(values) == denominator and all(np.isfinite(value) for value in values)
        if fully_available:
            count: int | float = sum(bool(value) for value in values)
            rate = _percent(float(count), denominator)
        else:
            count = float("nan")
            rate = float("nan")
        summary[f"ours_p{budget}_success_count"] = count
        summary[f"ours_p{budget}_feasibility_rate_percent"] = rate

    deployed_values = [_parse_float(row.get("deployed_setting_success")) for row in relax_rows]
    if len(deployed_values) == denominator and all(np.isfinite(value) for value in deployed_values):
        deployed_count: int | float = sum(bool(value) for value in deployed_values)
        deployed_rate = _percent(float(deployed_count), denominator)
    else:
        deployed_count = float("nan")
        deployed_rate = float("nan")
    summary["deployed_success_count"] = deployed_count
    summary["deployed_feasibility_rate_percent"] = deployed_rate
    return summary


def _validate_summary(rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> None:
    rebuilt = _summarize(rows)
    for field in SUMMARY_FIELDS:
        left, right = rebuilt[field], summary[field]
        if isinstance(left, float) and math.isnan(left):
            if not (isinstance(right, float) and math.isnan(right)):
                raise AssertionError(f"summary mismatch in {field}")
        elif left != right:
            raise AssertionError(f"summary mismatch in {field}: {left!r} != {right!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline TORO Pump recovery-feasibility CSV generator"
    )
    parser.add_argument("--sequence-file", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/Com_1_5_TORO"))
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-backtracks", type=int, default=20000)
    parser.add_argument("--num-prob-candidates", type=int, default=99)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run only the first two cycles of the first loaded run",
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be positive")
    if args.max_backtracks <= 0:
        raise SystemExit("--max-backtracks must be positive")
    if args.num_prob_candidates < DEPLOYED_PROBABILISTIC_BUDGET:
        raise SystemExit(
            f"--num-prob-candidates must be at least {DEPLOYED_PROBABILISTIC_BUDGET} "
            "to evaluate the audited trials=5 setting"
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "config_audit.csv"
    per_cycle_path = output_dir / "per_cycle_results.csv"
    summary_path = output_dir / "recovery_summary.csv"
    _atomic_write_csv(audit_path, AUDIT_FIELDS, _config_audit_rows())
    if args.audit_only:
        print(f"Wrote {audit_path}")
        return 0

    if not args.resume and not args.overwrite and (per_cycle_path.exists() or summary_path.exists()):
        raise SystemExit(
            f"result CSV already exists under {output_dir}; use --resume or --overwrite"
        )

    runs = _load_replay_runs(args.sequence_file, args.log_dir)
    if args.smoke_test:
        runs = [ReplayRun(runs[0].run_id, runs[0].windows[:2], runs[0].source)]

    existing = _read_csv(per_cycle_path) if args.resume else []
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in existing:
        try:
            rows_by_key[(str(row["run_id"]), int(row["cycle_index"]))] = row
        except (KeyError, TypeError, ValueError):
            continue

    total_cycles = sum(len(run.windows) for run in runs)
    completed = 0
    for run_index, run in enumerate(runs):
        # A partial run must be replayed from cycle zero to reconstruct the same
        # planner foot state.  Existing keys are replaced, never duplicated.
        for key in [key for key in rows_by_key if key[0] == run.run_id]:
            del rows_by_key[key]

        start_l, start_r, target_l, target_r = _initial_poses()
        state_available = True
        for cycle_index, window in enumerate(run.windows):
            if not state_available:
                break
            row, selected, next_l, next_r = _solve_cycle(
                run_id=run.run_id,
                run_index=run_index,
                cycle_index=cycle_index,
                label_window=window,
                start_l=start_l,
                start_r=start_r,
                target_l=target_l,
                target_r=target_r,
                base_seed=args.base_seed,
                temperature=args.temperature,
                max_backtracks=args.max_backtracks,
                num_prob_candidates=args.num_prob_candidates,
            )
            rows_by_key[(run.run_id, cycle_index)] = row
            ordered_rows = [rows_by_key[key] for key in sorted(rows_by_key)]
            _atomic_write_csv(per_cycle_path, PER_CYCLE_FIELDS, ordered_rows)
            summary = _summarize(ordered_rows)
            _validate_summary(ordered_rows, summary)
            _atomic_write_csv(summary_path, SUMMARY_FIELDS, [summary])
            completed += 1
            print(
                f"[{completed}/{total_cycles}] run={run.run_id} cycle={cycle_index} "
                f"relax={row['relax_success']} greedy={row['greedy_success']}"
            )
            if selected is None or next_l is None or next_r is None:
                state_available = False
                if cycle_index + 1 < len(run.windows):
                    raise RuntimeError(
                        f"offline replay stopped after {run.run_id} cycle {cycle_index}: "
                        "the next measured foot-pose snapshot is absent from the archived log, "
                        "and the audited deployed recovery did not provide a state transition"
                    )
            else:
                start_l = _normalize_replayed_pose5(next_l)
                start_r = _normalize_replayed_pose5(next_r)

    ordered_rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    final_summary = _summarize(ordered_rows)
    _validate_summary(ordered_rows, final_summary)
    _atomic_write_csv(per_cycle_path, PER_CYCLE_FIELDS, ordered_rows)
    _atomic_write_csv(summary_path, SUMMARY_FIELDS, [final_summary])
    print(f"Wrote {per_cycle_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
