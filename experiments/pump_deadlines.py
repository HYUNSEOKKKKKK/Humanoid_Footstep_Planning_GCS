#!/usr/bin/env python3
"""Pump/TORO deadline comparison for Reviewer Comment 10.2.

This file is intentionally standalone.  It does not call the interactive
``main_pump_TORO_optapi.py`` entry point, so an offline benchmark never opens a
GUI or sends UDP packets to TORO.

Two commands are provided:

``run``
    Replay one or more Pump symbol sequences with Proposed, Graph-based, and
    Optimization-based planners.  Every planner receives the same symbol
    window and canonical start poses.  The first-swing side selected by the
    deployed Proposed rollout is treated as the shared support-phase input for
    both baselines.  The uncensored time to the first validated feasible plan
    is recorded once, then all deadline statistics are computed post hoc.

``summarize``
    Read arbitrary per-cycle CSV files, automatically detect columns
    equivalent to sequence_id, cycle_id, planner, ok, and time_sec, validate
    paired coverage, and create the requested detailed/compact/LaTeX files.

Examples
--------

Run a two-cycle smoke test with the Proposed paper's Pump protocol::

    python -m experiments.pump_deadlines run --paper-legacy --smoke-test --overwrite

Run the complete paper-protocol replay.  This is also the default when no
sequence source is supplied (the observation limit is not a planning deadline;
it only bounds how long late feasible solutions are observed)::

    python -m experiments.pump_deadlines run --paper-legacy --observation-limit-sec 10 --overwrite

Summarize one or more existing raw CSV files::

    python -m experiments.pump_deadlines summarize raw_a.csv raw_b.csv raw_c.csv

The three requested files are written under ``outputs/Com10_2`` by default.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import heapq
import itertools
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np


PLANNER_ORDER = ("Proposed", "Graph-based", "Optimization-based")
DEFAULT_DEADLINES_SEC = (1.2, 1.0, 0.75, 0.5, 0.3)
DEFAULT_OUTPUT_DIR = Path("outputs/Com10_2")
DEFAULT_INITIAL_WINDOW = ("All", "UR", "UL", "DR", "C", "UR")
DEFAULT_WINDOW_LEN = 6
DEFAULT_OBSERVATION_LIMIT_SEC = 10.0
DEFAULT_RECOVERY_CANDIDATES = 100
PAPER_LEGACY_PROTOCOL_ID = "proposed_paper_pump_fixed6_53keys"
# Recovered from paper-run commit 4a1cc06a6883062f2dd26e27c3796a7410bcc3fe.
# The compact JSON serialization of all 54 windows has SHA-256:
# da2f0b13948b395ede3fadded994e187a25f7be8ce895a641d0753b4bdd8d2c4
PAPER_LEGACY_SEQUENCE_KEYS = (
    "1", "9", "7", "5", "7", "9", "5", "3", "1", "9", "7", "5",
    "1", "3", "5", "9", "1", "5", "7", "3", "5", "9", "7", "3",
    "1", "9", "5", "3", "1", "3", "7", "9", "7", "9", "7", "9",
    "7", "3", "1", "3", "1", "5", "7", "9", "7", "3", "5", "3",
    "0", "0", "0", "0", "0",
)

KEY_TO_LABEL = {
    "0": "All",
    "1": "DL",
    "2": "D",
    "3": "DR",
    "4": "L",
    "5": "C",
    "6": "R",
    "7": "UL",
    "8": "U",
    "9": "UR",
}

RAW_FIELDS = (
    "sequence_id",
    "cycle_id",
    "scenario_hash",
    "protocol",
    "planner",
    "ok",
    "time_sec",
    "status",
    "stop_reason",
    "label_window",
    "start_left_pose",
    "start_right_pose",
    "first_mover",
    "observation_limit_sec",
    "setup_time_sec",
    "solver_time_sec",
    "num_candidates",
    "details",
)

DETAILED_FIELDS = (
    "planner",
    "deadline_sec",
    "total_replanning_cycles",
    "feasible_count",
    "infeasible_count",
    "feasibility_rate_percent",
    "on_time_success_count",
    "on_time_success_rate_percent",
    "missed_deadline_count",
    "late_but_feasible_count",
    "sequence_count",
    "complete_sequence_success_count",
    "complete_sequence_success_rate_percent",
)

COMPACT_FIELDS = (
    "planner",
    "deadline_sec",
    "total_replanning_cycles",
    "feasibility_rate_percent",
    "on_time_success_rate_percent",
    "missed_deadline_count",
    "late_but_feasible_count",
    "complete_sequence_success_rate_percent",
)


class Com10Error(RuntimeError):
    """Base class for actionable experiment errors."""


class SchemaDetectionError(Com10Error):
    """Raised when a raw CSV does not contain enough per-cycle information."""


class DataValidationError(Com10Error):
    """Raised when detected data are malformed or not paired."""


@dataclass(frozen=True)
class NormalizedCycle:
    sequence_id: str
    cycle_id: str
    planner: str
    ok: bool
    time_sec: float
    source_file: str
    source_row: int
    scenario_hash: str = ""
    symbol_window: str = ""


@dataclass
class ReplaySequence:
    sequence_id: str
    windows: list[list[str]]
    source: str


@dataclass
class CycleSolveResult:
    planner: str
    ok: bool
    time_sec: float = math.nan
    status: str = ""
    stop_reason: str = ""
    setup_time_sec: float = math.nan
    solver_time_sec: float = math.nan
    num_candidates: int = 0
    details: str = ""
    next_left: Optional[np.ndarray] = None
    next_right: Optional[np.ndarray] = None
    first_mover: str = ""


@dataclass
class ColumnDetection:
    sequence: Optional[str] = None
    cycle: Optional[str] = None
    planner: Optional[str] = None
    ok: Optional[str] = None
    time: Optional[str] = None
    time_scale: float = 1.0
    inferred_planner: Optional[str] = None
    scenario_hash: Optional[str] = None
    symbol_window: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence,
            "cycle_id": self.cycle,
            "planner": self.planner or (
                f"<inferred:{self.inferred_planner}>"
                if self.inferred_planner
                else None
            ),
            "ok": self.ok,
            "first_feasible_time": self.time,
            "time_scale_to_seconds": self.time_scale,
            "scenario_hash": self.scenario_hash,
            "symbol_window": self.symbol_window,
        }


def _canonical_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _natural_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
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


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise SchemaDetectionError(f"CSV file does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(stream, dialect=dialect)
        headers = [str(item) for item in (reader.fieldnames or [])]
        rows = [dict(row) for row in reader]
    if not headers:
        raise SchemaDetectionError(f"CSV has no header: {path}")
    if not rows:
        raise SchemaDetectionError(
            f"CSV has a header but no per-cycle rows: {path}; "
            f"discovered columns={headers}"
        )
    return headers, rows


SEQUENCE_ALIASES = (
    "sequence_id",
    "sequence",
    "seq_id",
    "seq",
    "run_id",
    "episode_id",
    "episode",
    "trial_id",
    "run_json",
)
CYCLE_ALIASES = (
    "cycle_id",
    "cycle_index",
    "cycle",
    "replanning_cycle",
    "replan_cycle",
    "tick_id",
    "tick",
    "frame_id",
    "step_id",
)
PLANNER_ALIASES = (
    "planner",
    "planner_name",
    "method",
    "method_name",
    "algorithm",
    "approach",
)
OK_ALIASES = (
    "ok",
    "success",
    "is_success",
    "feasible",
    "is_feasible",
    "solved",
    "plan_found",
)
TIME_SECOND_ALIASES = (
    "first_feasible_time_sec",
    "time_to_first_feasible_sec",
    "first_solution_time_sec",
    "first_feasible_sec",
)
TIME_CANONICAL_ALIASES = (
    "time_sec",
    "time_s",
)
TIME_GENERIC_SECOND_ALIASES = (
    "planning_time_sec",
    "elapsed_sec",
    "runtime_sec",
    "runtime_s",
)
TIME_MILLISECOND_ALIASES = (
    "first_feasible_time_ms",
    "time_to_first_feasible_ms",
    "first_solution_time_ms",
    "planning_time_ms",
    "elapsed_ms",
    "runtime_ms",
    "time_ms",
)
SCENARIO_HASH_ALIASES = (
    "scenario_hash",
    "problem_hash",
    "task_hash",
    "cycle_hash",
)
SYMBOL_WINDOW_ALIASES = (
    "label_window",
    "symbol_window",
    "symbols",
    "pump_window",
)


def _find_alias(
    headers: Sequence[str],
    aliases: Sequence[str],
    *,
    role: str,
) -> Optional[str]:
    canonical_to_original: dict[str, list[str]] = defaultdict(list)
    for header in headers:
        canonical_to_original[_canonical_name(header)].append(header)
    collisions = {
        canonical: originals
        for canonical, originals in canonical_to_original.items()
        if len(originals) > 1
    }
    if collisions:
        raise SchemaDetectionError(
            f"normalized CSV header collision while detecting {role}: "
            + json.dumps(collisions, ensure_ascii=False)
        )
    matches: list[str] = []
    for alias in aliases:
        match = canonical_to_original.get(_canonical_name(alias), [])
        if match and match[0] not in matches:
            matches.append(match[0])
    if len(matches) > 1:
        raise SchemaDetectionError(
            f"ambiguous columns for {role}: {matches}; keep one canonical "
            "column or rename the non-input diagnostics"
        )
    return matches[0] if matches else None


def _normalize_planner(value: str) -> str:
    canonical = _canonical_name(value)
    if canonical in {
        "proposed",
        "ours",
        "ourmethod",
        "gcs",
        "gcsbased",
        "torooptapi",
    }:
        return "Proposed"
    if canonical in {
        "graph",
        "graphbased",
        "astar",
        "a",
        "ihmc",
        "ihmcastar",
    }:
        return "Graph-based"
    if canonical in {
        "optimization",
        "optimizationbased",
        "optimisationbased",
        "mip",
        "miqp",
        "miqcqp",
        "mico",
    }:
        return "Optimization-based"
    raise DataValidationError(
        f"unknown planner value {value!r}; expected one of "
        f"{', '.join(PLANNER_ORDER)} (common Ours/A*/MIP aliases are accepted)"
    )


def _infer_planner_from_path(path: Path) -> Optional[str]:
    canonical = _canonical_name(path.stem)
    candidates: list[str] = []
    if any(token in canonical for token in ("proposed", "ours", "gcs")):
        candidates.append("Proposed")
    if any(
        token in canonical
        for token in ("graphbased", "graph", "astar", "ihmc")
    ):
        candidates.append("Graph-based")
    if any(
        token in canonical
        for token in (
            "optimizationbased",
            "optimisationbased",
            "optimization",
            "optimisation",
            "miqcqp",
            "mip",
        )
    ):
        candidates.append("Optimization-based")
    return candidates[0] if len(set(candidates)) == 1 else None


def _detect_columns(
    path: Path,
    headers: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> ColumnDetection:
    detected = ColumnDetection(
        sequence=_find_alias(headers, SEQUENCE_ALIASES, role="sequence_id"),
        cycle=_find_alias(headers, CYCLE_ALIASES, role="cycle_id"),
        planner=_find_alias(headers, PLANNER_ALIASES, role="planner"),
        ok=_find_alias(headers, OK_ALIASES, role="ok/feasible"),
        scenario_hash=_find_alias(
            headers, SCENARIO_HASH_ALIASES, role="scenario_hash"
        ),
        symbol_window=_find_alias(
            headers, SYMBOL_WINDOW_ALIASES, role="symbol_window"
        ),
    )
    # Prefer an explicitly named first-feasible timestamp.  If unavailable,
    # prefer the requested canonical time_sec over generic final-runtime names.
    detected.time = _find_alias(
        headers, TIME_SECOND_ALIASES, role="explicit first-feasible seconds"
    )
    if detected.time is None:
        detected.time = _find_alias(
            headers, TIME_CANONICAL_ALIASES, role="canonical time seconds"
        )
    if detected.time is None:
        detected.time = _find_alias(
            headers,
            TIME_GENERIC_SECOND_ALIASES,
            role="generic elapsed/runtime seconds",
        )
    if detected.time is None:
        detected.time = _find_alias(
            headers,
            TIME_MILLISECOND_ALIASES,
            role="first-feasible milliseconds",
        )
        if detected.time is not None:
            detected.time_scale = 0.001

    if detected.planner is None:
        detected.inferred_planner = _infer_planner_from_path(path)

    missing: list[str] = []
    if detected.sequence is None:
        missing.append(
            "sequence_id (aliases: sequence, seq_id, run_id, episode_id, run_json)"
        )
    if detected.cycle is None:
        missing.append(
            "cycle_id (aliases: cycle_index, replanning_cycle, tick, frame_id)"
        )
    if detected.planner is None and detected.inferred_planner is None:
        missing.append(
            "planner (aliases: method, algorithm), or an unambiguous "
            "Proposed/Graph/AStar/MIP filename"
        )
    if detected.ok is None:
        missing.append(
            "feasibility/success flag (aliases: ok, success, feasible, solved)"
        )
    if detected.time is None:
        missing.append(
            "time to first feasible solution (seconds or milliseconds; aliases: "
            "time_sec, runtime_s, elapsed_sec, time_to_first_feasible_ms)"
        )

    if missing:
        diagnostic = {
            "file": str(path),
            "discovered_columns": list(headers),
            "normalized_columns": {
                header: _canonical_name(header) for header in headers
            },
            "detected_mapping": detected.as_dict(),
            "missing_information": missing,
        }
        raise SchemaDetectionError(
            "insufficient per-cycle CSV schema:\n"
            + json.dumps(diagnostic, indent=2, ensure_ascii=False)
        )
    return detected


TRUE_VALUES = {
    "1",
    "true",
    "t",
    "yes",
    "y",
    "ok",
    "success",
    "succeeded",
    "feasible",
    "optimal",
    "reached",
    "solved",
}
FALSE_VALUES = {
    "0",
    "false",
    "f",
    "no",
    "n",
    "fail",
    "failed",
    "failure",
    "infeasible",
    "timeout",
    "timedout",
}


def _parse_bool(value: Any, *, context: str) -> bool:
    canonical = _canonical_name(str(value))
    if canonical in {_canonical_name(item) for item in TRUE_VALUES}:
        return True
    if canonical in {_canonical_name(item) for item in FALSE_VALUES}:
        return False
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        numeric = math.nan
    if numeric == 1.0:
        return True
    if numeric == 0.0:
        return False
    raise DataValidationError(
        f"{context}: cannot parse success/feasibility value {value!r}"
    )


def _parse_time(value: Any, *, scale: float, context: str) -> float:
    text = str(value).strip()
    if text == "":
        return math.nan
    try:
        result = float(text) * float(scale)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"{context}: cannot parse first-feasible time {value!r}"
        ) from exc
    if math.isfinite(result) and result < 0.0:
        raise DataValidationError(
            f"{context}: first-feasible time must be nonnegative, got {result}"
        )
    return result


def _canonical_symbol_window(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
    if isinstance(parsed, (list, tuple)):
        return json.dumps(
            [str(item) for item in parsed],
            separators=(",", ":"),
            ensure_ascii=False,
        )
    return re.sub(r"\s+", " ", text)


def load_normalized_cycles(paths: Sequence[Path]) -> list[NormalizedCycle]:
    normalized: list[NormalizedCycle] = []
    for path in paths:
        headers, rows = _read_csv_rows(path)
        detected = _detect_columns(path, headers, rows)
        for row_number, row in enumerate(rows, start=2):
            context = f"{path}:{row_number}"
            sequence_id = str(row.get(detected.sequence or "", "")).strip()
            cycle_id = str(row.get(detected.cycle or "", "")).strip()
            planner_text = (
                str(row.get(detected.planner, "")).strip()
                if detected.planner
                else str(detected.inferred_planner or "")
            )
            if not sequence_id:
                raise DataValidationError(f"{context}: empty sequence identifier")
            if not cycle_id:
                raise DataValidationError(f"{context}: empty cycle identifier")
            planner = _normalize_planner(planner_text)
            ok = _parse_bool(
                row.get(detected.ok or "", ""),
                context=f"{context} column {detected.ok!r}",
            )
            first_time = _parse_time(
                row.get(detected.time or "", ""),
                scale=detected.time_scale,
                context=f"{context} column {detected.time!r}",
            )
            if ok and not math.isfinite(first_time):
                raise DataValidationError(
                    f"{context}: ok=true requires a finite first-feasible time; "
                    f"column {detected.time!r} contained "
                    f"{row.get(detected.time or '', '')!r}"
                )
            normalized.append(
                NormalizedCycle(
                    sequence_id=sequence_id,
                    cycle_id=cycle_id,
                    planner=planner,
                    ok=ok,
                    time_sec=first_time,
                    source_file=str(path),
                    source_row=row_number,
                    scenario_hash=(
                        str(row.get(detected.scenario_hash, "")).strip()
                        if detected.scenario_hash
                        else ""
                    ),
                    symbol_window=(
                        _canonical_symbol_window(
                            row.get(detected.symbol_window, "")
                        )
                        if detected.symbol_window
                        else ""
                    ),
                )
            )
    if not normalized:
        raise DataValidationError("no per-cycle rows were loaded")
    _validate_cycle_data(normalized)
    return sorted(
        normalized,
        key=lambda row: (
            PLANNER_ORDER.index(row.planner),
            _natural_key(row.sequence_id),
            _natural_key(row.cycle_id),
        ),
    )


def _validate_cycle_data(rows: Sequence[NormalizedCycle]) -> None:
    locations: dict[tuple[str, str, str], NormalizedCycle] = {}
    coverage: dict[str, set[tuple[str, str]]] = {
        planner: set() for planner in PLANNER_ORDER
    }
    for row in rows:
        key = (row.sequence_id, row.cycle_id, row.planner)
        previous = locations.get(key)
        if previous is not None:
            raise DataValidationError(
                "duplicate sequence/cycle/planner row: "
                f"{key}; first={previous.source_file}:{previous.source_row}, "
                f"again={row.source_file}:{row.source_row}"
            )
        locations[key] = row
        coverage[row.planner].add((row.sequence_id, row.cycle_id))

    present = {row.planner for row in rows}
    missing_planners = [item for item in PLANNER_ORDER if item not in present]
    if missing_planners:
        raise DataValidationError(
            "the comparison requires all three planners; missing "
            + ", ".join(missing_planners)
        )

    reference = coverage["Proposed"]
    mismatches: dict[str, dict[str, list[list[str]]]] = {}
    for planner in PLANNER_ORDER[1:]:
        missing = sorted(reference - coverage[planner], key=lambda x: tuple(map(_natural_key, x)))
        extra = sorted(coverage[planner] - reference, key=lambda x: tuple(map(_natural_key, x)))
        if missing or extra:
            mismatches[planner] = {
                "missing_vs_Proposed_first_20": [list(item) for item in missing[:20]],
                "extra_vs_Proposed_first_20": [list(item) for item in extra[:20]],
            }
    if mismatches:
        raise DataValidationError(
            "planner rows are not paired on the same sequence_id/cycle_id set:\n"
            + json.dumps(mismatches, indent=2, ensure_ascii=False)
        )

    grouped: dict[tuple[str, str], list[NormalizedCycle]] = defaultdict(list)
    for row in rows:
        grouped[(row.sequence_id, row.cycle_id)].append(row)
    identity_mismatches: dict[str, Any] = {}
    for key, paired_rows in grouped.items():
        hashes = {row.scenario_hash for row in paired_rows if row.scenario_hash}
        windows = {row.symbol_window for row in paired_rows if row.symbol_window}
        hash_presence = sum(bool(row.scenario_hash) for row in paired_rows)
        window_presence = sum(bool(row.symbol_window) for row in paired_rows)
        issues: dict[str, Any] = {}
        if len(hashes) > 1:
            issues["scenario_hash_values"] = sorted(hashes)
        if 0 < hash_presence < len(paired_rows):
            issues["scenario_hash_presence"] = (
                f"{hash_presence}/{len(paired_rows)} planners"
            )
        if len(windows) > 1:
            issues["symbol_window_values"] = sorted(windows)
        if 0 < window_presence < len(paired_rows):
            issues["symbol_window_presence"] = (
                f"{window_presence}/{len(paired_rows)} planners"
            )
        if issues:
            identity_mismatches[f"{key[0]}::{key[1]}"] = issues
    if identity_mismatches:
        sample = dict(list(identity_mismatches.items())[:20])
        raise DataValidationError(
            "paired IDs refer to different or only partially identified Pump "
            "cycle inputs:\n"
            + json.dumps(sample, indent=2, ensure_ascii=False)
        )

    cycle_ids_by_sequence: dict[str, set[str]] = defaultdict(set)
    for sequence_id, cycle_id in reference:
        cycle_ids_by_sequence[sequence_id].add(cycle_id)
    if not cycle_ids_by_sequence:
        raise DataValidationError("no paired sequence/cycle keys remain")


def _percent(numerator: int, denominator: int) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else math.nan


def summarize_deadlines(
    rows: Sequence[NormalizedCycle],
    deadlines_sec: Sequence[float] = DEFAULT_DEADLINES_SEC,
) -> list[dict[str, Any]]:
    _validate_cycle_data(rows)
    deadlines = [float(value) for value in deadlines_sec]
    if not deadlines or any(
        not math.isfinite(value) or value <= 0.0 for value in deadlines
    ):
        raise DataValidationError("every planning deadline must be finite and positive")
    if len(set(deadlines)) != len(deadlines):
        raise DataValidationError("planning deadlines must be unique")

    detailed: list[dict[str, Any]] = []
    for planner in PLANNER_ORDER:
        planner_rows = [row for row in rows if row.planner == planner]
        total = len(planner_rows)
        feasible_count = sum(row.ok for row in planner_rows)
        sequence_ids = sorted(
            {row.sequence_id for row in planner_rows}, key=_natural_key
        )
        for deadline in deadlines:
            on_time_by_row = {
                (row.sequence_id, row.cycle_id): bool(
                    row.ok
                    and math.isfinite(row.time_sec)
                    and row.time_sec <= deadline
                )
                for row in planner_rows
            }
            on_time_count = sum(on_time_by_row.values())
            late_count = sum(
                row.ok
                and math.isfinite(row.time_sec)
                and row.time_sec > deadline
                for row in planner_rows
            )
            missed_count = total - on_time_count
            infeasible_count = total - feasible_count
            if on_time_count + late_count != feasible_count:
                raise AssertionError(
                    "metric invariant failed: on_time + late != feasible"
                )
            if on_time_count + missed_count != total:
                raise AssertionError(
                    "metric invariant failed: on_time + missed != total"
                )
            if missed_count != late_count + infeasible_count:
                raise AssertionError(
                    "metric invariant failed: missed != late + infeasible"
                )
            complete_count = 0
            for sequence_id in sequence_ids:
                sequence_keys = [
                    key for key in on_time_by_row if key[0] == sequence_id
                ]
                if sequence_keys and all(on_time_by_row[key] for key in sequence_keys):
                    complete_count += 1
            detailed.append(
                {
                    "planner": planner,
                    "deadline_sec": deadline,
                    "total_replanning_cycles": total,
                    "feasible_count": feasible_count,
                    "infeasible_count": infeasible_count,
                    "feasibility_rate_percent": _percent(feasible_count, total),
                    "on_time_success_count": on_time_count,
                    "on_time_success_rate_percent": _percent(on_time_count, total),
                    "missed_deadline_count": missed_count,
                    "late_but_feasible_count": late_count,
                    "sequence_count": len(sequence_ids),
                    "complete_sequence_success_count": complete_count,
                    "complete_sequence_success_rate_percent": _percent(
                        complete_count, len(sequence_ids)
                    ),
                }
            )
        planner_summaries = [
            row for row in detailed if row["planner"] == planner
        ]
        by_increasing_deadline = sorted(
            planner_summaries, key=lambda row: float(row["deadline_sec"])
        )
        previous_on_time = -1
        previous_complete = -1
        previous_late = math.inf
        for summary in by_increasing_deadline:
            on_time = int(summary["on_time_success_count"])
            complete = int(summary["complete_sequence_success_count"])
            late = int(summary["late_but_feasible_count"])
            if (
                on_time < previous_on_time
                or complete < previous_complete
                or late > previous_late
            ):
                raise AssertionError(
                    "deadline monotonicity invariant failed for "
                    f"{planner}"
                )
            previous_on_time = on_time
            previous_complete = complete
            previous_late = late
    return detailed


def _format_csv_value(field_name: str, value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        if field_name == "deadline_sec":
            return f"{value:.6g}"
        if field_name.endswith("_percent"):
            return f"{value:.6f}"
        return f"{value:.9f}"
    return value


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _build_latex_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol_note: str = "",
) -> str:
    lines = [
        "% Auto-generated by Com10_2.py.",
    ]
    if protocol_note:
        lines.append(f"% Protocol: {protocol_note}")
    lines.extend(
        [
        "% Missed = all cycles without an on-time feasible plan "
        "(infeasible + late feasible).",
        "% Complete sequence = every expected replanning cycle on time.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        (
            r"Planner & Deadline [s] & Cycles & Feasible [\%] & "
            r"On-time [\%] & Missed & Late feasible & Complete seq. [\%] \\"
        ),
        r"\hline",
        ]
    )
    for row in rows:
        lines.append(
            "{} & {:.2f} & {} & {:.2f} & {:.2f} & {} & {} & {:.2f} \\\\".format(
                _latex_escape(str(row["planner"])),
                float(row["deadline_sec"]),
                int(row["total_replanning_cycles"]),
                float(row["feasibility_rate_percent"]),
                float(row["on_time_success_rate_percent"]),
                int(row["missed_deadline_count"]),
                int(row["late_but_feasible_count"]),
                float(row["complete_sequence_success_rate_percent"]),
            )
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            (
                r"\caption{Pump/TORO replanning performance under post-hoc "
                r"planning deadlines.}"
            ),
            r"\label{tab:pump_deadline_comparison}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_summary_outputs(
    rows: Sequence[NormalizedCycle],
    output_dir: Path,
    deadlines_sec: Sequence[float] = DEFAULT_DEADLINES_SEC,
) -> tuple[Path, Path, Path]:
    detailed = summarize_deadlines(rows, deadlines_sec)
    compact = [
        {field_name: row[field_name] for field_name in COMPACT_FIELDS}
        for row in detailed
    ]
    detailed_serialized = [
        {
            field_name: _format_csv_value(field_name, row[field_name])
            for field_name in DETAILED_FIELDS
        }
        for row in detailed
    ]
    compact_serialized = [
        {
            field_name: _format_csv_value(field_name, row[field_name])
            for field_name in COMPACT_FIELDS
        }
        for row in compact
    ]

    detailed_path = output_dir / "pump_deadline_summary_detailed.csv"
    compact_path = output_dir / "pump_deadline_summary_compact.csv"
    latex_path = output_dir / "pump_deadline_latex_table.txt"
    sequence_ids = {row.sequence_id for row in rows}
    protocol_note = (
        "Proposed paper Pump preset; fixed window 6; "
        "initial plus 53 keys = 54 cycles."
        if sequence_ids == {PAPER_LEGACY_PROTOCOL_ID}
        else ""
    )
    _atomic_write_csv(detailed_path, DETAILED_FIELDS, detailed_serialized)
    _atomic_write_csv(compact_path, COMPACT_FIELDS, compact_serialized)
    _atomic_write_text(
        latex_path,
        _build_latex_table(detailed, protocol_note=protocol_note),
    )
    return detailed_path, compact_path, latex_path


def _pose5(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    return np.asarray([x, y, z, math.cos(yaw), math.sin(yaw)], dtype=float)


def _normalize_pose5(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < 5 or not np.all(np.isfinite(array[:5])):
        raise ValueError(f"expected finite pose5, got {value!r}")
    return _pose5(
        float(array[0]),
        float(array[1]),
        math.atan2(float(array[4]), float(array[3])),
        z=0.0,
    )


def _initial_poses() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.pi / 2.0
    left = _pose5(-0.115, 0.0, yaw)
    right = _pose5(0.115, 0.0, yaw)
    return left.copy(), right.copy(), left.copy(), right.copy()


def _pairs_from_solution(
    solution: Mapping[str, Any],
    edge: np.ndarray,
    path: Sequence[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(solution["y"], dtype=float).reshape(-1)
    zp_left = np.asarray(solution["zp_L"], dtype=float)
    zp_right = np.asarray(solution["zp_R"], dtype=float)
    edge = np.asarray(edge, dtype=int)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for source, target in zip(path[:-1], path[1:]):
        matches = np.where(
            (edge[:, 0] == int(source)) & (edge[:, 1] == int(target))
        )[0]
        if not matches.size:
            continue
        edge_index = int(matches[0])
        denominator = max(1e-9, float(y[edge_index]))
        pairs.append(
            (
                zp_left[edge_index] / denominator,
                zp_right[edge_index] / denominator,
            )
        )
    return pairs


def _candidate_seed(
    base_seed: int,
    sequence_index: int,
    cycle_index: int,
    candidate_index: int,
) -> int:
    seed_sequence = np.random.SeedSequence(
        [
            int(base_seed),
            int(sequence_index),
            int(cycle_index),
            int(candidate_index),
        ]
    )
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def solve_proposed_cycle(
    label_window: Sequence[str],
    start_left: np.ndarray,
    start_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    *,
    observation_limit_sec: float,
    base_seed: int,
    sequence_index: int,
    cycle_index: int,
    recovery_candidates: int = DEFAULT_RECOVERY_CANDIDATES,
) -> CycleSolveResult:
    """Run the current Pump OptAPI pipeline headlessly until first feasibility."""
    import mosek

    from src.graph.centroid import compute_centroids_from_vertices
    from src.model.gcs_model_ddr import build_gcs_model_ddr_TORO_optapi
    from src.solver.path import path_to_y, relaxation_max, relaxation_prob
    from src.solver.stochastic import (
        _is_feasible_opt,
        _solve_fixed_once_ddr_optapi,
    )
    from src.solver.utils import (
        get_relaxed_y_ddr_optapi,
        set_parameters_ddr_optapi,
        solve_relaxed_ddr_optapi,
    )
    from src.terrain_ddr import Terrain_GCS_DDR
    from src.utils import postprocess_solution

    result = CycleSolveResult(planner="Proposed", ok=False)
    wall_start = time.perf_counter()
    deadline = wall_start + float(observation_limit_sec)
    task = None
    solver_elapsed = 0.0

    def remaining() -> float:
        return deadline - time.perf_counter()

    def set_remaining_limit() -> bool:
        value = remaining()
        if value <= 0.0:
            return False
        task.putdouparam(
            mosek.dparam.optimizer_max_time, max(0.001, float(value))
        )
        return True

    try:
        labels = list(label_window)
        if not labels:
            raise ValueError("label_window is empty")
        terrain = Terrain_GCS_DDR(*(labels + ["All"]))
        vertices = terrain.get_vertices()
        edge = np.asarray(terrain.get_edges(), dtype=int)
        modes = terrain.get_mode_list()
        vertex_names = [stone.label for stone in terrain.stepping_stones]
        centroid_mat, _ = compute_centroids_from_vertices(vertices)

        task, variables, indices, parameters, _ = (
            build_gcs_model_ddr_TORO_optapi(
                vertices,
                edge,
                math.pi / 2.0,
                modes,
                start_left,
                start_right,
                target_left,
                target_right,
                v_names=vertex_names,
            )
        )
        vertex_count = centroid_mat.shape[1]
        geometry = np.empty(2 * vertex_count, dtype=float)
        geometry[0::2] = centroid_mat[0, :]
        geometry[1::2] = centroid_mat[1, :]
        parameters_relaxed = {
            "geometry": geometry,
            "start_L": start_left,
            "start_R": start_right,
            "target_L": target_left,
            "target_R": target_right,
            "M_scalar": 1.0,
            "y_relax": np.zeros(int(indices["n_edge"]), dtype=float),
        }
        set_parameters_ddr_optapi(
            task, variables, indices, parameters, parameters_relaxed
        )
        result.setup_time_sec = time.perf_counter() - wall_start
        if not set_remaining_limit():
            result.status = result.stop_reason = "observation_timeout_during_setup"
            return result

        solve_start = time.perf_counter()
        relax_info, relax_objective = solve_relaxed_ddr_optapi(
            task,
            variables,
            indices,
            parameters,
            M_large=1e3,
            log_level=0,
        )
        solver_elapsed += time.perf_counter() - solve_start
        relax_ok = (
            _is_feasible_opt(
                relax_info.get("pstatus"), relax_info.get("sstatus")
            )
            and relax_objective is not None
            and math.isfinite(float(relax_objective))
        )
        if not relax_ok:
            result.status = (
                f"relaxed_failed:{relax_info.get('pstatus', '')}|"
                f"{relax_info.get('sstatus', '')}"
            )
            result.stop_reason = (
                "observation_timeout"
                if remaining() <= 0.0
                else "relaxed_infeasible_or_unknown"
            )
            return result

        relaxed_y = get_relaxed_y_ddr_optapi(task, variables)
        if (
            relaxed_y.size != int(indices["n_edge"])
            or not np.all(np.isfinite(relaxed_y))
        ):
            result.status = "invalid_relaxed_flow"
            result.stop_reason = "relaxed_solution_missing"
            return result

        candidate_paths: list[tuple[str, Callable[[], list[int]]]] = [
            (
                "greedy",
                lambda: relaxation_max(
                    relaxed_y,
                    edge,
                    int(indices["vertex_number"]),
                    max_backtracks=20000,
                ),
            )
        ]
        for candidate_index in range(1, int(recovery_candidates)):
            seed = _candidate_seed(
                base_seed,
                sequence_index,
                cycle_index,
                candidate_index,
            )
            candidate_paths.append(
                (
                    f"probabilistic_{candidate_index}",
                    lambda seed=seed: relaxation_prob(
                        relaxed_y,
                        edge,
                        int(indices["vertex_number"]),
                        temperature=1.0,
                        seed=seed,
                        max_backtracks=20000,
                    ),
                )
            )

        candidate_errors: list[str] = []
        for candidate_number, (tag, path_factory) in enumerate(
            candidate_paths, start=1
        ):
            result.num_candidates = candidate_number
            if remaining() <= 0.0:
                result.status = "observation_timeout"
                result.stop_reason = "no_feasible_fixed_path_before_limit"
                break
            try:
                path = path_factory()
                fixed_y = path_to_y(
                    edge, path, int(indices["n_edge"])
                )
                if not set_remaining_limit():
                    result.status = "observation_timeout"
                    result.stop_reason = "no_feasible_fixed_path_before_limit"
                    break
                solve_start = time.perf_counter()
                info, objective, solution = _solve_fixed_once_ddr_optapi(
                    task,
                    variables,
                    indices,
                    centroid_mat=centroid_mat,
                    y_relax=fixed_y,
                    M_small=1e-6,
                    start_L=start_left,
                    start_R=start_right,
                    target_L=target_left,
                    target_R=target_right,
                    log_level=0,
                )
                solver_elapsed += time.perf_counter() - solve_start
                feasible = (
                    _is_feasible_opt(
                        info.get("pstatus"), info.get("sstatus")
                    )
                    and objective is not None
                    and math.isfinite(float(objective))
                    and isinstance(solution, dict)
                    and bool(solution)
                )
                # Match the Proposed paper timer: planning ends when the first
                # solver-feasible fixed solution returns.  We still validate
                # that candidate below; if validation fails it is not accepted.
                first_feasible_time = time.perf_counter() - wall_start
                if not feasible:
                    candidate_errors.append(
                        f"{tag}:{info.get('pstatus', '')}|"
                        f"{info.get('sstatus', '')}"
                    )
                    continue

                # Match the deployed pipeline's validation boundary: a fixed
                # solve is accepted only if the existing postprocessor and
                # first state-pair extraction both succeed.
                postprocess_solution(
                    solution,
                    edge,
                    path,
                    np.asarray(start_left[:2], dtype=float),
                    np.asarray(start_right[:2], dtype=float),
                    np.asarray(target_left[:2], dtype=float),
                    np.asarray(target_right[:2], dtype=float),
                    target_left,
                    target_right,
                )
                pairs = _pairs_from_solution(solution, edge, path)
                if not pairs:
                    candidate_errors.append(f"{tag}:no_next_state_pair")
                    continue
                first_mode = (
                    modes[int(path[1]) - 1] if len(path) > 1 else ""
                )
                if first_mode not in {"SL", "SR"}:
                    candidate_errors.append(
                        f"{tag}:unexpected_first_mode={first_mode!r}"
                    )
                    continue

                result.ok = True
                result.time_sec = first_feasible_time
                result.status = (
                    f"feasible:{info.get('pstatus', '')}|"
                    f"{info.get('sstatus', '')}"
                )
                result.stop_reason = "first_feasible"
                result.solver_time_sec = solver_elapsed
                result.next_left = _normalize_pose5(pairs[0][0])
                result.next_right = _normalize_pose5(pairs[0][1])
                result.first_mover = "L" if first_mode == "SL" else "R"
                result.details = (
                    f"candidate={tag}; objective={float(objective):.12g}; "
                    f"path_length={max(0, len(path) - 1)}"
                )
                return result
            except Exception as exc:
                candidate_errors.append(
                    f"{tag}:{type(exc).__name__}:{exc}"
                )

        if not result.status:
            result.status = "no_feasible_fixed_path"
            result.stop_reason = "candidate_budget_exhausted"
        result.details = "; ".join(candidate_errors[:8])
        return result
    except Exception as exc:
        result.status = f"error:{type(exc).__name__}"
        result.stop_reason = "exception"
        result.details = str(exc)
        return result
    finally:
        result.solver_time_sec = solver_elapsed
        if task is not None:
            try:
                task.dispose()
            except Exception:
                pass


@dataclass
class PumpAutomaton:
    terrain: Any
    edges: np.ndarray
    modes: list[str]
    stones: list[Any]
    outgoing: dict[int, list[int]]
    source: int
    goal: int


def _build_pump_automaton(label_window: Sequence[str]) -> PumpAutomaton:
    from src.terrain_ddr import Terrain_GCS_DDR

    labels = list(label_window)
    if not labels:
        raise ValueError("label_window is empty")
    terrain = Terrain_GCS_DDR(*(labels + ["All"]))
    edges = np.asarray(terrain.get_edges(), dtype=int) - 1
    modes = list(terrain.get_mode_list())
    stones = list(terrain.stepping_stones)
    outgoing: dict[int, list[int]] = defaultdict(list)
    for source, target in edges:
        outgoing[int(source)].append(int(target))
    return PumpAutomaton(
        terrain=terrain,
        edges=edges,
        modes=modes,
        stones=stones,
        outgoing=dict(outgoing),
        source=0,
        goal=len(stones) - 1,
    )


def _yaw_from_pose5(pose: Sequence[float]) -> float:
    value = np.asarray(pose, dtype=float).reshape(-1)
    return math.atan2(float(value[4]), float(value[3]))


def _center_inside_stone(step: Any, stone: Any, eps: float = 1e-9) -> bool:
    """Match the Pump convex programs' point-in-region semantics."""
    x = float(step.get_x())
    y = float(step.get_y())
    cx, cy = map(float, stone.center)
    return (
        abs(x - cx) <= 0.5 * float(stone.width) + eps
        and abs(y - cy) <= 0.5 * float(stone.height) + eps
    )


def solve_graph_cycle(
    label_window: Sequence[str],
    start_left: np.ndarray,
    start_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    *,
    first_mover: str,
    observation_limit_sec: float,
    astar_root: Path,
    max_iterations: int = 200000,
) -> CycleSolveResult:
    """Layered A* using the existing IHMC lattice and the exact Pump automaton.

    The legacy VAE A* checker treats all stones as a static union.  Here the
    temporal Pump vertex is part of the search state, so repeated labels remain
    distinct and the order cannot be skipped.  Once the temporal goal is
    reached, A* returns both feet to the same home double-support target used by
    Proposed.
    """
    root = astar_root.expanduser().resolve()
    if not (root / "ihmc_footstep_planning").is_dir():
        return CycleSolveResult(
            planner="Graph-based",
            ok=False,
            status="missing_graph_planner",
            stop_reason="dependency_missing",
            details=f"not found: {root / 'ihmc_footstep_planning'}",
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Imports are intentionally outside the timed region, like shared-library
    # loading in all three planner processes.
    from ihmc_footstep_planning.graph import DiscreteFootstep, FootstepGraphNode
    from ihmc_footstep_planning.graph.discrete_footstep_tools import (
        construct_step_in_previous_step_frame,
    )
    from ihmc_footstep_planning.graph.lattice_point import LatticePoint
    from ihmc_footstep_planning.geometry import wrap_to_pi
    from ihmc_footstep_planning.parameters import (
        DefaultFootstepPlannerParameters,
    )
    from ihmc_footstep_planning.robot_side import RobotSide
    from ihmc_footstep_planning.step_cost import FootstepCostCalculator
    from ihmc_footstep_planning.step_expansion import (
        ParameterBasedStepExpansion,
    )

    result = CycleSolveResult(planner="Graph-based", ok=False)
    wall_start = time.perf_counter()
    deadline = wall_start + float(observation_limit_sec)
    try:
        # The repository's 5 cm lattice cannot represent the Pump pad boundary
        # at +/-0.11 m and incorrectly empties the search.  A 1 cm lattice is
        # still the same graph-based method while resolving the actual geometry.
        LatticePoint.grid_size_xy = 0.01
        automaton = _build_pump_automaton(label_window)
        params = DefaultFootstepPlannerParameters()
        foot_polygon = [
            (-0.075, -0.035),
            (-0.075, 0.035),
            (0.075, 0.035),
            (0.075, -0.035),
        ]
        foot_polygons = {
            RobotSide.LEFT: foot_polygon,
            RobotSide.RIGHT: foot_polygon,
        }
        expansion = ParameterBasedStepExpansion(params, foot_polygons)
        expansion.initialize()
        cost_calculator = FootstepCostCalculator(params)

        left = DiscreteFootstep(
            float(start_left[0]),
            float(start_left[1]),
            _yaw_from_pose5(start_left),
            RobotSide.LEFT,
        )
        right = DiscreteFootstep(
            float(start_right[0]),
            float(start_right[1]),
            _yaw_from_pose5(start_right),
            RobotSide.RIGHT,
        )
        target_l = DiscreteFootstep(
            float(target_left[0]),
            float(target_left[1]),
            _yaw_from_pose5(target_left),
            RobotSide.LEFT,
        )
        target_r = DiscreteFootstep(
            float(target_right[0]),
            float(target_right[1]),
            _yaw_from_pose5(target_right),
            RobotSide.RIGHT,
        )
        if first_mover == "L":
            root_node = FootstepGraphNode(left, right)
        elif first_mover == "R":
            root_node = FootstepGraphNode(right, left)
        else:
            raise ValueError(
                f"first_mover must be 'L' or 'R', got {first_mover!r}"
            )

        result.setup_time_sec = time.perf_counter() - wall_start
        if time.perf_counter() >= deadline:
            result.status = result.stop_reason = "observation_timeout_during_setup"
            return result

        def side_name(step: Any) -> str:
            return (
                "L"
                if step.get_robot_side() is RobotSide.LEFT
                else "R"
            )

        def feet_by_side(node: Any) -> tuple[Any, Any]:
            first = node.get_first_step()
            second = node.get_second_step()
            if first.get_robot_side() is RobotSide.LEFT:
                return first, second
            return second, first

        def noncrossing(node: Any) -> bool:
            node_l, node_r = feet_by_side(node)
            return node_l.get_x() + 0.2 <= node_r.get_x() + 1e-12

        def step_distance(first: Any, second: Any) -> float:
            return math.hypot(
                first.get_x() - second.get_x(),
                first.get_y() - second.get_y(),
            )

        def edge_kinematics_ok(before: Any, after: Any) -> bool:
            old_l, old_r = feet_by_side(before)
            new_l, new_r = feet_by_side(after)
            return (
                noncrossing(before)
                and noncrossing(after)
                and step_distance(new_l, old_r) <= 0.33 + 1e-12
                and step_distance(new_r, old_l) <= 0.33 + 1e-12
            )

        def terminal_node(node: Any) -> Optional[Any]:
            # The Pump GCS SL/SR->Do edge updates both foot variables
            # atomically.  Preserve that semantics instead of adding two
            # baseline-only return steps.
            if node.get_first_step_side() is RobotSide.LEFT:
                goal_node = FootstepGraphNode(target_l, target_r)
            else:
                goal_node = FootstepGraphNode(target_r, target_l)
            return goal_node if edge_kinematics_ok(node, goal_node) else None

        reverse: list[list[int]] = [
            [] for _ in range(len(automaton.stones))
        ]
        for source, destinations in automaton.outgoing.items():
            for destination in destinations:
                reverse[destination].append(source)
        large_distance = 10**9
        remaining_edges = [large_distance] * len(automaton.stones)
        remaining_edges[automaton.goal] = 0
        breadth_queue = [automaton.goal]
        for destination in breadth_queue:
            for source in reverse[destination]:
                if remaining_edges[source] == large_distance:
                    remaining_edges[source] = (
                        remaining_edges[destination] + 1
                    )
                    breadth_queue.append(source)

        def heuristic(node: Any, temporal_vertex: int) -> float:
            destinations = [
                target
                for target in automaton.outgoing.get(temporal_vertex, [])
                if target != automaton.goal
            ]
            temporal_distance = remaining_edges[temporal_vertex]
            base = (
                0.0
                if temporal_distance >= large_distance
                else 10.0 * temporal_distance
            )
            if not destinations:
                return base
            stance = node.get_second_step()
            # Prefer the next temporal layer over optional extra contacts in
            # the same symbol.  The existing graph planner also uses weighted,
            # non-admissible guidance.
            lookahead = max(destinations)
            stone = automaton.stones[lookahead]
            center_x, center_y = stone.center
            delta_x = max(
                abs(stance.get_x() - center_x)
                - 0.5 * float(stone.width),
                0.0,
            )
            delta_y = max(
                abs(stance.get_y() - center_y)
                - 0.5 * float(stone.height),
                0.0,
            )
            reach_hint = math.hypot(delta_x, delta_y)
            yaw_hint = abs(
                wrap_to_pi(stance.get_yaw() - math.pi / 2.0)
            )
            return base + 30.0 * reach_hint + yaw_hint

        start_state = (root_node, automaton.source)
        best_cost: dict[tuple[Any, int], float] = {start_state: 0.0}
        parent: dict[tuple[Any, int], tuple[Any, int]] = {}
        queue: list[tuple[float, int, tuple[Any, int]]] = []
        tie = itertools.count()
        heapq.heappush(
            queue,
            (
                heuristic(root_node, automaton.source),
                next(tie),
                start_state,
            ),
        )
        expanded = 0
        generated = 0
        accepted = 0
        solver_start = time.perf_counter()
        closed: set[tuple[Any, int]] = set()

        if not noncrossing(root_node):
            result.status = "invalid_start_noncrossing"
            result.stop_reason = "invalid_paired_input"
            result.solver_time_sec = time.perf_counter() - solver_start
            return result

        def expand_fast(node: Any) -> tuple[list[Any], bool]:
            swing_side = node.get_first_step_side()
            stance = node.get_second_step()
            unique: dict[Any, None] = {}
            for offset_index, (
                step_length,
                step_width,
                step_yaw,
            ) in enumerate(
                zip(
                    expansion.x_offsets,
                    expansion.y_offsets,
                    expansion.yaw_offsets,
                )
            ):
                if (
                    offset_index % 64 == 0
                    and time.perf_counter() >= deadline
                ):
                    return list(unique), True
                candidate_step = construct_step_in_previous_step_frame(
                    step_length,
                    swing_side.negate_if_right_side(step_width),
                    swing_side.negate_if_right_side(step_yaw),
                    stance,
                )
                unique.setdefault(
                    FootstepGraphNode(stance, candidate_step), None
                )
            return list(unique), False

        while queue and expanded < int(max_iterations):
            if time.perf_counter() >= deadline:
                result.status = "observation_timeout"
                result.stop_reason = "no_feasible_path_before_limit"
                break
            priority, _, state = heapq.heappop(queue)
            del priority
            node, temporal_vertex = state
            if state in closed:
                continue
            closed.add(state)
            state_cost = best_cost.get(state)
            if state_cost is None:
                continue
            expanded += 1

            if automaton.goal in automaton.outgoing.get(
                temporal_vertex, []
            ):
                goal_node = terminal_node(node)
                generated += 1
                if goal_node is not None:
                    result.ok = True
                    result.time_sec = time.perf_counter() - wall_start
                    result.status = "feasible"
                    result.stop_reason = "first_feasible"
                    result.solver_time_sec = (
                        time.perf_counter() - solver_start
                    )
                    result.num_candidates = expanded
                    result.first_mover = first_mover
                    result.details = (
                        f"expanded_states={expanded}; "
                        f"generated_states={generated}; "
                        f"accepted_states={accepted}; "
                        f"lattice_offsets={len(expansion.x_offsets)}; "
                        "grid_xy=0.01; terminal=atomic_Do"
                    )
                    return result

            destinations = [
                target
                for target in automaton.outgoing.get(
                    temporal_vertex, []
                )
                if target != automaton.goal
            ]
            children, expansion_timed_out = expand_fast(node)
            if expansion_timed_out:
                result.status = "observation_timeout"
                result.stop_reason = "timeout_during_expansion"
                break

            for destination in destinations:
                required_mode = automaton.modes[destination]
                required_side = (
                    "L" if required_mode == "SL" else "R"
                )
                region = automaton.stones[destination]
                for candidate_index, child in enumerate(children):
                    if (
                        candidate_index % 64 == 0
                        and time.perf_counter() >= deadline
                    ):
                        result.status = "observation_timeout"
                        result.stop_reason = "timeout_during_validation"
                        queue.clear()
                        break
                    candidate = child.get_second_step()
                    generated += 1
                    if side_name(candidate) != required_side:
                        continue
                    if not _center_inside_stone(candidate, region):
                        continue
                    if not edge_kinematics_ok(node, child):
                        continue
                    edge_cost = cost_calculator.compute_cost(
                        candidate,
                        node.get_second_step(),
                        node.get_first_step(),
                    )
                    child_state = (child, destination)
                    child_cost = state_cost + float(edge_cost)
                    if child_cost + 1e-12 >= best_cost.get(
                        child_state, math.inf
                    ):
                        continue
                    best_cost[child_state] = child_cost
                    parent[child_state] = state
                    accepted += 1
                    heapq.heappush(
                        queue,
                        (
                            child_cost + heuristic(child, destination),
                            next(tie),
                            child_state,
                        ),
                    )
                if result.status == "observation_timeout":
                    break

        result.solver_time_sec = time.perf_counter() - solver_start
        result.num_candidates = expanded
        if not result.status:
            if not queue:
                result.status = "open_list_empty"
                result.stop_reason = "no_feasible_path"
            else:
                result.status = "max_iterations"
                result.stop_reason = "search_limit"
        result.details = (
            f"expanded_states={expanded}; generated_states={generated}; "
            f"accepted_states={accepted}; "
            f"lattice_offsets={len(expansion.x_offsets)}; grid_xy=0.01"
        )
        return result
    except Exception as exc:
        result.status = f"error:{type(exc).__name__}"
        result.stop_reason = "exception"
        result.details = str(exc)
        return result


def _wrap_angle(value: float) -> float:
    wrapped = (float(value) + math.pi) % (2.0 * math.pi) - math.pi
    if math.isclose(wrapped, -math.pi, abs_tol=1e-12):
        return math.pi
    return wrapped


def solve_optimization_cycle(
    label_window: Sequence[str],
    start_left: np.ndarray,
    start_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    *,
    first_mover: str,
    observation_limit_sec: float,
    mip_root: Path,
    trig_segments: int = 6,
) -> CycleSolveResult:
    """Run the existing MIQCQP through a non-invasive temporal Model proxy.

    The original MIP implementation assigns every step to an arbitrary member
    of a static region union.  The proxy captures its named variables and adds
    Pump temporal-flow, support-side, and hard final double-support constraints
    immediately before ``solve``.  ``mioMaxNumSolutions=1`` plus a MOSEK data
    callback gives the time of the first integer-feasible complete plan.
    """
    root = mip_root.expanduser().resolve()
    source_root = root / "src"
    if not (source_root / "micqp_planner").is_dir():
        return CycleSolveResult(
            planner="Optimization-based",
            ok=False,
            status="missing_optimization_planner",
            stop_reason="dependency_missing",
            details=f"not found: {source_root / 'micqp_planner'}",
        )
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    # Imports are outside the timed interval for the same reason as the A*
    # imports: process/library loading is not planner work for a cycle.
    import mosek
    from mosek.fusion import (
        AccSolutionStatus,
        Domain,
        Expr,
        Model as FusionModel,
    )

    from micqp_planner.config import PlannerConfig
    from micqp_planner.terrain import (
        FootPose,
        SteppingStone,
        regions_from_stepping_stones,
    )
    import micqp_planner.mosek_mico as mico

    result = CycleSolveResult(planner="Optimization-based", ok=False)
    wall_start = time.perf_counter()
    absolute_deadline = wall_start + float(observation_limit_sec)

    try:
        if first_mover not in {"L", "R"}:
            raise ValueError(
                f"first_mover must be 'L' or 'R', got {first_mover!r}"
            )
        automaton = _build_pump_automaton(label_window)
        number_of_symbols = max(0, len(label_window) - 1)
        if number_of_symbols <= 0:
            raise ValueError(
                "Pump window needs at least one future symbol after its start token"
            )

        # The MIP formulation assumes forward +x, left +y, and step index 2
        # moves its internal L foot.  Pump faces world +y.  For an actual-L
        # swing, R(-pi/2) supplies that frame.  For an actual-R swing, reflect
        # across world y=x and swap foot roles: actual R then becomes internal
        # L without placing nominal yaw on the +/-pi discontinuity.
        swap_roles = first_mover == "R"
        if swap_roles:
            def transform_xy(x: float, y: float) -> tuple[float, float]:
                return float(y), float(x)

            def transform_yaw(theta: float) -> float:
                return _wrap_angle(math.pi / 2.0 - float(theta))

            region_yaw = math.pi / 2.0
        else:
            def transform_xy(x: float, y: float) -> tuple[float, float]:
                return float(y), -float(x)

            def transform_yaw(theta: float) -> float:
                return _wrap_angle(float(theta) - math.pi / 2.0)

            region_yaw = -math.pi / 2.0

        def transform_pose(pose: Sequence[float]) -> Any:
            value = _normalize_pose5(pose)
            x, y = transform_xy(float(value[0]), float(value[1]))
            theta = transform_yaw(_yaw_from_pose5(value))
            return FootPose(x, y, 0.0, theta)

        actual_start_l = transform_pose(start_left)
        actual_start_r = transform_pose(start_right)
        actual_target_l = transform_pose(target_left)
        actual_target_r = transform_pose(target_right)
        if swap_roles:
            internal_start_l, internal_start_r = (
                actual_start_r,
                actual_start_l,
            )
            internal_target_l, internal_target_r = (
                actual_target_r,
                actual_target_l,
            )
        else:
            internal_start_l, internal_start_r = (
                actual_start_l,
                actual_start_r,
            )
            internal_target_l, internal_target_r = (
                actual_target_l,
                actual_target_r,
            )

        temporal_stones: list[Any] = []
        temporal_names: list[str] = []
        temporal_modes: list[str] = []
        for vertex_index, stone in enumerate(automaton.stones):
            center_x, center_y = transform_xy(
                float(stone.center[0]),
                float(stone.center[1]),
            )
            name = (
                f"v{vertex_index:03d}_{stone.label}_{stone.mode}"
                if vertex_index != automaton.goal
                else f"v{vertex_index:03d}_goal_All_Do"
            )
            temporal_stones.append(
                SteppingStone(
                    name=name,
                    center=(center_x, center_y, 0.0),
                    width=float(stone.width),
                    height=float(stone.height),
                    yaw=region_yaw,
                )
            )
            temporal_names.append(name)
            physical_mode = (
                "D" if stone.mode == "Do" else str(stone.mode)
            )
            if swap_roles:
                physical_mode = {
                    "SL": "SR",
                    "SR": "SL",
                }.get(physical_mode, physical_mode)
            temporal_modes.append(physical_mode)
        regions = regions_from_stepping_stones(
            temporal_stones, z_epsilon=1e-4
        )

        predecessors: list[set[int]] = [
            set() for _ in temporal_stones
        ]
        predecessors[automaton.source].add(automaton.source)
        predecessors[automaton.goal].add(automaton.goal)
        for source, target in automaton.edges:
            predecessors[int(target)].add(int(source))

        # Keep the Optimization-based planner's established VAE benchmark
        # horizon (N=14).  A Pump path needs at least one landing per symbol,
        # plus two fixed initial and two final double-support contacts.  Extra
        # indices may traverse the automaton's within-symbol SL->SR->SL edges;
        # this avoids unfairly reducing MIP to exactly one landing per symbol.
        minimum_horizon = 2 + number_of_symbols + 2
        horizon = max(14, minimum_horizon)

        @dataclass
        class TemporalSpec:
            first_incumbent_wall_sec: float = math.nan
            first_incumbent_solver_sec: float = math.nan
            first_incumbent_objective: float = math.nan
            injected: bool = False

        spec = TemporalSpec()

        class TemporalModelProxy:
            def __init__(self, *args: Any, **kwargs: Any):
                self._model = FusionModel(*args, **kwargs)
                self._variables: dict[str, Any] = {}

            def __getattr__(self, name: str) -> Any:
                return getattr(self._model, name)

            def variable(self, *args: Any, **kwargs: Any) -> Any:
                variable = self._model.variable(*args, **kwargs)
                if args and isinstance(args[0], str):
                    self._variables[args[0]] = variable
                return variable

            @staticmethod
            def sum_scalars(values: Iterable[Any]) -> Any:
                items = list(values)
                if not items:
                    return Expr.constTerm(0.0)
                if len(items) == 1:
                    return items[0]
                return Expr.add(items)

            def inject_constraints(self) -> None:
                assignment = self._variables["H"]
                x_var = self._variables["x"]
                y_var = self._variables["y"]
                z_var = self._variables["z"]
                theta_var = self._variables["theta"]
                region_count = len(regions)

                for step_index in (0, 1):
                    self._model.constraint(
                        assignment.index(
                            automaton.source, step_index
                        ),
                        Domain.equalsTo(1.0),
                    )
                for step_index in (horizon - 2, horizon - 1):
                    self._model.constraint(
                        assignment.index(
                            automaton.goal, step_index
                        ),
                        Domain.equalsTo(1.0),
                    )

                for step_index in range(2, horizon - 2):
                    wanted_mode = (
                        "SL" if step_index % 2 == 0 else "SR"
                    )
                    allowed = [
                        assignment.index(region_index, step_index)
                        for region_index, mode in enumerate(
                            temporal_modes
                        )
                        if mode == wanted_mode
                    ]
                    # N=14 is an upper horizon.  Once every symbol has been
                    # traversed, the plan may enter the terminal Do vertex
                    # early and remain there through its explicit self-edge.
                    # The predecessor constraints below still make it
                    # impossible to skip any symbol layer.
                    allowed.append(
                        assignment.index(automaton.goal, step_index)
                    )
                    self._model.constraint(
                        self.sum_scalars(allowed),
                        Domain.equalsTo(1.0),
                    )

                for step_index in range(2, horizon):
                    for destination in range(region_count):
                        allowed_predecessors = predecessors[destination]
                        if not allowed_predecessors:
                            self._model.constraint(
                                assignment.index(
                                    destination, step_index
                                ),
                                Domain.equalsTo(0.0),
                            )
                            continue
                        previous_sum = self.sum_scalars(
                            assignment.index(
                                source, step_index - 1
                            )
                            for source in sorted(
                                allowed_predecessors
                            )
                        )
                        self._model.constraint(
                            Expr.sub(
                                assignment.index(
                                    destination, step_index
                                ),
                                previous_sum,
                            ),
                            Domain.lessThan(0.0),
                        )

                final_left_index = (
                    horizon - 1
                    if (horizon - 1) % 2 == 0
                    else horizon - 2
                )
                final_right_index = (
                    horizon - 1
                    if (horizon - 1) % 2 == 1
                    else horizon - 2
                )
                for step_index, pose in (
                    (final_left_index, internal_target_l),
                    (final_right_index, internal_target_r),
                ):
                    for variable, target_value in (
                        (x_var, pose.x),
                        (y_var, pose.y),
                        (z_var, pose.z),
                        (theta_var, pose.theta),
                    ):
                        self._model.constraint(
                            Expr.sub(
                                variable.index(step_index),
                                float(target_value),
                            ),
                            Domain.equalsTo(0.0),
                        )

                self._model.acceptedSolutionStatus(
                    AccSolutionStatus.Feasible
                )
                self._model.setSolverParam(
                    "mioMaxNumSolutions", 1
                )
                self._model.setSolverParam(
                    "mioMaxTime",
                    max(
                        0.001,
                        absolute_deadline - time.perf_counter(),
                    ),
                )

                def data_callback(
                    caller: Any,
                    double_info: Any,
                    integer_info: Any,
                    long_info: Any,
                ) -> bool:
                    del integer_info, long_info
                    if (
                        caller == mosek.callbackcode.new_int_mio
                        and not math.isfinite(
                            spec.first_incumbent_wall_sec
                        )
                    ):
                        spec.first_incumbent_wall_sec = (
                            time.perf_counter() - wall_start
                        )
                        spec.first_incumbent_solver_sec = float(
                            double_info[
                                mosek.dinfitem.optimizer_time
                            ]
                        )
                        try:
                            spec.first_incumbent_objective = float(
                                double_info[
                                    mosek.dinfitem.mio_obj_int
                                ]
                            )
                        except Exception:
                            pass
                    return False

                self._model.setDataCallbackHandler(data_callback)
                spec.injected = True

            def solve(self, *args: Any, **kwargs: Any) -> Any:
                self.inject_constraints()
                return self._model.solve(*args, **kwargs)

        configuration = PlannerConfig()
        configuration.solver.time_limit_s = float(observation_limit_sec)
        configuration.solver.verbose = False
        configuration.solver.mip_gap = 1e-4
        configuration.trig.num_segments = int(trig_segments)
        configuration.disable_trim = True
        configuration.enforce_trim_prefix = False
        configuration.obj.q_trim = 0.0
        configuration.reach.lr_sep_min = 0.20
        configuration.reach.lr_sep_max = 0.35
        configuration.reach.same_foot_span = 0.65
        configuration.reach.max_dtheta = math.radians(30.0)
        bounds = {
            "x": (-0.8, 0.8),
            "y": (-0.8, 0.8),
            "z": (-0.05, 0.05),
            "theta": (-math.pi, math.pi),
        }

        result.setup_time_sec = time.perf_counter() - wall_start
        original_model = mico.Model
        mico.Model = TemporalModelProxy
        try:
            mip_result = mico.solve_miqcqp_mosek(
                regions,
                internal_start_l,
                internal_start_r,
                internal_target_l,
                internal_target_r,
                horizon,
                configuration,
                abs_bounds=bounds,
            )
        finally:
            mico.Model = original_model

        selected_names = [
            footstep.region for footstep in mip_result.footsteps
        ]
        name_to_vertex = {
            name: index for index, name in enumerate(temporal_names)
        }
        selected_vertices = [
            name_to_vertex.get(name, -1) for name in selected_names
        ]
        path_valid = (
            len(selected_vertices) == horizon
            and all(vertex >= 0 for vertex in selected_vertices)
            and selected_vertices[:2]
            == [automaton.source, automaton.source]
            and selected_vertices[-2:]
            == [automaton.goal, automaton.goal]
        )
        if path_valid:
            for step_index in range(2, horizon):
                if (
                    selected_vertices[step_index - 1]
                    not in predecessors[selected_vertices[step_index]]
                ):
                    path_valid = False
                    break

        status_ok = str(mip_result.status).lower() in {
            "optimal",
            "feasible",
            "solved",
            "time_limit",
        }
        result.ok = bool(
            spec.injected
            and path_valid
            and status_ok
            and math.isfinite(spec.first_incumbent_wall_sec)
        )
        result.first_mover = first_mover
        result.solver_time_sec = spec.first_incumbent_solver_sec
        if result.ok:
            result.time_sec = spec.first_incumbent_wall_sec
            result.setup_time_sec = max(
                0.0,
                result.time_sec
                - (
                    result.solver_time_sec
                    if math.isfinite(result.solver_time_sec)
                    else 0.0
                ),
            )
            result.status = str(mip_result.status)
            result.stop_reason = "first_feasible"
        else:
            result.status = str(mip_result.status)
            result.stop_reason = (
                "observation_timeout"
                if time.perf_counter() >= absolute_deadline
                else "no_valid_first_incumbent"
            )
        result.num_candidates = 1 if math.isfinite(
            spec.first_incumbent_wall_sec
        ) else 0
        result.details = (
            f"N={horizon}; regions={len(regions)}; "
            f"temporal_path_valid={path_valid}; "
            f"first_incumbent_objective="
            f"{spec.first_incumbent_objective}; "
            f"selected_regions={json.dumps(selected_names)}"
        )
        return result
    except Exception as exc:
        result.status = f"error:{type(exc).__name__}"
        result.stop_reason = "exception"
        result.details = str(exc)
        return result


def _valid_label_window(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and all(
            isinstance(item, str) and item in set(KEY_TO_LABEL.values())
            for item in value
        )
    )


def _windows_from_keys(
    keys: Sequence[str],
    *,
    initial_window: Sequence[str] = DEFAULT_INITIAL_WINDOW,
    window_len: int = DEFAULT_WINDOW_LEN,
) -> list[list[str]]:
    window = deque(
        [str(item) for item in initial_window], maxlen=int(window_len)
    )
    windows = [list(window)]
    for raw_key in keys:
        key = str(raw_key)
        if key not in KEY_TO_LABEL:
            raise DataValidationError(f"unknown Pump key {key!r}")
        window.append(KEY_TO_LABEL[key])
        windows.append(list(window))
    return windows


def _sequence_from_mapping(
    value: Mapping[str, Any],
    fallback_id: str,
) -> ReplaySequence:
    sequence_id = str(
        value.get("sequence_id", value.get("run_id", fallback_id))
    )
    if "label_windows" in value:
        windows = value["label_windows"]
        if (
            not isinstance(windows, list)
            or not windows
            or not all(_valid_label_window(item) for item in windows)
        ):
            raise DataValidationError(
                f"{sequence_id}: label_windows must be a nonempty list "
                "of valid Pump label lists"
            )
        return ReplaySequence(
            sequence_id,
            [list(item) for item in windows],
            "JSON label_windows",
        )
    keys = value.get("sequence_keys", value.get("keys"))
    if keys is None:
        raise DataValidationError(
            f"{sequence_id}: expected label_windows or sequence_keys"
        )
    return ReplaySequence(
        sequence_id,
        _windows_from_keys(
            [str(item) for item in keys],
            initial_window=value.get(
                "initial_window", DEFAULT_INITIAL_WINDOW
            ),
            window_len=int(
                value.get("window_len", DEFAULT_WINDOW_LEN)
            ),
        ),
        "JSON sequence_keys",
    )


def _parse_sequence_file(path: Path) -> list[ReplaySequence]:
    if not path.is_file():
        raise DataValidationError(f"sequence file does not exist: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "runs" in data:
            return [
                _sequence_from_mapping(
                    item, f"{path.stem}_{index:03d}"
                )
                for index, item in enumerate(data["runs"])
            ]
        if isinstance(data, dict):
            return [_sequence_from_mapping(data, path.stem)]
        if (
            isinstance(data, list)
            and data
            and all(_valid_label_window(item) for item in data)
        ):
            return [
                ReplaySequence(
                    path.stem,
                    [list(item) for item in data],
                    "JSON label_windows",
                )
            ]
        if isinstance(data, list) and all(
            str(item) in KEY_TO_LABEL for item in data
        ):
            return [
                ReplaySequence(
                    path.stem,
                    _windows_from_keys([str(item) for item in data]),
                    "JSON sequence_keys",
                )
            ]
        raise DataValidationError(
            f"unsupported sequence JSON structure: {path}"
        )

    text = path.read_text(encoding="utf-8", errors="replace")
    literal_values: list[Any] = []
    windows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("["):
            continue
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            continue
        literal_values.append(parsed)
        if _valid_label_window(parsed):
            windows.append(list(parsed))
    if windows:
        return [
            ReplaySequence(
                path.stem,
                windows,
                "archived Pump label-window replay",
            )
        ]
    for parsed in literal_values:
        if isinstance(parsed, list) and all(
            str(item) in KEY_TO_LABEL for item in parsed
        ):
            return [
                ReplaySequence(
                    path.stem,
                    _windows_from_keys(
                        [str(item) for item in parsed]
                    ),
                    "archived Pump sequence_keys replay",
                )
            ]
    raise DataValidationError(
        f"no Pump label windows or key sequence found in {path}"
    )


def load_replay_sequences(
    *,
    paper_legacy: bool,
    sequence_file: Optional[Path],
    log_dir: Optional[Path],
    generated_sequences: int,
    cycles_per_sequence: int,
    base_seed: int,
) -> list[ReplaySequence]:
    selected_sources = sum(
        [
            paper_legacy,
            sequence_file is not None,
            log_dir is not None,
            generated_sequences > 0,
        ]
    )
    if selected_sources > 1:
        raise DataValidationError(
            "choose only one of --paper-legacy, --sequence-file, "
            "--log-dir, or --generate-sequences"
        )
    if paper_legacy or selected_sources == 0:
        paper_windows = _windows_from_keys(
            PAPER_LEGACY_SEQUENCE_KEYS,
            initial_window=DEFAULT_INITIAL_WINDOW,
            window_len=6,
        )
        if (
            len(PAPER_LEGACY_SEQUENCE_KEYS) != 53
            or len(paper_windows) != 54
            or any(len(window) != 6 for window in paper_windows)
            or paper_windows[-1]
            != ["DR", "All", "All", "All", "All", "All"]
        ):
            raise AssertionError("paper legacy Pump preset integrity failure")
        return [
            ReplaySequence(
                PAPER_LEGACY_PROTOCOL_ID,
                paper_windows,
                (
                    "Proposed paper Pump protocol: fixed window_len=6, "
                    "initial + 53 deterministic keys"
                ),
            )
        ]
    if generated_sequences > 0:
        if cycles_per_sequence <= 0:
            raise DataValidationError(
                "--cycles-per-sequence must be positive"
            )
        generated: list[ReplaySequence] = []
        group_a = ["1", "5", "7"]
        group_b = ["3", "5", "9"]
        weights = [0.4, 0.2, 0.4]
        for sequence_index in range(int(generated_sequences)):
            rng = random.Random(
                int(base_seed) + 1000003 * sequence_index
            )
            keys = [
                rng.choices(
                    group_a if index % 2 == 0 else group_b,
                    weights=weights,
                    k=1,
                )[0]
                for index in range(int(cycles_per_sequence))
            ]
            generated.append(
                ReplaySequence(
                    f"seed_{base_seed}_sequence_{sequence_index:03d}",
                    _windows_from_keys(keys),
                    "deterministic generated Pump sequence",
                )
            )
        return generated

    if log_dir is not None:
        directory = log_dir.expanduser().resolve()
        paths = sorted(
            [
                path
                for path in directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".txt", ".log", ".json"}
            ],
            key=lambda path: _natural_key(path.name),
        )
        if not paths:
            raise DataValidationError(
                f"no .txt/.log/.json sequence files found in {directory}"
            )
        sequences: list[ReplaySequence] = []
        for path in paths:
            sequences.extend(_parse_sequence_file(path))
    else:
        if sequence_file is None:
            raise AssertionError("explicit sequence source unexpectedly missing")
        sequences = _parse_sequence_file(sequence_file)

    used_ids: set[str] = set()
    for sequence in sequences:
        original = sequence.sequence_id
        suffix = 1
        while sequence.sequence_id in used_ids:
            sequence.sequence_id = f"{original}_{suffix}"
            suffix += 1
        used_ids.add(sequence.sequence_id)
    return sequences


def _scenario_hash(
    sequence_id: str,
    cycle_id: int,
    label_window: Sequence[str],
    start_left: np.ndarray,
    start_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    first_mover: str,
) -> str:
    payload = {
        "sequence_id": str(sequence_id),
        "cycle_id": int(cycle_id),
        "label_window": list(label_window),
        "start_left": np.asarray(start_left, dtype=float).round(12).tolist(),
        "start_right": np.asarray(start_right, dtype=float).round(12).tolist(),
        "target_left": np.asarray(target_left, dtype=float).round(12).tolist(),
        "target_right": np.asarray(target_right, dtype=float).round(12).tolist(),
        "first_mover": str(first_mover),
        "state_policy": "shared_Proposed_reconstructed_pose",
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_array(value: Sequence[float]) -> str:
    return json.dumps(
        [round(float(item), 12) for item in value],
        separators=(",", ":"),
    )


def _raw_row(
    *,
    sequence_id: str,
    cycle_id: int,
    protocol: str,
    label_window: Sequence[str],
    start_left: np.ndarray,
    start_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    first_mover: str,
    observation_limit_sec: float,
    solve_result: CycleSolveResult,
) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "cycle_id": cycle_id,
        "scenario_hash": _scenario_hash(
            sequence_id,
            cycle_id,
            label_window,
            start_left,
            start_right,
            target_left,
            target_right,
            first_mover,
        ),
        "protocol": protocol,
        "planner": solve_result.planner,
        "ok": int(bool(solve_result.ok)),
        "time_sec": (
            f"{float(solve_result.time_sec):.17g}"
            if math.isfinite(float(solve_result.time_sec))
            else "nan"
        ),
        "status": solve_result.status,
        "stop_reason": solve_result.stop_reason,
        "label_window": json.dumps(
            list(label_window), separators=(",", ":")
        ),
        "start_left_pose": _json_array(start_left),
        "start_right_pose": _json_array(start_right),
        "first_mover": first_mover,
        "observation_limit_sec": f"{float(observation_limit_sec):.9g}",
        "setup_time_sec": (
            f"{float(solve_result.setup_time_sec):.12g}"
            if math.isfinite(float(solve_result.setup_time_sec))
            else "nan"
        ),
        "solver_time_sec": (
            f"{float(solve_result.solver_time_sec):.12g}"
            if math.isfinite(float(solve_result.solver_time_sec))
            else "nan"
        ),
        "num_candidates": int(solve_result.num_candidates),
        "details": solve_result.details,
    }


def run_experiment(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    deadlines = tuple(float(value) for value in args.deadlines)
    if args.observation_limit_sec <= max(deadlines):
        raise DataValidationError(
            "--observation-limit-sec must be greater than the largest "
            "reported deadline so late-but-feasible solutions can be observed"
        )
    if args.recovery_candidates <= 0:
        raise DataValidationError("--recovery-candidates must be positive")

    sequences = load_replay_sequences(
        paper_legacy=args.paper_legacy,
        sequence_file=args.sequence_file,
        log_dir=args.log_dir,
        generated_sequences=args.generate_sequences,
        cycles_per_sequence=args.cycles_per_sequence,
        base_seed=args.base_seed,
    )
    if args.exclude_initial_cycle:
        sequences = [
            ReplaySequence(
                sequence.sequence_id,
                sequence.windows[1:],
                sequence.source + "; initial cycle excluded",
            )
            for sequence in sequences
        ]
    if args.max_sequences is not None:
        sequences = sequences[: int(args.max_sequences)]
    if args.max_cycles is not None:
        sequences = [
            ReplaySequence(
                sequence.sequence_id,
                sequence.windows[: int(args.max_cycles)],
                sequence.source,
            )
            for sequence in sequences
        ]
    if args.smoke_test:
        sequences = [
            ReplaySequence(
                sequences[0].sequence_id,
                sequences[0].windows[:2],
                sequences[0].source + "; smoke test",
            )
        ]
    sequences = [sequence for sequence in sequences if sequence.windows]
    if not sequences:
        raise DataValidationError("no replanning cycles selected")

    output_dir = args.output_dir.expanduser().resolve()
    raw_path = output_dir / "pump_planner_cycle_results.csv"
    requested_paths = [
        raw_path,
        output_dir / "pump_deadline_summary_detailed.csv",
        output_dir / "pump_deadline_summary_compact.csv",
        output_dir / "pump_deadline_latex_table.txt",
    ]
    existing = [path for path in requested_paths if path.exists()]
    if existing and not args.overwrite:
        raise DataValidationError(
            "output already exists; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    expected_cycles = sum(len(sequence.windows) for sequence in sequences)
    completed_cycles = 0
    for sequence_index, sequence in enumerate(sequences):
        start_left, start_right, target_left, target_right = _initial_poses()
        for cycle_index, label_window in enumerate(sequence.windows):
            cycle_start_left = start_left.copy()
            cycle_start_right = start_right.copy()
            proposed = solve_proposed_cycle(
                label_window,
                cycle_start_left,
                cycle_start_right,
                target_left,
                target_right,
                observation_limit_sec=args.observation_limit_sec,
                base_seed=args.base_seed,
                sequence_index=sequence_index,
                cycle_index=cycle_index,
                recovery_candidates=args.recovery_candidates,
            )
            first_mover = proposed.first_mover

            raw_rows.append(
                _raw_row(
                    sequence_id=sequence.sequence_id,
                    cycle_id=cycle_index,
                    protocol=sequence.source,
                    label_window=label_window,
                    start_left=cycle_start_left,
                    start_right=cycle_start_right,
                    target_left=target_left,
                    target_right=target_right,
                    first_mover=first_mover,
                    observation_limit_sec=args.observation_limit_sec,
                    solve_result=proposed,
                )
            )

            if not (
                proposed.ok
                and first_mover in {"L", "R"}
                and proposed.next_left is not None
                and proposed.next_right is not None
            ):
                _atomic_write_csv(raw_path, RAW_FIELDS, raw_rows)
                raise DataValidationError(
                    "Proposed did not provide the canonical first-swing/"
                    "next-state transition required for a paired replay at "
                    f"sequence={sequence.sequence_id!r}, cycle={cycle_index}; "
                    f"status={proposed.status!r}, "
                    f"stop_reason={proposed.stop_reason!r}. The run was "
                    "stopped instead of fabricating baseline failures."
                )

            graph = solve_graph_cycle(
                label_window,
                cycle_start_left,
                cycle_start_right,
                target_left,
                target_right,
                first_mover=first_mover,
                observation_limit_sec=args.observation_limit_sec,
                astar_root=args.astar_root,
                max_iterations=args.graph_max_iterations,
            )
            optimization = solve_optimization_cycle(
                label_window,
                cycle_start_left,
                cycle_start_right,
                target_left,
                target_right,
                first_mover=first_mover,
                observation_limit_sec=args.observation_limit_sec,
                mip_root=args.mip_root,
                trig_segments=args.mip_trig_segments,
            )

            for baseline in (graph, optimization):
                raw_rows.append(
                    _raw_row(
                        sequence_id=sequence.sequence_id,
                        cycle_id=cycle_index,
                        protocol=sequence.source,
                        label_window=label_window,
                        start_left=cycle_start_left,
                        start_right=cycle_start_right,
                        target_left=target_left,
                        target_right=target_right,
                        first_mover=first_mover,
                        observation_limit_sec=args.observation_limit_sec,
                        solve_result=baseline,
                    )
                )

            start_left = _normalize_pose5(proposed.next_left)
            start_right = _normalize_pose5(proposed.next_right)

            completed_cycles += 1
            _atomic_write_csv(raw_path, RAW_FIELDS, raw_rows)
            print(
                f"[{completed_cycles}/{expected_cycles}] "
                f"sequence={sequence.sequence_id} cycle={cycle_index} "
                f"Proposed={int(proposed.ok)} "
                f"Graph={int(graph.ok)} "
                f"Optimization={int(optimization.ok)}"
            )

    normalized = load_normalized_cycles([raw_path])
    detailed_path, compact_path, latex_path = write_summary_outputs(
        normalized, output_dir, deadlines
    )
    return raw_path, detailed_path, compact_path, latex_path


def _parse_deadlines(values: Sequence[float]) -> tuple[float, ...]:
    deadlines = tuple(float(value) for value in values)
    if not deadlines or any(
        not math.isfinite(value) or value <= 0.0 for value in deadlines
    ):
        raise DataValidationError(
            "deadlines must be finite positive seconds"
        )
    if len(set(deadlines)) != len(deadlines):
        raise DataValidationError("deadlines must be unique")
    return deadlines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pump/TORO Proposed vs Graph-based vs Optimization-based "
            "deadline experiment"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="auto-detect and summarize existing per-cycle CSV files",
    )
    summarize_parser.add_argument(
        "inputs", nargs="+", type=Path, help="one or more raw CSV files"
    )
    summarize_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    summarize_parser.add_argument(
        "--deadlines",
        type=float,
        nargs="+",
        default=list(DEFAULT_DEADLINES_SEC),
        metavar="SEC",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="run all three planners headlessly and then summarize",
    )
    source_group = run_parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--paper-legacy",
        action="store_true",
        help=(
            "use the Proposed paper Pump protocol: fixed window 6 and "
            "the original deterministic 53-key sequence (default)"
        ),
    )
    source_group.add_argument("--sequence-file", type=Path)
    source_group.add_argument("--log-dir", type=Path)
    source_group.add_argument(
        "--generate-sequences",
        type=int,
        default=0,
        metavar="COUNT",
    )
    run_parser.add_argument(
        "--cycles-per-sequence", type=int, default=100
    )
    run_parser.add_argument("--base-seed", type=int, default=123)
    run_parser.add_argument(
        "--exclude-initial-cycle", action="store_true"
    )
    run_parser.add_argument("--max-sequences", type=int)
    run_parser.add_argument("--max-cycles", type=int)
    run_parser.add_argument("--smoke-test", action="store_true")
    run_parser.add_argument(
        "--observation-limit-sec",
        type=float,
        default=DEFAULT_OBSERVATION_LIMIT_SEC,
    )
    run_parser.add_argument(
        "--recovery-candidates",
        type=int,
        default=DEFAULT_RECOVERY_CANDIDATES,
    )
    run_parser.add_argument(
        "--graph-max-iterations", type=int, default=200000
    )
    run_parser.add_argument("--mip-trig-segments", type=int, default=6)
    run_parser.add_argument(
        "--astar-root",
        type=Path,
        default=Path("baselines/astar"),
    )
    run_parser.add_argument(
        "--mip-root",
        type=Path,
        default=Path("baselines/mip"),
    )
    run_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    run_parser.add_argument(
        "--deadlines",
        type=float,
        nargs="+",
        default=list(DEFAULT_DEADLINES_SEC),
        metavar="SEC",
    )
    run_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.deadlines = _parse_deadlines(args.deadlines)
        if args.command == "summarize":
            normalized = load_normalized_cycles(
                [path.expanduser().resolve() for path in args.inputs]
            )
            if not any(
                row.scenario_hash or row.symbol_window
                for row in normalized
            ):
                print(
                    "WARNING: paired coverage is verified by sequence_id/"
                    "cycle_id only; input CSVs contain no scenario_hash or "
                    "symbol window for semantic alignment.",
                    file=sys.stderr,
                )
            paths = write_summary_outputs(
                normalized,
                args.output_dir.expanduser().resolve(),
                args.deadlines,
            )
            for path in paths:
                print(f"Wrote {path}")
            return 0

        raw_path, detailed_path, compact_path, latex_path = run_experiment(
            args
        )
        for path in (
            raw_path,
            detailed_path,
            compact_path,
            latex_path,
        ):
            print(f"Wrote {path}")
        return 0
    except Com10Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
