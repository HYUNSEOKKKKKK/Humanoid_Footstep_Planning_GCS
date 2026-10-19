"""Batch runner (no argparse) for a directory of stepping-stone JSON terrains.

Matches your A* runner style:
  - Fixed config in code (no CLI args)
  - Prints per-file summary + global success-rate summary
  - Plots ONLY the last successful run

Terrain JSON schema supported:
  - Stones-only:
      {"terrain": {"stones": [ {name, center, width, height, yaw_deg}, ... ]}}
  - Optionally you can include per-file stance width:
      {"stance_width": 0.20, "terrain": {"stones": [...]}}

If start/goal are missing, they are inferred automatically from the first/last stone
inside `micqp_planner.terrain.load_problem_json`.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import List
import csv
from datetime import datetime

# --- project root / import path (PyCharm-friendly) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../footstep_mip
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from micqp_planner.config import PlannerConfig  # noqa: E402
from micqp_planner.terrain import load_problem_json, SteppingStone  # noqa: E402
from micqp_planner.mosek_mico import solve_miqcqp_mosek  # noqa: E402
from micqp_planner.plot import plot_plan_3d  # noqa: E402


def _load_stones_only_json(path: Path) -> List[SteppingStone]:
    """Load stepping stones from the 'stones-only' JSON format.

    We keep this local to the runner because the optimizer only needs regions,
    but the visualizer needs the actual stone geometry.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    stones_raw = data.get("terrain", {}).get("stones", [])
    stones: List[SteppingStone] = []
    for s in stones_raw:
        stones.append(
            SteppingStone(
                name=str(s.get("name", f"stone_{len(stones)+1}")),
                center=(float(s["center"][0]), float(s["center"][1]), float(s["center"][2])),
                width=float(s["width"]),
                height=float(s["height"]),
                yaw=math.radians(float(s.get("yaw_deg", 0.0))),
            )
        )
    return stones

def append_result_csv(csv_path: str, row: dict, fieldnames: list):
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not p.exists()) or (p.stat().st_size == 0)
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)

def main() -> None:
    # ========================
    # CONFIG (edit these)
    # ========================
    data_dir = Path("data/terrains/alpha/0.0")
    pattern = "run_*.json"
    N = 14

    # Success criterion: count as OK only if the final (active) double-support is on/near the goal.
    # We check the last ACTIVE left step and last ACTIVE right step.
    # - Region-based (robust): both feet are assigned to the goal stone (last stone in the JSON).
    # - Distance-based (fallback): both feet are within this Euclidean tolerance of goal_left/goal_right.
    GOAL_TOL_M = 0.5

    out_dir = PROJECT_ROOT / "out" / "batch"
    out_dir.mkdir(parents=True, exist_ok=True)


    # ========================
    # CSV logging (one file per execution)
    # ========================
    result_dir = Path("outputs/mip")
    result_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    results_csv = result_dir / f"optimization_based_{stamp}.csv"

    fieldnames = [
        "planner", "alpha", "t_max_sec", "run_json",
        "ok", "stop_reason", "time_sec",
        "status", "objective", "runtime_s",
        "N", "goal_tol_m",
    ]


    cfg = PlannerConfig()
    cfg.solver.verbose = True
    cfg.solver.time_limit_s = 10.0
    cfg.trig.num_segments = 6

    # Keep trimming disabled for stability/consistency during benchmarking
    cfg.disable_trim = False
    cfg.enforce_trim_prefix = True
    cfg.obj.q_trim = -1.0  # 시작은 -5, 더 줄이고 싶으면 -10

    # ========================

    run_files = sorted(data_dir.glob(pattern))
    if not run_files:
        raise FileNotFoundError(f"No json files matched: {data_dir / pattern}")

    results = []
    last_success = None


    for rf in run_files:
        print("\n" + "=" * 80)
        print(f"Running: {rf.name}")

        regions, start_left, start_right, goal_left, goal_right = load_problem_json(str(rf))
        stones = _load_stones_only_json(rf)
        print("goal_left :", goal_left.x, goal_left.y, goal_left.z, goal_left.theta)
        print("goal_right:", goal_right.x, goal_right.y, goal_right.z, goal_right.theta)

        t0 = time.perf_counter()
        res = solve_miqcqp_mosek(
            regions=regions,
            start_left=start_left,
            start_right=start_right,
            goal_left=goal_left,
            goal_right=goal_right,
            cfg=cfg,
            N=N,
        )
        elapsed = time.perf_counter() - t0

        status_ok = (res.status in {"optimal", "time_limit", "feasible"}) and bool(res.active_indices)

        # --- goal reach check (only count as success if goal is reached) ---
        goal_region_name = stones[-1].name if stones else None

        def _last_active_step(foot: str):
            idxs = [j for j in res.active_indices if (j < len(res.footsteps)) and (res.footsteps[j].foot == foot)]
            if not idxs:
                return None
            return res.footsteps[max(idxs)]

        stepL = _last_active_step("L")
        stepR = _last_active_step("R")

        goal_reached = False
        if stepL is not None and stepR is not None:
            # Region-based success: both last active feet are on the goal stone.
            if goal_region_name is not None:
                goal_reached = (stepL.region == goal_region_name) and (stepR.region == goal_region_name)

            # Distance-based fallback (or additional acceptance): near the inferred goal poses.
            if not goal_reached:
                dL = math.sqrt((stepL.x - goal_left.x) ** 2 + (stepL.y - goal_left.y) ** 2 + (stepL.z - goal_left.z) ** 2)
                dR = math.sqrt((stepR.x - goal_right.x) ** 2 + (stepR.y - goal_right.y) ** 2 + (stepR.z - goal_right.z) ** 2)
                goal_reached = (dL <= GOAL_TOL_M) and (dR <= GOAL_TOL_M)

        ok = bool(status_ok and goal_reached)

        results.append(
            {
                "run_json": str(rf),
                "ok": ok,
                "status": res.status,
                "objective": float(res.objective) if res.objective == res.objective else float("nan"),
                "runtime_s": float(res.runtime_s) if res.runtime_s is not None else float(elapsed),
            }
        )
        # ---- stop_reason 정의 (Graph/Ours와 맞추기) ----
        if ok:
            stop_reason = "reached"
        elif res.status == "time_limit":
            stop_reason = "timeout"
        elif res.status in {"infeasible", "unknown"}:
            stop_reason = res.status
        else:
            stop_reason = res.status

        # ---- CSV row (append) ----
        row = {
            "planner": "Optimization-based",
            "t_max_sec": float(cfg.solver.time_limit_s),
            "run_json": rf.name,
            "ok": int(bool(ok)),
            "stop_reason": stop_reason,
            "time_sec": float(res.runtime_s) if res.runtime_s is not None else float(elapsed),

            "status": str(res.status),
            "objective": float(res.objective) if (res.objective == res.objective) else float("nan"),
            "runtime_s": float(res.runtime_s) if res.runtime_s is not None else float("nan"),

            "N": int(N),
            "goal_tol_m": float(GOAL_TOL_M),
        }
        append_result_csv(str(results_csv), row, fieldnames)

        print(f"status   = {res.status}")
        print(f"objective= {res.objective}")
        print(f"elapsed  = {elapsed:.3f}s")

        if res.active_indices:
            print(f"\nFOOTSTEPS = [  # {rf.name}")
            for j in res.active_indices:
                if j < len(res.footsteps):
                    fs = res.footsteps[j]
                    print(f"    ('{fs.foot}', {fs.x:8.3f}, {fs.y:8.3f}, {fs.z:8.3f}, {math.degrees(fs.theta):8.2f}),")
            print("]")


        if ok:
            last_success = (rf, stones, res)

    # ===== summary =====
    ok_cnt = sum(1 for r in results if r["ok"])
    total = len(results)

    print("\n================ Summary ================")
    print(f"OK: {ok_cnt}/{total}")
    for r in results:
        name = Path(r["run_json"]).name
        if r["ok"]:
            print(f"[OK]   {name} | status={r['status']} obj={r['objective']:.6g} time={r['runtime_s']:.3f}s")
        else:
            print(f"[FAIL] {name} | status={r['status']} time={r['runtime_s']:.3f}s")

    # ===== plot only last success =====
    if last_success is None:
        print("\nNo successful run to plot.")
        return

    rf, stones, res = last_success
    out_png = out_dir / f"{rf.stem}_plan.png"
    plot_plan_3d(
        stones,
        res,
        out_png=str(out_png),
        title=f"MIQCQP plan ({rf.name})",
        thickness=0.04,
        pad_xy=0.0,
        view_top=False,
        show=True,
    )
    print(f"Wrote plot: {out_png}")


if __name__ == "__main__":
    main()