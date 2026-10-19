"""Run GCS, A*, and MIP on shared terrain files and save a comparison CSV.

This runner records the preserved terrain-runner settings. It is a functional
comparison; reproducing a published figure additionally requires its run selection
and normalization protocol (see docs/reproduction.md).
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load_astar():
    sys.path.insert(0, str(ROOT / "baselines/astar"))
    spec = importlib.util.spec_from_file_location("terrain_astar_runner", ROOT / "baselines/astar/examples/plot_astar_stones_3d.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_astar(terrain, timeout):
    m = load_astar()
    args = SimpleNamespace(stance_width=0.20, xy_tol=0.10, yaw_tol_deg=10.0,
                           max_iter=50000, time_limit_sec=timeout, print_every=0)
    foot = [(-0.075, -0.035), (-0.075, 0.035), (0.075, 0.035), (0.075, -0.035)]
    polygons = {m.RobotSide.LEFT: foot, m.RobotSide.RIGHT: foot}
    result = m.solve_one(str(terrain), args, polygons, m.DefaultFootstepPlannerParameters())
    return bool(result["ok"]), result["elapsed_sec"], result.get("stop_reason", "")


def run_mip(terrain, timeout, horizon):
    sys.path.insert(0, str(ROOT / "baselines/mip/src"))
    from micqp_planner.config import PlannerConfig
    from micqp_planner.terrain import load_problem_json
    from micqp_planner.mosek_mico import solve_miqcqp_mosek
    regions, left, right, goal_left, goal_right = load_problem_json(str(terrain))
    config = PlannerConfig()
    config.solver.verbose = False
    config.solver.time_limit_s = timeout
    config.trig.num_segments = 6
    config.disable_trim = False
    config.enforce_trim_prefix = True
    config.obj.q_trim = -1.0
    start = time.perf_counter()
    result = solve_miqcqp_mosek(regions=regions, start_left=left, start_right=right,
                                goal_left=goal_left, goal_right=goal_right, cfg=config, N=horizon)
    elapsed = time.perf_counter() - start
    active = [result.footsteps[i] for i in result.active_indices if i < len(result.footsteps)]
    final = {foot: next((s for s in reversed(active) if s.foot == foot), None) for foot in ("L", "R")}
    stones = json.loads(terrain.read_text())["terrain"]["stones"]
    last_region = stones[-1].get("name", f"stone_{len(stones)}")
    reached = all(final.values()) and all(s.region == last_region for s in final.values())
    if not reached and all(final.values()):
        reached = all(math.dist((final[f].x, final[f].y, final[f].z), (g.x, g.y, g.z)) <= 0.5
                      for f, g in [("L", goal_left), ("R", goal_right)])
    ok = result.status in {"optimal", "time_limit", "feasible"} and bool(active) and reached
    return bool(ok), float(result.runtime_s if result.runtime_s is not None else elapsed), str(result.status)


def run_gcs(terrain, timeout, output):
    output.mkdir(parents=True, exist_ok=True)
    log = output / "run.log"
    with log.open("w") as stream:
        result = subprocess.run([sys.executable, str(ROOT / "main_VAE_optapi.py"),
                                 "--terrain", str(terrain), "--trial_budget_sec", str(timeout),
                                 "--no-show", "--output-dir", str(output)],
                                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    csvs = sorted(output.glob("vae_batch_results_optapi_*.csv"))
    if result.returncode or not csvs:
        return False, float("nan"), f"see {log}"
    row = next(csv.DictReader(csvs[-1].open()))
    return row["ok"] == "1", float(row["time_sec"]), row.get("error", "") or "reached"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", type=Path, help="one terrain JSON")
    parser.add_argument("--terrain-dir", type=Path, default=ROOT / "data/terrains/alpha/0.0")
    parser.add_argument("--max-terrains", type=int, default=1, help="0 means all terrains")
    parser.add_argument("--planners", nargs="+", choices=("gcs", "astar", "mip"), default=["gcs", "astar", "mip"])
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--mip-horizon", type=int, default=14)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/comparison")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.time_limit <= 0 or args.max_terrains < 0:
        parser.error("time limit must be positive and max terrains nonnegative")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.is_relative_to(ROOT / "data"):
        parser.error("select an output outside preserved data/")
    csv_path = args.output_dir / "comparison.csv"
    if csv_path.exists() and not args.overwrite:
        parser.error("output already exists; select another directory or pass --overwrite")
    terrains = [args.terrain.resolve()] if args.terrain else sorted(args.terrain_dir.glob("run_*.json"))
    if not args.terrain and args.max_terrains:
        terrains = terrains[:args.max_terrains]
    if not terrains or any(not p.is_file() for p in terrains):
        parser.error("no valid terrain files found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    rows = []
    errors = 0
    for i, terrain in enumerate(terrains):
        for planner in args.planners:
            try:
                if planner == "gcs":
                    ok, elapsed, status = run_gcs(terrain, args.time_limit, args.output_dir / f"gcs_{i:03d}")
                else:
                    with (args.output_dir / f"{planner}_{i:03d}.log").open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
                        if planner == "astar":
                            ok, elapsed, status = run_astar(terrain, args.time_limit)
                        else:
                            ok, elapsed, status = run_mip(terrain, args.time_limit, args.mip_horizon)
            except Exception as exc:
                ok, elapsed, status = False, float("nan"), f"error: {type(exc).__name__}: {exc}"
                errors += 1
            row = {"planner": planner, "terrain": str(terrain.relative_to(ROOT)) if terrain.is_relative_to(ROOT) else str(terrain),
                   "terrain_sha256": hashlib.sha256(terrain.read_bytes()).hexdigest(), "ok": int(ok),
                   "reported_time_sec": elapsed, "status": status}
            rows.append(row)
            print(f"{planner}: {terrain.name}: ok={ok}, time={elapsed:.3f}s, status={status}")
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
    manifest = {"protocol": "preserved-terrain-runners-v1", "python": sys.version,
                "time_limit_sec": args.time_limit, "mip_horizon": args.mip_horizon,
                "gcs_stance_width": 0.25, "astar_stance_width": 0.20,
                "note": "Original runner settings; this is not a certified reconstruction of Figure 4. Read docs/reproduction.md.",
                "results": rows}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    print(f"Saved {csv_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
