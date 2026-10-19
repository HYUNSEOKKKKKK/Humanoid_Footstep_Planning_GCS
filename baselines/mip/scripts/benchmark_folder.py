#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import argparse
import csv
import math
import json
from typing import Optional, List

from micqp_planner.config import PlannerConfig
from micqp_planner.terrain import load_problem_json, FootPose, SteppingStone
try:
    from micqp_planner.mosek_mico import solve_miqcqp_mosek
except Exception:
    solve_miqcqp_mosek = None



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder with terrain JSON files")
    ap.add_argument("--pattern", default="*.json", help="Glob pattern")
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--out", default="out/benchmark.csv")
    ap.add_argument("--time-limit", type=float, default=30.0)
    args = ap.parse_args()

    cfg = PlannerConfig()
    cfg.solver.time_limit_s = float(args.time_limit)
    cfg.solver.verbose = False

    files = sorted(Path(args.folder).glob(args.pattern))
    if not files:
        raise RuntimeError("No JSON files found.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in files:
        try:
            regions, sl, sr, goal_left, goal_right = load_problem_json(str(f))
            if sl is None or sr is None or goal_left is None or goal_right is None:
                rows.append({"file": f.name, "status": "missing_start_goal", "runtime_s": "", "objective": ""})
                continue
            if solve_miqcqp_mosek is None:
                raise RuntimeError('MOSEK backend not available. Install mosek and set a license.')
            res = solve_miqcqp_mosek(regions, sl, sr, goal_left, goal_right, N=int(args.N), cfg=cfg)
            rows.append({"file": f.name, "status": res.status, "runtime_s": f"{res.runtime_s:.4f}",
                         "objective": f"{res.objective:.6g}" if res.objective == res.objective else ""})
        except Exception as e:
            rows.append({"file": f.name, "status": f"error:{type(e).__name__}", "runtime_s": "", "objective": ""})

    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["file", "status", "runtime_s", "objective"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok = sum(1 for r in rows if r["status"] in ("optimal", "time_limit"))
    print(f"Wrote {out_path} ({ok}/{len(rows)} solved)")

if __name__ == "__main__":
    main()
