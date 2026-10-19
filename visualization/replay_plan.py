"""Replay a saved GCS plan in the TORO kinematic viewer, or render it to MP4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco
from toro_footstep_walk import (run_render, run_interactive, build_mjcf_from_json_stones,
                                align_footsteps_to_stone1, build_schedule, set_footstep_pose)
from toro_real_walk import build_joint_map

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "outputs/gcs/plan.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/visualization/toro.mp4")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--check", action="store_true", help="load model and evaluate kinematic poses without a display")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--step-duration", type=float, default=0.4)
    args = parser.parse_args()
    data = json.loads(args.plan.read_text())
    steps = [(s["foot"], s["x"], s["y"], s["z"], s["yaw_deg"]) for s in data["footsteps"]]
    if len(steps) < 3 or {s[0] for s in steps[:2]} != {"L", "R"}:
        parser.error("plan must start with left and right initial foot poses")
    if args.fps < 1 or args.width < 2 or args.height < 2 or args.step_duration <= 0:
        parser.error("FPS, image dimensions, and step duration must be positive")
    if args.check:
        xml = build_mjcf_from_json_stones(str(args.plan), xy_scale=1.0)
        model = mujoco.MjModel.from_xml_string(xml)
        state = mujoco.MjData(model)
        joints = build_joint_map(model)
        aligned = align_footsteps_to_stone1(steps, str(args.plan), xy_scale=1.0)
        schedule, total = build_schedule(aligned, T_init=0.1, T_step=args.step_duration)
        for time in np.linspace(0, total, 30):
            set_footstep_pose(model, state, joints, schedule, float(time))
            if not np.isfinite(state.qpos).all():
                raise RuntimeError("non-finite kinematic pose")
        print(f"Loaded TORO model and checked 30 poses from {len(steps)} planned foot poses.")
    elif args.interactive:
        run_interactive(footsteps=steps, json_path=str(args.plan), xy_scale=1.0,
                        T_init=0.1, T_step=args.step_duration)
    else:
        if args.output.resolve().is_relative_to((ROOT / "data").resolve()) or args.output.resolve().is_relative_to((ROOT / "paper").resolve()):
            parser.error("select an output outside preserved data/ and paper/")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        run_render(footsteps=steps, output=str(args.output), json_path=str(args.plan), xy_scale=1.0,
                   fps=args.fps, width=args.width, height=args.height, T_init=0.1, T_step=args.step_duration)


if __name__ == "__main__":
    main()
