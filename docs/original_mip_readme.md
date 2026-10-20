# MOSEK-based MIQCQP Footstep Planner (Reproduction Package)

This repo is a runnable reproduction-style implementation of the formulation in:
**Robert Deits & Russ Tedrake (2014)**, *Footstep Planning on Uneven Terrain with Mixed-Integer Convex Optimization*.

This bundle is configured to run **with MOSEK** (Fusion API) by default.
Indicator constraints from the paper are implemented via **Big-M** linearizations,
and quadratic terms are represented with conic constraints (Mixed-Integer Conic Optimization / MICO).

## 0) Requirements
- A working **MOSEK** installation (pip package `mosek`)
- A valid **MOSEK license** (Personal Academic License is free for students/faculty)

In the integrated `Humanoid_Footstep_Planning_GCS` project, the MIP runner
auto-detects `<repository>/.local/licenses/mosek.lic` when no license environment
variable is set. To configure the shared `footstep_planner` environment for both
GCS and MIP, follow the [current installation guide](../README.md#1-install).
The sections below preserve the original standalone package workflow.

## 1) Create conda environment (name: footstep_mip)

From the repo root:

```bash
conda env create -f env-mosek.yml
conda activate footstep_mip
```

## 2) Run the example

```bash
# default (PyCharm-friendly): runs the easy example and writes to ./out
python scripts/run_planner.py

# or specify a terrain explicitly
python scripts/run_planner.py --terrain examples/stepping_stones_example.json --N 12
```

Outputs:
- `out/plan.json`
- `out/plan.png`

## 3) Run your own terrain JSON

If your JSON contains `terrain.stones` (each with center/width/height/yaw_deg), it will be loaded and plotted.
You can also override start/goal from CLI:

```bash
python scripts/run_planner.py \
  --terrain your_terrain.json --N 12 \
  --start-left  0,0.10,0,0 \
  --start-right 0,-0.10,0,0 \
  --goal 2.0,2.5,0,0
```

Pose format: `x,y,z,theta` (theta in radians).

## 4) Batch benchmark over a folder

```bash
python scripts/benchmark_folder.py --folder terrains --pattern "*.json" --N 12 --time-limit 30
```

Produces a CSV summary in the output folder.

---

### Notes / Caveats
- To stay conservative (avoid infeasible solutions due to trig approximation), increase `--segments`
  or slightly reduce reachability radii in `src/micqp_planner/config.py`.
- Default solver backend is **MOSEK**; you can still try `--solver gurobi` if you happen to have it installed.


### Easy mode (default)
The script enables an *easy mode* by default (larger reachability radii, trimming disabled, and fewer trig segments) so you can confirm your environment and MOSEK license quickly. To switch back, edit `scripts/run_planner.py` and set `EASY_MODE = False`.
