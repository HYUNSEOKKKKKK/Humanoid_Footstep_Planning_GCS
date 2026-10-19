# Footstep Planner using Graph-of-Convex-Sets (GCS)

This repository provides a pipeline for **GCS-based footstep
planning**, including:

-   Graph-of-Convex-Sets formulation for discrete--continuous footstep
    optimization\
-   Sliding-window replanning for pump-it-up dynamic stepping for TORO\
-   Terrain generation via CVAE & WaveFunctionCollapse (WFC)\


------------------------------------------------------------------------

## 1. Environment Setup

We recommend using a conda environment due to several dependencies required by this project.

If you don't have conda installed, install Miniconda (or Miniforge/Mambaforge) as follows:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

After installation, restart your terminal or run:
```
source ~/.bashrc
```
If you already have conda installed, clone the repository and create the environment:
```
git clone https://github.com/HYUNSEOKKKKKK/footstep_planner.git
cd footstep_planner
conda env create -f environment.yml
conda activate footstep_planner
```


This creates a fully configured environment required to run the footstep planner

------------------------------------------------------------------------

## 2. Project Structure

    footstep_planner/
    ├── configs/                       # Experiment and terrain configuration files (.json)
    ├── cpp/                           # C++ implementation and pybind11 bindings (optional)
    ├── python/                        # Utility Python scripts (deprecated or legacy helper code)
    ├── src/                           # Main source code for GCS model, solver, and visualization
    │   ├── data/                      # Terrain generation, stepping-stone data, VAE outputs
    │   ├── graph/                     # GCS graph construction (vertices, edges, modes)
    │   ├── model/                     # Optimization model builders (Fusion API / OptAPI)
    │   ├── solver/                    # Relaxed/fixed solve logic, greedy fallback, stochastic trials
    │   └── viz/                       # Visualization utilities for footsteps, stones, trajectories
    ├── environment.yml                # Conda environment specification
    ├── main.py                        # Basic example entrypoint
    ├── main_interactive_fusionapi.py  # Interactive planning example using Fusion API
    ├── main_longfootsteps_optapi.py   # Long sequence stepping planning using OptAPI
    ├── main_pump_TORO_fusionapi.py    # Pump It Up scenario using Fusion API
    ├── main_pump_TORO_optapi.py       # Pump It Up scenario using OptAPI
    ├── main_replanning_fusionapi.py   # Per-cycle replanning example (Fusion API)
    ├── main_replanning_optapi.py      # Per-cycle replanning example (OptAPI)
    ├── main_VAE(fusionapi).py         # VAE-based terrain generation + planning demo
    ├── main_WFC(fusionapi).py         # Wave Function Collapse terrain generation + planning
    ├── pump_dancepad.png              # Example image used in documentation
    └── README.md



------------------------------------------------------------------------

## 3. How to run 

All commands below assume that the conda environment is already created and activated:

### 3.1 TORO Pump-It-Up demo 

To reproduce the TORO Pump-It-Up demo shown in the paper (with a 1.2 s replanning interval), run:

```
python main_pump_TORO_optapi.py
```
This script:
-   builds the GCS-based footstep planning problem for the Pump-It-Up dance
-   solves it using the MOSEK Optimizer API (OptAPI), and
-   logs cost information corresponding to the results reported in the paper.

By default, the time interval for replanning is controlled by:
`STEP_PERIOD_S = 1.2`
This matches the 1.2-second replanning period used in the paper.

The demo runs on a fixed sequence of **28 dance-pad labels**, and the label order can be modified via the variable:
`sequence_key`

During execution, the script prints (per replanning cycle):

- the next **6 upcoming step labels** currently considered,
- the **[relaxed] Cost** returned by the relaxation,
- the **[fixed] Cost** after recovering a discrete solution.

If you want to adjust more advanced behaviors (e.g. graph construction, edge rules, solver settings, timing structure, or logging format),  
please refer to:
`run_main_DDR_sliding_TORO_optapi.py`
This function contains the core logic for sliding-window replanning, label-window updates, GCS model building, and the relaxed→fixed solve pipeline.

### 3.2 Other entry points

You can run other examples by calling the corresponding `main_*.py` script:

------------------------------------------------------------------------

## 4. Contact

hyunseak@kaist.ac.kr
