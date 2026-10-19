# Reproduction guide and provenance

Run all commands from the repository root with `footstep_planner` active.
Reference files are in `data/` and `paper/`; new runs go to `outputs/`.
`paper/artifact_manifest.json` records the preserved files' SHA-256 hashes.

## What has been reproduced

| Paper component | Preserved source or results | Reproduction status |
|---|---|---|
| GCS relaxation and path recovery | `src/`, `main_VAE_optapi.py` | Licensed offline planning verified on a fixed terrain. |
| CVAE benchmark | `data/terrains/alpha/{0.0,0.5,1.0}` | All 300 local input JSONs preserved; sampling and training have separate commands. |
| Fig. 4, three-planner success/time curves | A*/MIP implementations and historical baseline CSVs | Final GCS run CSVs and exact final figure run selection have not been located. The comparison command is a functional runner, not an exact reconstruction of this figure. |
| Fig. 5, duplication sensitivity | `data/reference/gcs/Com10_3_extended/` | Raw trials and experiment code preserved. Final figure normalization/selection still requires confirmation; raw success counts must not be presented as its normalized bars. |
| Table II, edge-rule relaxation gaps | `data/reference/gcs/Com10_3_extended/raw_results.csv` | Recalculation from successful trials gives 25.66% and 32.66%, matching the manuscript. |
| Offline Pump deadline comparison | `data/reference/gcs/Com10_2/pump_planner_cycle_results.csv` | Summaries are regenerated from all 54 recorded cycles per planner. |
| Fig. 7, deployment timing | Final figure preserved in `paper/source/Fig/data_result.pdf` | Exact raw deployment timing trace has not been established. |
| Fig. 8, relaxed cost and gap | `data/reference/pump_logs/{3,...,9}.txt` | Cost pairs can be exported and plotted. Logs include startup records; the final paper's exact 100-trial selection is not yet established. |
| TORO hardware experiment | Final supplementary video; original UDP planner code | Hardware controller is an external DLR system. MuJoCo replay is a kinematic visualization. |

The submitted figure PDFs and LaTeX source are preserved in `paper/source/`.
They establish the submitted content but do not replace missing raw result
provenance. Source commits, including the pre-integration GCS revision, are in
`paper/source_repositories.json` and are preserved in the local research archives. The public repository starts
with a curated snapshot on `main`.

## Recompute preserved numerical results

```bash
python -m experiments.verify_artifacts
python -m experiments.reproduce_reference
```

Outputs:

- `outputs/reference/table_ii.csv`: medians recomputed from the raw successful sensitivity trials.
- `outputs/reference/pump_deadlines/`: detailed, compact, and LaTeX deadline summaries.
- `outputs/reference/pump_cost_pairs.csv`: all paired cost records, including startup windows.
- `outputs/reference/pump_cost_pairs.png`: a plot of those records, with tiny negative numerical gaps clipped only for display.

The exported raw gap is `100 * (C_round - C_relax) / C_relax`. Its original
value is retained in the CSV. The display plot is explicitly a plot of the
preserved logs, rather than a claim of exact Fig. 8 reconstruction.

## Run new experiments

The following commands execute the preserved experimental implementations.
They can take substantially longer than the introductory example. These runs
require a valid MOSEK license; measured runtimes depend on CPU, solver threads,
background load, and recovery seeds.

### Terrain comparison

```bash
python -m experiments.compare_terrains --terrain-dir data/terrains/alpha/0.0 --max-terrains 0 --output-dir outputs/compare_alpha00
python -m experiments.compare_terrains --terrain-dir data/terrains/alpha/0.5 --max-terrains 0 --output-dir outputs/compare_alpha05
python -m experiments.compare_terrains --terrain-dir data/terrains/alpha/1.0 --max-terrains 0 --output-dir outputs/compare_alpha10
```

The runner preserves the historical GCS, A*, and MIP terrain-runner settings:
GCS stance width 0.25 m, A* stance width 0.20 m, A* position tolerance 0.10 m,
and the original MIP goal-region/0.5 m fallback acceptance rule. It records
these settings in a manifest. Sharing terrain files alone does not establish
the identical feasibility constraints required by a final-paper comparison.
Before labeling new curves as Fig. 4 reproduction, recover the final settings
and run-selection metadata and align the interfaces explicitly.

The paper normalizes success over terrains solved by at least one planner.
That requires joining all planners by **terrain identity and difficulty**, not
dividing each planner by its own successful trials. Preserve failed trials,
and do not substitute the wall time of a whole batch for time to the first
feasible solution. `reported_time_sec` in the functional runner follows each
historical runner's timing boundary; it is not a newly certified timing metric.

### Graph construction and recovery

```bash
# One nominal case first; omit the range limits for the full benchmark.
python -m experiments.graph_sensitivity_extended --run --config nominal --terrain-start 0 --terrain-end 1 --output-dir outputs/graph_smoke

# All stored graph variants, 100 terrains each.
python -m experiments.graph_sensitivity_extended --run --output-dir outputs/graph_full

# Offline Pump comparison, two-cycle installation check.
python -m experiments.pump_deadlines run --paper-legacy --smoke-test --output-dir outputs/pump_smoke

# Complete preserved fixed-six-window, 53-key replay protocol.
python -m experiments.pump_deadlines run --paper-legacy --output-dir outputs/pump_full
```

The `paper-legacy` protocol is recovered from an earlier deployment revision.
It is named explicitly because the final manuscript also discusses a window of
four labels. Do not silently reinterpret one protocol as the other.

Additional review experiments remain available:

| Current module | Original filename |
|---|---|
| `experiments.pump_deadlines` | `Com10_2.py` |
| `experiments.graph_sensitivity` | `Com10_3.py` |
| `experiments.graph_sensitivity_extended` | `Com10_3_extended.py` |
| `experiments.horizon_region_sweep` | `Com_1_2.py` |
| `experiments.regioncount_sweep` | `Com_1_2_regioncount_extended.py` |
| `experiments.path_recovery` | `Com_1_5.py` |
| `experiments.pump_recovery` | `Com_1_5_TORO.py` |
| `experiments.time_expanded_horizon` | `main_VAE_optapi_Com1_2.py` |

Use `python -m MODULE --help` to inspect experiment-specific budgets and output
options. Historical manifests retain their original file names and paths.

## Data and environment preservation

Both CVAE checkpoints were imported without changing their weights. The
alpha datasets and region-count datasets are stored independently. The
region-count archive contains 100 terrains each for N=3,4,5 and 30 for N=6;
do not assume every historical condition contains 100 trials.

CPU sampling is deterministic for a fixed seed, model, and pinned environment,
but is not asserted to recreate the older GPU-generated dataset exactly.
The fixed benchmark files are therefore the reference inputs.

`requirements.lock.txt` records the validated Linux Python environment.
The root `environment.yml` defines one `footstep_planner` environment for GCS,
CVAE, A*, MIP, and TORO/MuJoCo replay. No personal MOSEK license is tracked.
