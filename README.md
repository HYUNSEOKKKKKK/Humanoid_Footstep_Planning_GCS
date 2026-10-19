# Humanoid Footstep Planning using Graphs of Convex Sets

**Hyunseok Kim, George Mesesan, Robert Schuller, Jinoh Lee, and Hae-Won Park**

Accepted for publication in **IEEE Robotics and Automation Letters (RA-L), 2026**, submission **26-2531**.

[Project page](https://hyunseokkkkkk.github.io/Humanoid_Footstep_Planning_GCS/) · [Accepted manuscript](paper/26-2531_03_MS.pdf) · [Supplementary video](paper/Video_Final.mp4) · [Reproduction guide](docs/reproduction.md)

This branch contains the GCS footstep planner, pretrained CVAE terrain generator,
A* and MIP comparison implementations, benchmark terrains, archived experimental
results, and a TORO kinematic visualization. A normal clone includes all runtime
Python sources and resources. No sibling repositories, ROS, RaiSim, or robot
connection are needed for the offline examples.

## 1. Install

Validated on **Ubuntu 24.04, Linux x86_64, Python 3.10**. The default environment
uses CPU PyTorch; a GPU is not required. Run commands from the repository root.

```bash
git clone https://github.com/HYUNSEOKKKKKK/Humanoid_Footstep_Planning_GCS.git
cd Humanoid_Footstep_Planning_GCS
conda env create -f environment.yml
conda activate footstep_planner
python -m pip check
python -m experiments.verify_artifacts
```

The Conda file installs Python, FFmpeg, and the pinned Python dependencies in
`requirements.lock.txt`. `requirements.txt` lists the direct dependencies. The
original C++ prototype is optional; there is no build step for the Python workflow.
The same `footstep_planner` environment runs GCS, CVAE, both comparison baselines,
and the TORO/MuJoCo visualization.

**MOSEK requires your own valid license** to solve GCS and MIP problems. Place it
at `~/mosek/mosek.lic`, or configure another location:

```bash
export MOSEKLM_LICENSE_FILE="$HOME/mosek/mosek.lic"
```

Licenses are obtained from [MOSEK](https://www.mosek.com/products/academic-licenses/)
and are not included in this repository. Terrain sampling, saved-result analysis,
and visualization of an existing plan work without a solver license.

## 2. Plan footsteps on a benchmark terrain

```bash
python main_VAE_optapi.py --terrain data/terrains/alpha/0.0/run_000.json --no-show
```

This loads a fixed terrain, builds the GCS problem, solves the convex relaxation,
recovers a discrete path, and solves the fixed-path problem. It writes:

- `outputs/gcs/plan.json`: terrain and ordered left/right foot poses, with position in metres and yaw in degrees.
- `outputs/gcs/plan.png`: terrain and planned footsteps.
- `outputs/gcs/vae_batch_results_optapi_*.csv`: feasibility, objectives, and measured runtime.

Omit `--no-show` to display the plot on a desktop. The default directory run uses
one terrain. To run an entire difficulty level:

```bash
python main_VAE_optapi.py --terrain_dir data/terrains/alpha/0.5 --max-terrains 0 --no-show --output-dir outputs/gcs_alpha05
```

The four-region benchmark uses nominal copy counts `[5, 4, 4, 5]`. The default
adapts to the number of input stones; `--copy_counts` overrides it explicitly.
Some generated terrains are infeasible under the planning constraints.

## 3. Run the comparison implementations

```bash
python -m experiments.compare_terrains --terrain data/terrains/alpha/0.0/run_000.json
```

This runs GCS, the IHMC-inspired A* implementation, and the Deits–Tedrake-inspired
MIP implementation on the same input. Results and per-planner logs are saved in
`outputs/comparison/`; `comparison.csv` records terrain hashes, feasibility,
reported runtime, and status. A failed planning trial is recorded as a result.
Use another output directory for a new run, or `--overwrite` to replace results.

```bash
python -m experiments.compare_terrains --terrain-dir data/terrains/alpha/0.5 --max-terrains 0 --output-dir outputs/comparison_alpha05
```

This command preserves the historical terrain-runner settings. It is a functional
comparison, not a certified reconstruction of the final paper's Fig. 4. In
particular, the historical GCS/A* runners use different initial stance widths.
See the [experiment-to-paper mapping](docs/reproduction.md) before interpreting
comparative success rates or timings.

## 4. Replay a plan or save a video

Check the saved plan and TORO model without a display:

```bash
python visualization/replay_plan.py --plan outputs/gcs/plan.json --check
```

Open the viewer on a desktop:

```bash
python visualization/replay_plan.py --plan outputs/gcs/plan.json --interactive
```

Save an MP4 on Linux with an EGL graphics driver:

```bash
MUJOCO_GL=egl python visualization/replay_plan.py --plan outputs/gcs/plan.json --output outputs/visualization/toro.mp4
```

The video uses the **saved planner output**, with geometry scale 1.0. It is a
kinematic replay, not a whole-body dynamics controller or a reproduction of the
hardware experiment. A supplied example, `examples/gcs_plan.json`, can be replayed
without running MOSEK. The real TORO deployment uses the external DLR control
stack described in the paper. Robot UDP transmission is disabled by default;
integration details are in [docs/hardware.md](docs/hardware.md).

## 5. Use or retrain the CVAE

For comparisons, use the **preserved benchmark JSONs** in `data/terrains/alpha/`:
100 terrains each for alpha 0.0, 0.5, and 1.0. They do not need to be regenerated.

To sample **new** terrains from the pretrained checkpoint:

```bash
python terrain_generation/generate.py --alpha 0.5 --stones 4 --samples 100 --output-dir outputs/new_terrains_alpha05
```

This does not retrain the model or overwrite the preserved benchmark. It saves
the seeds, model hash, and generated-file hashes in a manifest. Changing sampling
device or checkpoint can change the terrain sequence; new samples are not
asserted to match the historical benchmark byte-for-byte.

Train a new model explicitly:

```bash
python terrain_generation/train.py --epochs 50 --output outputs/checkpoints/my_cvae.pt
```

Training uses 40,000 synthetic conditional samples, latent dimension 8, batch
size 512, learning rate 2e-4, and seed 123 by default. The shipped checkpoints
remain separate from newly trained models.

## 6. Reproduce saved statistics and run checks

```bash
python -m experiments.reproduce_reference
python -m unittest discover -s tests -v
```

The first command recomputes **Table II** from raw sensitivity trials, reproduces
the saved offline Pump deadline summaries, and exports the cost pairs from the
preserved Pump logs. These steps do not require a solver license. See
[docs/reproduction.md](docs/reproduction.md) for full experiment commands,
normalization rules, and the remaining provenance gaps for the final figures.

## Repository layout

```text
src/                         GCS graph, optimization, recovery, and plotting code
main*.py                     Direct planning/demo entry points
terrain_generation/          CVAE model, train/sample commands, reference checkpoints
baselines/astar/             Graph-search comparison implementation
baselines/mip/               Mixed-integer comparison implementation
experiments/                 Offline benchmarks and reference analysis
visualization/               TORO assets and MuJoCo kinematic replay
data/terrains/               Preserved benchmark inputs
data/reference/              Original experimental CSVs and Pump logs
data/legacy/                 Earlier input samples retained for provenance
paper/                       Final submitted paper, video, source, and artifact hashes
website/                     GitHub Pages project page and deployment instructions
outputs/                     New runtime results; ignored by Git
docs/                        Reproduction details, source mapping, and hardware scope
```

This repository starts with the curated paper reproduction snapshot on `main`.
The original source commits and local changes are recorded in
`paper/source_repositories.json`; the complete former Git histories are preserved
in the author's local research archives. Historical result manifests retain their
original absolute path labels; executable code uses the integrated layout.

## Citation

```bibtex
@article{kim2026humanoidgcs,
  title   = {Humanoid Footstep Planning using Graphs of Convex Sets},
  author  = {Kim, Hyunseok and Mesesan, George and Schuller, Robert and Lee, Jinoh and Park, Hae-Won},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  note    = {Accepted for publication}
}
```

DOI, volume, and page numbers will be added when available. See
[LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md) for licensing and provenance.
