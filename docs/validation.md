# Local validation

Validated on 2026-10-19 using a newly created `footstep_planner` Conda
environment from the root environment specification. Python dependencies are
pinned in `requirements.lock.txt`; CPU PyTorch is sufficient.

- `pip check`: no broken requirements.
- Six offline unittest checks: artifact hashes, all 300 benchmark terrain schemas,
  repeatable sampling, checkpoint preservation and output protections, TORO model
  replay, and opt-in robot UDP behavior.
- GCS, A*, and MIP: all three returned feasible plans on
  `data/terrains/alpha/0.0/run_000.json`, using a valid local MOSEK license.
- Graph-construction experiment: one nominal case executed successfully.
- Pump replay: both smoke-test cycles succeeded for all three planners.
- CVAE sampling and one small training epoch: completed, with original checkpoints
  preserved. The small training check is not a full retraining of the paper model.
- TORO replay: model loaded and 30 kinematic poses checked. A 15-frame MP4
  was rendered using EGL and FFmpeg from a freshly solved GCS plan.
- Table II: recomputed from raw trials and rounded to 25.66% and 32.66%.
- Pump summaries: regenerated from raw cycle CSVs; 707 recorded cost pairs exported
  from the seven preserved window logs.

These are installation and integration checks. A new full 300-terrain timing
benchmark and a physical TORO experiment were not executed during packaging.
The outstanding final-figure provenance is documented in `reproduction.md`.
