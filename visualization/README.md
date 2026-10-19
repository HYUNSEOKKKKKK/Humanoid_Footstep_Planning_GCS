# TORO / MuJoCo visualization

Use the repository's shared `footstep_planner` environment. The TORO model,
meshes, original visualization scripts, and saved-plan replay entry point are
included here. See the [root README](../README.md) for installation.

From the repository root, check the supplied example plan without a display:

```bash
conda activate footstep_planner
python visualization/replay_plan.py --plan examples/gcs_plan.json --check
```

Open the interactive viewer:

```bash
python visualization/replay_plan.py --plan examples/gcs_plan.json --interactive
```

Render an MP4 on Linux with an EGL graphics driver:

```bash
MUJOCO_GL=egl python visualization/replay_plan.py --plan examples/gcs_plan.json --output outputs/visualization/toro.mp4
```

To visualize a newly solved plan, use `--plan outputs/gcs/plan.json`. The renderer
loads the saved terrain and foot poses and creates a kinematic replay. The
supplied example can be viewed and rendered without a MOSEK license.

The paper's original [supplementary video](../paper/Video_Final.mp4) is preserved
separately from newly rendered results. The [project page](https://hyunseokkkkkk.github.io/Humanoid_Footstep_Planning_GCS/)
also provides an embedded player.

`walk_viewer.py` is an earlier generic `dm_control` humanoid demonstration with
an optional `dm_control` dependency. The paper's TORO workflow uses `replay_plan.py`.
