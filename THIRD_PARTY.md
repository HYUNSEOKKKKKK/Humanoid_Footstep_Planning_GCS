# Sources and third-party material

The public repository contains a curated snapshot of these sources. Complete
former Git histories are preserved locally. Exact source commits and original
local modifications are listed in
`paper/source_repositories.json`.

| Component | Imported source | Notes |
|---|---|---|
| GCS planner | `HYUNSEOKKKKKK/footstep_planner`, `471e2eb` | Original planner, examples, and reviewer experiments. |
| CVAE generator | `HYUNSEOKKKKKK/CVAE_steppingstone`, `06eee7d` | Generator code, two checkpoints, plus preserved local benchmark inputs. |
| A* baseline | `HYUNSEOKKKKKK/footstep_planning_astar`, `1ddb496` | Python implementation inspired by/translated from IHMC planner components; not the full official IHMC stack. |
| MIP baseline | `HYUNSEOKKKKKK/footstep_planning_mip`, `ed9ec2f` | Project implementation inspired by Deits and Tedrake (Humanoids 2014); includes conic/Big-M implementation choices. |
| TORO visualization | `HYUNSEOKKKKKK/footstep_planning_visualize`, `859f643` | Kinematic visualization and associated TORO meshes/models, imported as supplied. |

The baseline implementations should be cited together with their source
methods; they are not represented as verbatim official upstream releases.
Original source notices must be retained in downstream distributions.

TORO model/mesh assets in `visualization/model/` and the submitted manuscript,
LaTeX figures, and supplementary video in `paper/` are preserved research
materials. They are not relicensed by the project's MIT license. The imported
visualization repository did not contain a standalone asset license; a
separate redistribution/reuse license for those assets has not been established
by this integration.

MOSEK is a separately licensed solver. PyTorch, NumPy, SciPy, Matplotlib,
MuJoCo, MeshCat, and other installed dependencies retain their own licenses.
The optional pybind11 dependency is obtained from its original upstream and
retains its BSD-style license.
