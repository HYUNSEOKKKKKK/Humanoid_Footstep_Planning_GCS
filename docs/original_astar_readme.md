# ihmc_py_planner (Step 0)

This is a Python-first translation of the *core* IHMC footstep graph-search planner pieces
you uploaded:

- `AStarIterationData.java` -> `graph_search/a_star_iteration_data.py`
- `AStarFootstepPlannerIterationConductor.java` -> `graph_search/a_star_conductor.py`
- `LatticePoint.java` -> `graph/lattice_point.py`
- `DiscreteFootstep.java` -> `graph/discrete_footstep.py`
- `DiscreteFootstepTools.java` (subset) -> `graph/discrete_footstep_tools.py`
- `FootstepGraphNode.java` -> `graph/footstep_graph_node.py`
- `FootstepExpansion.java` -> `step_expansion/footstep_expansion.py`
- `ParameterBasedStepExpansion.java` -> `step_expansion/parameter_based_step_expansion.py`
- `PartialExpansionManager.java` -> `step_expansion/partial_expansion_manager.py`
- `FootstepCostCalculator.java` -> `step_cost/footstep_cost_calculator.py` (snapping & terrain are stubs)

## Run demo
```bash
python examples/run_astar_flat.py
```

## Next steps (Step 1+)
To match the paper/IHMC output, you will want to port next:
- `FootstepPlannerHeuristicCalculator.java` (exact heuristic)
- `DefaultFootstepPlannerParameters*.java` + JSON loader for the real parameter set
- step checking: reachability, collision, cliff/obstacle checks (stepChecking/*)
- snapping: planar-region snapper + wiggling (footstepSnapping/*, polygonSnapping/*)


## Using IHMC DefaultFootstepPlannerParameters.json
The loader accepts IHMC's human-readable keys like "Min step length" and "Max step reach".
