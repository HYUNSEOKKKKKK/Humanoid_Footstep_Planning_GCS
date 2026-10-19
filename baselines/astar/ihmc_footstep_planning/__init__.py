"""Lightweight, Python-first reimplementation of IHMC's footstep graph-search planner.

This package is *not* affiliated with IHMC. It is intended to reproduce the core
graph-search mechanics from the IHMC open-source codebase, with minimal dependencies.

Current scope (Step 0):
- LatticePoint / DiscreteFootstep / FootstepGraphNode (discrete grid state)
- ParameterBasedStepExpansion (offset-based successor generation)
- AStarFootstepPlannerIterationConductor (one-iteration conductor + a small DirectedGraph)
- FootstepCostCalculator (ported structure; snapping/terrain hooks are stubs)
"""
from .robot_side import RobotSide
