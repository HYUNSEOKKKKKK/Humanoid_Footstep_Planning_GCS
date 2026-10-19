from __future__ import annotations
from typing import Protocol, Optional
from ..graph import DiscreteFootstep, FootstepGraphNode

class IdealStepCalculatorInterface(Protocol):
    def compute_ideal_step(self, stance_step: DiscreteFootstep, start_of_swing: DiscreteFootstep) -> DiscreteFootstep:
        ...

    def compute_ideal_step_for_node(self, stance_node: FootstepGraphNode) -> DiscreteFootstep:
        ...
