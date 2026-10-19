from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class DefaultFootstepPlannerParameters:
    """A minimal Python mirror of IHMC's DefaultFootstepPlannerParametersReadOnly fields used in Step 0/1.

    Notes:
    - IHMC stores parameters via StoredPropertySet with human-readable keys (e.g., "Min step width").
    - This class supports both IHMC-style keys and snake_case/camelCase variants.
    """

    # --- Expansion / reachability (used by ParameterBasedStepExpansion) ---
    min_step_length: float = -0.25
    min_step_width: float = 0.2
    min_step_yaw: float = -0.0             #-0.27rad -> -15도

    max_step_width: float = 0.3
    max_step_yaw: float = 0.54                #0.27rad -> +15도
    max_step_reach: float = 0.25 # radius constraint around ideal width

    ideal_footstep_width: float = 0.23
    min_clearance_from_stance: float = 0.0

    max_branch_factor: int = -1  # <=0 disables partial expansion
    enable_expansion_mask: bool = False

    # --- Cost weights (used by FootstepCostCalculator) ---
    yaw_weight: float = 0.2
    forward_weight: float = 0.2
    lateral_weight: float = 0.2
    cost_per_step: float = 0.15
    step_up_weight: float = 0.2
    step_down_weight: float = 0.2
    roll_weight: float = 0.2
    pitch_weight: float = 0.2

    # --- A* parameters (not used in Step 0 demo, but needed soon) ---
    astar_heuristics_weight: float = 1.5
    final_turn_proximity: float = 0.2

    @staticmethod
    def from_json(path: str | Path) -> "DefaultFootstepPlannerParameters":
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        def get(*keys, default=None):
            for k in keys:
                if k in data:
                    return data[k]
            return default

        return DefaultFootstepPlannerParameters(
            astar_heuristics_weight=float(get("AStar heuristics weight", "astarHeuristicsWeight", "astar_heuristics_weight", default=1.5)),
            final_turn_proximity=float(get("Final turn proximity", "finalTurnProximity", "final_turn_proximity", default=0.2)),
            max_branch_factor=int(get("Max branch factor", "maxBranchFactor", "max_branch_factor", default=-1)),
            enable_expansion_mask=bool(get("Enable expansion mask", "enableExpansionMask", "enable_expansion_mask", default=True)),

            ideal_footstep_width=float(get("Ideal footstep width", "idealFootstepWidth", "ideal_footstep_width", default=0.22)),
            min_clearance_from_stance=float(get("Min clearance from stance", "minClearanceFromStance", "min_clearance_from_stance", default=0.0)),

            min_step_length=float(get("Min step length", "minStepLength", "min_step_length", default=-0.25)),
            min_step_width=float(get("Min step width", "minStepWidth", "min_step_width", default=0.15)),
            min_step_yaw=float(get("Min step yaw", "minStepYaw", "min_step_yaw", default=-0.3)),

            max_step_width=float(get("Max step width", "maxStepWidth", "max_step_width", default=0.4)),
            max_step_reach=float(get("Max step reach", "maxStepReach", "max_step_reach", default=0.45)),
            max_step_yaw=float(get("Max step yaw", "maxStepYaw", "max_step_yaw", default=0.6)),

            yaw_weight=float(get("Yaw weight", "yawWeight", "yaw_weight", default=0.2)),
            forward_weight=float(get("Forward weight", "forwardWeight", "forward_weight", default=0.2)),
            lateral_weight=float(get("Lateral weight", "lateralWeight", "lateral_weight", default=0.2)),
            cost_per_step=float(get("Cost per step", "costPerStep", "cost_per_step", default=0.15)),
            step_up_weight=float(get("Step up weight", "stepUpWeight", "step_up_weight", default=0.2)),
            step_down_weight=float(get("Step down weight", "stepDownWeight", "step_down_weight", default=0.2)),
            roll_weight=float(get("Roll weight", "rollWeight", "roll_weight", default=0.2)),
            pitch_weight=float(get("Pitch weight", "pitchWeight", "pitch_weight", default=0.2)),
        )

    # Java-style getters (used by our ports for easier diffing)
    def getAStarHeuristicsWeight(self): return self.astar_heuristics_weight
    def getFinalTurnProximity(self): return self.final_turn_proximity
    def getMaxBranchFactor(self): return self.max_branch_factor
    def getEnableExpansionMask(self): return self.enable_expansion_mask

    def getIdealFootstepWidth(self): return self.ideal_footstep_width
    def getMinClearanceFromStance(self): return self.min_clearance_from_stance

    def getMinStepLength(self): return self.min_step_length
    def getMinStepWidth(self): return self.min_step_width
    def getMinStepYaw(self): return self.min_step_yaw

    def getMaxStepWidth(self): return self.max_step_width
    def getMaxStepReach(self): return self.max_step_reach
    def getMaxStepYaw(self): return self.max_step_yaw

    def getYawWeight(self): return self.yaw_weight
    def getForwardWeight(self): return self.forward_weight
    def getLateralWeight(self): return self.lateral_weight
    def getCostPerStep(self): return self.cost_per_step
    def getStepUpWeight(self): return self.step_up_weight
    def getStepDownWeight(self): return self.step_down_weight
    def getRollWeight(self): return self.roll_weight
    def getPitchWeight(self): return self.pitch_weight
