from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..parameters import DefaultFootstepPlannerParameters
from ..robot_side import RobotSide
from ..graph import DiscreteFootstep, FootstepGraphNode
from ..graph.discrete_footstep_tools import compute_distance_between_foot_polygons, construct_step_in_previous_step_frame
from .partial_expansion_manager import PartialExpansionManager
from .ideal_step import IdealStepCalculatorInterface

@dataclass
class ParameterBasedStepExpansion:
    parameters: DefaultFootstepPlannerParameters
    foot_polygons: Dict[RobotSide, List[Tuple[float, float]]]
    ideal_step_calculator: Optional[IdealStepCalculatorInterface] = None

    # cached full expansion offsets
    x_offsets: List[float] = field(default_factory=list)
    y_offsets: List[float] = field(default_factory=list)
    yaw_offsets: List[float] = field(default_factory=list)

    partial_expansion_enabled: bool = False
    expansion_managers: Dict[FootstepGraphNode, PartialExpansionManager] = field(default_factory=dict)

    def initialize(self) -> None:
        # Mirrors IHMC ParameterBasedStepExpansion.initialize()
        self.x_offsets.clear()
        self.y_offsets.clear()
        self.yaw_offsets.clear()

        from ..graph.lattice_point import LatticePoint

        max_reach_sq = self.parameters.getMaxStepReach() ** 2

        x = self.parameters.getMinStepLength()
        while x <= self.parameters.getMaxStepReach() + 1e-12:
            y = self.parameters.getMinStepWidth()
            while y <= self.parameters.getMaxStepWidth() + 1e-12:
                # IHMC uses reach around (x, y - idealFootstepWidth)
                rel_y_to_ideal = y - self.parameters.getIdealFootstepWidth()
                reach_sq = x * x + rel_y_to_ideal * rel_y_to_ideal
                if reach_sq > max_reach_sq:
                    y += LatticePoint.grid_size_xy
                    continue

                yaw = self.parameters.getMinStepYaw()
                while yaw <= self.parameters.getMaxStepYaw() + 1e-12:
                    # clearance check between feet polygons
                    stance = DiscreteFootstep(0.0, 0.0, 0.0, RobotSide.RIGHT)
                    candidate = DiscreteFootstep(x, y, yaw, RobotSide.LEFT)
                    dist = compute_distance_between_foot_polygons(stance, candidate, self.foot_polygons)
                    if dist >= self.parameters.getMinClearanceFromStance():
                        self.x_offsets.append(x)
                        self.y_offsets.append(y)
                        self.yaw_offsets.append(yaw)
                    yaw += LatticePoint.grid_size_yaw

                y += LatticePoint.grid_size_xy
            x += LatticePoint.grid_size_xy

        self.partial_expansion_enabled = self.parameters.getMaxBranchFactor() > 0
        self.expansion_managers.clear()

    def do_full_expansion(self, node_to_expand: FootstepGraphNode, full_expansion_to_pack: List[FootstepGraphNode]) -> None:
        full_expansion_to_pack.clear()
        step_side = node_to_expand.get_first_step_side()

        for i in range(len(self.x_offsets)):
            step_length = self.x_offsets[i]
            # In IHMC: step width/yaw are negated if RIGHT side
            step_width = step_side.negate_if_right_side(self.y_offsets[i])
            step_yaw = step_side.negate_if_right_side(self.yaw_offsets[i])

            child_step = construct_step_in_previous_step_frame(step_length, step_width, step_yaw, node_to_expand.get_second_step())
            child_node = FootstepGraphNode(node_to_expand.get_second_step(), child_step)
            if child_node not in full_expansion_to_pack:
                full_expansion_to_pack.append(child_node)

        # Optional: expansion mask (requires an ideal step calculator)
        if self.ideal_step_calculator is not None and self.parameters.getEnableExpansionMask():
            self._apply_mask(full_expansion_to_pack, node_to_expand)

    def do_iterative_expansion(self, stance_step: FootstepGraphNode, expansion_to_pack: List[FootstepGraphNode]) -> bool:
        if self.partial_expansion_enabled:
            manager = self.expansion_managers.get(stance_step)
            if manager is None:
                manager = PartialExpansionManager(self.parameters)
                full = []
                self.do_full_expansion(stance_step, full)
                manager.initialize(full)
                self.expansion_managers[stance_step] = manager
            manager.pack_partial_expansion(expansion_to_pack)
            return not manager.finished_expansion()
        else:
            self.do_full_expansion(stance_step, expansion_to_pack)
            return False

    # -------------------- expansion mask (ported structure; simplified) --------------------
    def _apply_mask(self, list_to_filter: List[FootstepGraphNode], stance_node: FootstepGraphNode) -> None:
        # This mask in IHMC reduces expansions far from the ideal step.
        # Here we implement the same logic pattern (using Manhattan distances) but keep the masks as simple sets.
        ideal_step = self.ideal_step_calculator.compute_ideal_step_for_node(stance_node)  # type: ignore
        min_xy = self._compute_min_xy_manhattan_distance(list_to_filter, ideal_step)
        min_yaw = self._compute_min_yaw_distance(list_to_filter, ideal_step)

        # In IHMC these come from parameters (as bit masks). We use a conservative default:
        xy_mask: Set[int] = set(range(0, 999))
        yaw_mask: Set[int] = set(range(0, 999))

        filtered = []
        for node in list_to_filter:
            xy_d = ideal_step.compute_xy_manhattan_distance(node.get_second_step()) - min_xy
            if xy_d not in xy_mask:
                continue
            yaw_d = ideal_step.compute_yaw_index_distance(node.get_second_step()) - min_yaw
            if yaw_d not in yaw_mask:
                continue
            filtered.append(node)

        list_to_filter.clear()
        list_to_filter.extend(filtered)

    @staticmethod
    def _compute_min_xy_manhattan_distance(list_to_filter: List[FootstepGraphNode], ideal_step: DiscreteFootstep) -> int:
        return min(ideal_step.compute_xy_manhattan_distance(n.get_second_step()) for n in list_to_filter) if list_to_filter else 0

    @staticmethod
    def _compute_min_yaw_distance(list_to_filter: List[FootstepGraphNode], ideal_step: DiscreteFootstep) -> int:
        return min(ideal_step.compute_yaw_index_distance(n.get_second_step()) for n in list_to_filter) if list_to_filter else 0
