from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List
import math


@dataclass
class TrigApproxConfig:
    """
    Piecewise-linear approximation settings for sin/cos.
    We approximate over theta in [-pi, pi] with L equal-length segments.

    Paper: uses segment binaries S_{l,j}, C_{l,j} with constraints (7)-(8).
    """
    num_segments: int = 8
    theta_min: float = -math.pi
    theta_max: float = math.pi

@dataclass
class ReachabilityConfig:
    """
    Geometry constraints for footstep feasibility (Option A):
      - Consecutive L-R separation bounds (SOC + linear non-crossing)
      - Same-foot span bounds (SOC)
      - Linear bounds in z and yaw per step
    """
    # L-R distance upper bound (double-support geometry)
    lr_sep_max: float = 0.35     # meters

    # minimum lateral width to prevent crossing (assume +y is left)
    lr_sep_min: float = 0.20     # meters

    # same-foot movement bound (L->L, R->R) in XY
    same_foot_span: float = 0.65  # meters (tune)

    # additional linear displacement bounds
    max_dz: float = 0.3
    max_dtheta: float = math.radians(30.0)


@dataclass
class BoundsConfig:
    """
    Global bounds f_min <= f_j <= f_max and delta bounds Δf_min <= f_j-f_{j-1} <= Δf_max.
    Paper lists these generically (Section II-D). We auto-fit abs bounds from terrain,
    but keep delta bounds explicit.
    """
    # delta bounds (x,y,z,theta) per step
    dx_max: float = 0.50
    dy_max: float = 0.50
    dz_max: float = 0.30
    dtheta_max: float = math.pi  # generous; reachability also limits yaw effect

    # absolute bounds margins (added around terrain bbox)
    abs_xy_margin: float = 1.0
    abs_z_margin: float = 0.5


@dataclass
class ObjectiveConfig:
    """
    Paper objective (Section II-D):
        (f_N - g)^T Qg (f_N - g) + sum_j q_t t_j + sum_j (f_{j+1}-f_j)^T Qr (f_{j+1}-f_j)
    """
    # goal weight
    qg_pos: float = 50.0
    qg_yaw: float = 10.0

    # relative step smoothness weight
    qr_pos: float = 10.0
    qr_yaw: float = 1.0

    # trim incentive (negative -> encourages trimming / fewer active steps)
    q_trim: float = -3.0


@dataclass
class SolverConfig:
    time_limit_s: float = 10.0
    mip_gap: float = 1e-4
    verbose: bool = True


@dataclass
class PlannerConfig:
    trig: TrigApproxConfig = field(default_factory=TrigApproxConfig)
    reach: ReachabilityConfig = field(default_factory=ReachabilityConfig)
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    obj: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    # Whether to force trim variables to form a monotone *suffix* (0...0, 1...1).
    # NOTE: The field name is kept for backward compatibility with older runners.
    enforce_trim_prefix: bool = False

    # DEBUG/EASE: disable trimming variables entirely (fix t_j=0).
    # Set to False if you want to reproduce the paper-style trim behavior.
    disable_trim: bool = True
