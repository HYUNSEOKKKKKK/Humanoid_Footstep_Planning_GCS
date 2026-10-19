from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import math
import time

import numpy as np

from .config import PlannerConfig
from .terrain import SafeRegion, FootPose
from .piecewise_trig import build_piecewise_linear_sin_cos

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:  # pragma: no cover
    gp = None
    GRB = None


@dataclass
class Footstep:
    idx: int
    foot: str  # "L" or "R"
    x: float
    y: float
    z: float
    theta: float  # radians
    s: float
    c: float
    region: str
    trim: bool


@dataclass
class SolveResult:
    status: str
    runtime_s: float
    objective: float
    footsteps: List[Footstep]
    active_indices: List[int]


def _require_gurobi():
    if gp is None:
        raise ImportError("gurobipy is required. Install Gurobi + `pip install gurobipy`.")


def solve_miqcqp(regions: List[SafeRegion],
                start_left: FootPose,
                start_right: FootPose,
                goal: FootPose,
                N: int,
                cfg: PlannerConfig,
                abs_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> SolveResult:
    """
    Solve the MIQCQP footstep planning problem following Deits & Tedrake (2014).

    Args:
        regions: convex safe regions A f <= b, f=[x,y,z,theta]
        start_left, start_right: initial fixed foot poses (f1,f2 in the paper)
        goal: desired goal pose for the LAST footstep
        N: total number of footsteps in the optimization (INCLUDING the first two fixed ones)
        cfg: PlannerConfig
        abs_bounds: optional dict with keys x,y,z,theta -> (min,max). If omitted, wide defaults are used.

    Returns:
        SolveResult with full footsteps and also active_indices after trimming a prefix where t_j==1.
    """
    _require_gurobi()
    if N < 3:
        raise ValueError("N must be >= 3 (includes 2 fixed initial feet).")
    R = len(regions)
    if R == 0:
        raise ValueError("Need at least one safe region.")

    # Bounds
    if abs_bounds is None:
        abs_bounds = {
            "x": (-10.0, 10.0),
            "y": (-10.0, 10.0),
            "z": (-2.0, 2.0),
            "theta": (-math.pi, math.pi),
        }

    trig = build_piecewise_linear_sin_cos(cfg.trig.num_segments, cfg.trig.theta_min, cfg.trig.theta_max)
    L = cfg.trig.num_segments

    t0 = time.perf_counter()
    m = gp.Model("miqcqp_footstep_planner")

    m.Params.TimeLimit = float(cfg.solver.time_limit_s)
    m.Params.MIPGap = float(cfg.solver.mip_gap)
    m.Params.OutputFlag = 1 if cfg.solver.verbose else 0

    # Decision variables per step j
    x = m.addVars(N, lb=abs_bounds["x"][0], ub=abs_bounds["x"][1], name="x")
    y = m.addVars(N, lb=abs_bounds["y"][0], ub=abs_bounds["y"][1], name="y")
    z = m.addVars(N, lb=abs_bounds["z"][0], ub=abs_bounds["z"][1], name="z")
    th = m.addVars(N, lb=abs_bounds["theta"][0], ub=abs_bounds["theta"][1], name="theta")

    s = m.addVars(N, lb=-1.0, ub=1.0, name="s")  # approx sin
    c = m.addVars(N, lb=-1.0, ub=1.0, name="c")  # approx cos

    # region assignment binaries H_{r,j}
    H = m.addVars(R, N, vtype=GRB.BINARY, name="H")

    # trig segment binaries S_{l,j}, C_{l,j}
    Sseg = m.addVars(L, N, vtype=GRB.BINARY, name="Sseg")
    Cseg = m.addVars(L, N, vtype=GRB.BINARY, name="Cseg")

    # trim binaries t_j (paper Eq. (10))
    t = m.addVars(N, vtype=GRB.BINARY, name="t")

    # Fix the first two footsteps to initial foot poses (paper Section II-D)
    m.addConstr(x[0] == float(start_left.x))
    m.addConstr(y[0] == float(start_left.y))
    m.addConstr(z[0] == float(start_left.z))
    m.addConstr(th[0] == float(start_left.theta))

    m.addConstr(x[1] == float(start_right.x))
    m.addConstr(y[1] == float(start_right.y))
    m.addConstr(z[1] == float(start_right.z))
    m.addConstr(th[1] == float(start_right.theta))

    # Don't allow trimming of the two fixed initial feet
    m.addConstr(t[0] == 0)
    m.addConstr(t[1] == 0)

    # Optional: enforce trim as a prefix (not in the paper; helps avoid "holes")
    if cfg.enforce_trim_prefix:
        for j in range(2, N - 1):
            m.addConstr(t[j] >= t[j + 1], name=f"trim_prefix_{j}")

    # Safe region constraints (paper Eq. (2), II-A)
    # Each step belongs to exactly one region.
    for j in range(N):
        m.addConstr(gp.quicksum(H[r, j] for r in range(R)) == 1, name=f"one_region_{j}")
        for r, reg in enumerate(regions):
            A = reg.A
            b = reg.b
            for k in range(A.shape[0]):
                lhs = A[k, 0] * x[j] + A[k, 1] * y[j] + A[k, 2] * z[j] + A[k, 3] * th[j]
                m.addGenConstrIndicator(H[r, j], True, lhs <= float(b[k]),
                                        name=f"reg_{r}_{j}_{k}")

    # Piecewise linear sin/cos (paper Eqs. (7)-(8), II-B)
    for j in range(N):
        m.addConstr(gp.quicksum(Sseg[l, j] for l in range(L)) == 1, name=f"one_Sseg_{j}")
        m.addConstr(gp.quicksum(Cseg[l, j] for l in range(L)) == 1, name=f"one_Cseg_{j}")
        for l in range(L):
            # sin segment constraints
            m.addGenConstrIndicator(Sseg[l, j], True, th[j] >= float(trig.phi[l]), name=f"sin_phi_lo_{l}_{j}")
            m.addGenConstrIndicator(Sseg[l, j], True, th[j] <= float(trig.phi[l + 1]), name=f"sin_phi_hi_{l}_{j}")
            m.addGenConstrIndicator(Sseg[l, j], True, s[j] == float(trig.g_sin[l]) * th[j] + float(trig.h_sin[l]),
                                    name=f"sin_line_{l}_{j}")

            # cos segment constraints
            m.addGenConstrIndicator(Cseg[l, j], True, th[j] >= float(trig.phi[l]), name=f"cos_phi_lo_{l}_{j}")
            m.addGenConstrIndicator(Cseg[l, j], True, th[j] <= float(trig.phi[l + 1]), name=f"cos_phi_hi_{l}_{j}")
            m.addGenConstrIndicator(Cseg[l, j], True, c[j] == float(trig.g_cos[l]) * th[j] + float(trig.h_cos[l]),
                                    name=f"cos_line_{l}_{j}")

    # Δ bounds on step differences (paper Section II-D)
    dx_max = float(cfg.bounds.dx_max)
    dy_max = float(cfg.bounds.dy_max)
    dz_max = float(cfg.bounds.dz_max)
    dth_max = float(cfg.bounds.dtheta_max)

    for j in range(1, N):
        m.addConstr(x[j] - x[j - 1] <= dx_max, name=f"dx_hi_{j}")
        m.addConstr(x[j] - x[j - 1] >= -dx_max, name=f"dx_lo_{j}")
        m.addConstr(y[j] - y[j - 1] <= dy_max, name=f"dy_hi_{j}")
        m.addConstr(y[j] - y[j - 1] >= -dy_max, name=f"dy_lo_{j}")
        m.addConstr(z[j] - z[j - 1] <= dz_max, name=f"dz_hi_{j}")
        m.addConstr(z[j] - z[j - 1] >= -dz_max, name=f"dz_lo_{j}")
        m.addConstr(th[j] - th[j - 1] <= dth_max, name=f"dth_hi_{j}")
        m.addConstr(th[j] - th[j - 1] >= -dth_max, name=f"dth_lo_{j}")

    # Additional linear bounds on dz and dtheta (mentioned in II-B, implemented explicitly)
    for j in range(1, N):
        m.addConstr(z[j] - z[j - 1] <= float(cfg.reach.max_dz), name=f"reach_dz_hi_{j}")
        m.addConstr(z[j] - z[j - 1] >= -float(cfg.reach.max_dz), name=f"reach_dz_lo_{j}")
        m.addConstr(th[j] - th[j - 1] <= float(cfg.reach.max_dtheta), name=f"reach_dth_hi_{j}")
        m.addConstr(th[j] - th[j - 1] >= -float(cfg.reach.max_dtheta), name=f"reach_dth_lo_{j}")

    # Approximate reachability (paper Eqs. (5)-(6), II-B)
    # For each transition j -> j+1, enforce both circle constraints using p1,p2 and d1,d2.
    p_list = [(cfg.reach.p1, cfg.reach.d1), (cfg.reach.p2, cfg.reach.d2)]
    for j in range(0, N - 1):
        for i, (p, d) in enumerate(p_list):
            px, py = float(p[0]), float(p[1])
            # R(theta_j) p = [ c*px - s*py, s*px + c*py ]
            vx = (x[j + 1] - x[j]) - (c[j] * px - s[j] * py)
            vy = (y[j + 1] - y[j]) - (s[j] * px + c[j] * py)
            m.addQConstr(vx * vx + vy * vy <= float(d * d), name=f"reach_{i}_{j}")

    # Trim constraints (paper Eq. (10), II-C)
    # If t_j == 1, fix step j to the initial pose of that foot (based on parity).
    for j in range(2, N):
        if j % 2 == 0:  # same foot as step 0 (left)
            m.addGenConstrIndicator(t[j], True, x[j] == float(start_left.x), name=f"trim_x_L_{j}")
            m.addGenConstrIndicator(t[j], True, y[j] == float(start_left.y), name=f"trim_y_L_{j}")
            m.addGenConstrIndicator(t[j], True, z[j] == float(start_left.z), name=f"trim_z_L_{j}")
            m.addGenConstrIndicator(t[j], True, th[j] == float(start_left.theta), name=f"trim_th_L_{j}")
        else:  # right
            m.addGenConstrIndicator(t[j], True, x[j] == float(start_right.x), name=f"trim_x_R_{j}")
            m.addGenConstrIndicator(t[j], True, y[j] == float(start_right.y), name=f"trim_y_R_{j}")
            m.addGenConstrIndicator(t[j], True, z[j] == float(start_right.z), name=f"trim_z_R_{j}")
            m.addGenConstrIndicator(t[j], True, th[j] == float(start_right.theta), name=f"trim_th_R_{j}")

    # Objective (paper II-D)
    qg_pos = float(cfg.obj.qg_pos)
    qg_yaw = float(cfg.obj.qg_yaw)
    qr_pos = float(cfg.obj.qr_pos)
    qr_yaw = float(cfg.obj.qr_yaw)
    q_trim = float(cfg.obj.q_trim)

    obj = gp.QuadExpr()

    # Goal cost on last step
    jN = N - 1
    obj += qg_pos * ((x[jN] - float(goal.x)) * (x[jN] - float(goal.x)) +
                     (y[jN] - float(goal.y)) * (y[jN] - float(goal.y)) +
                     (z[jN] - float(goal.z)) * (z[jN] - float(goal.z)))
    obj += qg_yaw * ((th[jN] - float(goal.theta)) * (th[jN] - float(goal.theta)))

    # Relative step smoothness
    for j in range(0, N - 1):
        obj += qr_pos * ((x[j + 1] - x[j]) * (x[j + 1] - x[j]) +
                         (y[j + 1] - y[j]) * (y[j + 1] - y[j]) +
                         (z[j + 1] - z[j]) * (z[j + 1] - z[j]))
        obj += qr_yaw * ((th[j + 1] - th[j]) * (th[j + 1] - th[j]))

    # Trim linear incentive
    for j in range(2, N):
        obj += q_trim * t[j]

    m.setObjective(obj, GRB.MINIMIZE)

    m.optimize()
    runtime = time.perf_counter() - t0

    status_map = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "inf_or_unbd",
        GRB.UNBOUNDED: "unbounded",
    }
    status = status_map.get(m.Status, f"status_{m.Status}")

    footsteps: List[Footstep] = []
    if status not in ("optimal", "time_limit"):
        return SolveResult(status=status, runtime_s=runtime, objective=float("nan"), footsteps=[], active_indices=[])

    # Extract chosen region per step
    for j in range(N):
        # which region is active
        r_star = max(range(R), key=lambda r: H[r, j].X)
        region_name = regions[r_star].name
        foot = "L" if (j % 2 == 0) else "R"
        footsteps.append(Footstep(
            idx=j,
            foot=foot,
            x=float(x[j].X), y=float(y[j].X), z=float(z[j].X), theta=float(th[j].X),
            s=float(s[j].X), c=float(c[j].X),
            region=region_name,
            trim=(t[j].X > 0.5),
        ))

    # Paper: remove any footsteps at the beginning for which t_j is true.
    # We implement "prefix trim": drop consecutive trimmed steps starting at j=2.
    active_indices = [0, 1]
    j = 2
    while j < N and footsteps[j].trim:
        j += 1
    active_indices.extend(list(range(j, N)))

    return SolveResult(status=status, runtime_s=runtime, objective=float(m.ObjVal), footsteps=footsteps, active_indices=active_indices)
