from __future__ import annotations

"""MOSEK (Fusion) backend.

This reproduces the same Deits & Tedrake (2014) MIQCQP formulation used in
`miqcqp.py`, but avoids indicator constraints by using Big-M linearizations.

We also represent squared-norm objective terms using rotated quadratic cones,
so the resulting model is a Mixed-Integer Conic Optimization (MICO) problem,
which MOSEK supports.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import math
import time
import sys
import traceback

import numpy as np

from .config import PlannerConfig
from .terrain import SafeRegion, FootPose
from .piecewise_trig import build_piecewise_linear_sin_cos
from .miqcqp import Footstep, SolveResult

try:
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, SolutionStatus, SolutionType
except Exception:  # pragma: no cover
    Model = None
    Domain = None
    Expr = None
    ObjectiveSense = None
    SolutionStatus = None


def _require_mosek():
    if Model is None:
        raise ImportError(
            "MOSEK Fusion API is required for the --solver mosek backend. "
            "Install `mosek` (pip/conda) and set a MOSEK license."
        )


def _ct(v: float):
    """Scalar constant expression (size 1)."""
    return Expr.constTerm(float(v))


def _big_m_linear(row: np.ndarray,
                  rhs: float,
                  bounds: Dict[str, Tuple[float, float]]) -> float:
    """Compute a valid Big-M for constraint: row*[x,y,z,th] <= rhs.

    We use variable bounds to upper-bound the LHS.
    M >= max_{f in bounds} (row*f - rhs). Then:
        row*f <= rhs + M*(1-h)
    disables the constraint when h=0.
    """
    xL, xU = bounds["x"]
    yL, yU = bounds["y"]
    zL, zU = bounds["z"]
    tL, tU = bounds["theta"]
    lb = np.array([xL, yL, zL, tL], dtype=float)
    ub = np.array([xU, yU, zU, tU], dtype=float)
    # maximize row·f over a box occurs at bounds depending on sign
    f_star = np.where(row >= 0.0, ub, lb)
    lhs_max = float(np.dot(row, f_star))
    return max(0.0, lhs_max - float(rhs))


def _big_m_affine(expr_min: float, expr_max: float) -> float:
    """Big-M for |expr| <= M*(1-b) style constraints.

    If expr is known to lie in [expr_min, expr_max], then |expr| <= M where
    M = max(|expr_min|, |expr_max|).
    """
    return max(abs(float(expr_min)), abs(float(expr_max)))


def _range_of_linear(g: float, h: float, t_bounds: Tuple[float, float]) -> Tuple[float, float]:
    tL, tU = map(float, t_bounds)
    v1 = g * tL + h
    v2 = g * tU + h
    return (min(v1, v2), max(v1, v2))


def solve_miqcqp_mosek(regions: List[SafeRegion],
                       start_left: FootPose,
                       start_right: FootPose,
                       goal_left: FootPose,
                       goal_right: FootPose,
                       N: int,
                       cfg: PlannerConfig,
                       abs_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> SolveResult:
    """MOSEK Fusion implementation of the MIQCQP footstep planner.

    Notes:
      * Indicator constraints in the paper are implemented via Big-M.
      * Quadratic constraints are implemented as quadratic-cone constraints.
      * Quadratic objective terms are modeled with rotated quadratic cones + linear objective.
    """
    _require_mosek()
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
    L = int(cfg.trig.num_segments)

    t0 = time.perf_counter()
    M = Model("miqcqp_footstep_planner_mosek")
    # Print MOSEK log to stdout when verbose (helps debugging infeasibility / licensing).
    if cfg.solver.verbose:
        try:
            M.setLogHandler(sys.stdout)
        except Exception:
            pass
    try:
        # Solver params
        if cfg.solver.time_limit_s is not None:
            M.setSolverParam('mioMaxTime', float(cfg.solver.time_limit_s))
        if cfg.solver.mip_gap is not None:
            M.setSolverParam('mioTolRelGap', float(cfg.solver.mip_gap))

        # Decision variables (vectors of length N)
        x = M.variable("x", N, Domain.inRange(abs_bounds["x"][0], abs_bounds["x"][1]))
        y = M.variable("y", N, Domain.inRange(abs_bounds["y"][0], abs_bounds["y"][1]))
        z = M.variable("z", N, Domain.inRange(abs_bounds["z"][0], abs_bounds["z"][1]))
        th = M.variable("theta", N, Domain.inRange(abs_bounds["theta"][0], abs_bounds["theta"][1]))

        s = M.variable("s", N, Domain.inRange(-1.0, 1.0))
        c = M.variable("c", N, Domain.inRange(-1.0, 1.0))

        H = M.variable("H", [R, N], Domain.binary())
        Sseg = M.variable("Sseg", [L, N], Domain.binary())
        Cseg = M.variable("Cseg", [L, N], Domain.binary())
        t = M.variable("t", N, Domain.binary())

        # Fix first two steps (left then right)
        M.constraint(Expr.sub(x.index(0), float(start_left.x)), Domain.equalsTo(0.0))
        M.constraint(Expr.sub(y.index(0), float(start_left.y)), Domain.equalsTo(0.0))
        M.constraint(Expr.sub(z.index(0), float(start_left.z)), Domain.equalsTo(0.0))
        M.constraint(Expr.sub(th.index(0), float(start_left.theta)), Domain.equalsTo(0.0))

        M.constraint(Expr.sub(x.index(1), float(start_right.x)), Domain.equalsTo(0.0))
        M.constraint(Expr.sub(y.index(1), float(start_right.y)), Domain.equalsTo(0.0))
        M.constraint(Expr.sub(z.index(1), float(start_right.z)), Domain.equalsTo(0.0))
        M.constraint(Expr.sub(th.index(1), float(start_right.theta)), Domain.equalsTo(0.0))

        # No trimming of fixed steps
        M.constraint(t.index(0), Domain.equalsTo(0.0))
        M.constraint(t.index(1), Domain.equalsTo(0.0))

        # Optional: disable trimming entirely (fix all t[j]=0).
        if getattr(cfg, "disable_trim", False):
            for j in range(2, N):
                M.constraint(t.index(j), Domain.equalsTo(0.0))


        # Optional: enforce trimming pattern (monotone t).
        # NOTE: In this implementation, t[j]=1 means "trimmed / inactive".
        # A *suffix* trim is usually what you want: once a step is trimmed, all later steps are trimmed too.
        # This enforces: t[2] <= t[3] <= ... <= t[N-1].
        if (not getattr(cfg, 'disable_trim', False)) and cfg.enforce_trim_prefix:
            for j in range(2, N - 1):
                # t[j+1] - t[j] >= 0
                M.constraint(Expr.sub(t.index(j + 1), t.index(j)), Domain.greaterThan(0.0))



        # ---------------- Safe region constraints (Big-M) ----------------
        for j in range(N):
            # sum_r H[r,j] == 1
            M.constraint(Expr.sum(H.slice([0, j], [R, j + 1])), Domain.equalsTo(1.0))
            for r, reg in enumerate(regions):
                A = reg.A
                bvec = reg.b
                for k in range(A.shape[0]):
                    row = A[k, :].astype(float)
                    rhs = float(bvec[k])
                    bigM = _big_m_linear(row, rhs, abs_bounds)
                    # row·f <= rhs + M*(1 - H[r,j])  <=>  row·f + M*H[r,j] <= rhs + M
                    rowf = Expr.add(
                        Expr.add(Expr.mul(row[0], x.index(j)), Expr.mul(row[1], y.index(j))),
                        Expr.add(Expr.mul(row[2], z.index(j)), Expr.mul(row[3], th.index(j)))
                    )
                    lhs = Expr.add(rowf, Expr.mul(bigM, H.index(r, j)))
                    M.constraint(lhs, Domain.lessThan(rhs + bigM))

        # ---------------- Piecewise sin/cos (Big-M) ----------------
        thL, thU = abs_bounds["theta"]
        for j in range(N):
            M.constraint(Expr.sum(Sseg.slice([0, j], [L, j + 1])), Domain.equalsTo(1.0))
            M.constraint(Expr.sum(Cseg.slice([0, j], [L, j + 1])), Domain.equalsTo(1.0))

            for l in range(L):
                phi_lo = float(trig.phi[l])
                phi_hi = float(trig.phi[l + 1])

                # theta bounds for segment selection
                M_lo = max(0.0, phi_lo - thL)  # so theta >= phi_lo - M*(1-b)
                M_hi = max(0.0, thU - phi_hi)  # so theta <= phi_hi + M*(1-b)

                # theta >= phi_lo - M*(1 - S)
                M.constraint(
                    Expr.add(th.index(j), Expr.mul(M_lo, Expr.sub(1.0, Sseg.index(l, j)))),
                    Domain.greaterThan(phi_lo)
                )
                # theta <= phi_hi + M*(1 - S)  <=>  theta - M*(1-S) <= phi_hi
                M.constraint(
                    Expr.sub(th.index(j), Expr.mul(M_hi, Expr.sub(1.0, Sseg.index(l, j)))),
                    Domain.lessThan(phi_hi)
                )

                # sin line: s = g*theta + h when active
                g = float(trig.g_sin[l]); h = float(trig.h_sin[l])
                # range of (s - (g*th+h)) over bounds
                # s in [-1,1], g*th+h in [min,max]
                gt_min, gt_max = _range_of_linear(g, h, (thL, thU))
                expr_min = -1.0 - gt_max
                expr_max = 1.0 - gt_min
                M_eq = _big_m_affine(expr_min, expr_max)

                diff = Expr.sub(s.index(j), Expr.add(Expr.mul(g, th.index(j)), h))
                # diff <= M*(1-S)  <=>  diff - M*(1-S) <= 0
                M.constraint(Expr.sub(diff, Expr.mul(M_eq, Expr.sub(1.0, Sseg.index(l, j)))),
                             Domain.lessThan(0.0))
                # diff >= -M*(1-S) <=> diff + M*(1-S) >= 0
                M.constraint(Expr.add(diff, Expr.mul(M_eq, Expr.sub(1.0, Sseg.index(l, j)))),
                             Domain.greaterThan(0.0))

                # cos segment constraints (same phi bounds)
                M.constraint(
                    Expr.add(th.index(j), Expr.mul(M_lo, Expr.sub(1.0, Cseg.index(l, j)))),
                    Domain.greaterThan(phi_lo)
                )
                M.constraint(
                    Expr.sub(th.index(j), Expr.mul(M_hi, Expr.sub(1.0, Cseg.index(l, j)))),
                    Domain.lessThan(phi_hi)
                )
                g = float(trig.g_cos[l]); h = float(trig.h_cos[l])
                gt_min, gt_max = _range_of_linear(g, h, (thL, thU))
                expr_min = -1.0 - gt_max
                expr_max = 1.0 - gt_min
                M_eq = _big_m_affine(expr_min, expr_max)
                diff = Expr.sub(c.index(j), Expr.add(Expr.mul(g, th.index(j)), h))
                M.constraint(Expr.sub(diff, Expr.mul(M_eq, Expr.sub(1.0, Cseg.index(l, j)))),
                             Domain.lessThan(0.0))
                M.constraint(Expr.add(diff, Expr.mul(M_eq, Expr.sub(1.0, Cseg.index(l, j)))),
                             Domain.greaterThan(0.0))

        # ---------------- Δ bounds (linear) ----------------
        dx_max = float(cfg.bounds.dx_max)
        dy_max = float(cfg.bounds.dy_max)
        dz_max = float(cfg.bounds.dz_max)
        dth_max = float(cfg.bounds.dtheta_max)

        for j in range(1, N):
            M.constraint(Expr.sub(x.index(j), x.index(j - 1)), Domain.inRange(-dx_max, dx_max))
            M.constraint(Expr.sub(y.index(j), y.index(j - 1)), Domain.inRange(-dy_max, dy_max))
            M.constraint(Expr.sub(z.index(j), z.index(j - 1)), Domain.inRange(-dz_max, dz_max))
            M.constraint(Expr.sub(th.index(j), th.index(j - 1)), Domain.inRange(-dth_max, dth_max))

            M.constraint(Expr.sub(z.index(j), z.index(j - 1)), Domain.inRange(-float(cfg.reach.max_dz), float(cfg.reach.max_dz)))
            M.constraint(Expr.sub(th.index(j), th.index(j - 1)), Domain.inRange(-float(cfg.reach.max_dtheta), float(cfg.reach.max_dtheta)))

        # ---------------- L/R relative constraints (SOC + linear) ----------------
        # Assumption: +y is "left" in world/foot frame.
        w_max = float(getattr(cfg.reach, "lr_sep_max", 0.35))  # max L-R distance
        w_min = float(getattr(cfg.reach, "lr_sep_min", 0.20))  # min lateral width (non-crossing)

        for j in range(0, N - 1):
            dx = Expr.sub(x.index(j + 1), x.index(j))
            dy = Expr.sub(y.index(j + 1), y.index(j))

            # (H-1') max separation (SOC): ||[dx, dy]|| <= w_max
            M.constraint(Expr.vstack(_ct(w_max), dx, dy), Domain.inQCone())

            # non-crossing + min width (linear), depends on which foot is the "previous" one
            if (j % 2) == 0:
                # prev = Left, next = Right  => Right must be on -y side
                # y_{j+1} - y_j <= -w_min  <=> (y_{j+1}-y_j) + w_min <= 0
                M.constraint(Expr.add(dy, _ct(w_min)), Domain.lessThan(0.0))
            else:
                # prev = Right, next = Left => Left must be on +y side
                # y_{j+1} - y_j >= +w_min
                M.constraint(dy, Domain.greaterThan(w_min))

        # ---------------- Same-foot step span (SOC) ----------------
        same_span = float(getattr(cfg.reach, "same_foot_span", 0.60))  # tune this
        for j in range(2, N):
            dx2 = Expr.sub(x.index(j), x.index(j - 2))
            dy2 = Expr.sub(y.index(j), y.index(j - 2))
            M.constraint(Expr.vstack(_ct(same_span), dx2, dy2), Domain.inQCone())

        # ---------------- Trim constraints (Big-M equality) ----------------
        # If t[j]=1, fix (x,y,z,theta) to a reference pose.
        #
        # IMPORTANT: To make trimming behave like "use fewer steps", the most robust choice is
        # to fix trimmed (inactive) steps to the *goal* pose (per foot), i.e., "stand at the goal"
        # after you reach it early. This avoids the awkward reachability interaction you get when
        # fixing trimmed steps to the *initial* pose.
        if not getattr(cfg, "disable_trim", False):
            xL, xU = abs_bounds["x"]
            yL, yU = abs_bounds["y"]
            zL, zU = abs_bounds["z"]
            tL, tU = abs_bounds["theta"]
            Mx = xU - xL
            My = yU - yL
            Mz = zU - zL
            Mth = tU - tL

            for j in range(2, N):
                # Trimmed steps are fixed to the goal pose (per-foot): once you reach the goal,
                # remaining steps can stay there.
                if j % 2 == 0:
                    ref = goal_left
                else:
                    ref = goal_right

                for var, refv, bigM in (
                    (x.index(j), float(ref.x), float(Mx)),
                    (y.index(j), float(ref.y), float(My)),
                    (z.index(j), float(ref.z), float(Mz)),
                    (th.index(j), float(ref.theta), float(Mth)),
                ):
                    # var - ref <= M*(1-t)
                    M.constraint(
                        Expr.sub(
                            Expr.sub(var, refv),
                            Expr.mul(bigM, Expr.sub(1.0, t.index(j)))
                        ),
                        Domain.lessThan(0.0)
                    )
                    # ref - var <= M*(1-t)
                    M.constraint(
                        Expr.sub(
                            Expr.sub(refv, var),
                            Expr.mul(bigM, Expr.sub(1.0, t.index(j)))
                        ),
                        Domain.lessThan(0.0)
                    )

        # ---------------- Objective (conic + linear) ----------------
        # Model each squared term via rotated cone: u >= sum squares, minimize weighted sum of u.
        qg_pos = float(cfg.obj.qg_pos)
        qg_yaw = float(cfg.obj.qg_yaw)
        qr_pos = float(cfg.obj.qr_pos)
        qr_yaw = float(cfg.obj.qr_yaw)
        q_trim = float(cfg.obj.q_trim)

        u_terms = []
        w_terms = []

        
        # Goal terms (same-foot goals)
        # We apply the goal cost to the *last left* step and the *last right* step,
        # which encourages finishing in double support on/near the final stone.
        jL = (N - 1) if ((N - 1) % 2 == 0) else (N - 2)
        jR = (N - 1) if ((N - 1) % 2 == 1) else (N - 2)

        def _add_goal_terms(tag: str, j: int, g: FootPose):
            # position
            ug = M.variable(f"u_goal_pos_{tag}", 1, Domain.greaterThan(0.0))
            dxg = Expr.sub(x.index(j), float(g.x))
            dyg = Expr.sub(y.index(j), float(g.y))
            dzg = Expr.sub(z.index(j), float(g.z))
            M.constraint(
                Expr.hstack([Expr.mul(1.0, ug), _ct(0.5), dxg, dyg, dzg]),
                Domain.inRotatedQCone(),
            )
            u_terms.append(ug)
            w_terms.append(qg_pos)

            # yaw
            ugth = M.variable(f"u_goal_yaw_{tag}", 1, Domain.greaterThan(0.0))
            dthg = Expr.sub(th.index(j), float(g.theta))
            M.constraint(
                Expr.hstack([Expr.mul(1.0, ugth), _ct(0.5), dthg]),
                Domain.inRotatedQCone(),
            )
            u_terms.append(ugth)
            w_terms.append(qg_yaw)

        _add_goal_terms("L", jL, goal_left)
        _add_goal_terms("R", jR, goal_right)

        # Same-foot smoothness (L-to-L and R-to-R)
        q_same_pos = float(getattr(cfg.obj, "q_same_pos", qr_pos))
        q_same_yaw = float(getattr(cfg.obj, "q_same_yaw", qr_yaw))

        for j in range(2, N):
            usp = M.variable(f"u_same_pos_{j}", 1, Domain.greaterThan(0.0))
            dx = Expr.sub(x.index(j), x.index(j - 2))
            dy = Expr.sub(y.index(j), y.index(j - 2))
            dz = Expr.sub(z.index(j), z.index(j - 2))
            M.constraint(
                Expr.hstack([Expr.mul(1.0, usp), _ct(0.5), dx, dy, dz]),
                Domain.inRotatedQCone(),
            )
            u_terms.append(usp)
            w_terms.append(q_same_pos)

            usy = M.variable(f"u_same_yaw_{j}", 1, Domain.greaterThan(0.0))
            dth = Expr.sub(th.index(j), th.index(j - 2))
            M.constraint(
                Expr.hstack([Expr.mul(1.0, usy), _ct(0.5), dth]),
                Domain.inRotatedQCone(),
            )
            u_terms.append(usy)
            w_terms.append(q_same_yaw)

# Linear trim incentive
        # add q_trim * sum_{j=2..N-1} t[j]
        # We'll implement as an extra variable vtrim = sum(t[2:]) and include it.
        vtrim = M.variable("v_trim", 1, Domain.greaterThan(0.0))
        M.constraint(Expr.sub(vtrim, Expr.sum(t.slice(2, N))), Domain.equalsTo(0.0))

        # Objective = sum_i w_i * u_i + q_trim * vtrim
        obj = Expr.mul(w_terms[0], u_terms[0])
        for wi, ui in zip(w_terms[1:], u_terms[1:]):
            obj = Expr.add(obj, Expr.mul(wi, ui))
        obj = Expr.add(obj, Expr.mul(q_trim, vtrim))

        M.objective("obj", ObjectiveSense.Minimize, obj)

        M.solve()
        runtime = time.perf_counter() - t0

        solsta = None
        try:
            solsta = M.getPrimalSolutionStatus()
        except Exception:
            solsta = None

        if solsta not in (SolutionStatus.Optimal, SolutionStatus.Feasible):
            return SolveResult(status="failed", runtime_s=float(runtime), objective=float("nan"),
                               footsteps=[], active_indices=[])

        status = "solved"
        if solsta is not None:
            if solsta == SolutionStatus.Optimal:
                status = "optimal"
            elif solsta == SolutionStatus.Feasible:
                status = "feasible"


        # Get solution values.
        # NOTE: For MIP, MOSEK may return a *primal feasible* integer solution without proving optimality
        # within the time/gap limits. Fusion's `.level()` on Default solution can throw in that case,
        # so we try to explicitly read the integer solution when possible.
        def _level_safe(var):
            try:
                return np.array(var.level(), dtype=float)
            except Exception:
                try:
                    return np.array(var.level(SolutionType.Integer), dtype=float)
                except Exception:
                    raise

        # Get solution values (may throw if no integer solution exists)
        x_val = _level_safe(x)
        y_val = _level_safe(y)
        z_val = _level_safe(z)
        th_val = _level_safe(th)
        s_val = _level_safe(s)
        c_val = _level_safe(c)
        t_val = _level_safe(t)
        # Safer extraction for 2D variables: Fusion's internal ordering is not always obvious.
        H_val = np.zeros((R, N), dtype=float)
        for r in range(R):
            for j in range(N):
                H_val[r, j] = float(_level_safe(H.index(r, j))[0])

        footsteps: List[Footstep] = []
        for j in range(N):
            r_star = int(np.argmax(H_val[:, j]))
            foot = "L" if (j % 2 == 0) else "R"
            footsteps.append(Footstep(
                idx=j,
                foot=foot,
                x=float(x_val[j]), y=float(y_val[j]), z=float(z_val[j]), theta=float(th_val[j]),
                s=float(s_val[j]), c=float(c_val[j]),
                region=regions[r_star].name,
                trim=(t_val[j] > 0.5),
            ))

        # Active steps for visualization.
        # When trimming is enabled with the monotone constraint, t is a *suffix* (0...0, 1...1),
        # meaning the plan reaches the goal and then "stays" there for the remaining steps.
        #
        # For plotting, we keep steps up to the first trimmed index, plus one extra step so the
        # final double-support on the goal is visible.
        if (not getattr(cfg, 'disable_trim', False)) and cfg.enforce_trim_prefix:
            # first trimmed index after j>=2
            k = 2
            while k < N and (not footsteps[k].trim):
                k += 1

            if k >= N:
                active_indices = list(range(N))
            else:
                k_end = min(N - 1, k + 1)
                active_indices = list(range(k_end + 1))
        else:
            active_indices = [j for j in range(N) if not footsteps[j].trim]

        # Objective value (best effort)
        obj_val = float("nan")
        try:
            obj_val = float(M.primalObjValue())
        except Exception:
            pass
        if not (obj_val == obj_val):  # NaN check
            try:
                obj_val = 0.0
                for wi, ui in zip(w_terms, u_terms):
                    obj_val += float(wi) * float(_level_safe(ui)[0])
                obj_val += q_trim * float(_level_safe(vtrim)[0])
            except Exception:
                obj_val = float("nan")


        return SolveResult(status=status, runtime_s=float(runtime), objective=float(obj_val),
                           footsteps=footsteps, active_indices=active_indices)

    except Exception:
        runtime = time.perf_counter() - t0
        # Best-effort status; MOSEK may throw on infeasible or other termination.
        print("[MIQCQP][MOSEK] solve failed. See traceback below.", file=sys.stderr)
        traceback.print_exc()
        return SolveResult(status="failed", runtime_s=float(runtime), objective=float('nan'), footsteps=[], active_indices=[])
    finally:
        try:
            M.dispose()
        except Exception:
            pass
