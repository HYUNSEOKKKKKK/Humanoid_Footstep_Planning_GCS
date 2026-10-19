## Vision-based SINGLE-FOOT footstep optimizer for the hopper -- decoupled from the raisim tester via JSON.
##
## Pipeline:  [raisim tester] --(press 'p')--> gcs_regions.json --> [THIS] --> footsteps.json --> [tester set_target_world]
## Runs in the `footstep_planner` conda env (has mosek 11). It does NOT import or modify any existing
## footstep_planner module -- standalone, so the two different conda envs never have to meet (they only swap JSON).
##
## WHY NOT the existing GCS biped model (build_gcs_model_short_optapi):
##   that model HARDCODES a lateral stance d_lat=0.25 (the two feet are forced 0.5 m apart). Our stepping-stone
##   grid is only 3*0.15 = 0.45 m wide, so a degenerate "feet coincident" hopper is INFEASIBLE there, and d_lat is
##   a local var (no runtime override). So we keep the SAME convex-region idea (each tile = {x : A x <= b}) but a
##   single-foot reach model. The integer region-SELECTION that GCS provides is unnecessary here -- vision already
##   hands us the target tiles in order (near->far). Add MICP selection later if the path becomes ambiguous.
##
## The problem (small convex SOCP):  choose a foothold p_i in each ordered target tile i to
##     minimize   sum_i ||p_i - p_{i-1}||        (shortest chained path -> places the foot toward the NEXT tile when
##                + W_CENTER * sum_i ||p_i-c_i||  zig-zagging, toward the centre when going straight = context-optimal)
##     s.t.       A_i p_i <= b_i                 (foot lands ON tile i -- the vision-measured convex region)
##                p_0 = current foot
##                reach (foot HEADING frame, forward-biased BOX = what the low-level controller can actually do):
##                   REACH_FWD_MIN <= (p_i - p_{i-1}).fwd <= REACH_FWD_MAX
##                   |(p_i - p_{i-1}).lft| <= REACH_LAT_MAX
##
## Usage (in the footstep_planner env):  python main_visionbased.py

import json
import math
import numpy as np
import mosek.fusion as mf

# ============================== EDIT HERE ==============================
REGIONS_IN    = "/tmp/gcs_regions.json"     # vision output (raisim tester, 'p' key)
FOOTSTEPS_OUT = "/tmp/footsteps.json"       # plan for the tester (set_target_world)

# Low-level controller REACHABLE set per HOP, in the foot heading frame (m). *** MEASURE these on the frozen policy ***
# (from the run we saw: forward ~0.05-0.20, lateral up to ~0.30; start conservative, widen once measured).
REACH_FWD_MIN = 0.05
REACH_FWD_MAX = 0.30
REACH_LAT_MAX = 0.18

W_CENTER      = 0.25      # objective weight on staying near each tile centre (small = let the path optimum decide)
YAW_MODE      = "face_next"  # "face_next" = aim each foothold at the NEXT one | "grid" = constant (face down the grid)
# ======================================================================


def main():
    with open(REGIONS_IN) as f:
        data = json.load(f)

    fwd = np.asarray(data["fwd"], float)
    lft = np.asarray(data["lft"], float)
    grid_yaw = float(data["grid_yaw"])
    foot = np.asarray(data["foot"], float)
    regs = data["regions"]                      # already ordered near -> far along the path
    N = len(regs)
    print("loaded {} '{}' regions ({} mode) from {}".format(N, data.get("target_color"), data.get("region_mode"), REGIONS_IN))

    if N == 0:
        json.dump({"target_color": data.get("target_color"), "frame": "world", "grid_yaw": grid_yaw, "footsteps": []},
                  open(FOOTSTEPS_OUT, "w"), indent=2)
        print("no regions -> wrote empty plan."); return

    A = [np.asarray(r["A"], float) for r in regs]      # (4,2) each:  A x <= b
    b = [np.asarray(r["b"], float) for r in regs]      # (4,)
    ctr = [np.asarray(r["center"], float) for r in regs]

    with mf.Model("hopper_footsteps") as M:
        P = [M.variable("p{}".format(i), 2, mf.Domain.unbounded()) for i in range(N)]
        tstep = M.variable("tstep", N, mf.Domain.greaterThan(0.0))   # ||p_i - p_{i-1}||
        tcen = M.variable("tcen", N, mf.Domain.greaterThan(0.0))     # ||p_i - center_i||

        for i in range(N):
            prev = foot if i == 0 else None
            # foot lands ON tile i (the vision-measured convex region)
            M.constraint("reg{}".format(i), mf.Expr.mul(mf.Matrix.dense(A[i].tolist()), P[i]),
                         mf.Domain.lessThan(b[i].tolist()))
            # centering term:  ||p_i - center_i|| <= tcen_i
            M.constraint(mf.Expr.vstack(tcen.index(i), mf.Expr.sub(P[i], ctr[i].tolist())), mf.Domain.inQCone())
            # step vector p_i - p_{i-1}
            d = mf.Expr.sub(P[i], foot.tolist()) if i == 0 else mf.Expr.sub(P[i], P[i - 1])
            # path-length term:  ||p_i - p_{i-1}|| <= tstep_i
            M.constraint(mf.Expr.vstack(tstep.index(i), d), mf.Domain.inQCone())
            # reach BOX in the heading frame (= the controller's coverable set)
            M.constraint("fwd{}".format(i), mf.Expr.dot(fwd.tolist(), d), mf.Domain.inRange(REACH_FWD_MIN, REACH_FWD_MAX))
            M.constraint("lat{}".format(i), mf.Expr.dot(lft.tolist(), d), mf.Domain.inRange(-REACH_LAT_MAX, REACH_LAT_MAX))

        M.objective(mf.ObjectiveSense.Minimize,
                    mf.Expr.add(mf.Expr.sum(tstep), mf.Expr.mul(float(W_CENTER), mf.Expr.sum(tcen))))
        M.solve()

        status = M.getPrimalSolutionStatus()
        if status not in (mf.SolutionStatus.Optimal, mf.SolutionStatus.Feasible):
            print("** SOLVE {} ** -- the red path likely exceeds the reach box. Check spacing or widen REACH_*."
                  .format(status))
            # report the worst gap so you know what to relax
            pc = [foot] + [ctr[i] for i in range(N)]
            for i in range(N):
                dd = pc[i + 1] - pc[i]
                print("   center-to-center step {}: fwd={:+.3f} lat={:+.3f} (limits fwd[{},{}] |lat|<={})"
                      .format(i, float(dd @ fwd), float(dd @ lft), REACH_FWD_MIN, REACH_FWD_MAX, REACH_LAT_MAX))
            return

        Psol = np.array([P[i].level() for i in range(N)])

    # yaw per foothold
    steps = []
    for i in range(N):
        p = Psol[i]
        if YAW_MODE == "face_next" and i < N - 1:
            dxy = Psol[i + 1] - p
            yaw = float(math.atan2(dxy[1], dxy[0])) if np.linalg.norm(dxy) > 1e-6 else grid_yaw
        else:
            yaw = grid_yaw
        off = float(np.linalg.norm(p - ctr[i])) * 100.0    # how far the optimal foothold sits from the tile centre
        steps.append({"x": float(p[0]), "y": float(p[1]), "yaw": yaw,
                      "row": regs[i].get("row"), "col": regs[i].get("col"), "off_center_cm": round(off, 1)})
        print("  step {:2d}  tile(row{},col{})  p=({:+.3f},{:+.3f})  yaw={:+.3f}  off-center={:.1f}cm"
              .format(i, regs[i].get("row"), regs[i].get("col"), p[0], p[1], yaw, off))

    out = {"target_color": data.get("target_color"), "frame": "world", "grid_yaw": grid_yaw,
           "reach": {"fwd_min": REACH_FWD_MIN, "fwd_max": REACH_FWD_MAX, "lat_max": REACH_LAT_MAX},
           "yaw_mode": YAW_MODE, "footsteps": steps}
    json.dump(out, open(FOOTSTEPS_OUT, "w"), indent=2)
    print("planned {} footsteps -> {}".format(len(steps), FOOTSTEPS_OUT))


if __name__ == "__main__":
    main()
