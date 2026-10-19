from __future__ import annotations
import time
import numpy as np
from mosek.fusion import Model, Domain, Expr, Matrix, ObjectiveSense

from src.graph import centroid_and_scaled_vertices
from src.solver.fusion_ops import eq0, le0, ge0, cone2, sum_expr, sum_pick_1d
from src.graph.geometry import rect_A_b_from_vertices
from src.graph.centroid import compute_edge_directions


def build_gcs_model_short(v, edge, step_span, alpha, mode_list, x_ref, y_ref):

    t0 = time.perf_counter()
    edge = edge.astype(int) - 1
    n_edge = int(edge.shape[0])
    vertex_number = int(max(edge[:, 0].max(), edge[:, 1].max())) + 1
    ca = float(np.cos(alpha))

    # ===== Convex region =====
    centroid, v_scaled = centroid_and_scaled_vertices(v, scale=1.0)
    centroid_xy = centroid[:2,:].T
    # print(centroid_xy)
    u_ref_x, u_ref_y, has_ref = compute_edge_directions(centroid_xy, edge, eps=1e-8, assume_zero_based=True)
    # print(u_ref_x)
    # print(u_ref_y)
    # print(has_ref)
    const_struct = []
    for ii, vtx in enumerate(v_scaled):
        A, b = rect_A_b_from_vertices(vtx)
        const_struct.append({
            "A": A,
            "b": b + A @ centroid[:2, ii],
            "center_z": centroid[2, ii]
        })

    XREF = Matrix.dense(2,1, [float(x_ref[0]),float(x_ref[1])])
    YREF = Matrix.dense(2,1, [float(y_ref[0]),float(y_ref[1])])


    # ===== Model & Variables =====
    M = Model("GCS_footstepplanning_opless")

    y    = M.variable("y",     n_edge,    Domain.inRange(0.0, 1.0))
    z_L  = M.variable("z_L",  [n_edge,5], Domain.unbounded())
    z_R  = M.variable("z_R",  [n_edge,5], Domain.unbounded())
    zp_L = M.variable("zp_L", [n_edge,5], Domain.unbounded())
    zp_R = M.variable("zp_R", [n_edge,5], Domain.unbounded())

    rho_L  = M.variable("rho_L",  n_edge, Domain.greaterThan(0.0))
    rho_R  = M.variable("rho_R",  n_edge, Domain.greaterThan(0.0))
    rhoz_L = M.variable("rhoz_L", n_edge, Domain.greaterThan(0.0))
    rhoz_R = M.variable("rhoz_R", n_edge, Domain.greaterThan(0.0))

    X_L  = [ M.variable(f"X_L_{e}",  [5,5], Domain.inPSDCone(5))  for e in range(n_edge) ]
    X_R  = [ M.variable(f"X_R_{e}",  [5,5], Domain.inPSDCone(5))  for e in range(n_edge) ]
    X_L2 = [ M.variable(f"X_L2_{e}", [7,7], Domain.inPSDCone(7)) for e in range(n_edge) ]
    X_R2 = [ M.variable(f"X_R2_{e}", [7,7], Domain.inPSDCone(7)) for e in range(n_edge) ]

    # parameter
    start_L  = M.parameter("start_L",  5)
    start_R  = M.parameter("start_R",  5)
    target_L = M.parameter("target_L", 5)
    target_R = M.parameter("target_R", 5)

    y_relax  = M.parameter("y_relax",  n_edge)
    M_scalar = M.parameter("M_scalar")


    flow_out_vertices = edge[:, 0].astype(int)   # 0-based  <<< CHANGED
    flow_in_vertices  = edge[:, 1].astype(int)   # 0-based  <<< CHANGED


    # ========================= Constraints ==============================

    # (A) || y - y_relax ||_inf <= M_scalar  →  -M <= y - y_relax <= M
    ones_e = Matrix.dense(np.ones((n_edge,1)))
    Mvec   = Expr.mul(ones_e, M_scalar)
    le0(M, Expr.sub(Expr.sub(y,       y_relax), Mvec))  # y - y_relax - M <= 0
    le0(M, Expr.sub(Expr.sub(y_relax, y),       Mvec))  # y_relax - y - M <= 0

    # (B) source/target flow
    src_edges  = np.where(flow_out_vertices == 0)[0]
    sink_edges = np.where(flow_in_vertices  == vertex_number-1)[0]
    for e in src_edges:
        for k in range(5):
            eq0(M, Expr.sub(z_L.index(e,k), Expr.mul(y.index(e), start_L.index(k))))
            eq0(M, Expr.sub(z_R.index(e,k), Expr.mul(y.index(e), start_R.index(k))))

    for e in sink_edges:
        for k in range(5):
            eq0(M, Expr.sub(zp_L.index(e,k), Expr.mul(y.index(e), target_L.index(k))))
            eq0(M, Expr.sub(zp_R.index(e,k), Expr.mul(y.index(e), target_R.index(k))))


    # (C) Source and Target vertex Flow conservation
    # source sum == 1, target sum == 1
    M.constraint(sum_pick_1d(y, src_edges), Domain.equalsTo(1.0))
    M.constraint(sum_pick_1d(y, sink_edges), Domain.equalsTo(1.0))

    # (D) In = Out flow conservation (middle vertices)
    for v_id in range(1, vertex_number - 1):
        in_idx = np.where(flow_in_vertices == v_id)[0]
        out_idx = np.where(flow_out_vertices == v_id)[0]

        # 1) sum of y
        M.constraint(Expr.sub(sum_pick_1d(y, in_idx), sum_pick_1d(y, out_idx)),
                     Domain.equalsTo(0.0))

        # 2) sum of zp (In edge) == sum of z (Out edge)
        for k in range(5):
            # sum(zp_L(in,k)) == sum(z_L(out,k))
            in_sum_L = Expr.add([zp_L.index(e, k) for e in in_idx]) if in_idx.size > 0 else Expr.const(0.0)
            out_sum_L = Expr.add([z_L.index(e, k) for e in out_idx]) if out_idx.size > 0 else Expr.const(0.0)
            M.constraint(Expr.sub(in_sum_L, out_sum_L), Domain.equalsTo(0.0))

            # sum(zp_R(in,k)) == sum(z_R(out,k))
            in_sum_R = Expr.add([zp_R.index(e, k) for e in in_idx]) if in_idx.size > 0 else Expr.const(0.0)
            out_sum_R = Expr.add([z_R.index(e, k) for e in out_idx]) if out_idx.size > 0 else Expr.const(0.0)
            M.constraint(Expr.sub(in_sum_R, out_sum_R), Domain.equalsTo(0.0))

    # (E) outflow of y (middle vertices)
    for v_id in range(1, vertex_number-1):
        out_idx = np.where(flow_out_vertices == v_id)[0]
        M.constraint(sum_pick_1d(y, out_idx), Domain.lessThan(1.0))

    # (F) Inside Convex region (depends on from/to mode)
    for e in range(n_edge):
        from_v = flow_out_vertices[e]; to_v = flow_in_vertices[e]
        mode_from = mode_list[from_v]; mode_to = mode_list[to_v]

        A_from = const_struct[from_v]["A"]; b_from = const_struct[from_v]["b"]; z_from = const_struct[from_v]["center_z"]
        A_to   = const_struct[to_v]["A"];   b_to   = const_struct[to_v]["b"];   z_to   = const_struct[to_v]["center_z"]

        B_from = Matrix.dense(b_from.reshape(-1,1))
        B_to   = Matrix.dense(b_to.reshape(-1,1))
        AfromM = Matrix.dense(A_from)
        AtoM   = Matrix.dense(A_to)

        if mode_from == "D" and mode_to == "SL":
            lhs = Expr.sub(Expr.mul(AtoM, Expr.vstack(zp_L.index(e,0), zp_L.index(e,1))),
                           Expr.mul(B_to, y.index(e)))
            le0(M, lhs)
            eq0(M, Expr.sub(zp_L.index(e,2), Expr.mul(y.index(e), z_to)))
        if mode_from == "D" and mode_to == "SR":
            lhs = Expr.sub(Expr.mul(AtoM, Expr.vstack(zp_R.index(e,0), zp_R.index(e,1))),
                           Expr.mul(B_to, y.index(e)))
            le0(M, lhs)
            eq0(M, Expr.sub(zp_R.index(e,2), Expr.mul(y.index(e), z_to)))

        if mode_from == "SL" and mode_to == "SR":
            lhs1 = Expr.sub(Expr.mul(AfromM, Expr.vstack(zp_L.index(e,0), zp_L.index(e,1))),
                            Expr.mul(B_from, y.index(e)))
            lhs2 = Expr.sub(Expr.mul(AtoM,   Expr.vstack(zp_R.index(e,0), zp_R.index(e,1))),
                            Expr.mul(B_to,   y.index(e)))
            le0(M, lhs1); le0(M, lhs2)
            eq0(M, Expr.sub(zp_L.index(e,2), Expr.mul(y.index(e), z_from)))
            eq0(M, Expr.sub(zp_R.index(e,2), Expr.mul(y.index(e), z_to)))

        if mode_from == "SR" and mode_to == "SL":
            lhs1 = Expr.sub(Expr.mul(AfromM, Expr.vstack(zp_R.index(e,0), zp_R.index(e,1))),
                            Expr.mul(B_from, y.index(e)))
            lhs2 = Expr.sub(Expr.mul(AtoM,   Expr.vstack(zp_L.index(e,0), zp_L.index(e,1))),
                            Expr.mul(B_to,   y.index(e)))
            le0(M, lhs1); le0(M, lhs2)
            eq0(M, Expr.sub(zp_R.index(e,2), Expr.mul(y.index(e), z_from)))
            eq0(M, Expr.sub(zp_L.index(e,2), Expr.mul(y.index(e), z_to)))

        if mode_from == "SL" and mode_to == "D":
            lhs = Expr.sub(Expr.mul(AfromM, Expr.vstack(z_L.index(e,0), z_L.index(e,1))),
                           Expr.mul(B_from, y.index(e)))
            le0(M, lhs)
            eq0(M, Expr.sub(z_L.index(e,2), Expr.mul(y.index(e), z_from)))
        if mode_from == "SR" and mode_to == "D":
            lhs = Expr.sub(Expr.mul(AfromM, Expr.vstack(z_R.index(e,0), z_R.index(e,1))),
                           Expr.mul(B_from, y.index(e)))
            le0(M, lhs)
            eq0(M, Expr.sub(z_R.index(e,2), Expr.mul(y.index(e), z_from)))

    # (G) z(4:5) : |var| ≤ y  → var - y ≤ 0,  -var - y ≤ 0
    for e in range(n_edge):
        for var in (z_L, z_R, zp_L, zp_R):
            for k in (3,4):  # 0-based: cos,sin
                le0(M, Expr.sub(var.index(e,k), y.index(e)))
                le0(M, Expr.sub(Expr.neg(var.index(e,k)), y.index(e)))

    # (H) step span: ||(zp - z)_{1:2}|| ≤ step_span * y
    for e in range(n_edge):
        cone2(M, Expr.mul(step_span, y.index(e)),
              Expr.sub(zp_L.index(e,0), z_L.index(e,0)),
              Expr.sub(zp_L.index(e,1), z_L.index(e,1)))
        cone2(M, Expr.mul(step_span, y.index(e)),
              Expr.sub(zp_R.index(e,0), z_R.index(e,0)),
              Expr.sub(zp_R.index(e,1), z_R.index(e,1)))

    # (H-1) relative foot : ||(zp_L - z_R)_{1:2}|| ≤ 0.35 * y
    for e in range(n_edge):
        cone2(M, Expr.mul(0.35, y.index(e)),
              Expr.sub(zp_L.index(e,0), z_R.index(e,0)),
              Expr.sub(zp_L.index(e,1), z_R.index(e,1)))
        cone2(M, Expr.mul(0.35, y.index(e)),
              Expr.sub(zp_R.index(e,0), z_L.index(e,0)),
              Expr.sub(zp_R.index(e,1), z_L.index(e,1)))


        # (I) Inward constraint (TORO Constraint)
        SIN_COL = 4  # 0-based index: [x,y,z,cos,sin]
        zpR_sin = zp_R.slice([0, SIN_COL], [n_edge, SIN_COL + 1]).reshape(n_edge)
        zL_sin = z_L.slice([0, SIN_COL], [n_edge, SIN_COL + 1]).reshape(n_edge)
        zpL_sin = zp_L.slice([0, SIN_COL], [n_edge, SIN_COL + 1]).reshape(n_edge)
        zR_sin = z_R.slice([0, SIN_COL], [n_edge, SIN_COL + 1]).reshape(n_edge)

        M.constraint(Expr.sub(zpR_sin, zL_sin), Domain.lessThan(0.0))  # zp_R <= z_L
        M.constraint(Expr.sub(zpL_sin, zR_sin), Domain.greaterThan(0.0))  # zp_L >= z_R



        # ===== (J) 5x5 SDP 블록 (회전/연결) — 전역축 일치 버전 =====
        # 전역←로컬 변환행렬 E (main 에서 만든 x_ref, y_ref 사용)
        E11 = float(x_ref[0])  # ex.x
        E21 = float(x_ref[1])  # ex.y
        E12 = float(y_ref[0])  # ey.x
        E22 = float(y_ref[1])  # ey.y

        # [c_g; s_g] = [[E11,E12],[E21,E22]] * [c; s]

        def cg_from_local(c_var, s_var):
            return Expr.add(Expr.mul(E11, c_var), Expr.mul(E12, s_var))  # c_g

        def sg_from_local(c_var, s_var):
            return Expr.add(Expr.mul(E21, c_var), Expr.mul(E22, s_var))  # s_g

        src_edges_set = set(map(int, src_edges))  # 0-based source에서 나가는 엣지 집합

        for e in range(n_edge):
            XL, XR = X_L[e], X_R[e]

            # X[0,0] == 1
            eq0(M, Expr.sub(XL.index(0, 0), 1.0))
            eq0(M, Expr.sub(XR.index(0, 0), 1.0))

            # 단위원: (c_g,s_g), (c'_g,s'_g) 가 단위벡터
            eq0(M, Expr.sub(Expr.add(XL.index(1, 1), XL.index(2, 2)), 1.0))
            eq0(M, Expr.sub(Expr.add(XR.index(1, 1), XR.index(2, 2)), 1.0))
            eq0(M, Expr.sub(Expr.add(XL.index(3, 3), XL.index(4, 4)), 1.0))
            eq0(M, Expr.sub(Expr.add(XR.index(3, 3), XR.index(4, 4)), 1.0))

            # cos*cos' + sin*sin' >= cos(alpha)   (전역 기준: c_g c'_g + s_g s'_g)
            ge0(M, Expr.sub(Expr.add(XL.index(1, 3), XL.index(2, 4)), Expr.mul(ca, y.index(e))))
            ge0(M, Expr.sub(Expr.add(XR.index(1, 3), XR.index(2, 4)), Expr.mul(ca, y.index(e))))

            # --- row0 매핑: [1, c_g, s_g, c'_g, s'_g] ---
            if e in src_edges_set:
                # (x0 행) 현재는 start_*, 다음은 zp_*  (전부 전역으로 변환해서 넣음)
                # Left
                eq0(M, Expr.sub(XL.index(0, 1), cg_from_local(start_L.index(3), start_L.index(4))))  # c_g
                eq0(M, Expr.sub(XL.index(0, 2), sg_from_local(start_L.index(3), start_L.index(4))))  # s_g
                eq0(M, Expr.sub(XL.index(0, 3), cg_from_local(zp_L.index(e, 3), zp_L.index(e, 4))))  # c'_g
                eq0(M, Expr.sub(XL.index(0, 4), sg_from_local(zp_L.index(e, 3), zp_L.index(e, 4))))  # s'_g
                # Right
                eq0(M, Expr.sub(XR.index(0, 1), cg_from_local(start_R.index(3), start_R.index(4))))
                eq0(M, Expr.sub(XR.index(0, 2), sg_from_local(start_R.index(3), start_R.index(4))))
                eq0(M, Expr.sub(XR.index(0, 3), cg_from_local(zp_R.index(e, 3), zp_R.index(e, 4))))
                eq0(M, Expr.sub(XR.index(0, 4), sg_from_local(zp_R.index(e, 3), zp_R.index(e, 4))))
            else:
                # 일반 edge: 현재 z_* → 다음 zp_*  (전역으로 변환해서 넣음)
                # Left
                eq0(M, Expr.sub(XL.index(0, 1), cg_from_local(z_L.index(e, 3), z_L.index(e, 4))))
                eq0(M, Expr.sub(XL.index(0, 2), sg_from_local(z_L.index(e, 3), z_L.index(e, 4))))
                eq0(M, Expr.sub(XL.index(0, 3), cg_from_local(zp_L.index(e, 3), zp_L.index(e, 4))))
                eq0(M, Expr.sub(XL.index(0, 4), sg_from_local(zp_L.index(e, 3), zp_L.index(e, 4))))
                # Right
                eq0(M, Expr.sub(XR.index(0, 1), cg_from_local(z_R.index(e, 3), z_R.index(e, 4))))
                eq0(M, Expr.sub(XR.index(0, 2), sg_from_local(z_R.index(e, 3), z_R.index(e, 4))))
                eq0(M, Expr.sub(XR.index(0, 3), cg_from_local(zp_R.index(e, 3), zp_R.index(e, 4))))
                eq0(M, Expr.sub(XR.index(0, 4), sg_from_local(zp_R.index(e, 3), zp_R.index(e, 4))))

        def mccormick_bind(M, u, uL, uU, v, vL, vU, w):
            # w >= uL*v + vL*u - uL*vL
            M.constraint(Expr.sub(
                w, Expr.add(Expr.add(Expr.mul(uL, v), Expr.mul(vL, u)), -uL * vL)
            ), Domain.greaterThan(0.0))
            # w >= uU*v + vU*u - uU*vU
            M.constraint(Expr.sub(
                w, Expr.add(Expr.add(Expr.mul(uU, v), Expr.mul(vU, u)), -uU * vU)
            ), Domain.greaterThan(0.0))
            # w <= uU*v + vL*u - uU*vL
            M.constraint(Expr.sub(
                Expr.add(Expr.add(Expr.mul(uU, v), Expr.mul(vL, u)), -uU * vL), w
            ), Domain.greaterThan(0.0))
            # w <= uL*v + vU*u - uL*vU
            M.constraint(Expr.sub(
                Expr.add(Expr.add(Expr.mul(uL, v), Expr.mul(vU, u)), -uL * vU), w
            ), Domain.greaterThan(0.0))

    # # ----------- X_R tightness RLT -----------
    # if e in src_edges_set:
    #     # 현재 값은 start_*에서 상수(혹은 파라미터), 다음은 zp_*
    #     cg = cg_from_local(start_L.index(3), start_L.index(4))  # c_g
    #     sg = sg_from_local(start_L.index(3), start_L.index(4))  # s_g
    #     cgp = cg_from_local(zp_L.index(e, 3), zp_L.index(e, 4))  # c'_g
    #     sgp = sg_from_local(zp_L.index(e, 3), zp_L.index(e, 4))  # s'_g
    # else:
    #     cg = cg_from_local(z_L.index(e, 3), z_L.index(e, 4))
    #     sg = sg_from_local(z_L.index(e, 3), z_L.index(e, 4))
    #     cgp = cg_from_local(zp_L.index(e, 3), zp_L.index(e, 4))
    #     sgp = sg_from_local(zp_L.index(e, 3), zp_L.index(e, 4))
    #
    # # 안전한 박스 경계(전역 cos/sin은 [-1,1] 가정; E는 정규직교)
    # for expr in (cg, sg, cgp, sgp):
    #     M.constraint(expr, Domain.inRange(-1.0, 1.0))
    #
    # # 맵: (u, v, w=XL[i,j]) 6쌍
    # pairs_XL = [
    #     (cg, sg, XL.index(1, 2)),  # c_g s_g
    #     (cg, cgp, XL.index(1, 3)),  # c_g c'_g
    #     (cg, sgp, XL.index(1, 4)),  # c_g s'_g
    #     (sg, cgp, XL.index(2, 3)),  # s_g c'_g
    #     (sg, sgp, XL.index(2, 4)),  # s_g s'_g
    #     (cgp, sgp, XL.index(3, 4)),  # c'_g s'_g
    # ]
    #
    # # src edge면 상수×변수(혹은 상수×상수)인 것들은 "선형 등식"으로 더 타이트하게 가능
    # if e in src_edges_set:
    #     # 현재(cg,sg)는 start_L에서 상수 → 레벨값을 한 번 읽어서 등식으로 묶으면 끝
    #     cg0 = float(cg.level()[0]) if hasattr(cg, "level") else None
    #     sg0 = float(sg.level()[0]) if hasattr(sg, "level") else None
    #     # (1,2)=cg*sg는 상수
    #     if cg0 is not None and sg0 is not None:
    #         eq0(M, Expr.sub(XL.index(1, 2), cg0 * sg0))
    #     else:
    #         mccormick_bind(M, cg, -1.0, 1.0, sg, -1.0, 1.0, XL.index(1, 2))
    #
    #     # (1,3),(1,4) : cg0*변수
    #     if cg0 is not None:
    #         eq0(M, Expr.sub(XL.index(1, 3), Expr.mul(cg0, cgp)))
    #         eq0(M, Expr.sub(XL.index(1, 4), Expr.mul(cg0, sgp)))
    #     else:
    #         mccormick_bind(M, cg, -1.0, 1.0, cgp, -1.0, 1.0, XL.index(1, 3))
    #         mccormick_bind(M, cg, -1.0, 1.0, sgp, -1.0, 1.0, XL.index(1, 4))
    #
    #     # (2,3),(2,4) : sg0*변수
    #     if sg0 is not None:
    #         eq0(M, Expr.sub(XL.index(2, 3), Expr.mul(sg0, cgp)))
    #         eq0(M, Expr.sub(XL.index(2, 4), Expr.mul(sg0, sgp)))
    #     else:
    #         mccormick_bind(M, sg, -1.0, 1.0, cgp, -1.0, 1.0, XL.index(2, 3))
    #         mccormick_bind(M, sg, -1.0, 1.0, sgp, -1.0, 1.0, XL.index(2, 4))
    #
    #     # (3,4)=c'_g s'_g : 둘 다 변수 → 맥코믹
    #     mccormick_bind(M, cgp, -1.0, 1.0, sgp, -1.0, 1.0, XL.index(3, 4))
    #
    # else:
    #     # 일반 엣지: 6개 전부 맥코믹
    #     for u, v, w in pairs_XL:
    #         mccormick_bind(M, u, -1.0, 1.0, v, -1.0, 1.0, w)
    #
    # # ----------- X_R tightness RLT -----------
    # if e in src_edges_set:
    #     # 현재 값은 start_*에서 상수(혹은 파라미터), 다음은 zp_*
    #     cg = cg_from_local(start_R.index(3), start_R.index(4))  # c_g
    #     sg = sg_from_local(start_R.index(3), start_R.index(4))  # s_g
    #     cgp = cg_from_local(zp_R.index(e, 3), zp_R.index(e, 4))  # c'_g
    #     sgp = sg_from_local(zp_R.index(e, 3), zp_R.index(e, 4))  # s'_g
    # else:
    #     cg = cg_from_local(z_R.index(e, 3), z_R.index(e, 4))
    #     sg = sg_from_local(z_R.index(e, 3), z_R.index(e, 4))
    #     cgp = cg_from_local(zp_R.index(e, 3), zp_R.index(e, 4))
    #     sgp = sg_from_local(zp_R.index(e, 3), zp_R.index(e, 4))
    #
    # # 안전한 박스 경계(전역 cos/sin은 [-1,1] 가정; E는 정규직교)
    # for expr in (cg, sg, cgp, sgp):
    #     M.constraint(expr, Domain.inRange(-1.0, 1.0))
    #
    # # 맵: (u, v, w=XL[i,j]) 6쌍
    # pairs_XR = [
    #     (cg, sg, XR.index(1, 2)),  # c_g s_g
    #     (cg, cgp, XR.index(1, 3)),  # c_g c'_g
    #     (cg, sgp, XR.index(1, 4)),  # c_g s'_g
    #     (sg, cgp, XR.index(2, 3)),  # s_g c'_g
    #     (sg, sgp, XR.index(2, 4)),  # s_g s'_g
    #     (cgp, sgp, XR.index(3, 4)),  # c'_g s'_g
    # ]
    #
    # # src edge면 상수×변수(혹은 상수×상수)인 것들은 "선형 등식"으로 더 타이트하게 가능
    # if e in src_edges_set:
    #     # 현재(cg,sg)는 start_L에서 상수 → 레벨값을 한 번 읽어서 등식으로 묶으면 끝
    #     cg0 = float(cg.level()[0]) if hasattr(cg, "level") else None
    #     sg0 = float(sg.level()[0]) if hasattr(sg, "level") else None
    #     # (1,2)=cg*sg는 상수
    #     if cg0 is not None and sg0 is not None:
    #         eq0(M, Expr.sub(XR.index(1, 2), cg0 * sg0))
    #     else:
    #         mccormick_bind(M, cg, -1.0, 1.0, sg, -1.0, 1.0, XR.index(1, 2))
    #
    #     # (1,3),(1,4) : cg0*변수
    #     if cg0 is not None:
    #         eq0(M, Expr.sub(XR.index(1, 3), Expr.mul(cg0, cgp)))
    #         eq0(M, Expr.sub(XR.index(1, 4), Expr.mul(cg0, sgp)))
    #     else:
    #         mccormick_bind(M, cg, -1.0, 1.0, cgp, -1.0, 1.0, XR.index(1, 3))
    #         mccormick_bind(M, cg, -1.0, 1.0, sgp, -1.0, 1.0, XR.index(1, 4))
    #
    #     # (2,3),(2,4) : sg0*변수
    #     if sg0 is not None:
    #         eq0(M, Expr.sub(XR.index(2, 3), Expr.mul(sg0, cgp)))
    #         eq0(M, Expr.sub(XR.index(2, 4), Expr.mul(sg0, sgp)))
    #     else:
    #         mccormick_bind(M, sg, -1.0, 1.0, cgp, -1.0, 1.0, XR.index(2, 3))
    #         mccormick_bind(M, sg, -1.0, 1.0, sgp, -1.0, 1.0, XR.index(2, 4))
    #
    #     # (3,4)=c'_g s'_g : 둘 다 변수 → 맥코믹
    #     mccormick_bind(M, cgp, -1.0, 1.0, sgp, -1.0, 1.0, XR.index(3, 4))
    #
    # else:
    #     # 일반 엣지: 6개 전부 맥코믹
    #     for u, v, w in pairs_XR:
    #         mccormick_bind(M, u, -1.0, 1.0, v, -1.0, 1.0, w)


    # ===== (K) 7x7 SDP 블록 (반평면 회전 제약; 게이팅 없음, 전역축 일치) =====
    # - 전역 좌표계: MATLAB X/Y
    # - 로컬 yaw(c,s)는 start 기준 로컬축 -> 전역으로 변환한 (c_g,s_g)만 사용
    # - row0: [1, x, y, c_g, s_g, x', y']

    # ---- 전역←로컬 변환행렬 E 구성 (스칼라 상수) ----
    # main에서 이미 만든 x_ref, y_ref 를 그대로 사용
    E11 = float(x_ref[0])  # ex.x
    E21 = float(x_ref[1])  # ex.y
    E12 = float(y_ref[0])  # ey.x
    E22 = float(y_ref[1])  # ey.y

    # 참고: [c_g; s_g] = [[E11,E12],[E21,E22]] * [c; s]

    def cg_from_local(c_var, s_var):
        # c_g = ex.x * c + ey.x * s
        return Expr.add(Expr.mul(E11, c_var), Expr.mul(E12, s_var))

    def sg_from_local(c_var, s_var):
        # s_g = ex.y * c + ey.y * s
        return Expr.add(Expr.mul(E21, c_var), Expr.mul(E22, s_var))


    # ---- 파라미터 ----
    d_lat = 0.25
    src_edges_set = set(map(int, src_edges))
    sink_edges_set = set(map(int, sink_edges))

    def w_perp_from(X):
        # w_perp = -s_g*(x' - x) + c_g*(y' - y)
        # (XL2: -n_right^T (pR' - pL), XR2: +n_left^T (pL' - pR))
        return Expr.add(Expr.sub(X.index(4, 1), X.index(4, 5)),
                        Expr.sub(X.index(3, 6), X.index(3, 2)))

    for e in range(n_edge):
        XL2, XR2 = X_L2[e], X_R2[e]

        # X[0,0]=1
        eq0(M, Expr.sub(XL2.index(0, 0), 1.0))
        eq0(M, Expr.sub(XR2.index(0, 0), 1.0))

        # 단위원 등호
        eq0(M, Expr.sub(Expr.add(XL2.index(3, 3), XL2.index(4, 4)), 1.0))
        eq0(M, Expr.sub(Expr.add(XR2.index(3, 3), XR2.index(4, 4)), 1.0))

        # --- row0 매핑: c_g, s_g로 매핑 (전역 프레임 일치) ---
        if e in src_edges_set:
            # XL2 row0 = [1, xL, yL, c_g(L), s_g(L), xL', yL']
            eq0(M, Expr.sub(XL2.index(0, 1), start_L.index(0)))
            eq0(M, Expr.sub(XL2.index(0, 2), start_L.index(1)))
            eq0(M, Expr.sub(XL2.index(0, 3), cg_from_local(start_L.index(3), start_L.index(4))))
            eq0(M, Expr.sub(XL2.index(0, 4), sg_from_local(start_L.index(3), start_L.index(4))))
            eq0(M, Expr.sub(XL2.index(0, 5), zp_R.index(e, 0)))
            eq0(M, Expr.sub(XL2.index(0, 6), zp_R.index(e, 1)))
            # XR2 row0 = [1, xR, yR, c_g(R), s_g(R), xR', yR']
            eq0(M, Expr.sub(XR2.index(0, 1), start_R.index(0)))
            eq0(M, Expr.sub(XR2.index(0, 2), start_R.index(1)))
            eq0(M, Expr.sub(XR2.index(0, 3), cg_from_local(start_R.index(3), start_R.index(4))))
            eq0(M, Expr.sub(XR2.index(0, 4), sg_from_local(start_R.index(3), start_R.index(4))))
            eq0(M, Expr.sub(XR2.index(0, 5), zp_L.index(e, 0)))
            eq0(M, Expr.sub(XR2.index(0, 6), zp_L.index(e, 1)))

        elif e in sink_edges_set:
            # XL2 row0 = [1, xL, yL, c_g(L), s_g(L), xL', yL']
            eq0(M, Expr.sub(XL2.index(0, 1), z_L.index(e, 0)))
            eq0(M, Expr.sub(XL2.index(0, 2), z_L.index(e, 1)))
            eq0(M, Expr.sub(XL2.index(0, 3), cg_from_local(z_L.index(e, 3), z_L.index(e, 4))))
            eq0(M, Expr.sub(XL2.index(0, 4), sg_from_local(z_L.index(e, 3), z_L.index(e, 4))))
            eq0(M, Expr.sub(XL2.index(0, 5), target_R.index(0)))
            eq0(M, Expr.sub(XL2.index(0, 6), target_R.index(1)))
            # XR2 row0 = [1, xR, yR, c_g(R), s_g(R), xR', yR']
            eq0(M, Expr.sub(XR2.index(0, 1), z_R.index(e, 0)))
            eq0(M, Expr.sub(XR2.index(0, 2), z_R.index(e, 1)))
            eq0(M, Expr.sub(XR2.index(0, 3), cg_from_local(z_R.index(e, 3), z_R.index(e, 4))))
            eq0(M, Expr.sub(XR2.index(0, 4), sg_from_local(z_R.index(e, 3), z_R.index(e, 4))))
            eq0(M, Expr.sub(XR2.index(0, 5), target_L.index(0)))
            eq0(M, Expr.sub(XR2.index(0, 6), target_L.index(1)))

        else:
            # XL2 row0 = [1, xL, yL, c_g(L), s_g(L), xL', yL']
            eq0(M, Expr.sub(XL2.index(0, 1), z_L.index(e, 0)))
            eq0(M, Expr.sub(XL2.index(0, 2), z_L.index(e, 1)))
            eq0(M, Expr.sub(XL2.index(0, 3), cg_from_local(z_L.index(e, 3), z_L.index(e, 4))))
            eq0(M, Expr.sub(XL2.index(0, 4), sg_from_local(z_L.index(e, 3), z_L.index(e, 4))))
            eq0(M, Expr.sub(XL2.index(0, 5), zp_R.index(e, 0)))
            eq0(M, Expr.sub(XL2.index(0, 6), zp_R.index(e, 1)))
            # XR2 row0 = [1, xR, yR, c_g(R), s_g(R), xR', yR']
            eq0(M, Expr.sub(XR2.index(0, 1), z_R.index(e, 0)))
            eq0(M, Expr.sub(XR2.index(0, 2), z_R.index(e, 1)))
            eq0(M, Expr.sub(XR2.index(0, 3), cg_from_local(z_R.index(e, 3), z_R.index(e, 4))))
            eq0(M, Expr.sub(XR2.index(0, 4), sg_from_local(z_R.index(e, 3), z_R.index(e, 4))))
            eq0(M, Expr.sub(XR2.index(0, 5), zp_L.index(e, 0)))
            eq0(M, Expr.sub(XR2.index(0, 6), zp_L.index(e, 1)))

        # --- 모드별 반평면 하드 제약 ---
        to_v = int(flow_in_vertices[e])
        mode = mode_list[to_v]

        if mode == "SL":
            # 왼발: 왼쪽 반평면 → w_perp_L ≥ d_lat * y[e]
            # ⇔ (d_lat*y[e] - w_perp_L) ≤ 0
            wperp_L = w_perp_from(XR2)
            M.constraint(
                Expr.sub(Expr.mul(d_lat, y.index(e)), wperp_L),
                Domain.lessThan(0.0)
            )
        elif mode == "SR":
            # 오른발: 오른쪽 반평면 → w_perp_R ≤ - d_lat * y[e]
            # ⇔ (w_perp_R + d_lat*y[e]) ≤ 0
            wperp_R = w_perp_from(XL2)
            M.constraint(
                Expr.add(wperp_R, Expr.mul(d_lat, y.index(e))),
                Domain.lessThan(0.0)
            )
        elif mode == "D":
            wperp_L = w_perp_from(XR2)
            M.constraint(
                Expr.sub(Expr.mul(d_lat, y.index(e)), wperp_L),
                Domain.lessThan(0.0)
            )
            wperp_R = w_perp_from(XL2)
            M.constraint(
                Expr.add(wperp_R, Expr.mul(d_lat, y.index(e))),
                Domain.lessThan(0.0)
            )
        else:
            pass


    # --- McCormick helper (Expr-only; 이전 오류 방지 버전) ---
    def mccormick(M, v, w, vw, vL, vU, wL, wU):
        # 1) vw ≥ vL*w + wL*v - vL*wL
        M.constraint(
            Expr.sub(
                vw,
                Expr.add(Expr.mul(vL, w), Expr.add(Expr.mul(wL, v), -vL * wL))
            ),
            Domain.greaterThan(0.0)
        )
        # 2) vw ≥ vU*w + wU*v - vU*wU
        M.constraint(
            Expr.sub(
                vw,
                Expr.add(Expr.mul(vU, w), Expr.add(Expr.mul(wU, v), -vU * wU))
            ),
            Domain.greaterThan(0.0)
        )
        # 3) vw ≤ vU*w + wL*v - vU*wL    ⇔ (...) - vw ≥ 0
        M.constraint(
            Expr.sub(
                Expr.add(Expr.add(Expr.mul(vU, w), Expr.mul(wL, v)), -vU * wL),
                vw
            ),
            Domain.greaterThan(0.0)
        )
        # 4) vw ≤ vL*w + wU*v - vL*wU    ⇔ (...) - vw ≥ 0
        M.constraint(
            Expr.sub(
                Expr.add(Expr.add(Expr.mul(vL, w), Expr.mul(wU, v)), -vL * wU),
                vw
            ),
            Domain.greaterThan(0.0)
        )

    # --- 하나의 7x7 PSD 블록 X에 대해 모든 교차항에 RLT 걸기 ---
    # row0 = [1, x, y, cg, sg, xp, yp]  => 인덱스 0..6
    def apply_full_rlt_block(M, X, bounds):
        # row0 스칼라 Expr
        x = X.index(0, 1);
        y = X.index(0, 2)
        cg = X.index(0, 3);
        sg = X.index(0, 4)
        xp = X.index(0, 5);
        yp = X.index(0, 6)

        vars_list = [x, y, cg, sg, xp, yp]
        idx_in_mat = [1, 2, 3, 4, 5, 6]  # X의 행/열 인덱스
        lows, highs = zip(*bounds)  # [(xL,xU), (yL,yU), ..., (ypL,ypU)]

        # 모든 i<j 쌍에 대해: X[idx_i, idx_j] ≈ vars[i]*vars[j]
        for i in range(6):
            vi, vL, vU, ii = vars_list[i], lows[i], highs[i], idx_in_mat[i]
            for j in range(i + 1, 6):
                vj, wL, wU, jj = vars_list[j], lows[j], highs[j], idx_in_mat[j]
                vw = X.index(ii, jj)  # 대칭행렬이라 (jj,ii)와 동일
                mccormick(M, vi, vj, vw, vL, vU, wL, wU)

        # (옵션) 대각 상한도 같이 캡: X[ii,ii] ≤ max(|L|,|U|)^2  (폭주 방지)
        caps = [max(abs(l), abs(u)) ** 2 for (l, u) in bounds]
        for k, ii in enumerate(idx_in_mat):
            M.constraint(Expr.sub(X.index(ii, ii), caps[k]), Domain.lessThan(0.0))


    # (L) Coupling tightness
    for e in range(n_edge):
        XL, XR = X_L[e], X_R[e];
        XL2, XR2 = X_L2[e], X_R2[e];

        eq0(M, Expr.sub(XL.index(0,1), XL2.index(0,3)))
        eq0(M, Expr.sub(XL.index(0,2), XL2.index(0,4)))
        eq0(M, Expr.sub(XL.index(1,1), XL2.index(3,3)))
        eq0(M, Expr.sub(XL.index(1,2), XL2.index(3,4)))
        eq0(M, Expr.sub(XL.index(2,2), XL2.index(4,4)))

        eq0(M, Expr.sub(XR.index(0, 1), XR2.index(0, 3)))
        eq0(M, Expr.sub(XR.index(0, 2), XR2.index(0, 4)))
        eq0(M, Expr.sub(XR.index(1, 1), XR2.index(3, 3)))
        eq0(M, Expr.sub(XR.index(1, 2), XR2.index(3, 4)))
        eq0(M, Expr.sub(XR.index(2, 2), XR2.index(4, 4)))



    # =============================== Objective ===================================
    terms = []


    #  (A) rho related objective term
    Mbig = 1.0
    for e in range(n_edge):
        m = mode_list[flow_in_vertices[e]]

        # deltas
        dLx = Expr.sub(zp_L.index(e, 0), z_L.index(e, 0))
        dLy = Expr.sub(zp_L.index(e, 1), z_L.index(e, 1))
        dRx = Expr.sub(zp_R.index(e, 0), z_R.index(e, 0))
        dRy = Expr.sub(zp_R.index(e, 1), z_R.index(e, 1))
        hL = Expr.sub(zp_L.index(e, 2), z_L.index(e, 2))
        hR = Expr.sub(zp_R.index(e, 2), z_R.index(e, 2))

        if m == "SL":
            # 오른발(스탠스) 고정
            for k in (0, 1, 3, 4):
                eq0(M, Expr.sub(z_R.index(e, k), zp_R.index(e, k)))
            # 왼발만 이동 cone
            cone2(M, rho_L.index(e), dLx, dLy)
            eq0(M, rho_R.index(e))

        elif m == "SR":
            for k in (0, 1, 3, 4):
                eq0(M, Expr.sub(z_L.index(e, k), zp_L.index(e, k)))
            cone2(M, rho_R.index(e), dRx, dRy)
            eq0(M, rho_L.index(e))

        elif m == "D":
            # ✅ D는 둘 다 cone (이중집계가 걱정되면 나중에 w를 0.5로)
            cone2(M, rho_L.index(e), dLx, dLy)
            cone2(M, rho_R.index(e), dRx, dRy)

        # |h| <= rhoz (원하면 모드별 마스크도 가능)
        cone2(M, rhoz_L.index(e), hL)
        cone2(M, rhoz_R.index(e), hR)

        # indicator linking (비활성 엣지면 rho=0으로 수축)
        M.constraint(Expr.sub(rho_L.index(e), Expr.mul(Mbig, y.index(e))), Domain.lessThan(0.0))
        M.constraint(Expr.sub(rho_R.index(e), Expr.mul(Mbig, y.index(e))), Domain.lessThan(0.0))
        M.constraint(Expr.sub(rhoz_L.index(e), Expr.mul(Mbig, y.index(e))), Domain.lessThan(0.0))
        M.constraint(Expr.sub(rhoz_R.index(e), Expr.mul(Mbig, y.index(e))), Domain.lessThan(0.0))

    # --- masks by mode ---
    wL = np.zeros(n_edge);
    wR = np.zeros(n_edge)
    for e in range(n_edge):
        m_from = mode_list[flow_out_vertices[e]]
        m_to = mode_list[flow_in_vertices[e]]

        if m_from == "SR" and m_to == "D":
             wR[e] = 1.0
             wL[e] = 1.0
        elif m_from == "SL" and m_to == "D":
             wR[e] = 1.0
             wL[e] = 1.0
        elif m_from == "SR" and m_to == "SL":
            wL[e] = 1.0
        elif m_from == "D" and m_to == "SL":
            wL[e] = 1.0
        elif m_from == "SL" and m_to == "SR":
            wR[e] = 1.0
        elif m_from == "D" and m_to == "SR":
            wR[e] = 1.0
    # xy
    terms.append(Expr.sum(Expr.mulElm(wL, rho_L)))
    terms.append(Expr.sum(Expr.mulElm(wR, rho_R)))
    # z
    terms.append(Expr.add(Expr.sum(rhoz_L), Expr.sum(rhoz_R)))


    # ========================== L2^2(= sum of squares) 균등화 항 추가 =======================
    lam_eq = 2.0  # 작게 시작해서 조절

    # 보조변수 tL, tR >= 0  (각 원소가 rho^2를 상계)
    tL = M.variable("tL", n_edge, Domain.greaterThan(0.0))
    tR = M.variable("tR", n_edge, Domain.greaterThan(0.0))

    # t[e] >= rho[e]^2  를 회전 2차콘으로: (t, 1, rho) ∈ RQCone  =>  t*1 >= rho^2
    ONE = Expr.constTerm(1, 1.0)  # 혹은: Matrix.dense(1,1,[1.0])
    for e in range(n_edge):
        M.constraint(Expr.vstack(tL.index(e), ONE, rho_L.index(e)),
                     Domain.inRotatedQCone())
        M.constraint(Expr.vstack(tR.index(e), ONE, rho_R.index(e)),
                     Domain.inRotatedQCone())

    gamma = 3.0  # 시작/끝을 얼마나 더 작게 만들지 (1.5~3.0 사이에서 조절)
    wProfile_L = np.ones(n_edge)
    wProfile_R = np.ones(n_edge)

    for e in range(n_edge):
        m_from = mode_list[flow_out_vertices[e]]
        m_to = mode_list[flow_in_vertices[e]]

        # 첫 발 디딤: D -> SL (왼발이 움직임), D -> SR (오른발이 움직임)
        if m_from == "D" and m_to == "SL":
            wProfile_L[e] = gamma
        elif m_from == "D" and m_to == "SR":
            wProfile_R[e] = gamma

        # 마지막 멈춤: SL -> D (오른발이 마지막으로 움직여 정렬), SR -> D (왼발이 마지막)
        elif m_from == "SL" and m_to == "D":
            wProfile_L[e] = gamma
        elif m_from == "SR" and m_to == "D":
            wProfile_R[e] = gamma

    # numpy -> Matrix
    wPL = Matrix.dense(n_edge, 1, wProfile_L.tolist())
    wPR = Matrix.dense(n_edge, 1, wProfile_R.tolist())

    # 이제 L2^2 항에 가중치 곱해서 목적식에 추가
    # (기존에 넣은) lam_eq는 그대로 사용
    terms.append(Expr.mul(lam_eq, Expr.sum(Expr.mulElm(wPL, tL))))
    terms.append(Expr.mul(lam_eq, Expr.sum(Expr.mulElm(wPR, tR))))

    # ========================= rotation related cost (per-foot ref) =========================
    # - 왼발 yaw: [z_L(e,3), z_L(e,4)]  vs  ref_L = (zp_L - z_L)
    # - 오른발 yaw: [z_R(e,3), z_R(e,4)] vs  ref_R = (zp_R - z_R)
    lam_rot = 0.1
    lam_circ = 1.0 * lam_rot  # (옵션) 단위원 유도용 가중치

    ref_scale = 0.1
    inv_scale = 1.0 / ref_scale

    for e in range(n_edge):
        # --- 왼발 기준 참조 이동벡터: 현재 왼발 -> 다음 왼발 ---
        L_dx = Expr.sub(zp_L.index(e, 0), z_L.index(e, 0))
        L_dy = Expr.sub(zp_L.index(e, 1), z_L.index(e, 1))

        # --- 오른발 기준 참조 이동벡터: 현재 오른발 -> 다음 오른발 ---
        R_dx = Expr.sub(zp_R.index(e, 0), z_R.index(e, 0))
        R_dy = Expr.sub(zp_R.index(e, 1), z_R.index(e, 1))

        # --- 고정 스케일 정규화 (수치안정 & 볼록성 유지) ---
        L_dx_s = Expr.mul(inv_scale, L_dx)
        L_dy_s = Expr.mul(inv_scale, L_dy)
        R_dx_s = Expr.mul(inv_scale, R_dx)
        R_dy_s = Expr.mul(inv_scale, R_dy)

        # --- 전역 기준축 성분 (x_ref, y_ref는 전역 단위직교축) ---
        # ref_c = < ref_step_s, axis >
        L_c1 = Expr.add(Expr.mul(float(x_ref[0]), L_dx_s),
                        Expr.mul(float(x_ref[1]), L_dy_s))
        L_c2 = Expr.add(Expr.mul(float(y_ref[0]), L_dx_s),
                        Expr.mul(float(y_ref[1]), L_dy_s))

        R_c1 = Expr.add(Expr.mul(float(x_ref[0]), R_dx_s),
                        Expr.mul(float(x_ref[1]), R_dy_s))
        R_c2 = Expr.add(Expr.mul(float(y_ref[0]), R_dx_s),
                        Expr.mul(float(y_ref[1]), R_dy_s))

        # --- yaw 정렬 SOCP: t >= || [cos - ref_c1, sin - ref_c2] ||_2 ---
        tLrot = M.variable(f"tLrot_{e}", 1, Domain.greaterThan(0.0))
        tRrot = M.variable(f"tRrot_{e}", 1, Domain.greaterThan(0.0))

        # 왼발 yaw = z_L(e,3:5)
        cone2(M, tLrot,
              Expr.sub(z_L.index(e, 3), L_c1),
              Expr.sub(z_L.index(e, 4), L_c2))

        # 오른발 yaw = z_R(e,3:5)
        cone2(M, tRrot,
              Expr.sub(z_R.index(e, 3), R_c1),
              Expr.sub(z_R.index(e, 4), R_c2))

        terms.append(Expr.mul(lam_rot, Expr.add(tLrot, tRrot)))


    # step number
    terms.append(Expr.mul(0.1, Expr.sum(y)))


    # Trace norm minimize to tighten PSD relaxation
    I5 = Matrix.eye(5)
    I7 = Matrix.eye(7)

    def trace_of_5(X):
        return Expr.sum(Expr.mulElm(I5, X))
    def trace_of_7(X):
        return Expr.sum(Expr.mulElm(I7, X))

    lambda_nn = 1e-3 # 아주 작게 시작해서 튜닝

    for e in range(n_edge):
        terms.append(Expr.mul(lambda_nn, trace_of_5(X_L[e])))
        terms.append(Expr.mul(lambda_nn, trace_of_5(X_R[e])))
        terms.append(Expr.mul(lambda_nn, trace_of_7(X_L2[e])))
        terms.append(Expr.mul(lambda_nn, trace_of_7(X_R2[e])))



    M.objective(ObjectiveSense.Minimize, sum_expr(terms))

    build_time = time.perf_counter() - t0

    V = dict(
        y=y, z_L=z_L, z_R=z_R, zp_L=zp_L, zp_R=zp_R,
        rho_L=rho_L, rho_R=rho_R, rhoz_L=rhoz_L, rhoz_R=rhoz_R,
        X_L=X_L, X_R=X_R, X_L2=X_L2, X_R2=X_R2,
        start_L=start_L, start_R=start_R, target_L=target_L, target_R=target_R,
        y_relax=y_relax, M_scalar=M_scalar,

        # ux=ux_list, uy=uy_list, s_abs=sabs_list,

    )
    IDX = dict(n_edge=n_edge, vertex_number=vertex_number)
    PAR = dict(centroid=centroid)
    return M, V, IDX, PAR, build_time