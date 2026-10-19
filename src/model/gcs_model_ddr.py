from __future__ import annotations
import time
import numpy as np
from typing import Any, Dict, List, Tuple
from mosek.fusion import Model, Domain, Expr, Matrix, ObjectiveSense, AccSolutionStatus
import time
import numpy as np
from collections import namedtuple
import mosek
from src.graph import centroid_and_scaled_vertices
from src.graph.geometry import rect_A_b_from_vertices as _rect_A_b
from src.graph.centroid import compute_edge_directions
# fusion helpers (네 프로젝트에 이미 있음)
from src.solver.fusion_ops import eq0, le0, cone2, sum_expr, sum_pick_1d


def _rect_A_b(vertices_2x4: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """MATLAB extractABfromVertices(2D 분기)와 동일: convhull의 edge 법선으로 Ax<=b 구성."""
    V = vertices_2x4.T  # (4,2)
    cx, cy = V.mean(axis=0)
    # 사각형은 이미 볼록/정렬이라 순서대로 법선 만들면 충분
    A = []
    b = []
    for i in range(4):
        p1 = V[i]
        p2 = V[(i + 1) % 4]
        e = p2 - p1
        n = np.array([-e[1], e[0]], float)
        n /= np.linalg.norm(n) + 1e-12
        # 내부를 향하도록 뒤집기
        if np.dot(n, np.array([cx, cy]) - p1) > 0:
            n = -n
        A.append(n)
        b.append(float(np.dot(n, p1)))
    return np.asarray(A, float), np.asarray(b, float)


def build_gcs_model_ddr(
    v: List[np.ndarray],                 # 각 vertex: (3x4) [x;y;z], z=0
    edge_1based: np.ndarray,             # (E,2) 1-based
    alpha: float,                        # 인터페이스 유지용(내부 사용X)
    mode_list_in: List[str],             # ["Do","SL","SR",...]
    start_L: np.ndarray, start_R: np.ndarray,      # (5,) [x,y,z,cos,sin]
    target_L: np.ndarray, target_R: np.ndarray,    # (5,)
    v_names=None,
) -> Tuple[Model, Dict[str, Any], Dict[str, Any], Dict[str, Any], float]:

    t0 = time.perf_counter()
    edge_1based = np.array(edge_1based, dtype=int)
    edge = edge_1based - 1
    E = int(edge.shape[0])
    Vn = int(edge.max()) + 1
    mode_list = [("D" if m == "Do" else m) for m in mode_list_in]

    # ----- region polytope: origin 기준(센터 제거)로 A,b 생성 -----
    centroids = np.stack([vi.mean(axis=1) for vi in v], axis=1)  # (3,V)
    const_struct = []
    for i, vi in enumerate(v):
        xy = vi[:2, :] - centroids[:2, [i]]
        A, b = _rect_A_b(xy)
        const_struct.append({"A": A, "b": b})

    # ===== Model & variables =====
    M = Model("GCS_DDR")

    y    = M.variable("y",     E,        Domain.inRange(0.0, 1.0))
    z_L  = M.variable("z_L",  [E, 5],    Domain.unbounded())  # [x,y,z,cos,sin]
    z_R  = M.variable("z_R",  [E, 5],    Domain.unbounded())
    zp_L = M.variable("zp_L", [E, 5],    Domain.unbounded())
    zp_R = M.variable("zp_R", [E, 5],    Domain.unbounded())
    rho_L = M.variable("rho_L", E, Domain.greaterThan(0.0))
    rho_R = M.variable("rho_R", E, Domain.greaterThan(0.0))

    # parameters (기존 set_parameters 호환)
    P_start_L  = M.parameter("start_L",  5)
    P_start_R  = M.parameter("start_R",  5)
    P_target_L = M.parameter("target_L", 5)
    P_target_R = M.parameter("target_R", 5)
    P_geometry = M.parameter("geometry", 2 * Vn)   # [x1,y1,x2,y2,...]
    # y 고정용 링크
    P_y_relax  = M.parameter("y_relax",  E)
    P_M_scalar = M.parameter("M_scalar")

    # 초기값(있어도 되고 없어도 됨)
    P_start_L.setValue(np.asarray(start_L, float).reshape(-1))
    P_start_R.setValue(np.asarray(start_R, float).reshape(-1))
    P_target_L.setValue(np.asarray(target_L, float).reshape(-1))
    P_target_R.setValue(np.asarray(target_R, float).reshape(-1))
    P_geometry.setValue(centroids[:2, :].T.reshape(-1))
    P_y_relax.setValue(np.zeros(E))
    P_M_scalar.setValue(1.0)

    outV = edge[:, 0].astype(int)
    inV  = edge[:, 1].astype(int)


    # M.setSolverParam("num_threads", 12)  # or however many cores you want to use

    # ========================= Constraints ==============================

    # (A) || y - y_relax ||_inf <= M_scalar  →  -M <= y - y_relax <= M
    ones_e = Matrix.dense(np.ones((E, 1)))
    Mvec   = Expr.mul(ones_e, P_M_scalar)
    le0(M, Expr.sub(Expr.sub(y,       P_y_relax), Mvec))
    le0(M, Expr.sub(Expr.sub(P_y_relax, y),       Mvec))

    # (B) source/target flow
    src_e  = np.where(outV == 0)[0]
    sink_e = np.where(inV  == Vn - 1)[0]
    for e in src_e:
        for k in range(5):
            eq0(M, Expr.sub(z_L.index(e, k), Expr.mul(y.index(e), P_start_L.index(k))))
            eq0(M, Expr.sub(z_R.index(e, k), Expr.mul(y.index(e), P_start_R.index(k))))
    for e in sink_e:
        for k in range(5):
            eq0(M, Expr.sub(zp_L.index(e, k), Expr.mul(y.index(e), P_target_L.index(k))))
            eq0(M, Expr.sub(zp_R.index(e, k), Expr.mul(y.index(e), P_target_R.index(k))))

    # (C) Source and Target vertex Flow conservation
    M.constraint(sum_pick_1d(y, src_e),  Domain.equalsTo(1.0))
    M.constraint(sum_pick_1d(y, sink_e), Domain.equalsTo(1.0))

    # (D) In = Out flow conservation (middle vertices)
    for vid in range(1, Vn - 1):
        in_idx  = np.where(inV  == vid)[0]
        out_idx = np.where(outV == vid)[0]

        # 1) sum of y
        M.constraint(Expr.sub(sum_pick_1d(y, in_idx), sum_pick_1d(y, out_idx)),
                     Domain.equalsTo(0.0))

        # 2) sum of zp (In edge) == sum of z (Out edge)
        for k in range(5):
            # sum(zp_L(in,k)) == sum(z_L(out,k))
            inL  = Expr.add([zp_L.index(e, k) for e in in_idx]) if in_idx.size else Expr.constTerm(1, 0.0)
            outL = Expr.add([ z_L.index(e, k) for e in out_idx]) if out_idx.size else Expr.constTerm(1, 0.0)
            M.constraint(Expr.sub(inL, outL), Domain.equalsTo(0.0))

            # sum(zp_R(in,k)) == sum(z_R(out,k))
            inR  = Expr.add([zp_R.index(e, k) for e in in_idx]) if in_idx.size else Expr.constTerm(1, 0.0)
            outR = Expr.add([ z_R.index(e, k) for e in out_idx]) if out_idx.size else Expr.constTerm(1, 0.0)
            M.constraint(Expr.sub(inR, outR), Domain.equalsTo(0.0))

    # (E) outflow of y (middle vertices)
    for vid in range(1, Vn - 1):
        out_idx = np.where(outV == vid)[0]
        M.constraint(sum_pick_1d(y, out_idx), Domain.lessThan(1.0))

    # (F) Inside Convex region (depends on from/to mode)
    def y_times_b_plus_Ageo(b_vec, A_mat, g_slice, y_scalar):
        B = Matrix.dense(b_vec.reshape(-1, 1))
        term_b    = Expr.mul(B, y_scalar)
        term_Ageo = Expr.mul(Matrix.dense(A_mat), Expr.mul(g_slice, y_scalar))
        return Expr.add(term_b, term_Ageo)  # (m,1)

    for e in range(E):
        u = outV[e]; v = inV[e]
        mf = mode_list[u]; mt = mode_list[v]
        A_from, b_from = const_struct[u]["A"], const_struct[u]["b"]
        A_to,   b_to   = const_struct[v]["A"], const_struct[v]["b"]
        g_from = P_geometry.slice(2*u, 2*u+2)
        g_to   = P_geometry.slice(2*v, 2*v+2)
        yi = y.index(e)

        if mf == "D" and mt == "SL":
            lhs = Expr.sub(Expr.mul(Matrix.dense(A_to),
                                    Expr.vstack(zp_L.index(e, 0), zp_L.index(e, 1))),
                           y_times_b_plus_Ageo(b_to, A_to, g_to, yi))
            le0(M, lhs)
        if mf == "D" and mt == "SR":
            lhs = Expr.sub(Expr.mul(Matrix.dense(A_to),
                                    Expr.vstack(zp_R.index(e, 0), zp_R.index(e, 1))),
                           y_times_b_plus_Ageo(b_to, A_to, g_to, yi))
            le0(M, lhs)
        if mf == "SL" and mt == "SR":
            lhs1 = Expr.sub(Expr.mul(Matrix.dense(A_from),
                                     Expr.vstack(zp_L.index(e, 0), zp_L.index(e, 1))),
                            y_times_b_plus_Ageo(b_from, A_from, g_from, yi))
            lhs2 = Expr.sub(Expr.mul(Matrix.dense(A_to),
                                     Expr.vstack(zp_R.index(e, 0), zp_R.index(e, 1))),
                            y_times_b_plus_Ageo(b_to, A_to, g_to, yi))
            le0(M, lhs1); le0(M, lhs2)
        if mf == "SR" and mt == "SL":
            lhs1 = Expr.sub(Expr.mul(Matrix.dense(A_from),
                                     Expr.vstack(zp_R.index(e, 0), zp_R.index(e, 1))),
                            y_times_b_plus_Ageo(b_from, A_from, g_from, yi))
            lhs2 = Expr.sub(Expr.mul(Matrix.dense(A_to),
                                     Expr.vstack(zp_L.index(e, 0), zp_L.index(e, 1))),
                            y_times_b_plus_Ageo(b_to, A_to, g_to, yi))
            le0(M, lhs1); le0(M, lhs2)
        if mf == "SL" and mt == "D":
            lhs = Expr.sub(Expr.mul(Matrix.dense(A_from),
                                    Expr.vstack(z_L.index(e, 0), z_L.index(e, 1))),
                           y_times_b_plus_Ageo(b_from, A_from, g_from, yi))
            le0(M, lhs)
        if mf == "SR" and mt == "D":
            lhs = Expr.sub(Expr.mul(Matrix.dense(A_from),
                                    Expr.vstack(z_R.index(e, 0), z_R.index(e, 1))),
                           y_times_b_plus_Ageo(b_from, A_from, g_from, yi))
            le0(M, lhs)

    # (G) cos/sin gating:  |var[e,k]| ≤ y[e]  (k=3,4)
    for e in range(E):
        ye = y.index(e)
        for var in (z_L, z_R, zp_L, zp_R):
            for k in (3, 4):  # 0-based: cos,sin
                le0(M, Expr.sub(var.index(e, k), ye))  # var - y ≤ 0
                le0(M, Expr.sub(Expr.neg(var.index(e, k)), ye))  # -var - y ≤ 0

    # # (H) relative foot bound: ||zp_L - z_R||_2 ≤ r*y,  ||zp_R - z_L||_2 ≤ r*y
    # step_radius = 0.71
    # for e in range(E):
    #     ye = y.index(e)
    #     # left-next vs right-current (x,y)
    #     cone2(M, Expr.mul(step_radius, ye),
    #           Expr.sub(zp_L.index(e, 0), z_R.index(e, 0)),
    #           Expr.sub(zp_L.index(e, 1), z_R.index(e, 1)))
    #     # right-next vs left-current (x,y)
    #     cone2(M, Expr.mul(step_radius, ye),
    #           Expr.sub(zp_R.index(e, 0), z_L.index(e, 0)),
    #           Expr.sub(zp_R.index(e, 1), z_L.index(e, 1)))

    # (I) minimum safety for not crossing over
    step_radius = 0.30
    for e in range(E):
        u = outV[e]; v = inV[e]
        mf = mode_list[u]; mt = mode_list[v]
        ye = y.index(e)
        if mf == "D" and mt in ("SL","SR"):
            # (I) minimum safety for not crossing over
            xL = z_L.slice([0, 0], [E, 1]).reshape(E)
            xR = z_R.slice([0, 0], [E, 1]).reshape(E)
            le0(M, Expr.sub(Expr.add(xL, Expr.mul(0.14, y)), xR))

            # (H) relative foot bound: ||zp_L - z_R||_2 ≤ r*y,  ||zp_R - z_L||_2 ≤ r*y
            # left-next vs right-current (x,y)
            cone2(M, Expr.mul(step_radius, ye),
                  Expr.sub(zp_L.index(e, 0), z_R.index(e, 0)),
                  Expr.sub(zp_L.index(e, 1), z_R.index(e, 1)))
            # right-next vs left-current (x,y)
            cone2(M, Expr.mul(step_radius, ye),
                  Expr.sub(zp_R.index(e, 0), z_L.index(e, 0)),
                  Expr.sub(zp_R.index(e, 1), z_L.index(e, 1)))

    # ===== Objective (MATLAB과 동일 의미) =====
    terms = []

    # #  (A) rho related objective term
    for e in range(E):
        mt = mode_list[inV[e]]
        if mt == "D":
            cone2(M, rho_L.index(e),
                  Expr.sub(zp_L.index(e, 0), z_L.index(e, 0)),
                  Expr.sub(zp_L.index(e, 1), z_L.index(e, 1)))
            cone2(M, rho_R.index(e),
                  Expr.sub(zp_R.index(e, 0), z_R.index(e, 0)),
                  Expr.sub(zp_R.index(e, 1), z_R.index(e, 1)))
        elif mt == "SL":
            for k in (0, 1, 3, 4):
                eq0(M, Expr.sub(zp_R.index(e, k), z_R.index(e, k)))
            eq0(M, rho_R.index(e))
            cone2(M, rho_L.index(e),
                  Expr.sub(zp_L.index(e, 0), z_L.index(e, 0)),
                  Expr.sub(zp_L.index(e, 1), z_L.index(e, 1)))
        elif mt == "SR":
            for k in (0, 1, 3, 4):
                eq0(M, Expr.sub(zp_L.index(e, k), z_L.index(e, k)))
            eq0(M, rho_L.index(e))
            cone2(M, rho_R.index(e),
                  Expr.sub(zp_R.index(e, 0), z_R.index(e, 0)),
                  Expr.sub(zp_R.index(e, 1), z_R.index(e, 1)))

    # #  (A-1) rho related objective term
    wL = np.zeros(E); wR = np.zeros(E)
    for e in range(E):
        mf = mode_list[outV[e]]; mt = mode_list[inV[e]]
        if (mf == "SR" and mt == "D") or (mf == "SL" and mt == "D"):
            wL[e] = 1.0; wR[e] = 1.0
        elif (mf == "SR" and mt == "SL") or (mf == "D" and mt == "SL"):
            wL[e] = 1.0
        elif (mf == "SL" and mt == "SR") or (mf == "D" and mt == "SR"):
            wR[e] = 1.0
    terms += [Expr.sum(Expr.mulElm(wL, rho_L)), Expr.sum(Expr.mulElm(wR, rho_R))]


    ## (B) 회전 변화 SOC: ||(c,s)-(c',s')||_2
    t_rotL = M.variable("t_rotchg_L", E, Domain.greaterThan(0.0))
    t_rotR = M.variable("t_rotchg_R", E, Domain.greaterThan(0.0))
    for e in range(E):
        cone2(M, t_rotL.index(e),
              Expr.sub(z_L.index(e, 3), zp_L.index(e, 3)),
              Expr.sub(z_L.index(e, 4), zp_L.index(e, 4)))
        cone2(M, t_rotR.index(e),
              Expr.sub(z_R.index(e, 3), zp_R.index(e, 3)),
              Expr.sub(z_R.index(e, 4), zp_R.index(e, 4)))
    # ★ 아주 작게 가중치
    lam_rotchg = 0.01  # 1e-5 ~ 1e-3 사이에서 튜닝 추천
    terms.append(Expr.mul(lam_rotchg, Expr.add(Expr.sum(t_rotL), Expr.sum(t_rotR))))

    # ── 이름/모드 분리 준비 ──
    stone_start = [v_names[inV[e]] for e in range(E)]  # ← 스톤 라벨만
    stone_end = [v_names[outV[e]] for e in range(E)]

    # 모드: Terrain이 "Do"를 쓰면 여기서 통일
    def norm_mode(x: str) -> str:
        if x in ("Do", "DO", "Double", "DoubleSupport"): return "D"
        return x

    mode_in = [norm_mode(mode_list[inV[e]]) for e in range(E)]
    mode_out = [norm_mode(mode_list[outV[e]]) for e in range(E)]

    # ── (stone_start, stone_end) → 각도(deg) 룰 (명시된 것만 적용, 나머지는 pass) ──
    RULES_DEG = {
        # from DL
        ("DL", "UL"): 90, ("DL", "U"): 85, ("DL", "UR"): 80,
        ("DL", "L"): 90, ("DL", "C"): 80, ("DL", "R"): 75,
        # from DR
        ("DR", "UL"): 100, ("DR", "U"): 95, ("DR", "UR"): 90,
        ("DR", "L"): 105, ("DR", "C"): 100, ("DR", "R"): 90,
        # from C
        ("C", "UL"): 105, ("C", "U"): 90, ("C", "UR"): 75,

    }

    def deg2rad(d):
        import numpy as np; return np.deg2rad(d)

    # 전역 강도 (필요시 조정)
    LAM_BASE = 0.35

    # ── 참조(c*,s*)와 룰 매치 마스크 ──
    ref_c = np.zeros(E);
    ref_s = np.zeros(E)
    rule_mask = np.zeros(E)  # (stone_start,stone_end)가 RULES에 있으면 1

    for e in range(E):
        key = (stone_start[e], stone_end[e])
        if key in RULES_DEG:
            th = deg2rad(RULES_DEG[key])
            ref_c[e], ref_s[e] = np.cos(th), np.sin(th)
            rule_mask[e] = 1.0  # ← 지정한 출발 스톤 조합에만 켜짐

    # ── 모드 마스크: 어느 발에 적용할지 (너의 rho 규칙과 동일 논리) ──
    mL = np.zeros(E);
    mR = np.zeros(E)
    for e in range(E):
        mt = mode_in[e]  # SL, SR, D
        mf = mode_out[e]  # SL, SR, D
        if (mf == "SR" and mt == "SL") or (mf == "D" and mt == "SL"):
            mR[e] = 1.0
        elif (mf == "SL" and mt == "SR") or (mf == "D" and mt == "SR"):
            mL[e] = 1.0
        else:
            mR[e] = 0.0
            mL[e] = 0.0

    # ── 최종 가중치 = 룰매치 × 모드마스크 × 전역강도 ──
    wL = LAM_BASE * rule_mask * mL
    wR = LAM_BASE * rule_mask * mR

    # ── SOC slack ──
    t_yaw_L = M.variable("t_yaw_pref_L", E, Domain.greaterThan(0.0))
    t_yaw_R = M.variable("t_yaw_pref_R", E, Domain.greaterThan(0.0))

    # 제약: w=0인 엣지는 아예 끄기(선택이지만 디버깅에 깔끔)
    for e in range(E):
        if wL[e] != 0.0:
            cone2(M, t_yaw_L.index(e),
                  Expr.sub(z_L.index(e, 3), ref_c[e]),
                  Expr.sub(z_L.index(e, 4), ref_s[e]))
        else:
            eq0(M, t_yaw_L.index(e))  # 비용 영향 제거

        if wR[e] != 0.0:
            cone2(M, t_yaw_R.index(e),
                  Expr.sub(z_R.index(e, 3), ref_c[e]),
                  Expr.sub(z_R.index(e, 4), ref_s[e]))
        else:
            eq0(M, t_yaw_R.index(e))

    # ── 코스트 ──
    terms.append(Expr.add(
        Expr.sum(Expr.mulElm(wL, t_yaw_L)),
        Expr.sum(Expr.mulElm(wR, t_yaw_R))
    ))

    # step number
    terms.append(Expr.mul(0.1, Expr.sum(y)))

    M.objective(ObjectiveSense.Minimize, sum_expr(terms))
    M.acceptedSolutionStatus(AccSolutionStatus.Feasible)

    build_time = time.perf_counter() - t0

    V = dict(
        y=y, z_L=z_L, z_R=z_R, zp_L=zp_L, zp_R=zp_R,
        rho_L=rho_L, rho_R=rho_R,
        start_L=P_start_L, start_R=P_start_R, target_L=P_target_L, target_R=P_target_R,
        geometry=P_geometry, y_relax=P_y_relax, M_scalar=P_M_scalar,
    )
    IDX = dict(n_edge=E, vertex_number=Vn)
    PAR = dict(centroid=centroids)
    return M, V, IDX, PAR, build_time




def build_gcs_model_ddr_TORO(
    v: List[np.ndarray],                 # 각 vertex: (3x4) [x;y;z], z=0
    edge_1based: np.ndarray,             # (E,2) 1-based
    alpha: float,                        # 인터페이스 유지용(내부 사용X)
    mode_list_in: List[str],             # ["Do","SL","SR",...]
    start_L: np.ndarray, start_R: np.ndarray,      # (5,) [x,y,z,cos,sin]
    target_L: np.ndarray, target_R: np.ndarray,    # (5,)
    v_names=None,
) -> Tuple[Model, Dict[str, Any], Dict[str, Any], Dict[str, Any], float]:

    t0 = time.perf_counter()
    edge_1based = np.array(edge_1based, dtype=int)
    edge = edge_1based - 1
    E = int(edge.shape[0])
    Vn = int(edge.max()) + 1
    mode_list = [("D" if m == "Do" else m) for m in mode_list_in]

    # ----- region polytope: origin 기준(센터 제거)로 A,b 생성 -----
    centroids = np.stack([vi.mean(axis=1) for vi in v], axis=1)  # (3,V)
    const_struct = []
    for i, vi in enumerate(v):
        xy = vi[:2, :] - centroids[:2, [i]]
        A, b = _rect_A_b(xy)
        const_struct.append({"A": A, "b": b})

    # ===== Model & variables =====
    M = Model("GCS_DDR")
    y    = M.variable("y",     E,        Domain.inRange(0.0, 1.0))
    z_L  = M.variable("z_L",  [E, 5],    Domain.unbounded())  # [x,y,z,cos,sin]
    z_R  = M.variable("z_R",  [E, 5],    Domain.unbounded())
    zp_L = M.variable("zp_L", [E, 5],    Domain.unbounded())
    zp_R = M.variable("zp_R", [E, 5],    Domain.unbounded())
    rho_L = M.variable("rho_L", E, Domain.greaterThan(0.0))
    rho_R = M.variable("rho_R", E, Domain.greaterThan(0.0))

    # parameters (기존 set_parameters 호환)
    P_start_L  = M.parameter("start_L",  5)
    P_start_R  = M.parameter("start_R",  5)
    P_target_L = M.parameter("target_L", 5)
    P_target_R = M.parameter("target_R", 5)
    P_geometry = M.parameter("geometry", 2 * Vn)   # [x1,y1,x2,y2,...]
    # y 고정용 링크
    P_y_relax  = M.parameter("y_relax",  E)
    P_M_scalar = M.parameter("M_scalar")

    # 초기값(있어도 되고 없어도 됨)
    P_start_L.setValue(np.asarray(start_L, float).reshape(-1))
    P_start_R.setValue(np.asarray(start_R, float).reshape(-1))
    P_target_L.setValue(np.asarray(target_L, float).reshape(-1))
    P_target_R.setValue(np.asarray(target_R, float).reshape(-1))
    P_geometry.setValue(centroids[:2, :].T.reshape(-1))
    P_y_relax.setValue(np.zeros(E))
    P_M_scalar.setValue(1.0)

    outV = edge[:, 0].astype(int)
    inV  = edge[:, 1].astype(int)

    # ========================= Constraints ==============================

    # (A) || y - y_relax ||_inf <= M_scalar  →  -M <= y - y_relax <= M
    ones_e = Matrix.dense(np.ones((E, 1)))
    Mvec   = Expr.mul(ones_e, P_M_scalar)
    le0(M, Expr.sub(Expr.sub(y,       P_y_relax), Mvec))
    le0(M, Expr.sub(Expr.sub(P_y_relax, y),       Mvec))

    # (B) source/target flow
    src_e  = np.where(outV == 0)[0]
    sink_e = np.where(inV  == Vn - 1)[0]
    for e in src_e:
        for k in range(5):
            eq0(M, Expr.sub(z_L.index(e, k), Expr.mul(y.index(e), P_start_L.index(k))))
            eq0(M, Expr.sub(z_R.index(e, k), Expr.mul(y.index(e), P_start_R.index(k))))
    for e in sink_e:
        for k in range(5):
            eq0(M, Expr.sub(zp_L.index(e, k), Expr.mul(y.index(e), P_target_L.index(k))))
            eq0(M, Expr.sub(zp_R.index(e, k), Expr.mul(y.index(e), P_target_R.index(k))))

    # (C) Source and Target vertex Flow conservation
    M.constraint(sum_pick_1d(y, src_e),  Domain.equalsTo(1.0))
    M.constraint(sum_pick_1d(y, sink_e), Domain.equalsTo(1.0))

    # (D) In = Out flow conservation (middle vertices)
    for vid in range(1, Vn - 1):
        in_idx  = np.where(inV  == vid)[0]
        out_idx = np.where(outV == vid)[0]

        # 1) sum of y
        M.constraint(Expr.sub(sum_pick_1d(y, in_idx), sum_pick_1d(y, out_idx)),
                     Domain.equalsTo(0.0))

        # 2) sum of zp (In edge) == sum of z (Out edge)
        for k in range(5):
            # sum(zp_L(in,k)) == sum(z_L(out,k))
            inL  = Expr.add([zp_L.index(e, k) for e in in_idx]) if in_idx.size else Expr.constTerm(1, 0.0)
            outL = Expr.add([ z_L.index(e, k) for e in out_idx]) if out_idx.size else Expr.constTerm(1, 0.0)
            M.constraint(Expr.sub(inL, outL), Domain.equalsTo(0.0))

            # sum(zp_R(in,k)) == sum(z_R(out,k))
            inR  = Expr.add([zp_R.index(e, k) for e in in_idx]) if in_idx.size else Expr.constTerm(1, 0.0)
            outR = Expr.add([ z_R.index(e, k) for e in out_idx]) if out_idx.size else Expr.constTerm(1, 0.0)
            M.constraint(Expr.sub(inR, outR), Domain.equalsTo(0.0))

    # (E) outflow of y (middle vertices)
    for vid in range(1, Vn - 1):
        out_idx = np.where(outV == vid)[0]
        M.constraint(sum_pick_1d(y, out_idx), Domain.lessThan(1.0))

    # (F) Inside Convex region (depends on from/to mode)
    def y_times_b_plus_Ageo(b_vec, A_mat, g_slice, y_scalar):
        B = Matrix.dense(b_vec.reshape(-1, 1))
        term_b    = Expr.mul(B, y_scalar)
        term_Ageo = Expr.mul(Matrix.dense(A_mat), Expr.mul(g_slice, y_scalar))
        return Expr.add(term_b, term_Ageo)  # (m,1)

    z_centers = centroids[2, :]  # (V,) 각 vertex의 z 중심 (평면 높이)
    for e in range(E):
        u = outV[e]; v = inV[e]
        mf = mode_list[u]; mt = mode_list[v]
        A_from, b_from = const_struct[u]["A"], const_struct[u]["b"]
        A_to,   b_to   = const_struct[v]["A"], const_struct[v]["b"]
        z_from = float(z_centers[u])
        z_to   = float(z_centers[v])
        g_from = P_geometry.slice(2*u, 2*u+2)
        g_to   = P_geometry.slice(2*v, 2*v+2)
        yi = y.index(e)

        if mf == "D" and mt == "SL":
            lhs = Expr.sub(Expr.mul(Matrix.dense(A_to),
                                    Expr.vstack(zp_L.index(e, 0), zp_L.index(e, 1))),
                           y_times_b_plus_Ageo(b_to, A_to, g_to, yi))
            le0(M, lhs)
            # ★ z 제약: zp_L(e, z) - y(e) * z_to == 0
            eq0(M, Expr.sub(zp_L.index(e, 2), Expr.mul(yi, z_to)))
            # ★ z 제약: zp_R(e, z) - y(e) * z_from == 0
            eq0(M, Expr.sub(zp_R.index(e, 2), Expr.mul(yi, z_from)))

        if mf == "D" and mt == "SR":
            lhs = Expr.sub(Expr.mul(Matrix.dense(A_to),
                                    Expr.vstack(zp_R.index(e, 0), zp_R.index(e, 1))),
                           y_times_b_plus_Ageo(b_to, A_to, g_to, yi))
            le0(M, lhs)
            # ★ z 제약: zp_R(e, z) - y(e) * z_to == 0
            eq0(M, Expr.sub(zp_R.index(e, 2), Expr.mul(yi, z_to)))
            # ★ z 제약: zp_L(e, z) - y(e) * z_from == 0
            eq0(M, Expr.sub(zp_L.index(e, 2), Expr.mul(yi, z_from)))

        if mf == "SL" and mt == "SR":
            lhs1 = Expr.sub(Expr.mul(Matrix.dense(A_from),
                                     Expr.vstack(zp_L.index(e, 0), zp_L.index(e, 1))),
                            y_times_b_plus_Ageo(b_from, A_from, g_from, yi))
            lhs2 = Expr.sub(Expr.mul(Matrix.dense(A_to),
                                     Expr.vstack(zp_R.index(e, 0), zp_R.index(e, 1))),
                            y_times_b_plus_Ageo(b_to, A_to, g_to, yi))
            le0(M, lhs1); le0(M, lhs2)
            eq0(M, Expr.sub(zp_L.index(e, 2), Expr.mul(yi, z_from)))
            eq0(M, Expr.sub(zp_R.index(e, 2), Expr.mul(yi, z_to)))


        if mf == "SR" and mt == "SL":
            lhs1 = Expr.sub(Expr.mul(Matrix.dense(A_from),
                                     Expr.vstack(zp_R.index(e, 0), zp_R.index(e, 1))),
                            y_times_b_plus_Ageo(b_from, A_from, g_from, yi))
            lhs2 = Expr.sub(Expr.mul(Matrix.dense(A_to),
                                     Expr.vstack(zp_L.index(e, 0), zp_L.index(e, 1))),
                            y_times_b_plus_Ageo(b_to, A_to, g_to, yi))
            le0(M, lhs1); le0(M, lhs2)
            eq0(M, Expr.sub(zp_R.index(e, 2), Expr.mul(yi, z_from)))
            eq0(M, Expr.sub(zp_L.index(e, 2), Expr.mul(yi, z_to)))


        if mf == "SL" and mt == "D":
            lhs = Expr.sub(Expr.mul(Matrix.dense(A_from),
                                    Expr.vstack(z_L.index(e, 0), z_L.index(e, 1))),
                           y_times_b_plus_Ageo(b_from, A_from, g_from, yi))
            le0(M, lhs)
            eq0(M, Expr.sub(z_L.index(e, 2), Expr.mul(yi, z_from)))


        if mf == "SR" and mt == "D":
            lhs = Expr.sub(Expr.mul(Matrix.dense(A_from),
                                    Expr.vstack(z_R.index(e, 0), z_R.index(e, 1))),
                           y_times_b_plus_Ageo(b_from, A_from, g_from, yi))
            le0(M, lhs)
            eq0(M, Expr.sub(z_R.index(e, 2), Expr.mul(yi, z_from)))

    # (G) gating:  |var[e,k]| ≤ y[e]
    # 위치(x,y,z) 용 큰 bound (활성화되면 거의 무제한 느낌)
    M_pos = 10.0  # or step_span*5, etc. 네 스케일에 맞게 설정
    # cos,sin 용 bound
    M_ang = 1.0
    for e in range(E):
        ye = y.index(e)
        for var in (z_L, z_R, zp_L, zp_R):
            for k in range(5):
                # k=0,1,2 → 위치; k=3,4 → cos,sin
                M_k = M_pos if k in (0, 1, 2) else M_ang

                # |var[e,k]| ≤ M_k * y[e]
                # 1)  var - M_k * y ≤ 0
                le0(M, Expr.sub(
                    var.index(e, k),
                    Expr.mul(ye, M_k)
                ))
                # 2) -var - M_k * y ≤ 0  ↔  var ≥ -M_k * y
                le0(M, Expr.sub(
                    Expr.neg(var.index(e, k)),
                    Expr.mul(ye, M_k)
                ))

    # # (H) relative foot bound: ||zp_L - z_R||_2 ≤ r*y,  ||zp_R - z_L||_2 ≤ r*y
    # step_radius = 0.71
    # for e in range(E):
    #     ye = y.index(e)
    #     # left-next vs right-current (x,y)
    #     cone2(M, Expr.mul(step_radius, ye),
    #           Expr.sub(zp_L.index(e, 0), z_R.index(e, 0)),
    #           Expr.sub(zp_L.index(e, 1), z_R.index(e, 1)))
    #     # right-next vs left-current (x,y)
    #     cone2(M, Expr.mul(step_radius, ye),
    #           Expr.sub(zp_R.index(e, 0), z_L.index(e, 0)),
    #           Expr.sub(zp_R.index(e, 1), z_L.index(e, 1)))


    # # (I) Inward constraint (TORO Constraint)
    # for e in range(E):
    #     u = outV[e]; v = inV[e]
    #     mf = mode_list[u]; mt = mode_list[v]
    #     ye = y.index(e)
    #     if mf == "D" and mt in ("SL","SR"):
    #         SIN_COL = 4  # 0-based index: [x,y,z,cos,sin]
    #         zpR_sin = zp_R.slice([0, SIN_COL], [E, SIN_COL + 1]).reshape(E)
    #         zL_sin = z_L.slice([0, SIN_COL], [E, SIN_COL + 1]).reshape(E)
    #         zpL_sin = zp_L.slice([0, SIN_COL], [E, SIN_COL + 1]).reshape(E)
    #         zR_sin = z_R.slice([0, SIN_COL], [E, SIN_COL + 1]).reshape(E)
    #
    #         M.constraint(Expr.sub(zpR_sin, zL_sin), Domain.lessThan(0.0))  # zp_R <= z_L
    #         M.constraint(Expr.sub(zpL_sin, zR_sin), Domain.greaterThan(0.0))  # zp_L >= z_R


    # (I) minimum safety for not crossing over for all edges
    step_radius = 0.33

    u = outV[e]; v = inV[e]
    mf = mode_list[u]; mt = mode_list[v]
    ye = y.index(e)
    # (I) minimum safety for not crossing over
    xL = z_L.slice([0, 0], [E, 1]).reshape(E)
    xR = z_R.slice([0, 0], [E, 1]).reshape(E)
    le0(M, Expr.sub(Expr.add(xL, Expr.mul(0.2, y)), xR))

    # (H) relative foot bound: ||zp_L - z_R||_2 ≤ r*y,  ||zp_R - z_L||_2 ≤ r*y
    # left-next vs right-current (x,y)
    cone2(M, Expr.mul(step_radius, ye),
          Expr.sub(zp_L.index(e, 0), z_R.index(e, 0)),
          Expr.sub(zp_L.index(e, 1), z_R.index(e, 1)))
    # right-next vs left-current (x,y)
    cone2(M, Expr.mul(step_radius, ye),
          Expr.sub(zp_R.index(e, 0), z_L.index(e, 0)),
          Expr.sub(zp_R.index(e, 1), z_L.index(e, 1)))




    # ===== Objective (MATLAB과 동일 의미) =====
    terms = []

    # #  (A) rho related objective term
    for e in range(E):
        mt = mode_list[inV[e]]
        if mt == "D":
            cone2(M, rho_L.index(e),
                  Expr.sub(zp_L.index(e, 0), z_L.index(e, 0)),
                  Expr.sub(zp_L.index(e, 1), z_L.index(e, 1)))
            cone2(M, rho_R.index(e),
                  Expr.sub(zp_R.index(e, 0), z_R.index(e, 0)),
                  Expr.sub(zp_R.index(e, 1), z_R.index(e, 1)))
        elif mt == "SL":
            for k in (0, 1, 3, 4):
                eq0(M, Expr.sub(zp_R.index(e, k), z_R.index(e, k)))
            eq0(M, rho_R.index(e))
            cone2(M, rho_L.index(e),
                  Expr.sub(zp_L.index(e, 0), z_L.index(e, 0)),
                  Expr.sub(zp_L.index(e, 1), z_L.index(e, 1)))
        elif mt == "SR":
            for k in (0, 1, 3, 4):
                eq0(M, Expr.sub(zp_L.index(e, k), z_L.index(e, k)))
            eq0(M, rho_L.index(e))
            cone2(M, rho_R.index(e),
                  Expr.sub(zp_R.index(e, 0), z_R.index(e, 0)),
                  Expr.sub(zp_R.index(e, 1), z_R.index(e, 1)))

    # #  (A-1) rho related objective term
    wL = np.zeros(E); wR = np.zeros(E)
    for e in range(E):
        mf = mode_list[outV[e]]; mt = mode_list[inV[e]]
        if (mf == "SR" and mt == "D") or (mf == "SL" and mt == "D"):
            wL[e] = 1.0; wR[e] = 1.0
        elif (mf == "SR" and mt == "SL") or (mf == "D" and mt == "SL"):
            wL[e] = 1.0
        elif (mf == "SL" and mt == "SR") or (mf == "D" and mt == "SR"):
            wR[e] = 1.0
    terms += [Expr.sum(Expr.mulElm(wL, rho_L)), Expr.sum(Expr.mulElm(wR, rho_R))]

    # ========================= Objective Rotation term =========================
    ## (B) 회전 변화 SOC: ||(c,s)-(c',s')||_2
    t_rotL = M.variable("t_rotchg_L", E, Domain.greaterThan(0.0))
    t_rotR = M.variable("t_rotchg_R", E, Domain.greaterThan(0.0))
    for e in range(E):
        cone2(M, t_rotL.index(e),
              Expr.sub(z_L.index(e, 3), zp_L.index(e, 3)),
              Expr.sub(z_L.index(e, 4), zp_L.index(e, 4)))
        cone2(M, t_rotR.index(e),
              Expr.sub(z_R.index(e, 3), zp_R.index(e, 3)),
              Expr.sub(z_R.index(e, 4), zp_R.index(e, 4)))
    # ★ 아주 작게 가중치
    lam_rotchg = 0.1  # 1e-5 ~ 1e-3 사이에서 튜닝 추천
    terms.append(Expr.mul(lam_rotchg, Expr.add(Expr.sum(t_rotL), Expr.sum(t_rotR))))

    ## (C) Pump Terrain dependent term
    # ── 이름/모드 분리 준비 ──
    stone_start = [v_names[inV[e]] for e in range(E)]  # ← 스톤 라벨만
    stone_end = [v_names[outV[e]] for e in range(E)]

    # 모드: Terrain이 "Do"를 쓰면 여기서 통일
    def norm_mode(x: str) -> str:
        if x in ("Do", "DO", "Double", "DoubleSupport"): return "D"
        return x

    mode_in = [norm_mode(mode_list[inV[e]]) for e in range(E)]
    mode_out = [norm_mode(mode_list[outV[e]]) for e in range(E)]

    # ── (stone_start, stone_end) → 각도(deg) 룰 (명시된 것만 적용, 나머지는 pass) ──
    RULES_DEG = {
        # from DL
        ("DL", "UL"): 90, ("DL", "U"): 85, ("DL", "UR"): 80,
        ("DL", "L"): 90, ("DL", "C"): 80, ("DL", "R"): 75,
        # from DR
        ("DR", "UL"): 100, ("DR", "U"): 95, ("DR", "UR"): 90,
        ("DR", "L"): 105, ("DR", "C"): 100, ("DR", "R"): 90,
        # from C
        ("C", "UL"): 105, ("C", "U"): 90, ("C", "UR"): 75,
    }

    def deg2rad(d):
        import numpy as np; return np.deg2rad(d)

    # 전역 강도 (필요시 조정)
    LAM_BASE = 0.35

    # ── 참조(c*,s*)와 룰 매치 마스크 ──
    ref_c = np.zeros(E);
    ref_s = np.zeros(E)
    rule_mask = np.zeros(E)  # (stone_start,stone_end)가 RULES에 있으면 1

    for e in range(E):
        key = (stone_start[e], stone_end[e])
        if key in RULES_DEG:
            th = deg2rad(RULES_DEG[key])
            ref_c[e], ref_s[e] = np.cos(th), np.sin(th)
            rule_mask[e] = 1.0  # ← 지정한 출발 스톤 조합에만 켜짐

    # ── 모드 마스크: 어느 발에 적용할지 (너의 rho 규칙과 동일 논리) ──
    mL = np.zeros(E);
    mR = np.zeros(E)
    for e in range(E):
        mt = mode_in[e]  # SL, SR, D
        mf = mode_out[e]  # SL, SR, D
        if (mf == "SR" and mt == "SL") or (mf == "D" and mt == "SL"):
            mR[e] = 1.0
        elif (mf == "SL" and mt == "SR") or (mf == "D" and mt == "SR"):
            mL[e] = 1.0
        else:
            mR[e] = 0.0
            mL[e] = 0.0


    # ── 최종 가중치 = 룰매치 × 모드마스크 × 전역강도 ──
    wL = LAM_BASE * rule_mask * mL
    wR = LAM_BASE * rule_mask * mR

    # ── SOC slack ──
    t_yaw_L = M.variable("t_yaw_pref_L", E, Domain.greaterThan(0.0))
    t_yaw_R = M.variable("t_yaw_pref_R", E, Domain.greaterThan(0.0))

    # 제약: w=0인 엣지는 아예 끄기(선택이지만 디버깅에 깔끔)
    for e in range(E):
        if wL[e] != 0.0:
            cone2(M, t_yaw_L.index(e),
                  Expr.sub(z_L.index(e, 3), ref_c[e]),
                  Expr.sub(z_L.index(e, 4), ref_s[e]))
        else:
            eq0(M, t_yaw_L.index(e))  # 비용 영향 제거

        if wR[e] != 0.0:
            cone2(M, t_yaw_R.index(e),
                  Expr.sub(z_R.index(e, 3), ref_c[e]),
                  Expr.sub(z_R.index(e, 4), ref_s[e]))
        else:
            eq0(M, t_yaw_R.index(e))

    # ── 코스트 ──
    terms.append(Expr.add(
        Expr.sum(Expr.mulElm(wL, t_yaw_L)),
        Expr.sum(Expr.mulElm(wR, t_yaw_R))
    ))

    # step number
    terms.append(Expr.mul(0.1, Expr.sum(y)))

    M.objective(ObjectiveSense.Minimize, sum_expr(terms))
    M.acceptedSolutionStatus(AccSolutionStatus.Feasible)

    build_time = time.perf_counter() - t0

    V = dict(
        y=y, z_L=z_L, z_R=z_R, zp_L=zp_L, zp_R=zp_R,
        rho_L=rho_L, rho_R=rho_R,
        start_L=P_start_L, start_R=P_start_R, target_L=P_target_L, target_R=P_target_R,
        geometry=P_geometry, y_relax=P_y_relax, M_scalar=P_M_scalar,
    )
    IDX = dict(n_edge=E, vertex_number=Vn)
    PAR = dict(centroid=centroids)
    return M, V, IDX, PAR, build_time





OptModel = namedtuple("OptModel", ["env", "task", "idx", "const", "params"])

def build_gcs_model_ddr_TORO_optapi(
    v: list,                                 # 각 vertex: (3x4) [x;y;z], z=0
    edge_1based: np.ndarray,                 # (E,2) 1-based
    alpha: float,                            # 인터페이스 유지용(내부 사용X)
    mode_list_in: list,                      # ["Do","SL","SR",...]
    start_L: np.ndarray, start_R: np.ndarray,      # (5,) [x,y,z,cos,sin]
    target_L: np.ndarray, target_R: np.ndarray,    # (5,)
    v_names=None,
):
    """
    Fusion 기반 build_gcs_model_ddr_TORO와 입력/출력/초기 설정을 1:1 매칭한 Optimizer API 버전.
    - x_ref, y_ref, step_* 등 Fusion에 없는 개념/인자는 사용/노출하지 않음.
    - Fusion의 parameter 들은 Optimizer에서는 PAR dict 로 매핑해 동일 키로 보관.
    - 변수/도메인/초기 파라미터 값/리턴 구조를 Fusion과 동일하게 유지.
    """

    t0 = time.perf_counter()

    # ===== 입력 전처리 (Fusion과 동일) =====
    edge_1based = np.array(edge_1based, dtype=int)
    edge = edge_1based - 1
    E = int(edge.shape[0])
    Vn = int(edge.max()) + 1
    mode_list = [("D" if m == "Do" else m) for m in mode_list_in]

    # ----- region polytope: origin 기준(센터 제거)로 A,b 생성 (Fusion과 동일) -----
    v = [np.asarray(vi, dtype=float) for vi in v]
    centroids = np.stack([vi.mean(axis=1) for vi in v], axis=1)  # (3,V)
    const_struct = []
    for i, vi in enumerate(v):
        xy = vi[:2, :] - centroids[:2, [i]]   # 각 정사각형을 centroid 원점으로 이동
        A, b = _rect_A_b(xy)                  # ← Fusion과 동일한 헬퍼를 그대로 사용
        const_struct.append({"A": A, "b": b})
    z_centers = centroids[2,:]

    outV = edge[:, 0].astype(int)
    inV  = edge[:, 1].astype(int)

    # ===== MOSEK Optimizer Task 생성 =====
    env = mosek.Env()
    task = env.Task(0, 0)

    # (옵션) Fusion 기본 설정과 유사하게
    task.putintparam(mosek.iparam.intpnt_solve_form, mosek.solveform.primal)
    task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.on)

    # ===== 변수 구성 (Fusion과 1:1) =====
    def _alloc_block(ncols, bound_type="unbounded", lb=None, ub=None):
        start = task.getnumvar()
        task.appendvars(ncols)
        idx = list(range(start, start + ncols))
        if bound_type == "unbounded":
            bk = [mosek.boundkey.fr] * ncols; bl = [0.0]*ncols; bu = [0.0]*ncols
        elif bound_type == "range":
            assert lb is not None and ub is not None
            bk = [mosek.boundkey.ra] * ncols; bl = [lb]*ncols; bu = [ub]*ncols
        elif bound_type == "lower":
            assert lb is not None
            bk = [mosek.boundkey.lo] * ncols; bl = [lb]*ncols; bu = [0.0]*ncols
        else:
            raise ValueError("unknown bound_type")
        task.putvarboundlist(idx, bk, bl, bu)
        return slice(start, start + ncols)

    V = {}

    # y ∈ [0,1]^E
    V["y"] = _alloc_block(E, bound_type="range", lb=0.0, ub=1.0)

    # z_* , zp_* ∈ R^{E×5}  ( [x,y,z,cos,sin] ), Domain.unbounded
    n5 = E * 5
    V["z_L"]  = _alloc_block(n5, bound_type="unbounded")
    V["z_R"]  = _alloc_block(n5, bound_type="unbounded")
    V["zp_L"] = _alloc_block(n5, bound_type="unbounded")
    V["zp_R"] = _alloc_block(n5, bound_type="unbounded")

    # rho_* ≥ 0
    V["rho_L"] = _alloc_block(E, bound_type="lower", lb=0.0)
    V["rho_R"] = _alloc_block(E, bound_type="lower", lb=0.0)

    # ===== Fusion의 parameter에 해당하는 PAR 딕셔너리 구성 =====
    # Fusion:
    #   P_start_L, P_start_R, P_target_L, P_target_R,
    #   P_geometry(2*Vn), P_y_relax(E), P_M_scalar(1)
    PAR = {
        "start_L":  np.asarray(start_L,  dtype=float).reshape(-1),  # (5,)
        "start_R":  np.asarray(start_R,  dtype=float).reshape(-1),  # (5,)
        "target_L": np.asarray(target_L, dtype=float).reshape(-1),  # (5,)
        "target_R": np.asarray(target_R, dtype=float).reshape(-1),  # (5,)
        "geometry":  centroids[:2, :].T.reshape(-1),                # (2*Vn,) → [x1,y1,x2,y2,...]
        "y_relax":   np.zeros(E, dtype=float),                      # (E,)
        "M_scalar":  1.0,                                           # scalar
    }

    # ===== IDX / 메타 정보 (Fusion과 동일 의미) =====
    IDX = {
        "n_edge": E,
        "vertex_number": Vn,
        "outV": outV,
        "inV": inV,
        "mode_list": mode_list,
        "const_struct": const_struct,  # Fusion은 지역변수였지만, 하위 제약 빌딩에서 필요하면 유지
        "centroids": centroids,
    }

    # (필요 시) 디버그/헬퍼 인덱싱 유틸
    def _cols(s: slice): return list(range(s.start, s.stop))
    def _ek(base_start: int, e: int, k: int, width: int = 5) -> int:
        return base_start + e*width + k
    V["meta"] = {
        "cols": _cols,
        "ek": _ek,
    }

    M = task
    build_time = time.perf_counter() - t0  # Fusion과 동일하게 seconds 단위 float

    # ========================= Constraints (Optimizer API) ======================
    INF = 0.0  # Mosek에서는 -INF/ +INF를 boundkey로 처리하므로 실제 값은 0.0으로 둡니다.

    # (bk=UP, bu=?, bl은 사용안함 / bk=FX, bl=bu=0.0 형태로 세팅)

    # 헬퍼: 1개 행 추가 (계수: (cols, vals), bound 모드/값)
    def _add_row(cols, vals, bk, bl, bu):
        r0 = task.getnumcon()
        task.appendcons(1)
        # 한 행에 대한 계수 세팅
        task.putarow(r0, cols, vals)
        # 경계(부등식/등식) 세팅
        task.putconbound(r0, bk, bl, bu)
        return r0

    # 슬라이스/인덱싱 헬퍼 (이전 meta와 동일)
    cols = V["meta"]["cols"]
    ek = V["meta"]["ek"]

    y_cols = cols(V["y"])
    zL0, zR0 = V["z_L"].start, V["z_R"].start
    zpL0, zpR0 = V["zp_L"].start, V["zp_R"].start

    E = IDX["n_edge"]
    Vn = IDX["vertex_number"]
    outV = IDX["outV"]
    inV = IDX["inV"]

    # ---------------------------------------------------------------------------
    # (A) || y - y_relax ||_inf <= M_scalar
    #     →  y_e - yrelax_e <=  M_scalar     [<= UB1_e]
    #     → -y_e + yrelax_e <=  M_scalar     [<= UB2_e]
    #  - 우변(UB)은 파라미터에 의존하므로, 행 인덱스를 저장해두고 solve 전 set_parameters에서 갱신.
    # ---------------------------------------------------------------------------
    IDX.setdefault("con_A_inf", {"row_ub_pos": [], "row_ub_neg": []})

    for e in range(E):
        # 1)  y[e] - y_relax[e] <= M_scalar    →  bk=UP, ub = y_relax[e] + M_scalar
        r_pos = _add_row(
            cols=[y_cols[e]],
            vals=[+1.0],
            bk=mosek.boundkey.up,
            bl=0.0,
            bu=0.0,  # 임시, solve 전에 set_parameters에서 실제 (y_relax[e] + M_scalar)로 갱신
        )
        IDX["con_A_inf"]["row_ub_pos"].append(r_pos)

        # 2) -y[e] + y_relax[e] <= M_scalar    →  bk=UP, ub = -y_relax[e] + M_scalar
        r_neg = _add_row(
            cols=[y_cols[e]],
            vals=[-1.0],
            bk=mosek.boundkey.up,
            bl=0.0,
            bu=0.0,  # 임시, solve 전에 set_parameters에서 실제 (-y_relax[e] + M_scalar)로 갱신
        )
        IDX["con_A_inf"]["row_ub_neg"].append(r_neg)

    # ---------------------------------------------------------------------------
    # (B) source/target flow
    #     src_e:  z_L[e,k]  - P_start_L[k]  * y[e] = 0
    #             z_R[e,k]  - P_start_R[k]  * y[e] = 0
    #     sink_e: zp_L[e,k] - P_target_L[k] * y[e] = 0
    #             zp_R[e,k] - P_target_R[k] * y[e] = 0
    #  - 모두 등식(FX). 우변 0.0, 계수는 (1, -P_*[k]).
    #  - 파라미터 값(P_start_*, P_target_*)은 solve 전에 set_parameters에서
    #    계수 갱신 없이 RHS=0 그대로, 계수의 상수 값만 사용(=선형).
    # ---------------------------------------------------------------------------
    IDX.setdefault("con_flow", {"src_rows_L": [], "src_rows_R": [],
                                "sink_rows_L": [], "sink_rows_R": []})

    src_e = np.where(outV == 0)[0]
    sink_e = np.where(inV == Vn - 1)[0]

    # 시작 포즈 (src edges)
    for e in src_e:
        y_col = y_cols[e]
        for k in range(5):
            # z_L(e,k) - P_start_L[k] * y_e = 0
            zL_col = ek(zL0, e, k, 5)
            rL = _add_row(
                cols=[zL_col, y_col],
                vals=[+1.0, -float(PAR["start_L"][k])],
                bk=mosek.boundkey.fx,
                bl=0.0,
                bu=0.0,
            )
            IDX["con_flow"]["src_rows_L"].append(rL)

            # z_R(e,k) - P_start_R[k] * y_e = 0
            zR_col = ek(zR0, e, k, 5)
            rR = _add_row(
                cols=[zR_col, y_col],
                vals=[+1.0, -float(PAR["start_R"][k])],
                bk=mosek.boundkey.fx,
                bl=0.0,
                bu=0.0,
            )
            IDX["con_flow"]["src_rows_R"].append(rR)

    # 목표 포즈 (sink edges)
    for e in sink_e:
        y_col = y_cols[e]
        for k in range(5):
            # zp_L(e,k) - P_target_L[k] * y_e = 0
            zpL_col = ek(zpL0, e, k, 5)
            rL = _add_row(
                cols=[zpL_col, y_col],
                vals=[+1.0, -float(PAR["target_L"][k])],
                bk=mosek.boundkey.fx,
                bl=0.0,
                bu=0.0,
            )
            IDX["con_flow"]["sink_rows_L"].append(rL)

            # zp_R(e,k) - P_target_R[k] * y_e = 0
            zpR_col = ek(zpR0, e, k, 5)
            rR = _add_row(
                cols=[zpR_col, y_col],
                vals=[+1.0, -float(PAR["target_R"][k])],
                bk=mosek.boundkey.fx,
                bl=0.0,
                bu=0.0,
            )
            IDX["con_flow"]["sink_rows_R"].append(rR)
    # ========================= (C) Source/Target vertex Flow conservation =========================
    # sum_{e in src_e} y_e = 1
    # sum_{e in sink_e} y_e = 1
    src_e = np.where(outV == 0)[0]
    sink_e = np.where(inV == Vn - 1)[0]

    def _add_sum_eq_const(indices, const_val: float):
        if len(indices) == 0:
            # 빈 합 == 상수 는 일반적으로 불가능(0 == const). const==0 이면 skip 가능.
            # 여기서는 src/sink 모두 비어있지 않은 게 정상. 안전 위해 assert.
            assert abs(const_val) < 1e-12, "Empty sum cannot equal nonzero constant."
            return None
        r = _add_row(
            cols=[y_cols[e] for e in indices],
            vals=[1.0] * len(indices),
            bk=mosek.boundkey.fx,
            bl=const_val,
            bu=const_val,
        )
        return r

    IDX.setdefault("con_flow_sum", {"src": None, "sink": None})
    IDX["con_flow_sum"]["src"] = _add_sum_eq_const(list(src_e), 1.0)
    IDX["con_flow_sum"]["sink"] = _add_sum_eq_const(list(sink_e), 1.0)

    # ========================= (D) In = Out flow conservation (middle vertices) ===================
    # (1) sum y(in) - sum y(out) = 0
    # (2) sum_k zp(in,k) = sum_k z(out,k)  for k=0..4, L/R 각각
    for vid in range(1, Vn - 1):
        in_idx = np.where(inV == vid)[0]
        out_idx = np.where(outV == vid)[0]

        # -- (1) sum_y(in) - sum_y(out) = 0
        cols_y, vals_y = [], []
        if len(in_idx):
            cols_y += [y_cols[e] for e in in_idx]
            vals_y += [1.0] * len(in_idx)
        if len(out_idx):
            cols_y += [y_cols[e] for e in out_idx]
            vals_y += [-1.0] * len(out_idx)

        if cols_y:  # 둘 다 비면 0=0 이므로 생략
            _add_row(cols=cols_y, vals=vals_y,
                     bk=mosek.boundkey.fx, bl=0.0, bu=0.0)

        # -- (2) sum_e zp_L(e,k) - sum_e z_L(e,k) = 0  (k=0..4)
        #         sum_e zp_R(e,k) - sum_e z_R(e,k) = 0
        for k in range(5):
            # Left
            cols_L, vals_L = [], []
            for e in in_idx:  # + zp_L(e,k)
                cols_L.append(ek(zpL0, e, k, 5));
                vals_L.append(+1.0)
            for e in out_idx:  # - z_L(e,k)
                cols_L.append(ek(zL0, e, k, 5));
                vals_L.append(-1.0)
            if cols_L:
                _add_row(cols=cols_L, vals=vals_L,
                         bk=mosek.boundkey.fx, bl=0.0, bu=0.0)

            # Right
            cols_R, vals_R = [], []
            for e in in_idx:  # + zp_R(e,k)
                cols_R.append(ek(zpR0, e, k, 5));
                vals_R.append(+1.0)
            for e in out_idx:  # - z_R(e,k)
                cols_R.append(ek(zR0, e, k, 5));
                vals_R.append(-1.0)
            if cols_R:
                _add_row(cols=cols_R, vals=vals_R,
                         bk=mosek.boundkey.fx, bl=0.0, bu=0.0)

    # ========================= (E) outflow of y (middle vertices) ================================
    # sum_{e: outV==vid} y_e <= 1     for vid in 1..Vn-2
    for vid in range(1, Vn - 1):
        out_idx = np.where(outV == vid)[0]
        if len(out_idx) == 0:
            continue  # 0 <= 1 은 항상 참이므로 생략 가능
        _add_row(
            cols=[y_cols[e] for e in out_idx],
            vals=[1.0] * len(out_idx),
            bk=mosek.boundkey.up,  # ≤ 1.0
            bl=0.0,
            bu=1.0,
        )
    # ========================= (F) Inside Convex region (depends on from/to mode) =================

    # Fusion: A*p - y*(b + A*g) <= 0   (행마다)
    # Optimizer: 한 행씩 bk=UP, ub=0.0 로 넣고,
    #            계수는  [ A[r,0] on x , A[r,1] on y ,  -(b[r] + A[r,:]@g) on y_scalar ]
    # 여기서 p는 (x,y)이고, g는 vertex centroid (PAR["geometry"][2*vid:2*vid+2])

    IDX.setdefault("con_region", [])
    geom = PAR["geometry"].reshape(-1)  # [x1,y1,x2,y2,...]

    def _add_region_ineq(A: np.ndarray, b: np.ndarray,
                         g_vid: int,  # geometry에서 참조할 vertex id
                         x_col: int, y_col: int,  # p=(x,y)의 변수 컬럼 인덱스
                         y_scalar_col: int,  # y[e] 의 변수 컬럼
                         tag: str):
        """
        A (m,2), b (m,), g from PAR["geometry"][2*g_vid:2*g_vid+2]
        Adds m rows:  A[r,0]*x + A[r,1]*y + (-(b[r]+A[r,:]@g))*y_scalar <= 0
        """
        g = geom[2 * g_vid:2 * g_vid + 2]  # (2,)
        m = A.shape[0]
        row_ids = []
        for r in range(m):
            a0, a1 = float(A[r, 0]), float(A[r, 1])
            beta = float(b[r])
            # coeff on y_scalar is -(b[r] + A[r,:] @ g)
            rhs_coeff = -(beta + a0 * g[0] + a1 * g[1])

            r_id = _add_row(
                cols=[x_col, y_col, y_scalar_col],
                vals=[a0, a1, rhs_coeff],
                bk=mosek.boundkey.up,  # <= 0
                bl=0.0,
                bu=0.0,
            )
            # 행에 이름 달기(어느 edge/모드/행인지 추적 용이)
            task.putconname(r_id, f"REGION[{tag}]-r{r}")

            row_ids.append(r_id)

        IDX["con_region"].append(
            {
                "rows": row_ids,
                "g_vid": g_vid,
                "A": A,
                "b": b,
                "y_col": y_scalar_col,
                "x_col": x_col,  # ★ 추가
                "xy_col": y_col,  # ★ 추가 (혹은 "y_xy_col" 등)
                "tag": tag,
            }
        )

    E = IDX["n_edge"]
    Vn = IDX["vertex_number"]
    outV = IDX["outV"]
    inV = IDX["inV"]
    const_struct = IDX["const_struct"]

    zL0, zR0 = V["z_L"].start, V["z_R"].start
    zpL0, zpR0 = V["zp_L"].start, V["zp_R"].start
    y_cols = V["meta"]["cols"](V["y"])
    ek = V["meta"]["ek"]

    for e in range(E):
        u = int(outV[e]);
        v = int(inV[e])
        mf = IDX["mode_list"][u]
        mt = IDX["mode_list"][v]

        A_from = const_struct[u]["A"];
        b_from = const_struct[u]["b"]
        A_to = const_struct[v]["A"];
        b_to = const_struct[v]["b"]
        # ★ 여기서 z_from, z_to 바로 사용
        z_from = float(z_centers[u])
        z_to = float(z_centers[v])

        ycol = y_cols[e]

        # Helper: 각 발의 (x,y) 컬럼 인덱스
        def xy_cols_of(base_start, e, width=5):
            return ek(base_start, e, 0, width), ek(base_start, e, 1, width)  # (x_col, y_col)

        def z_col_of(base_start, e, width=5):
            return ek(base_start, e, 2, width)  # k=2: z

        if mf == "D" and mt == "SL":
            # A_to * zp_L_xy - y*(b_to + A_to*g_to) <= 0
            xcol, ycol_xy = xy_cols_of(zpL0, e)
            _add_region_ineq(A_to, b_to, g_vid=v,
                             x_col=xcol, y_col=ycol_xy, y_scalar_col=ycol,
                             tag=f"e{e}:D->SL(to=v{v})")
            # z: zp_L(e,z) - z_to * y[e] == 0
            zcol = z_col_of(zpL0, e)
            _add_row(cols=[zcol, ycol],vals=[1.0, -z_to],bk=mosek.boundkey.fx, bl=0.0, bu=0.0,)
            # z: zp_R(e,z) - z_from * y[e] == 0
            zcol = z_col_of(zpR0, e)
            _add_row(cols=[zcol, ycol],vals=[1.0, -z_from],bk=mosek.boundkey.fx, bl=0.0, bu=0.0,)

        if mf == "D" and mt == "SR":
            xcol, ycol_xy = xy_cols_of(zpR0, e)
            # ★ 디버그 출력
            g_dbg = geom[2 * v: 2 * v + 2]
            _add_region_ineq(A_to, b_to, g_vid=v,
                             x_col=xcol, y_col=ycol_xy, y_scalar_col=ycol,
                             tag=f"e{e}:D->SR(to=v{v})")
            # z: zp_R(e,z) - z_to * y[e] == 0
            zcol = z_col_of(zpR0, e)
            _add_row(cols=[zcol, ycol],vals=[1.0, -z_to],bk=mosek.boundkey.fx, bl=0.0, bu=0.0,)
            # z: zp_L(e,z) - z_from * y[e] == 0
            zcol = z_col_of(zpL0, e)
            _add_row(cols=[zcol, ycol],vals=[1.0, -z_from],bk=mosek.boundkey.fx, bl=0.0, bu=0.0,)

        if mf == "SL" and mt == "SR":
            # from: A_from * zp_L_xy - y*(b_from + A_from*g_from) <= 0
            xcol, ycol_xy = xy_cols_of(zpL0, e)
            _add_region_ineq(A_from, b_from, g_vid=u,
                             x_col=xcol, y_col=ycol_xy, y_scalar_col=ycol,
                             tag=f"e{e}:SL->SR(from=u{u})")
            # to:   A_to   * zp_R_xy - y*(b_to   + A_to  *g_to)   <= 0
            xcol, ycol_xy = xy_cols_of(zpR0, e)
            _add_region_ineq(A_to, b_to, g_vid=v,
                             x_col=xcol, y_col=ycol_xy, y_scalar_col=ycol,
                             tag=f"e{e}:SL->SR(to=v{v})")
            # z: zp_L(e,z) - z_from * y[e] == 0
            zcol = z_col_of(zpL0, e)
            _add_row(cols=[zcol, ycol],vals=[1.0, -z_from],bk=mosek.boundkey.fx, bl=0.0, bu=0.0,)
            # z: zp_R(e,z) - z_to * y[e] == 0
            zcol = z_col_of(zpR0, e)
            _add_row(cols=[zcol, ycol],vals=[1.0, -z_to],bk=mosek.boundkey.fx, bl=0.0, bu=0.0,)

        if mf == "SR" and mt == "SL":
            # from: use zp_R
            xcol, ycol_xy = xy_cols_of(zpR0, e)
            _add_region_ineq(A_from, b_from, g_vid=u,
                             x_col=xcol, y_col=ycol_xy, y_scalar_col=ycol,
                             tag=f"e{e}:SR->SL(from=u{u})")
            # to:   use zp_L
            xcol, ycol_xy = xy_cols_of(zpL0, e)
            _add_region_ineq(A_to, b_to, g_vid=v,
                             x_col=xcol, y_col=ycol_xy, y_scalar_col=ycol,
                             tag=f"e{e}:SR->SL(to=v{v})")
            # z: zp_R(e,z) - z_from * y[e] == 0
            zcol = z_col_of(zpR0, e)
            _add_row(cols=[zcol, ycol],vals=[1.0, -z_from],bk=mosek.boundkey.fx, bl=0.0, bu=0.0,)
            # z: zp_L(e,z) - z_to * y[e] == 0
            zcol = z_col_of(zpL0, e)
            _add_row(cols=[zcol, ycol],vals=[1.0, -z_to],bk=mosek.boundkey.fx, bl=0.0, bu=0.0,)

        if mf == "SL" and mt == "D":
            # A_from * z_L_xy - y*(b_from + A_from*g_from) <= 0
            xcol, ycol_xy = xy_cols_of(zL0, e)
            _add_region_ineq(A_from, b_from, g_vid=u,
                             x_col=xcol, y_col=ycol_xy, y_scalar_col=ycol,
                             tag=f"e{e}:SL->D(from=u{u})")
            # z: z_L(e,z) - z_from * y[e] == 0
            zcol = z_col_of(zL0, e)
            _add_row(cols=[zcol, ycol],vals=[1.0, -z_from],bk=mosek.boundkey.fx, bl=0.0, bu=0.0,)

        if mf == "SR" and mt == "D":
            xcol, ycol_xy = xy_cols_of(zR0, e)
            _add_region_ineq(A_from, b_from, g_vid=u,
                             x_col=xcol, y_col=ycol_xy, y_scalar_col=ycol,
                             tag=f"e{e}:SR->D(from=u{u})")
            # z: z_R(e,z) - z_from * y[e] == 0
            zcol = z_col_of(zR0, e)
            _add_row(cols=[zcol, ycol],vals=[1.0, -z_from],bk=mosek.boundkey.fx, bl=0.0, bu=0.0,)

    # ========================= (G) gating: |var[e,k]| <= y[e] (k=3,4) =====================
    # Fusion:
    #   le0( var - y ), le0( -var - y )
    # Optimizer:
    #   (var - y <= 0), (-var - y <= 0)  → bk=UP, ub=0.0
    zL0, zR0 = V["z_L"].start, V["z_R"].start
    zpL0, zpR0 = V["zp_L"].start, V["zp_R"].start
    y_cols = V["meta"]["cols"](V["y"])
    ek = V["meta"]["ek"]
    # === SOC 도메인(차원 3: [t, x, y])을 한 번만 만들어 재사용 ===
    dom_q3 = task.appendquadraticconedomain(3)

    def _add_qcone_triplet(t_coeffs, x_coeffs, y_coeffs):
        """
        SOC: ||[x, y]||_2 <= t  를 AFE/ACC로 추가한다.
        각 인자는 [(col, val), ...] 형태의 (열, 계수) 쌍 리스트.

        예)
          t_coeffs = [(y_col, step_radius)]          # t = step_radius * y[e]
          x_coeffs = [(zpLx, 1.0), (zRx, -1.0)]      # x = zp_Lx - z_Rx
          y_coeffs = [(zpLy, 1.0), (zRy, -1.0)]      # y = zp_Ly - z_Ry
        """
        # AFE 3개 추가 (t, x, y)
        task.appendafes(3)
        r0 = task.getnumafe() - 3
        r_t, r_x, r_y = r0, r0 + 1, r0 + 2

        def _put_row(r, coeffs):
            if coeffs:
                cols, vals = zip(*coeffs)
                task.putafefrow(r, list(cols), list(vals))
            else:
                # 비어 있으면 0 == 0 로우가 된다 (상수=0, 계수 없음)
                task.putafefrow(r, [], [])

        _put_row(r_t, t_coeffs)
        _put_row(r_x, x_coeffs)
        _put_row(r_y, y_coeffs)

        # ACC 추가: 이 3개 AFE 행이 dom_q3(QuadraticCone-3)에 속하도록
        task.appendacc(dom_q3, [r_t, r_x, r_y], None)

        # 필요하면 행 인덱스를 반환해서 추적할 수도 있음
        return (r_t, r_x, r_y)
        # 위치/각도용 bound 상수 (스케일에 맞게 조정 가능)
    M_pos = 10.0  # 예: 10m 정도면 사실상 널널. 필요하면 step_span 기반으로 바꿔도 됨
    M_ang = 1.0  # cos,sin 은 [-1,1] 유지

    for e in range(E):
        y_col = y_cols[e]
        # 각 var ∈ {z_L, z_R, zp_L, zp_R}, k ∈ {0,1,2,3,4}
        for base0 in (zL0, zR0, zpL0, zpR0):
            for k in (0, 1, 2, 3, 4):
                vcol = ek(base0, e, k, 5)

                # k별 M_k 선택: 0,1,2 → M_pos / 3,4 → M_ang
                M_k = M_pos if k in (0, 1, 2) else M_ang

                # 1) var - M_k * y <= 0   →  var ≤ M_k * y
                _add_row(
                    cols=[vcol, y_col],
                    vals=[+1.0, -M_k],
                    bk=mosek.boundkey.up,
                    bl=0.0,
                    bu=0.0,
                )

                # 2) -var - M_k * y <= 0  →  var ≥ -M_k * y
                _add_row(
                    cols=[vcol, y_col],
                    vals=[-1.0, -M_k],
                    bk=mosek.boundkey.up,
                    bl=0.0,
                    bu=0.0,
                )

    # ========================= (I) minimum safety & (H) relative foot bound =======================
    # Fusion에서 (I)는 mf=="D" and mt in {"SL","SR"} 에서만 추가:
    #   xL[e] + 0.2*y[e] - xR[e] <= 0
    # (H) SOC: || [ zp_L(e,0)-z_R(e,0),  zp_L(e,1)-z_R(e,1) ] ||_2 <= step_radius * y[e]
    #          || [ zp_R(e,0)-z_L(e,0),  zp_R(e,1)-z_L(e,1) ] ||_2 <= step_radius * y[e]
    step_radius_val = 0.33  # Fusion 블록과 동일 숫자로 고정 0.33 real demo

    for e in range(E):
        u = int(outV[e]);
        v = int(inV[e])
        mf = IDX["mode_list"][u]
        mt = IDX["mode_list"][v]
        y_col = y_cols[e]

        # (I) xL[e] + 0.2*y[e] - xR[e] <= 0
        xL_col = ek(zL0, e, 0, 5)
        xR_col = ek(zR0, e, 0, 5)
        _add_row(
            cols=[xL_col, y_col, xR_col],
            vals=[+1.0, +0.2, -1.0],
            bk=mosek.boundkey.up,
            bl=0.0, bu=0.0,
        )

        # (H-1) SOC: left-next vs right-current
        # t = step_radius * y[e]
        # x = zp_L(e,0) - z_R(e,0)
        # y = zp_L(e,1) - z_R(e,1)
        t_coeffs = [(y_col, float(step_radius_val))]
        x_coeffs = [(ek(zpL0, e, 0, 5), +1.0), (ek(zR0, e, 0, 5), -1.0)]
        y_coeffs = [(ek(zpL0, e, 1, 5), +1.0), (ek(zR0, e, 1, 5), -1.0)]
        _add_qcone_triplet(t_coeffs, x_coeffs, y_coeffs)

        # (H-2) SOC: right-next vs left-current
        # t = step_radius * y[e]
        # x = zp_R(e,0) - z_L(e,0)
        # y = zp_R(e,1) - z_L(e,1)
        t_coeffs = [(y_col, float(step_radius_val))]
        x_coeffs = [(ek(zpR0, e, 0, 5), +1.0), (ek(zL0, e, 0, 5), -1.0)]
        y_coeffs = [(ek(zpR0, e, 1, 5), +1.0), (ek(zL0, e, 1, 5), -1.0)]
        _add_qcone_triplet(t_coeffs, x_coeffs, y_coeffs)

    # ========================= Objective & rho-연계 제약 =========================
    task.putobjsense(mosek.objsense.minimize)

    E = IDX["n_edge"]
    outV = IDX["outV"]
    inV = IDX["inV"]
    modes = IDX["mode_list"]

    zL0, zR0 = V["z_L"].start, V["z_R"].start
    zpL0, zpR0 = V["zp_L"].start, V["zp_R"].start
    rhoL0, rhoR0 = V["rho_L"].start, V["rho_R"].start

    ek = V["meta"]["ek"]

    # --- (A) rho와 좌표 차이를 연결하는 제약 + 모드별 고정 등식 ---
    for e in range(E):
        mt = modes[int(inV[e])]

        if mt == "D":
            # cone: || [zp_Lx - z_Lx,  zp_Ly - z_Ly] ||_2 <= rho_L[e]
            t_coeffs = [(rhoL0 + e, 1.0)]
            x_coeffs = [(ek(zpL0, e, 0, 5), +1.0), (ek(zL0, e, 0, 5), -1.0)]
            y_coeffs = [(ek(zpL0, e, 1, 5), +1.0), (ek(zL0, e, 1, 5), -1.0)]
            _add_qcone_triplet(t_coeffs, x_coeffs, y_coeffs)

            # cone: || [zp_Rx - z_Rx,  zp_Ry - z_Ry] ||_2 <= rho_R[e]
            t_coeffs = [(rhoR0 + e, 1.0)]
            x_coeffs = [(ek(zpR0, e, 0, 5), +1.0), (ek(zR0, e, 0, 5), -1.0)]
            y_coeffs = [(ek(zpR0, e, 1, 5), +1.0), (ek(zR0, e, 1, 5), -1.0)]
            _add_qcone_triplet(t_coeffs, x_coeffs, y_coeffs)

        elif mt == "SL":
            # 고정: zp_R(k) == z_R(k)  (k in {0,1,3,4})
            for k in (0, 1, 3, 4):
                _add_row(
                    cols=[ek(zpR0, e, k, 5), ek(zR0, e, k, 5)],
                    vals=[+1.0, -1.0],
                    bk=mosek.boundkey.fx, bl=0.0, bu=0.0
                )
            # rho_R[e] == 0
            _add_row(cols=[rhoR0 + e], vals=[1.0], bk=mosek.boundkey.fx, bl=0.0, bu=0.0)

            # cone: || [zp_L - z_L]_xy ||_2 <= rho_L[e]
            t_coeffs = [(rhoL0 + e, 1.0)]
            x_coeffs = [(ek(zpL0, e, 0, 5), +1.0), (ek(zL0, e, 0, 5), -1.0)]
            y_coeffs = [(ek(zpL0, e, 1, 5), +1.0), (ek(zL0, e, 1, 5), -1.0)]
            _add_qcone_triplet(t_coeffs, x_coeffs, y_coeffs)

        elif mt == "SR":
            # 고정: zp_L(k) == z_L(k)  (k in {0,1,3,4})
            for k in (0, 1, 3, 4):
                _add_row(
                    cols=[ek(zpL0, e, k, 5), ek(zL0, e, k, 5)],
                    vals=[+1.0, -1.0],
                    bk=mosek.boundkey.fx, bl=0.0, bu=0.0
                )
            # rho_L[e] == 0
            _add_row(cols=[rhoL0 + e], vals=[1.0], bk=mosek.boundkey.fx, bl=0.0, bu=0.0)

            # cone: || [zp_R - z_R]_xy ||_2 <= rho_R[e]
            t_coeffs = [(rhoR0 + e, 1.0)]
            x_coeffs = [(ek(zpR0, e, 0, 5), +1.0), (ek(zR0, e, 0, 5), -1.0)]
            y_coeffs = [(ek(zpR0, e, 1, 5), +1.0), (ek(zR0, e, 1, 5), -1.0)]
            _add_qcone_triplet(t_coeffs, x_coeffs, y_coeffs)

    # --- (A-1) 선형 목적함수: sum(wL .* rho_L) + sum(wR .* rho_R) ---
    wL = np.zeros(E);
    wR = np.zeros(E)
    for e in range(E):
        mf = modes[int(outV[e])];
        mt = modes[int(inV[e])]
        if (mf == "SR" and mt == "D") or (mf == "SL" and mt == "D"):
            wL[e] = 1.0;
            wR[e] = 1.0
        elif (mf == "SR" and mt == "SL") or (mf == "D" and mt == "SL"):
            wL[e] = 1.0
        elif (mf == "SL" and mt == "SR") or (mf == "D" and mt == "SR"):
            wR[e] = 1.0

    obj_cols = []
    obj_vals = []

    # rho_L
    for e in range(E):
        if wL[e] != 0.0:
            obj_cols.append(rhoL0 + e)
            obj_vals.append(float(wL[e]))

    # rho_R
    for e in range(E):
        if wR[e] != 0.0:
            obj_cols.append(rhoR0 + e)
            obj_vals.append(float(wR[e]))

    if obj_cols:
        task.putclist(obj_cols, obj_vals)

    # ========================= Objective Rotation term =========================
    ## (B) 회전 변화 SOC: ||(c,s)-(c',s')||_2 minimizing change difference (z / zp)
    def _add_qcone_triplet_const(t_coeffs, x_coeffs, y_coeffs, g_t=0.0, g_x=0.0, g_y=0.0):
        """
        SOC: ||[x, y]||_2 <= t   를 AFE/ACC로 추가.
        각 행은 f = F x + g 로 표현되며, g_t/g_x/g_y로 '상수'를 부여할 수 있다.
        인자: [(col, val), ...] 형태의 리스트.
        """
        task.appendafes(3)
        r0 = task.getnumafe() - 3
        r_t, r_x, r_y = r0, r0 + 1, r0 + 2

        def _put_row(r, coeffs):
            if coeffs:
                cols, vals = zip(*coeffs)
                task.putafefrow(r, list(cols), list(vals))
            else:
                task.putafefrow(r, [], [])

        _put_row(r_t, t_coeffs)
        _put_row(r_x, x_coeffs)
        _put_row(r_y, y_coeffs)

        # g = [g_t, g_x, g_y]
        task.appendacc(dom_q3, [r_t, r_x, r_y], [float(g_t), float(g_x), float(g_y)])
        return (r_t, r_x, r_y)

    # ----- 변수 추가: t_rotchg_L, t_rotchg_R >= 0 -----
    def _alloc_block(ncols, bound_type="unbounded", lb=None, ub=None):
        start = task.getnumvar()
        task.appendvars(ncols)
        idx = list(range(start, start + ncols))
        if bound_type == "unbounded":
            bk = [mosek.boundkey.fr] * ncols;
            bl = [0.0] * ncols;
            bu = [0.0] * ncols
        elif bound_type == "lower":
            bk = [mosek.boundkey.lo] * ncols;
            bl = [lb] * ncols;
            bu = [0.0] * ncols
        elif bound_type == "range":
            bk = [mosek.boundkey.ra] * ncols;
            bl = [lb] * ncols;
            bu = [ub] * ncols
        else:
            raise ValueError("unknown bound_type")
        task.putvarboundlist(idx, bk, bl, bu)
        return slice(start, start + ncols)

    E = IDX["n_edge"]
    modes = IDX["mode_list"]
    zL0, zR0 = V["z_L"].start, V["z_R"].start
    zpL0, zpR0 = V["zp_L"].start, V["zp_R"].start
    ek = V["meta"]["ek"]

    # 없으면 생성(여러 번 호출 안전)
    if "t_rotL" not in V:
        V["t_rotL"] = _alloc_block(E, bound_type="lower", lb=0.0)
    if "t_rotR" not in V:
        V["t_rotR"] = _alloc_block(E, bound_type="lower", lb=0.0)

    tRotL0 = V["t_rotL"].start
    tRotR0 = V["t_rotR"].start

    # SOC: ||(c,s) - (c',s')||_2 <= t_rot*
    for e in range(E):
        # Left
        _add_qcone_triplet_const(
            t_coeffs=[(tRotL0 + e, 1.0)],
            x_coeffs=[(ek(zL0, e, 3, 5), +1.0), (ek(zpL0, e, 3, 5), -1.0)],
            y_coeffs=[(ek(zL0, e, 4, 5), +1.0), (ek(zpL0, e, 4, 5), -1.0)],
            g_t=0.0, g_x=0.0, g_y=0.0
        )
        # Right
        _add_qcone_triplet_const(
            t_coeffs=[(tRotR0 + e, 1.0)],
            x_coeffs=[(ek(zR0, e, 3, 5), +1.0), (ek(zpR0, e, 3, 5), -1.0)],
            y_coeffs=[(ek(zR0, e, 4, 5), +1.0), (ek(zpR0, e, 4, 5), -1.0)],
            g_t=0.0, g_x=0.0, g_y=0.0
        )

    # 아주 작은 가중치
    lam_rotchg = 0.1  # Fusion과 동일
    task.putobjsense(mosek.objsense.minimize)
    task.putclist(list(range(tRotL0, tRotL0 + E)), [lam_rotchg] * E)
    task.putclist(list(range(tRotR0, tRotR0 + E)), [lam_rotchg] * E)


    ## (C) Pump Terrain dependent term
    # ----- 스톤 라벨(from=in, to=out) -----
    outV = IDX["outV"];
    inV = IDX["inV"]
    stone_start = [v_names[int(inV[e])] for e in range(E)]
    stone_end = [v_names[int(outV[e])] for e in range(E)]

    # 모드 이름 정규화
    def _norm_mode(x: str) -> str:
        return "D" if x in ("Do", "DO", "Double", "DoubleSupport") else x

    mode_in = [_norm_mode(modes[int(inV[e])]) for e in range(E)]
    mode_out = [_norm_mode(modes[int(outV[e])]) for e in range(E)]

    # 룰(각도 deg)
    RULES_DEG = {
        ("DL", "UL"): 90, ("DL", "U"): 85, ("DL", "UR"): 80,
        ("DL", "L"): 90, ("DL", "C"): 80, ("DL", "R"): 75,
        ("DR", "UL"): 100, ("DR", "U"): 95, ("DR", "UR"): 90,
        ("DR", "L"): 105, ("DR", "C"): 100, ("DR", "R"): 90,
        ("C", "UL"): 105, ("C", "U"): 90, ("C", "UR"): 75,

    }
    deg2rad = lambda d: np.deg2rad(d)

    # 참조(c*, s*) & 룰 매치 마스크
    ref_c = np.zeros(E);
    ref_s = np.zeros(E);
    rule_mask = np.zeros(E)
    for e in range(E):
        key = (stone_start[e], stone_end[e])
        if key in RULES_DEG:
            th = deg2rad(RULES_DEG[key])
            ref_c[e], ref_s[e] = np.cos(th), np.sin(th)
            rule_mask[e] = 1.0

    # 어느 발에 적용할지(모드 마스크)
    mL = np.zeros(E);
    mR = np.zeros(E)
    for e in range(E):
        mt = mode_in[e]  # SL, SR, D (in)
        mf = mode_out[e]  # SL, SR, D (out)
        if (mf == "SR" and mt == "SL") or (mf == "D" and mt == "SL"):
            mR[e] = 1.0
        elif (mf == "SL" and mt == "SR") or (mf == "D" and mt == "SR"):
            mL[e] = 1.0
        else:
            mL[e] = 0.0;
            mR[e] = 0.0


    # 최종 가중치
    LAM_BASE = 0.35
    wL = LAM_BASE * rule_mask * mL
    wR = LAM_BASE * rule_mask * mR
    # ----- 변수 추가: t_yaw_pref_L, t_yaw_pref_R >= 0 -----
    if "t_yaw_L" not in V:
        V["t_yaw_L"] = _alloc_block(E, bound_type="lower", lb=0.0)
    if "t_yaw_R" not in V:
        V["t_yaw_R"] = _alloc_block(E, bound_type="lower", lb=0.0)

    tYawL0 = V["t_yaw_L"].start
    tYawR0 = V["t_yaw_R"].start

    # 제약: w=0 이면 t=0 고정, w>0 이면 SOC로 ||(c,s)-(c*,s*)||_2 <= t
    for e in range(E):
        if wL[e] != 0.0:
            # x = z_L(e,3) - ref_c[e], y = z_L(e,4) - ref_s[e]
            _add_qcone_triplet_const(
                t_coeffs=[(tYawL0 + e, 1.0)],
                x_coeffs=[(ek(zL0, e, 3, 5), +1.0)],
                y_coeffs=[(ek(zL0, e, 4, 5), +1.0)],
                g_t=0.0, g_x=+float(ref_c[e]), g_y=+float(ref_s[e])
            )
        else:
            # t_yaw_L[e] == 0
            task.appendcons(1)
            r = task.getnumcon() - 1
            task.putarow(r, [tYawL0 + e], [1.0])
            task.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        if wR[e] != 0.0:
            # x = z_R(e,3) - ref_c[e], y = z_R(e,4) - ref_s[e]
            _add_qcone_triplet_const(
                t_coeffs=[(tYawR0 + e, 1.0)],
                x_coeffs=[(ek(zR0, e, 3, 5), +1.0)],
                y_coeffs=[(ek(zR0, e, 4, 5), +1.0)],
                g_t=0.0, g_x=+float(ref_c[e]), g_y=+float(ref_s[e])
            )
        else:
            # t_yaw_R[e] == 0
            task.appendcons(1)
            r = task.getnumcon() - 1
            task.putarow(r, [tYawR0 + e], [1.0])
            task.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

    # 목적: sum(wL .* t_yaw_L) + sum(wR .* t_yaw_R)
    obj_cols, obj_vals = [], []
    for e in range(E):
        if wL[e] != 0.0:
            obj_cols.append(tYawL0 + e)
            obj_vals.append(float(wL[e]))
    for e in range(E):
        if wR[e] != 0.0:
            obj_cols.append(tYawR0 + e)
            obj_vals.append(float(wR[e]))
    if obj_cols:
        task.putclist(obj_cols, obj_vals)

    # ----- (추가) step number cost: 0.1 * sum(y) -----
    y_cols = V["meta"]["cols"](V["y"])
    if y_cols:
        task.putclist(y_cols, [0.1] * len(y_cols))

    # ----- 목적 sense -----
    task.putobjsense(mosek.objsense.minimize)

    # (Fusion의 M.acceptedSolutionStatus(Feasible) 은 Optimizer에 대응 없음 → 생략)

    # ----- build time -----
    build_time = time.perf_counter() - t0

    # ----- V/IDX/PAR: Fusion과 키를 맞춰서 반환 -----
    #  - V에는 변수 핸들을 slice로 제공
    #  - Fusion의 parameter 자리(start_L 등)는 키만 동일하게 유지 (값은 PAR 것을 가리키거나 None)
    V = dict(
        y=V["y"],
        z_L=V["z_L"], z_R=V["z_R"], zp_L=V["zp_L"], zp_R=V["zp_R"],
        rho_L=V["rho_L"], rho_R=V["rho_R"],
        # Fusion에서는 parameter 객체였던 것들: 키만 동일하게 유지
        start_L=PAR["start_L"],  # 또는 None (호출부 호환성에 따라 선택)
        start_R=PAR["start_R"],
        target_L=PAR["target_L"],
        target_R=PAR["target_R"],
        geometry=PAR["geometry"],
        y_relax=PAR["y_relax"],
        M_scalar=PAR["M_scalar"],
        # (옵션) 앞서 저장한 meta 유틸 계속 노출
        meta=V["meta"],
    )

    # Fusion과 동일 핵심 키 유지
    IDX = dict(
        n_edge=IDX["n_edge"],
        vertex_number=IDX["vertex_number"],
        outV=IDX["outV"],
        inV=IDX["inV"],
        mode_list=IDX["mode_list"],
        # (옵션) 이후 set_parameters에서 쓰는 인덱스/행 ID를 담아뒀다면 그대로 포함
        con_A_inf=IDX.get("con_A_inf"),
        con_flow=IDX.get("con_flow"),
        con_region=IDX.get("con_region"),
    )

    # Fusion에서 PAR는 centroid만 주지만, Optimizer 경로에서는 solve 전 파라미터 갱신을 위해
    # start/target/geometry/y_relax/M_scalar까지 이미 담아 둠. centroid 키도 맞춰서 포함.
    PAR = dict(
        centroid=centroids,  # Fusion 호환용
        start_L=PAR["start_L"],
        start_R=PAR["start_R"],
        target_L=PAR["target_L"],
        target_R=PAR["target_R"],
        geometry=PAR["geometry"],
        y_relax=PAR["y_relax"],
        M_scalar=PAR["M_scalar"],
    )

    # Optimizer에서는 M이 mosek.Task
    M = task
    return M, V, IDX, PAR, build_time

