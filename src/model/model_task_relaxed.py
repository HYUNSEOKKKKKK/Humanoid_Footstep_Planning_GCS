from __future__ import annotations
import time
import numpy as np
from array import array
import mosek, inspect



def add_soc2(task, t_aff, x_aff, y_aff, afe_base):
    """
    t_aff, x_aff, y_aff : 각각 (terms, const)
      - terms: [(coef, var_j), ...]  (list)
      - const: float
    """

    def put_affine_row(row_idx, aff):
        terms, const = aff
        if terms:
            subj = [int(j) for (_, j) in terms]
            val  = [float(c) for (c, _) in terms]
            # ★ 당신 환경: putafefrow(afe_idx, subj, val)  ← 3-인자 버전
            task.putafefrow(row_idx, subj, val)
        else:
            # 빈 행도 허용: 빈 리스트 전달
            task.putafefrow(row_idx, [], [])
        task.putafeg(row_idx, float(const))  # g(상수항) 별도 설정

    row_t = afe_base + 0
    row_x = afe_base + 1
    row_y = afe_base + 2
    # 파일 상단 전역(또는 build 함수 초반)에 한 번:
    DOM_QUAD3 = task.appendquadraticconedomain(3)  # (t, x, y) ∈ Q³
    # AFE 행 3개를 row_t, row_x, row_y에 맞게 putafefrow/putafeg로 채워둔 뒤…
    acc_idx = task.appendacc(DOM_QUAD3, [row_t, row_x, row_y], None)

    return (afe_base, acc_idx, afe_base + 3)





def build_gcs_task_relaxed(
    v, edge, step_span, alpha, mode_list, x_ref, y_ref, v_names
):
    """
    Fusion의 build_gcs_model(...)을 Optimizer(Task)로 1차 포팅 (릴랙스드용 MVP).
    - 파라미터는 '상수 변수'로 만들어 bounds로 고정/갱신
    - 회전/ρ 관련 목적은 2단계에서 추가 예정 (지금은 0.1 * sum(y))
    """
    t0 = time.perf_counter()

    # 0-based edge
    edge = edge.astype(int) - 1
    n_edge = int(edge.shape[0])
    vertex_number = int(max(edge[:, 0].max(), edge[:, 1].max())) + 1

    # ====== 기하 보조 ======
    # v: 각 정점의 사각형 꼭짓점들. 기존 코드에서 쓰던 동일 헬퍼 호출을 가정.
    from src.graph import centroid_and_scaled_vertices
    from src.graph.geometry import rect_A_b_from_vertices

    centroid, v_scaled = centroid_and_scaled_vertices(v, scale=1.0)  # (3, nV), list of rects
    centroid_xy = centroid[:2, :].T

    # inside-region 불등식 A x ≤ b는 (x - center)로 잡지 않고,
    # 기존 Fusion 코드처럼 b를 y와 곱한 형태로 사용
    const_struct = []
    for ii, vtx in enumerate(v_scaled):
        A, b = rect_A_b_from_vertices(vtx)  # A: (m,2), b:(m,)
        const_struct.append({
            "A": A,
            "b": b + A @ centroid[:2, ii],  # 중심 이동 반영
            "center_z": float(centroid[2, ii])
        })

    flow_out_vertices = edge[:, 0].astype(int)
    flow_in_vertices  = edge[:, 1].astype(int)
    src_edges = np.where(flow_in_vertices == 0)[0]
    sink_edges = np.where(flow_in_vertices == vertex_number - 1)[0]

    # ====== Task 생성 ======
    task = mosek.Task()
    task.putobjsense(mosek.objsense.minimize)

    idx = {}
    var_base = 0

    def alloc_vars(n, lb=None, ub=None, bk=mosek.boundkey.fr):
        nonlocal var_base
        start = var_base
        task.appendvars(n)

        for j in range(start, start + n):
            if bk == mosek.boundkey.fr:
                # 자유변수: 하한/상한 인자 무시됨
                task.putvarbound(j, mosek.boundkey.fr, 0.0, 0.0)

            elif bk == mosek.boundkey.ra:
                # 구간(bound) 반드시 유한값 필요
                assert lb is not None and ub is not None
                task.putvarbound(j, mosek.boundkey.ra, float(lb), float(ub))

            elif bk == mosek.boundkey.fx:
                # 일단 0으로 고정해두고 나중에 update_task_params에서 원하는 값으로 다시 고정
                task.putvarbound(j, mosek.boundkey.fx, 0.0, 0.0)

            elif bk == mosek.boundkey.lo:
                assert lb is not None
                task.putvarbound(j, mosek.boundkey.lo, float(lb), 0.0)  # 상한은 무시됨

            elif bk == mosek.boundkey.up:
                assert ub is not None
                task.putvarbound(j, mosek.boundkey.up, 0.0, float(ub))  # 하한은 무시됨

            else:
                raise ValueError(f"Unsupported bound key: {bk}")

        var_base += n
        return np.arange(start, start + n, dtype=int)

    # y in [0,1]
    idx_y = alloc_vars(n_edge, 0.0, 1.0, mosek.boundkey.ra)

    # z_* : (n_edge, 5) free
    def alloc_block(ne, dim):
        ids = []
        for _ in range(ne):
            ids.append(alloc_vars(dim, bk=mosek.boundkey.fr))
        return np.vstack(ids)   # shape (n_edge, dim)

    idx_zL  = alloc_block(n_edge, 5)
    idx_zR  = alloc_block(n_edge, 5)
    idx_zpL = alloc_block(n_edge, 5)
    idx_zpR = alloc_block(n_edge, 5)

    # ----- 파라미터용 '상수 변수' -----
    # y_relax (n_edge), M_scalar (1)
    idx_y_relax = alloc_vars(n_edge, bk=mosek.boundkey.fx)  # 값은 set_parameters에서 고정
    idx_M       = alloc_vars(1,      bk=mosek.boundkey.fx)

    # start_L/R, target_L/R (각 5)
    idx_sL = alloc_vars(5, bk=mosek.boundkey.fx)
    idx_sR = alloc_vars(5, bk=mosek.boundkey.fx)
    idx_tL = alloc_vars(5, bk=mosek.boundkey.fx)
    idx_tR = alloc_vars(5, bk=mosek.boundkey.fx)

    # ====== 선형 제약 생성 ======
    # 편하게 a_ij 세팅을 모아서 넣자
    con_base = 0
    def add_cons(m):
        nonlocal con_base
        start = con_base
        task.appendcons(m)
        con_base += m
        return np.arange(start, start+m, dtype=int)

    # # (A) || y - y_relax ||_inf <= M  ==>  y - y_relax - M <= 0  and  y_relax - y - M <= 0
    # con_A1 = add_cons(n_edge)
    # con_A2 = add_cons(n_edge)
    # # y coef (+1) in A1, (-1) in A2
    # for e in range(n_edge):
    #     task.putaij(con_A1[e], idx_y[e], +1.0)
    #     task.putaij(con_A1[e], idx_y_relax[e], -1.0)
    #     task.putaij(con_A1[e], idx_M[0], -1.0)
    #     task.putconbound(con_A1[e], mosek.boundkey.up, -0.0, 0.0)
    #
    #     task.putaij(con_A2[e], idx_y[e], -1.0)
    #     task.putaij(con_A2[e], idx_y_relax[e], +1.0)
    #     task.putaij(con_A2[e], idx_M[0], -1.0)
    #     task.putconbound(con_A2[e], mosek.boundkey.up, -0.0, 0.0)

    # (B) source/target linkage with Big-M (linear, no product)
    Mbig = 1e3  # 우선 고정; 나중에 M_large로 인자화 가능

    #   링크: (z - s) 상/하  → 2 * 5 * 2(발) = 20행/엣지
    #   수축: |z| <= M*y      → 2 * 5 * 2(발) = 20행/엣지
    #   합계: 40행/엣지
    con_B_src = add_cons(len(src_edges) * 40)
    row = 0
    for e in src_edges:
        # z_L 링크 (10행)
        for k in range(5):
            #  z_L - sL <= M*(1-y)  →  z_L - sL + M*y <= M
            task.putaij(con_B_src[row], idx_zL[e, k], +1.0)
            task.putaij(con_B_src[row], idx_sL[k], -1.0)
            task.putaij(con_B_src[row], idx_y[e], +Mbig)
            task.putconbound(con_B_src[row], mosek.boundkey.up, -0.0, Mbig);
            row += 1
            # -(z_L - sL) <= M*(1-y) → -z_L + sL + M*y <= M
            task.putaij(con_B_src[row], idx_zL[e, k], -1.0)
            task.putaij(con_B_src[row], idx_sL[k], +1.0)
            task.putaij(con_B_src[row], idx_y[e], +Mbig)
            task.putconbound(con_B_src[row], mosek.boundkey.up, -0.0, Mbig);
            row += 1

        # z_R 링크 (10행)
        for k in range(5):
            task.putaij(con_B_src[row], idx_zR[e, k], +1.0)
            task.putaij(con_B_src[row], idx_sR[k], -1.0)
            task.putaij(con_B_src[row], idx_y[e], +Mbig)
            task.putconbound(con_B_src[row], mosek.boundkey.up, -0.0, Mbig);
            row += 1

            task.putaij(con_B_src[row], idx_zR[e, k], -1.0)
            task.putaij(con_B_src[row], idx_sR[k], +1.0)
            task.putaij(con_B_src[row], idx_y[e], +Mbig)
            task.putconbound(con_B_src[row], mosek.boundkey.up, -0.0, Mbig);
            row += 1

        # 수축: |z_L| ≤ M*y (10행), |z_R| ≤ M*y (10행)
        for k in range(5):
            task.putaij(con_B_src[row], idx_zL[e, k], +1.0)
            task.putaij(con_B_src[row], idx_y[e], -Mbig)
            task.putconbound(con_B_src[row], mosek.boundkey.up, -0.0, 0.0);
            row += 1

            task.putaij(con_B_src[row], idx_zL[e, k], -1.0)
            task.putaij(con_B_src[row], idx_y[e], -Mbig)
            task.putconbound(con_B_src[row], mosek.boundkey.up, -0.0, 0.0);
            row += 1

            task.putaij(con_B_src[row], idx_zR[e, k], +1.0)
            task.putaij(con_B_src[row], idx_y[e], -Mbig)
            task.putconbound(con_B_src[row], mosek.boundkey.up, -0.0, 0.0);
            row += 1

            task.putaij(con_B_src[row], idx_zR[e, k], -1.0)
            task.putaij(con_B_src[row], idx_y[e], -Mbig)
            task.putconbound(con_B_src[row], mosek.boundkey.up, -0.0, 0.0);
            row += 1

    # sink도 동일하게 40행/엣지
    con_B_sink = add_cons(len(sink_edges) * 40)
    row = 0
    for e in sink_edges:
        for k in range(5):
            task.putaij(con_B_sink[row], idx_zpL[e, k], +1.0)
            task.putaij(con_B_sink[row], idx_tL[k], -1.0)
            task.putaij(con_B_sink[row], idx_y[e], +Mbig)
            task.putconbound(con_B_sink[row], mosek.boundkey.up, -0.0, Mbig);
            row += 1

            task.putaij(con_B_sink[row], idx_zpL[e, k], -1.0)
            task.putaij(con_B_sink[row], idx_tL[k], +1.0)
            task.putaij(con_B_sink[row], idx_y[e], +Mbig)
            task.putconbound(con_B_sink[row], mosek.boundkey.up, -0.0, Mbig);
            row += 1

        for k in range(5):
            task.putaij(con_B_sink[row], idx_zpR[e, k], +1.0)
            task.putaij(con_B_sink[row], idx_tR[k], -1.0)
            task.putaij(con_B_sink[row], idx_y[e], +Mbig)
            task.putconbound(con_B_sink[row], mosek.boundkey.up, -0.0, Mbig);
            row += 1

            task.putaij(con_B_sink[row], idx_zpR[e, k], -1.0)
            task.putaij(con_B_sink[row], idx_tR[k], +1.0)
            task.putaij(con_B_sink[row], idx_y[e], +Mbig)
            task.putconbound(con_B_sink[row], mosek.boundkey.up, -0.0, Mbig);
            row += 1

        for k in range(5):
            task.putaij(con_B_sink[row], idx_zpL[e, k], +1.0)
            task.putaij(con_B_sink[row], idx_y[e], -Mbig)
            task.putconbound(con_B_sink[row], mosek.boundkey.up, -0.0, 0.0);
            row += 1

            task.putaij(con_B_sink[row], idx_zpL[e, k], -1.0)
            task.putaij(con_B_sink[row], idx_y[e], -Mbig)
            task.putconbound(con_B_sink[row], mosek.boundkey.up, -0.0, 0.0);
            row += 1

            task.putaij(con_B_sink[row], idx_zpR[e, k], +1.0)
            task.putaij(con_B_sink[row], idx_y[e], -Mbig)
            task.putconbound(con_B_sink[row], mosek.boundkey.up, -0.0, 0.0);
            row += 1

            task.putaij(con_B_sink[row], idx_zpR[e, k], -1.0)
            task.putaij(con_B_sink[row], idx_y[e], -Mbig)
            task.putconbound(con_B_sink[row], mosek.boundkey.up, -0.0, 0.0);
            row += 1

    # (C) source/sink sum(y)=1
    con_src1 = add_cons(1)
    for e in src_edges:
        task.putaij(con_src1[0], idx_y[e], +1.0)
    task.putconbound(con_src1[0], mosek.boundkey.fx, 1.0, 1.0)

    con_sink1 = add_cons(1)
    for e in sink_edges:
        task.putaij(con_sink1[0], idx_y[e], +1.0)
    task.putconbound(con_sink1[0], mosek.boundkey.fx, 1.0, 1.0)

    # (D) 중간 vertex: in = out (y), 그리고 sum(zp(in)) = sum(z(out))
    cons_inout = []
    for v_id in range(1, vertex_number-1):
        in_idx  = np.where(flow_in_vertices  == v_id)[0]
        out_idx = np.where(flow_out_vertices == v_id)[0]

        # sum_y(in) - sum_y(out) = 0
        c = add_cons(1)
        for e in in_idx:  task.putaij(c[0], idx_y[e], +1.0)
        for e in out_idx: task.putaij(c[0], idx_y[e], -1.0)
        task.putconbound(c[0], mosek.boundkey.fx, 0.0, 0.0)
        cons_inout.append(c[0])

        # zL/zR 5개씩: sum(zp(in,k)) - sum(z(out,k)) = 0
        for k in range(5):
            cL = add_cons(1)
            for e in in_idx:  task.putaij(cL[0], idx_zpL[e, k], +1.0)
            for e in out_idx: task.putaij(cL[0], idx_zL[e, k],  -1.0)
            task.putconbound(cL[0], mosek.boundkey.fx, 0.0, 0.0)

            cR = add_cons(1)
            for e in in_idx:  task.putaij(cR[0], idx_zpR[e, k], +1.0)
            for e in out_idx: task.putaij(cR[0], idx_zR[e, k],  -1.0)
            task.putconbound(cR[0], mosek.boundkey.fx, 0.0, 0.0)

    # (E) 각 중간 vertex outflow sum(y) <= 1
    for v_id in range(1, vertex_number-1):
        out_idx = np.where(flow_out_vertices == v_id)[0]
        c = add_cons(1)
        for e in out_idx:
            task.putaij(c[0], idx_y[e], +1.0)
        task.putconbound(c[0], mosek.boundkey.up, -0.0, 1.0)

    # (F) 영역 포함 + z[2]=center_z*y  (모드별)
    for e in range(n_edge):
        from_v = flow_out_vertices[e]; to_v = flow_in_vertices[e]
        mode_from = mode_list[from_v]; mode_to = mode_list[to_v]

        A_from = const_struct[from_v]["A"]; b_from = const_struct[from_v]["b"]; z_from = const_struct[from_v]["center_z"]
        A_to   = const_struct[to_v]["A"];   b_to   = const_struct[to_v]["b"];   z_to   = const_struct[to_v]["center_z"]

        # helper: A * [x;y] <= b * y(e)
        def add_region_le(A, b, var_xy, y_e):
            m = A.shape[0]
            c = add_cons(m)
            for r in range(m):
                # A[r,0]*x + A[r,1]*y - b[r]*y_e <= 0
                task.putaij(c[r], var_xy[0], A[r,0])
                task.putaij(c[r], var_xy[1], A[r,1])
                task.putaij(c[r], y_e,      -float(b[r]))
                task.putconbound(c[r], mosek.boundkey.up, -0.0, 0.0)

        if mode_from == "D" and mode_to == "SL":
            add_region_le(A_to, b_to, (idx_zpL[e,0], idx_zpL[e,1]), idx_y[e])
            # z[2] = center_z * y
            c = add_cons(1)
            task.putaij(c[0], idx_zpL[e,2], +1.0)
            task.putaij(c[0], idx_y[e],     -z_to)
            task.putconbound(c[0], mosek.boundkey.fx, 0.0, 0.0)

        if mode_from == "D" and mode_to == "SR":
            add_region_le(A_to, b_to, (idx_zpR[e,0], idx_zpR[e,1]), idx_y[e])
            c = add_cons(1)
            task.putaij(c[0], idx_zpR[e,2], +1.0)
            task.putaij(c[0], idx_y[e],     -z_to)
            task.putconbound(c[0], mosek.boundkey.fx, 0.0, 0.0)

        if mode_from == "SL" and mode_to == "SR":
            add_region_le(A_from, b_from, (idx_zpL[e,0], idx_zpL[e,1]), idx_y[e])
            add_region_le(A_to,   b_to,   (idx_zpR[e,0], idx_zpR[e,1]), idx_y[e])
            for (var, zc) in [(idx_zpL[e,2], z_from), (idx_zpR[e,2], z_to)]:
                c = add_cons(1)
                task.putaij(c[0], var,   +1.0)
                task.putaij(c[0], idx_y[e], -zc)
                task.putconbound(c[0], mosek.boundkey.fx, 0.0, 0.0)

        if mode_from == "SR" and mode_to == "SL":
            add_region_le(A_from, b_from, (idx_zpR[e,0], idx_zpR[e,1]), idx_y[e])
            add_region_le(A_to,   b_to,   (idx_zpL[e,0], idx_zpL[e,1]), idx_y[e])
            for (var, zc) in [(idx_zpR[e,2], z_from), (idx_zpL[e,2], z_to)]:
                c = add_cons(1)
                task.putaij(c[0], var,   +1.0)
                task.putaij(c[0], idx_y[e], -zc)
                task.putconbound(c[0], mosek.boundkey.fx, 0.0, 0.0)

        if mode_from == "SL" and mode_to == "D":
            add_region_le(A_from, b_from, (idx_zL[e,0], idx_zL[e,1]), idx_y[e])
            c = add_cons(1)
            task.putaij(c[0], idx_zL[e,2], +1.0)
            task.putaij(c[0], idx_y[e],    -z_from)
            task.putconbound(c[0], mosek.boundkey.fx, 0.0, 0.0)

        if mode_from == "SR" and mode_to == "D":
            add_region_le(A_from, b_from, (idx_zR[e,0], idx_zR[e,1]), idx_y[e])
            c = add_cons(1)
            task.putaij(c[0], idx_zR[e,2], +1.0)
            task.putaij(c[0], idx_y[e],    -z_from)
            task.putconbound(c[0], mosek.boundkey.fx, 0.0, 0.0)

    # (G) |cos/sin| <= y  〈=〉  var - y <= 0,  -var - y <= 0
    for e in range(n_edge):
        for block in (idx_zL, idx_zR, idx_zpL, idx_zpR):
            for k in (3, 4):
                c1 = add_cons(1)
                task.putaij(c1[0], block[e, k], +1.0)
                task.putaij(c1[0], idx_y[e],    -1.0)
                task.putconbound(c1[0], mosek.boundkey.up, -0.0, 0.0)

                c2 = add_cons(1)
                task.putaij(c2[0], block[e, k], -1.0)
                task.putaij(c2[0], idx_y[e],    -1.0)
                task.putconbound(c2[0], mosek.boundkey.up, -0.0, 0.0)

    # ====== SOCP (H) step span & (H-1) relative foot ======
    # AFE를 누적해서 쓰자
    afe_base = 0
    task.appendafes(0)  # 초기화

    def ensure_afes(n_add):
        nonlocal afe_base
        cur = task.getnumafe()
        task.appendafes(n_add)
        return cur  # 추가 전 개수 = 새로 쓸 첫 row

    # step span: ||(zp - z)_{1:2}||_2 <= step_span * y
    for e in range(n_edge):
        t_aff = ([(step_span, idx_y[e])], 0.0)
        dxL   = ([(+1.0, idx_zpL[e,0]), (-1.0, idx_zL[e,0])], 0.0)
        dyL   = ([(+1.0, idx_zpL[e,1]), (-1.0, idx_zL[e,1])], 0.0)
        base = ensure_afes(3)
        add_soc2(task, t_aff, dxL, dyL, base)

        t_aff = ([(step_span, idx_y[e])], 0.0)
        dxR   = ([(+1.0, idx_zpR[e,0]), (-1.0, idx_zR[e,0])], 0.0)
        dyR   = ([(+1.0, idx_zpR[e,1]), (-1.0, idx_zR[e,1])], 0.0)
        base = ensure_afes(3)
        add_soc2(task, t_aff, dxR, dyR, base)

    # relative: ||(zp_L - z_R)_{1:2}|| <= 0.41 * y,  그리고 반대도
    rel = 0.41
    for e in range(n_edge):
        t_aff = ([(rel, idx_y[e])], 0.0)
        dx    = ([(+1.0, idx_zpL[e,0]), (-1.0, idx_zR[e,0])], 0.0)
        dy    = ([(+1.0, idx_zpL[e,1]), (-1.0, idx_zR[e,1])], 0.0)
        base  = ensure_afes(3)
        add_soc2(task, t_aff, dx, dy, base)

        t_aff = ([(rel, idx_y[e])], 0.0)
        dx    = ([(+1.0, idx_zpR[e,0]), (-1.0, idx_zL[e,0])], 0.0)
        dy    = ([(+1.0, idx_zpR[e,1]), (-1.0, idx_zL[e,1])], 0.0)
        base  = ensure_afes(3)
        add_soc2(task, t_aff, dx, dy, base)

    # ====== 목적함수 (MVP: step 수 최소화) ======
    # 0.1 * sum(y)
    for e in range(n_edge):
        task.putcj(idx_y[e], 0.1)

    build_time = time.perf_counter() - t0

    # ====== 인덱스 묶음 반환 ======
    idx.update(dict(
        n_edge=n_edge, vertex_number=vertex_number,
        y=idx_y, z_L=idx_zL, z_R=idx_zR, zp_L=idx_zpL, zp_R=idx_zpR,
        y_relax=idx_y_relax, M_scalar=idx_M,
        start_L=idx_sL, start_R=idx_sR, target_L=idx_tL, target_R=idx_tR,
        flow_out=flow_out_vertices, flow_in=flow_in_vertices,
        src_edges=src_edges, sink_edges=sink_edges,
    ))

    return task, idx, build_time
