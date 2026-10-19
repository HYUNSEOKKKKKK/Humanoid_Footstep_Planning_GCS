from __future__ import annotations
import time
import numpy as np
from src.graph import centroid_and_scaled_vertices
from src.graph.geometry import rect_A_b_from_vertices
from src.graph.centroid import compute_edge_directions
import mosek
from collections import namedtuple


OptModel = namedtuple("OptModel", ["env", "task", "idx", "const", "params"])

def build_gcs_model_VAE_optapi(v, edge, step_span, alpha, mode_list, x_ref, y_ref):
    """
    Fusion API 버전의 build_gcs_model_short 중 '변수/파라미터 정의' 구간을ㅇㄹㅇ
    MOSEK Optimizer API(Task) 형태로 1:1 대응시켜 생성합니다.

    반환:
      OptModel(env, task, idx, const, params)
        - idx: 각 변수의 열 인덱스(or 범위), barvar 핸들 모음
        - const: centroid/convex region 등 고정 상수들
        - params: '파라미터' 값을 담아두는 dict (Optimizer API에는 parameter 개념 없음)
    """
    t0 = time.perf_counter()

    # ==== 입력 전처리 (Fusion 코드와 동일) ====
    edge = edge.astype(int) - 1
    n_edge = int(edge.shape[0])
    vertex_number = int(max(edge[:, 0].max(), edge[:, 1].max())) + 1
    ca = float(np.cos(alpha))

    # Convex region
    centroid, v_scaled = centroid_and_scaled_vertices(v, scale=1.0)
    centroid_xy = centroid[:2, :].T
    u_ref_x, u_ref_y, has_ref = compute_edge_directions(
        centroid_xy, edge, eps=1e-8, assume_zero_based=True
    )

    def rect_halfspace_from_corners_xy(corners_xy: np.ndarray):
        g = corners_xy.mean(axis=0)  # (2,)
        P = corners_xy - g  # (4,2)

        # PCA로 축 추정
        C = P.T @ P
        w, V = np.linalg.eigh(C)
        u = V[:, np.argmax(w)]
        u = u / np.linalg.norm(u)
        v = np.array([-u[1], u[0]])
        R_T = np.vstack([u, v])  # (2,2)

        local = (R_T @ P.T)  # (2,4)
        hx = float(np.max(np.abs(local[0, :])))
        hy = float(np.max(np.abs(local[1, :])))

        A0 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float)
        b = np.array([hx, hx, hy, hy], float)

        A = A0 @ R_T  # (4,2)  (world frame halfspaces)
        b_shift = b + A @ g  # (4,)
        return A, b_shift

    const_struct = []
    for ii, vtx_world in enumerate(v):
        Vw = np.asarray(vtx_world)
        if Vw.shape == (4, 3): Vw = Vw.T
        corners_xy = Vw[:2, :].T  # (4,2)

        A, b_shift = rect_halfspace_from_corners_xy(corners_xy)
        const_struct.append({"A": A, "b": b_shift, "center_z": float(centroid[2, ii])})

    # ===== (for RLT) global bounds from all vertices =====
    all_xy = []
    for vv in v:  # v는 원래 입력 v (회전 후 vertex list)
        vv = np.asarray(vv)
        if vv.shape == (4, 3):
            vv = vv.T
        all_xy.append(vv[:2, :].T)  # (4,2)
    all_xy = np.vstack(all_xy)  # (4*V, 2)

    x_min, y_min = np.min(all_xy[:, 0]), np.min(all_xy[:, 1])
    x_max, y_max = np.max(all_xy[:, 0]), np.max(all_xy[:, 1])

    pad = 0.10  # 여유 (m). 필요하면 튜닝
    xL, xU = float(x_min - pad), float(x_max + pad)
    yL, yU = float(y_min - pad), float(y_max + pad)

    # Fusion의 BOUNDS = [(x,y,cg,sg,xp,yp)]
    BOUNDS7 = [
        (xL, xU),  # x
        (yL, yU),  # y
        (-1.0, 1.0),  # cg
        (-1.0, 1.0),  # sg
        (xL, xU),  # xp
        (yL, yU),  # yp
    ]

    # Fusion의 Matrix.dense 대신 numpy로 들고 있음
    XREF = np.array([float(x_ref[0]), float(x_ref[1])], dtype=float).reshape(2, 1)
    YREF = np.array([float(y_ref[0]), float(y_ref[1])], dtype=float).reshape(2, 1)

    # 플로우 인/아웃 (0-based 유지)
    flow_out_vertices = edge[:, 0].astype(int)
    flow_in_vertices  = edge[:, 1].astype(int)

    # ==== Optimizer API 시작 ====
    env = mosek.Env()
    task = env.Task(0, 0)

    # 가독성을 위한 헬퍼: 연속 열변수 블록 할당기
    def alloc_block(ncols, bound_type="unbounded", lb=None, ub=None):
        """
        ncols개의 열변수를 뒤에 추가하고, 경계 설정 후
        (start, end_exclusive) 튜플을 반환.
        """
        start = task.getnumvar()
        task.appendvars(ncols)
        idx = list(range(start, start + ncols))

        if bound_type == "unbounded":
            # (-inf, +inf)
            bk = [mosek.boundkey.fr] * ncols
            bl = [-0.0] * ncols   # 무시됨
            bu = [+0.0] * ncols   # 무시됨
        elif bound_type == "range":
            assert lb is not None and ub is not None
            bk = [mosek.boundkey.ra] * ncols
            bl = [lb] * ncols
            bu = [ub] * ncols
        elif bound_type == "lower":
            assert lb is not None
            bk = [mosek.boundkey.lo] * ncols
            bl = [lb] * ncols
            bu = [+0.0] * ncols  # 무시됨
        else:
            raise ValueError("unknown bound_type")

        task.putvarboundlist(idx, bk, bl, bu)
        return start, start + ncols  # 파이썬 슬라이스처럼 [start:end)

    idx = {}

    # --- y \in [0,1]^E ---
    idx["y"] = slice(*alloc_block(n_edge, bound_type="range", lb=0.0, ub=1.0))

    # --- z_L, z_R, zp_L, zp_R \in R^{E x 5} (unbounded) ---
    n5 = n_edge * 5
    idx["z_L"]  = slice(*alloc_block(n5, bound_type="unbounded"))
    idx["z_R"]  = slice(*alloc_block(n5, bound_type="unbounded"))
    idx["zp_L"] = slice(*alloc_block(n5, bound_type="unbounded"))
    idx["zp_R"] = slice(*alloc_block(n5, bound_type="unbounded"))

    # --- rho_* \ge 0 ---
    idx["rho_L"]  = slice(*alloc_block(n_edge, bound_type="lower", lb=0.0))
    idx["rho_R"]  = slice(*alloc_block(n_edge, bound_type="lower", lb=0.0))
    idx["rhoz_L"] = slice(*alloc_block(n_edge, bound_type="lower", lb=0.0))
    idx["rhoz_R"] = slice(*alloc_block(n_edge, bound_type="lower", lb=0.0))

    # ==== PSD 변수(Bar Vars) ====
    # Fusion:  for e in range(E): X_L_e ∈ S_+^{5}
    # Optimizer: 각 e마다 barvar(dim) 하나씩 생성 → barvar 인덱스 저장
    bar_X_L  = []
    bar_X_R  = []
    bar_X_L2 = []
    bar_X_R2 = []

    # 참고: appendbarvars에 리스트로 여러 개 한꺼번에 추가도 가능하지만,
    # 여기선 e별로 인덱스를 따로 들고 가기 위해 루프 사용
    for _ in range(n_edge):
        # 5x5
        task.appendbarvars([5])
        bar_X_L.append(task.getnumbarvar() - 1)
    for _ in range(n_edge):
        task.appendbarvars([5])
        bar_X_R.append(task.getnumbarvar() - 1)
    for _ in range(n_edge):
        # 7x7
        task.appendbarvars([7])
        bar_X_L2.append(task.getnumbarvar() - 1)
    for _ in range(n_edge):
        task.appendbarvars([7])
        bar_X_R2.append(task.getnumbarvar() - 1)

    idx["X_L_bar"]  = bar_X_L
    idx["X_R_bar"]  = bar_X_R
    idx["X_L2_bar"] = bar_X_L2
    idx["X_R2_bar"] = bar_X_R2

    # ==== '파라미터' 대용 컨테이너 ====
    # Optimizer API에는 parameter 타입이 없으므로 값을 저장해 두었다가
    # 제약/목적 계수에 반영할 때 사용하세요.
    params = {
        "start_L":  np.zeros(5, dtype=float),
        "start_R":  np.zeros(5, dtype=float),
        "target_L": np.zeros(5, dtype=float),
        "target_R": np.zeros(5, dtype=float),
        "y_relax":  np.zeros(n_edge, dtype=float),
        "M_scalar": 0.0,
    }

    # ==== 상수(후속 단계에서 재사용할 정보) ====
    const = {
        "t0": t0,
        "n_edge": n_edge,
        "vertex_number": vertex_number,
        "ca": ca,
        "centroid": centroid,
        "v_scaled": v_scaled,
        "const_struct": const_struct,
        "XREF": XREF,
        "YREF": YREF,
        "flow_out_vertices": flow_out_vertices,
        "flow_in_vertices": flow_in_vertices,
        "u_ref_x": u_ref_x,
        "u_ref_y": u_ref_y,
        "has_ref": has_ref,
        "step_span": step_span,
        "mode_list": mode_list,
    }
    # ==== bar var 키 이름 통일 ====
    idx["X_L"]  = idx.pop("X_L_bar")
    idx["X_R"]  = idx.pop("X_R_bar")
    idx["X_L2"] = idx.pop("X_L2_bar")
    idx["X_R2"] = idx.pop("X_R2_bar")

    # ==== V/IDX/PAR 구성 ====
    V = dict(
        y=idx["y"],
        z_L=idx["z_L"], z_R=idx["z_R"], zp_L=idx["zp_L"], zp_R=idx["zp_R"],
        rho_L=idx["rho_L"], rho_R=idx["rho_R"], rhoz_L=idx["rhoz_L"], rhoz_R=idx["rhoz_R"],
        X_L=idx["X_L"], X_R=idx["X_R"], X_L2=idx["X_L2"], X_R2=idx["X_R2"],

        # Fusion에선 parameter였던 것들: Optimizer API에선 값을 나중에 set_parameters에서 계수/경계로 반영
        start_L=None, start_R=None, target_L=None, target_R=None,
        y_relax=None, M_scalar=None,

        # 제약/코스트에서 나중에 갱신할 '행/열 인덱스'를 저장할 메타
        meta={}
    )

    IDX = dict(n_edge=n_edge, vertex_number=vertex_number)
    PAR = dict(centroid=centroid)

    # (참고) task를 M으로 사용하면, 이후 코드에서 Fusion과 동일 시그니처 유지 가능
    M = task

    build_time = time.perf_counter() - t0

    # 필요하면 여기서 바로 (A)(B)(C) 제약을 이어서 추가하고,
    # 마지막에 아래 형태로 반환:
    # return M, V, IDX, PAR, build_time

    # =========================
    # Constraints (A), (B), (C)
    # =========================

    def _slice_to_idx_list(slc: slice):
        return list(range(slc.start, slc.stop))

    def _ek_to_col(base_start: int, e: int, k: int, width: int = 5) -> int:
        """(e,k) -> 납작한 열 인덱스"""
        return base_start + e * width + k

    # 편의 핸들
    y_sl = V["y"]
    zL_sl = V["z_L"];
    zR_sl = V["z_R"]
    zpL_sl = V["zp_L"];
    zpR_sl = V["zp_R"]

    y_cols = _slice_to_idx_list(y_sl)

    n_edge = IDX["n_edge"]
    vertex_number = IDX["vertex_number"]

    # 0-based edge arrays (이미 build 앞쪽에서 만든 값 사용)
    src_edges = np.where(flow_out_vertices == 0)[0].astype(int)
    sink_edges = np.where(flow_in_vertices == vertex_number - 1)[0].astype(int)

    # 메타 저장소
    if "meta" not in V:
        V["meta"] = {}
    meta = V["meta"]

    # ----------------------------------------------------------------------
    # (A) || y - y_relax ||_inf <= M_scalar
    #  ->  y_e <= y_relax_e + M_scalar
    #  -> -y_e <= -(y_relax_e - M_scalar)
    # 여기선 계수(±1)는 고정, ub만 나중에 set_parameters_optapi에서 갱신
    # ----------------------------------------------------------------------
    A_rows_upper = []  # +y_e <= ub1_e
    A_rows_lower = []  # -y_e <= ub2_e

    for e, col in enumerate(y_cols):
        # (1) +y_e <= ub1_e
        M.appendcons(1)
        r1 = M.getnumcon() - 1
        M.putaij(r1, col, 1.0)
        # ub는 자리만 잡아둠(이후 갱신). (-inf, +big)로 우선 설정
        M.putconbound(r1, mosek.boundkey.up, -0.0, +1e30)
        A_rows_upper.append(r1)

        # (2) -y_e <= ub2_e
        M.appendcons(1)
        r2 = M.getnumcon() - 1
        M.putaij(r2, col, -1.0)
        M.putconbound(r2, mosek.boundkey.up, -0.0, +1e30)
        A_rows_lower.append(r2)

    meta["A_rows_upper"] = A_rows_upper
    meta["A_rows_lower"] = A_rows_lower
    meta["A_type"] = "inf_norm_y_minus_param"  # 참고 태그

    # ----------------------------------------------------------------------
    # (B) source/target flow
    #  - source: z_L(e,k) = y_e * start_L[k],  z_R(e,k) = y_e * start_R[k]
    #  - sink:   zp_L(e,k) = y_e * target_L[k],  zp_R(e,k) = y_e * target_R[k]
    # 여기선 행을 만들고, y 계수는 나중에 set_parameters_optapi에서
    #  -start_L[k], -start_R[k], -target_L[k], -target_R[k] 로 갱신
    # ----------------------------------------------------------------------
    B_rows_src_zL = []  # list of list: [ [row(e,0),...,row(e,4)], ... ]
    B_rows_src_zR = []
    for e in src_edges:
        rowsL = []
        rowsR = []
        y_col = y_sl.start + e
        for k in range(5):
            # z_L(e,k) - start_L[k] * y_e = 0
            M.appendcons(1)
            r = M.getnumcon() - 1
            zcol = _ek_to_col(zL_sl.start, e, k, width=5)
            M.putaij(r, zcol, 1.0)
            M.putaij(r, y_col, 0.0)  # 나중에 -start_L[k] 로 갱신
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)
            rowsL.append(r)

            # z_R(e,k) - start_R[k] * y_e = 0
            M.appendcons(1)
            r = M.getnumcon() - 1
            zcol = _ek_to_col(zR_sl.start, e, k, width=5)
            M.putaij(r, zcol, 1.0)
            M.putaij(r, y_col, 0.0)  # 나중에 -start_R[k] 로 갱신
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)
            rowsR.append(r)
        B_rows_src_zL.append(rowsL)
        B_rows_src_zR.append(rowsR)

    B_rows_sink_zpL = []
    B_rows_sink_zpR = []
    for e in sink_edges:
        rowsL = []
        rowsR = []
        y_col = y_sl.start + e
        for k in range(5):
            # zp_L(e,k) - target_L[k] * y_e = 0
            M.appendcons(1)
            r = M.getnumcon() - 1
            zcol = _ek_to_col(zpL_sl.start, e, k, width=5)
            M.putaij(r, zcol, 1.0)
            M.putaij(r, y_col, 0.0)  # 나중에 -target_L[k] 로 갱신
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)
            rowsL.append(r)

            # zp_R(e,k) - target_R[k] * y_e = 0
            M.appendcons(1)
            r = M.getnumcon() - 1
            zcol = _ek_to_col(zpR_sl.start, e, k, width=5)
            M.putaij(r, zcol, 1.0)
            M.putaij(r, y_col, 0.0)  # 나중에 -target_R[k] 로 갱신
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)
            rowsR.append(r)
        B_rows_sink_zpL.append(rowsL)
        B_rows_sink_zpR.append(rowsR)

    meta["src_edges"] = src_edges
    meta["sink_edges"] = sink_edges
    meta["B_rows_src_zL"] = B_rows_src_zL
    meta["B_rows_src_zR"] = B_rows_src_zR
    meta["B_rows_sink_zpL"] = B_rows_sink_zpL
    meta["B_rows_sink_zpR"] = B_rows_sink_zpR
    meta["y_col_start"] = y_sl.start  # y 열 시작 저장(갱신 시 편리)
    meta.setdefault("J_src_rows_L_X01", [])  # XL: X[0,1] = c_g (source)
    meta.setdefault("J_src_rows_L_X02", [])  # XL: X[0,2] = s_g (source)
    meta.setdefault("J_src_rows_R_X01", [])  # XR: X[0,1] = c_g (source)
    meta.setdefault("J_src_rows_R_X02", [])  # XR: X[0,2] = s_g (source)
    # ----------------------------------------------------------------------
    # (C) Source and Target vertex Flow conservation
    #    sum_{e in src} y_e = 1,   sum_{e in sink} y_e = 1
    # ----------------------------------------------------------------------
    # src
    M.appendcons(1)
    r_src = M.getnumcon() - 1
    if len(src_edges) > 0:
        cols = [y_sl.start + int(e) for e in src_edges]
        vals = [1.0] * len(cols)
        M.putaijlist([r_src] * len(cols), cols, vals)
    M.putconbound(r_src, mosek.boundkey.fx, 1.0, 1.0)

    # sink
    M.appendcons(1)
    r_sink = M.getnumcon() - 1
    if len(sink_edges) > 0:
        cols = [y_sl.start + int(e) for e in sink_edges]
        vals = [1.0] * len(cols)
        M.putaijlist([r_sink] * len(cols), cols, vals)
    M.putconbound(r_sink, mosek.boundkey.fx, 1.0, 1.0)

    # =========================
    # Constraints (D), (E)
    # =========================

    # (D) In = Out flow conservation (middle vertices)
    for v_id in range(1, vertex_number - 1):
        in_idx  = np.where(flow_in_vertices  == v_id)[0].astype(int)
        out_idx = np.where(flow_out_vertices == v_id)[0].astype(int)

        # (D-1) sum_y(in) - sum_y(out) = 0
        if in_idx.size > 0 or out_idx.size > 0:
            M.appendcons(1)
            r = M.getnumcon() - 1
            # +1 * y_in
            if in_idx.size > 0:
                cols = [y_sl.start + int(e) for e in in_idx]
                vals = [1.0] * len(cols)
                M.putaijlist([r]*len(cols), cols, vals)
            # -1 * y_out
            if out_idx.size > 0:
                cols = [y_sl.start + int(e) for e in out_idx]
                vals = [-1.0] * len(cols)
                M.putaijlist([r]*len(cols), cols, vals)
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # (D-2) sum(zp_L(in,k)) - sum(z_L(out,k)) = 0  (k=0..4)
        #       sum(zp_R(in,k)) - sum(z_R(out,k)) = 0
        for k in range(5):
            # L쪽
            if in_idx.size > 0 or out_idx.size > 0:
                M.appendcons(1)
                rL = M.getnumcon() - 1
                # +1 * zp_L(in,k)
                if in_idx.size > 0:
                    cols = [_ek_to_col(zpL_sl.start, int(e), k, width=5) for e in in_idx]
                    vals = [1.0] * len(cols)
                    M.putaijlist([rL]*len(cols), cols, vals)
                # -1 * z_L(out,k)
                if out_idx.size > 0:
                    cols = [_ek_to_col(zL_sl.start, int(e), k, width=5) for e in out_idx]
                    vals = [-1.0] * len(cols)
                    M.putaijlist([rL]*len(cols), cols, vals)
                M.putconbound(rL, mosek.boundkey.fx, 0.0, 0.0)

            # R쪽
            if in_idx.size > 0 or out_idx.size > 0:
                M.appendcons(1)
                rR = M.getnumcon() - 1
                # +1 * zp_R(in,k)
                if in_idx.size > 0:
                    cols = [_ek_to_col(zpR_sl.start, int(e), k, width=5) for e in in_idx]
                    vals = [1.0] * len(cols)
                    M.putaijlist([rR]*len(cols), cols, vals)
                # -1 * z_R(out,k)
                if out_idx.size > 0:
                    cols = [_ek_to_col(zR_sl.start, int(e), k, width=5) for e in out_idx]
                    vals = [-1.0] * len(cols)
                    M.putaijlist([rR]*len(cols), cols, vals)
                M.putconbound(rR, mosek.boundkey.fx, 0.0, 0.0)

    # (E) outflow of y (middle vertices): sum_y(out) <= 1
    for v_id in range(1, vertex_number - 1):
        out_idx = np.where(flow_out_vertices == v_id)[0].astype(int)
        if out_idx.size > 0:
            M.appendcons(1)
            r = M.getnumcon() - 1
            cols = [y_sl.start + int(e) for e in out_idx]
            vals = [1.0] * len(cols)
            M.putaijlist([r]*len(cols), cols, vals)
            # sum(out) <= 1.0  → up bound
            M.putconbound(r, mosek.boundkey.up, -0.0, 1.0)
        else:
            # out이 비어도 0 <= 1 제약은 항상 참이니 생략 가능(원하면 고정행 추가해도 됨)
            pass
    # =========================
    # Constraints (F): Inside Convex region (depends on from/to mode)
    # =========================

    def _add_region_membership_rows(M, A_mat, b_vec, xcol, ycol, y_e_col):
        """
        A_mat @ [x;y] - b * y_e <= 0  (각 행별로)
        xcol, ycol: (해당 엣지 e의) x,y 변수 열 인덱스
        y_e_col:    y_e 열 인덱스
        """
        m = A_mat.shape[0]
        for i in range(m):
            M.appendcons(1)
            r = M.getnumcon() - 1
            a1 = float(A_mat[i, 0]); a2 = float(A_mat[i, 1]); bi = float(b_vec[i])
            M.putaij(r, xcol, a1)
            M.putaij(r, ycol, a2)
            M.putaij(r, y_e_col, -bi)
            # A*[x;y] - b*y_e <= 0  → upper bound 0
            M.putconbound(r, mosek.boundkey.up, -0.0, 0.0)

    def _add_z_equality(M, zcol, y_e_col, z_const):
        """
        z - z_const * y_e = 0
        """
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, zcol, 1.0)
        M.putaij(r, y_e_col, -float(z_const))
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

    for e in range(n_edge):
        from_v = int(flow_out_vertices[e]);  to_v = int(flow_in_vertices[e])
        mode_from = mode_list[from_v];       mode_to = mode_list[to_v]

        A_from = const_struct[from_v]["A"]; b_from = const_struct[from_v]["b"]; z_from = const_struct[from_v]["center_z"]
        A_to   = const_struct[to_v]["A"];   b_to   = const_struct[to_v]["b"];   z_to   = const_struct[to_v]["center_z"]

        y_e_col = y_sl.start + e

        # --- 모드 조합별 처리 ---

        # D → SL : (to)의 SL 좌표 zp_L(e,0:1) ∈ Poly(to),  zp_L(e,2) = z_to * y_e
        if mode_from == "D" and mode_to == "SL":
            xcol, ycol = _ek_to_col(zpL_sl.start, e, 0, 5), _ek_to_col(zpL_sl.start, e, 1, 5)
            _add_region_membership_rows(M, A_to, b_to, xcol, ycol, y_e_col)
            zcol = _ek_to_col(zpL_sl.start, e, 2, 5)
            _add_z_equality(M, zcol, y_e_col, z_to)

        # D → SR : (to)의 SR 좌표 zp_R(e,0:1) ∈ Poly(to),  zp_R(e,2) = z_to * y_e
        if mode_from == "D" and mode_to == "SR":
            xcol, ycol = _ek_to_col(zpR_sl.start, e, 0, 5), _ek_to_col(zpR_sl.start, e, 1, 5)
            _add_region_membership_rows(M, A_to, b_to, xcol, ycol, y_e_col)
            zcol = _ek_to_col(zpR_sl.start, e, 2, 5)
            _add_z_equality(M, zcol, y_e_col, z_to)

        # SL → SR : (from)의 L 좌표 zp_L(e,0:1) ∈ Poly(from), z = z_from*y_e
        #           (to)의   R 좌표 zp_R(e,0:1) ∈ Poly(to),   z = z_to*y_e
        if mode_from == "SL" and mode_to == "SR":
            # from 쪽 (L)
            xcol, ycol = _ek_to_col(zpL_sl.start, e, 0, 5), _ek_to_col(zpL_sl.start, e, 1, 5)
            _add_region_membership_rows(M, A_from, b_from, xcol, ycol, y_e_col)
            zcol = _ek_to_col(zpL_sl.start, e, 2, 5)
            _add_z_equality(M, zcol, y_e_col, z_from)
            # to 쪽 (R)
            xcol, ycol = _ek_to_col(zpR_sl.start, e, 0, 5), _ek_to_col(zpR_sl.start, e, 1, 5)
            _add_region_membership_rows(M, A_to, b_to, xcol, ycol, y_e_col)
            zcol = _ek_to_col(zpR_sl.start, e, 2, 5)
            _add_z_equality(M, zcol, y_e_col, z_to)

        # SR → SL : (from)의 R 좌표 zp_R(e,0:1) ∈ Poly(from), z = z_from*y_e
        #           (to)의   L 좌표 zp_L(e,0:1) ∈ Poly(to),   z = z_to*y_e
        if mode_from == "SR" and mode_to == "SL":
            # from 쪽 (R)
            xcol, ycol = _ek_to_col(zpR_sl.start, e, 0, 5), _ek_to_col(zpR_sl.start, e, 1, 5)
            _add_region_membership_rows(M, A_from, b_from, xcol, ycol, y_e_col)
            zcol = _ek_to_col(zpR_sl.start, e, 2, 5)
            _add_z_equality(M, zcol, y_e_col, z_from)
            # to 쪽 (L)
            xcol, ycol = _ek_to_col(zpL_sl.start, e, 0, 5), _ek_to_col(zpL_sl.start, e, 1, 5)
            _add_region_membership_rows(M, A_to, b_to, xcol, ycol, y_e_col)
            zcol = _ek_to_col(zpL_sl.start, e, 2, 5)
            _add_z_equality(M, zcol, y_e_col, z_to)

        # SL → D : (from)의 L 좌표 z_L(e,0:1) ∈ Poly(from), z = z_from*y_e
        if mode_from == "SL" and mode_to == "D":
            xcol, ycol = _ek_to_col(zL_sl.start, e, 0, 5), _ek_to_col(zL_sl.start, e, 1, 5)
            _add_region_membership_rows(M, A_from, b_from, xcol, ycol, y_e_col)
            zcol = _ek_to_col(zL_sl.start, e, 2, 5)
            _add_z_equality(M, zcol, y_e_col, z_from)

        # SR → D : (from)의 R 좌표 z_R(e,0:1) ∈ Poly(from), z = z_from*y_e
        if mode_from == "SR" and mode_to == "D":
            xcol, ycol = _ek_to_col(zR_sl.start, e, 0, 5), _ek_to_col(zR_sl.start, e, 1, 5)
            _add_region_membership_rows(M, A_from, b_from, xcol, ycol, y_e_col)
            zcol = _ek_to_col(zR_sl.start, e, 2, 5)
            _add_z_equality(M, zcol, y_e_col, z_from)


    # =========================
    # Constraints (G), (H)
    # =========================

    # 핸들
    y_sl   = V["y"]
    zL_sl  = V["z_L"];   zR_sl  = V["z_R"]
    zpL_sl = V["zp_L"];  zpR_sl = V["zp_R"]

    n_edge = IDX["n_edge"]

    # -------------------------
    # (G) z(3:4): |var| <= y
    #    var - y <= 0,  -var - y <= 0
    # -------------------------
    for e in range(n_edge):
        y_col = y_sl.start + e
        for var_sl in (zL_sl, zR_sl, zpL_sl, zpR_sl):
            for k in (3, 4):  # cos,sin
                vcol = _ek_to_col(var_sl.start, e, k, width=5)
                # var - y <= 0
                M.appendcons(1)
                r1 = M.getnumcon() - 1
                M.putaij(r1, vcol,  1.0)
                M.putaij(r1, y_col, -1.0)
                M.putconbound(r1, mosek.boundkey.up, -0.0, 0.0)
                # -var - y <= 0   (== -(var + y) <= 0)
                M.appendcons(1)
                r2 = M.getnumcon() - 1
                M.putaij(r2, vcol, -1.0)
                M.putaij(r2, y_col, -1.0)
                M.putconbound(r2, mosek.boundkey.up, -0.0, 0.0)

    # -------------------------
    # (H) step span:
    #   || (zp - z)_{1:2} ||_2 <= step_span * y
    #   → 보조변수 t, dx, dy 생성 후
    #     t = step_span * y,  dx = (zp_x - z_x),  dy = (zp_y - z_y)
    #     (t, dx, dy) \in QuadCone
    #   L, R 각각 추가
    # -------------------------

    # 보조변수 블록 할당기 (네 alloc_block을 그대로 사용)
    def _alloc_block(ncols, bound_type="unbounded", lb=None, ub=None):
        start = M.getnumvar()
        M.appendvars(ncols)
        idxs = list(range(start, start + ncols))
        if bound_type == "unbounded":
            bk = [mosek.boundkey.fr] * ncols; bl = [-0.0]*ncols; bu = [+0.0]*ncols
        elif bound_type == "lower":
            bk = [mosek.boundkey.lo] * ncols; bl = [lb]*ncols; bu = [+0.0]*ncols
        elif bound_type == "range":
            bk = [mosek.boundkey.ra] * ncols; bl = [lb]*ncols; bu = [ub]*ncols
        else:
            raise ValueError("unknown bound_type")
        M.putvarboundlist(idxs, bk, bl, bu)
        return start, start + ncols

    def _alloc_block_local(M, ncols, bound_type="unbounded", lb=None, ub=None):
        start = M.getnumvar()
        M.appendvars(ncols)
        idxs = list(range(start, start + ncols))
        if bound_type == "unbounded":
            bk = [mosek.boundkey.fr] * ncols; bl = [-0.0]*ncols; bu = [+0.0]*ncols
        elif bound_type == "lower":
            bk = [mosek.boundkey.lo] * ncols; bl = [lb]*ncols; bu = [+0.0]*ncols
        elif bound_type == "range":
            bk = [mosek.boundkey.ra] * ncols; bl = [lb]*ncols; bu = [ub]*ncols
        else:
            raise ValueError("unknown bound_type")
        M.putvarboundlist(idxs, bk, bl, bu)
        return slice(start, start + ncols)

    # 각 edge마다 L/R용 보조변수(t,dx,dy) 3개씩
    tL_sl  = slice(*_alloc_block(n_edge, bound_type="lower", lb=0.0))  # t >= 0 (관례)
    dLx_sl = slice(*_alloc_block(n_edge, bound_type="unbounded"))
    dLy_sl = slice(*_alloc_block(n_edge, bound_type="unbounded"))

    tR_sl  = slice(*_alloc_block(n_edge, bound_type="lower", lb=0.0))
    dRx_sl = slice(*_alloc_block(n_edge, bound_type="unbounded"))
    dRy_sl = slice(*_alloc_block(n_edge, bound_type="unbounded"))

    # (H-1) 보조변수와 원 변수 연결: 등식 제약
    for e in range(n_edge):
        y_col = y_sl.start + e

        # --- L ---
        # tL_e - step_span*y_e = 0
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, tL_sl.start + e, 1.0)
        M.putaij(r, y_col, -float(step_span))
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # dLx_e - (zpL_x - zL_x) = 0
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, dLx_sl.start + e, 1.0)
        M.putaij(r, _ek_to_col(zpL_sl.start, e, 0, 5), -1.0)
        M.putaij(r, _ek_to_col(zL_sl.start,  e, 0, 5),  1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # dLy_e - (zpL_y - zL_y) = 0
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, dLy_sl.start + e, 1.0)
        M.putaij(r, _ek_to_col(zpL_sl.start, e, 1, 5), -1.0)
        M.putaij(r, _ek_to_col(zL_sl.start,  e, 1, 5),  1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # --- R ---
        # tR_e - step_span*y_e = 0
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, tR_sl.start + e, 1.0)
        M.putaij(r, y_col, -float(step_span))
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # dRx_e - (zpR_x - zR_x) = 0
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, dRx_sl.start + e, 1.0)
        M.putaij(r, _ek_to_col(zpR_sl.start, e, 0, 5), -1.0)
        M.putaij(r, _ek_to_col(zR_sl.start,  e, 0, 5),  1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # dRy_e - (zpR_y - zR_y) = 0
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, dRy_sl.start + e, 1.0)
        M.putaij(r, _ek_to_col(zpR_sl.start, e, 1, 5), -1.0)
        M.putaij(r, _ek_to_col(zR_sl.start,  e, 1, 5),  1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

    # ===== extra lateral ordering constraint (from Fusion) =====
    # (z_R.y - z_L.y) + 0.2*y <= 0
    for e in range(n_edge):
        M.appendcons(1)
        r = M.getnumcon() - 1
        # z_R(e,1) - z_L(e,1) + 0.2*y_e <= 0
        M.putaij(r, _ek_to_col(zR_sl.start, e, 1, 5), 1.0)
        M.putaij(r, _ek_to_col(zL_sl.start, e, 1, 5), -1.0)
        M.putaij(r, y_sl.start + e, 0.2)
        M.putconbound(r, mosek.boundkey.up, -0.0, 0.0)



    # (H-2) 쿼드콘 제약:  (t, dx, dy) ∈ Q^3
    # MOSEK: apex=0.0, 변수 인덱스 리스트로 추가
    for e in range(n_edge):
        M.appendcone(mosek.conetype.quad, 0.0,
                     [tL_sl.start + e, dLx_sl.start + e, dLy_sl.start + e])
        M.appendcone(mosek.conetype.quad, 0.0,
                     [tR_sl.start + e, dRx_sl.start + e, dRy_sl.start + e])

    # =========================
    # (H-1) relative foot SOC  +  (I) inward (TORO) constraints
    # =========================

    y_sl = V["y"]
    zL_sl = V["z_L"];
    zR_sl = V["z_R"]
    zpL_sl = V["zp_L"];
    zpR_sl = V["zp_R"]
    n_edge = IDX["n_edge"]

    # --- 보조변수: t_rL, dx_rL, dy_rL  /  t_rR, dx_rR, dy_rR
    t_rL_sl = _alloc_block_local(M, n_edge, bound_type="lower", lb=0.0)  # t >= 0
    dx_rL_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")
    dy_rL_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")

    t_rR_sl = _alloc_block_local(M, n_edge, bound_type="lower", lb=0.0)
    dx_rR_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")
    dy_rR_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")

    REL_SPAN = 0.35  # 주석엔 0.32였지만 기존 코드 값(0.35)을 따름
    SIN_COL = 4  # [x,y,z,cos,sin]에서 sin의 0-based index

    for e in range(n_edge):
        ycol = y_sl.start + e

        # --- left-relative: || [zpL_x - zR_x, zpL_y - zR_y] ||_2 <= 0.35 * y_e
        # t_rL_e - 0.35*y_e = 0
        M.appendcons(1);
        r = M.getnumcon() - 1
        M.putaij(r, t_rL_sl.start + e, 1.0)
        M.putaij(r, ycol, -REL_SPAN)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # dx_rL_e - (zpL_x - zR_x) = 0
        M.appendcons(1);
        r = M.getnumcon() - 1
        M.putaij(r, dx_rL_sl.start + e, 1.0)
        M.putaij(r, _ek_to_col(zpL_sl.start, e, 0, 5), -1.0)
        M.putaij(r, _ek_to_col(zR_sl.start, e, 0, 5), 1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # dy_rL_e - (zpL_y - zR_y) = 0
        M.appendcons(1);
        r = M.getnumcon() - 1
        M.putaij(r, dy_rL_sl.start + e, 1.0)
        M.putaij(r, _ek_to_col(zpL_sl.start, e, 1, 5), -1.0)
        M.putaij(r, _ek_to_col(zR_sl.start, e, 1, 5), 1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # cone: (t_rL, dx_rL, dy_rL) ∈ Q
        M.appendcone(mosek.conetype.quad, 0.0,
                     [t_rL_sl.start + e, dx_rL_sl.start + e, dy_rL_sl.start + e])

        # --- right-relative: || [zpR_x - zL_x, zpR_y - zL_y] ||_2 <= 0.35 * y_e
        # t_rR_e - 0.35*y_e = 0
        M.appendcons(1);
        r = M.getnumcon() - 1
        M.putaij(r, t_rR_sl.start + e, 1.0)
        M.putaij(r, ycol, -REL_SPAN)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # dx_rR_e - (zpR_x - zL_x) = 0
        M.appendcons(1);
        r = M.getnumcon() - 1
        M.putaij(r, dx_rR_sl.start + e, 1.0)
        M.putaij(r, _ek_to_col(zpR_sl.start, e, 0, 5), -1.0)
        M.putaij(r, _ek_to_col(zL_sl.start, e, 0, 5), 1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # dy_rR_e - (zpR_y - zL_y) = 0
        M.appendcons(1);
        r = M.getnumcon() - 1
        M.putaij(r, dy_rR_sl.start + e, 1.0)
        M.putaij(r, _ek_to_col(zpR_sl.start, e, 1, 5), -1.0)
        M.putaij(r, _ek_to_col(zL_sl.start, e, 1, 5), 1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # cone: (t_rR, dx_rR, dy_rR) ∈ Q
        M.appendcone(mosek.conetype.quad, 0.0,
                     [t_rR_sl.start + e, dx_rR_sl.start + e, dy_rR_sl.start + e])

    # --- (I) Inward constraint (TORO):  zp_R.sin <= z_L.sin,   zp_L.sin >= z_R.sin
    for e in range(n_edge):
        # zp_R.sin - z_L.sin <= 0
        M.appendcons(1);
        r = M.getnumcon() - 1
        M.putaij(r, _ek_to_col(zpR_sl.start, e, SIN_COL, 5), 1.0)
        M.putaij(r, _ek_to_col(zL_sl.start, e, SIN_COL, 5), -1.0)
        M.putconbound(r, mosek.boundkey.up, -0.0, 0.0)

        # zp_L.sin - z_R.sin >= 0
        M.appendcons(1);
        r = M.getnumcon() - 1
        M.putaij(r, _ek_to_col(zpL_sl.start, e, SIN_COL, 5), 1.0)
        M.putaij(r, _ek_to_col(zR_sl.start, e, SIN_COL, 5), -1.0)
        M.putconbound(r, mosek.boundkey.lo, 0.0, +0.0)  # lower bound 0

    # =========================
    # (J) 5x5 SDP block (rotation/connection) — global-axis matching
    # =========================

    n_edge = IDX["n_edge"]
    y_sl = V["y"]
    zL_sl, zR_sl = V["z_L"], V["z_R"]
    zpL_sl, zpR_sl = V["zp_L"], V["zp_R"]

    # 전역축 변환계수
    E11 = float(x_ref[0]);
    E21 = float(x_ref[1])
    E12 = float(y_ref[0]);
    E22 = float(y_ref[1])
    # meta 보장 + 저장
    if "meta" not in V:
        V["meta"] = {}
    meta = V["meta"]
    meta["E11"] = E11;
    meta["E12"] = E12
    meta["E21"] = E21;
    meta["E22"] = E22

    # (이미 쓰는 경우 있을 텐데) src_edges도 meta에 저장해 두면 편해요
    meta["src_edges"] = list(src_edges)  # 0-based
    # --- 대칭 선택 행렬 캐시: <S(i,j), X> = X_ij ---
    # (dim=5용을 한 번만 등록해서 재사용)
    if "meta" not in V: V["meta"] = {}
    meta = V["meta"]
    if "sel5" not in meta:
        meta["sel5"] = {}

    def _get_sel5(i: int, j: int):
        """returns symmetric selector for X[i,j] using LOWER triangle only"""
        key = (i, j) if i >= j else (j, i)
        if "sel5" not in V["meta"]:
            V["meta"]["sel5"] = {}
        if key in V["meta"]["sel5"]:
            return V["meta"]["sel5"][key]

        ii, jj = key  # guarantee ii >= jj
        # lower-tri single entry
        subi = [ii]
        subj = [jj]
        val = [1.0]  # NOTE: single entry only; no upper-tri term

        sidx = M.appendsparsesymmat(5, subi, subj, val)
        V["meta"]["sel5"][key] = sidx
        return sidx

    # 소스 엣지 집합(0-based)
    src_edges_set = set(int(e) for e in src_edges)

    for e in range(n_edge):
        barL = V["X_L"][e]
        barR = V["X_R"][e]
        ycol = y_sl.start + e

        def _put_X_ij_eq_coef(M, row, bar_idx, sel_idx, coef):
            # sel_idx: appendsparsesymmat(...)가 반환한 정수
            M.putbaraij(row, int(bar_idx), [int(sel_idx)], [float(coef)])

        # ---- (J-0) X[0,0] == 1 ----
        # Left
        M.appendcons(1);
        r = M.getnumcon() - 1
        _put_X_ij_eq_coef(M, r, barL, _get_sel5(0, 0), 1.0)
        M.putconbound(r, mosek.boundkey.fx, 1.0, 1.0)
        # Right
        M.appendcons(1);
        r = M.getnumcon() - 1
        _put_X_ij_eq_coef(M, r, barR, _get_sel5(0, 0), 1.0)
        M.putconbound(r, mosek.boundkey.fx, 1.0, 1.0)

        # ---- (J-1) unit-circle blocks: X[1,1]+X[2,2]=1,  X[3,3]+X[4,4]=1 ----
        # Left: (1,1)+(2,2)=1
        M.appendcons(1);
        r = M.getnumcon() - 1
        _put_X_ij_eq_coef(M, r, barL, _get_sel5(1, 1), 1.0)
        _put_X_ij_eq_coef(M, r, barL, _get_sel5(2, 2), 1.0)
        M.putconbound(r, mosek.boundkey.fx, 1.0, 1.0)
        # Left: (3,3)+(4,4)=1
        M.appendcons(1);
        r = M.getnumcon() - 1
        _put_X_ij_eq_coef(M, r, barL, _get_sel5(3, 3), 1.0)
        _put_X_ij_eq_coef(M, r, barL, _get_sel5(4, 4), 1.0)
        M.putconbound(r, mosek.boundkey.fx, 1.0, 1.0)

        # Right: (1,1)+(2,2)=1
        M.appendcons(1);
        r = M.getnumcon() - 1
        _put_X_ij_eq_coef(M, r, barR, _get_sel5(1, 1), 1.0)
        _put_X_ij_eq_coef(M, r, barR, _get_sel5(2, 2), 1.0)
        M.putconbound(r, mosek.boundkey.fx, 1.0, 1.0)
        # Right: (3,3)+(4,4)=1
        M.appendcons(1);
        r = M.getnumcon() - 1
        _put_X_ij_eq_coef(M, r, barR, _get_sel5(3, 3), 1.0)
        _put_X_ij_eq_coef(M, r, barR, _get_sel5(4, 4), 1.0)
        M.putconbound(r, mosek.boundkey.fx, 1.0, 1.0)

        # ---- (J-2) c*c' + s*s' >= cos(alpha)*y  → X[1,3] + X[2,4] - ca*y >= 0 ----
        # Left
        M.appendcons(1);
        r = M.getnumcon() - 1
        _put_X_ij_eq_coef(M, r, barL, _get_sel5(1, 3), 1.0)
        _put_X_ij_eq_coef(M, r, barL, _get_sel5(2, 4), 1.0)
        M.putaij(r, ycol, -float(ca))
        M.putconbound(r, mosek.boundkey.lo, 0.0, +0.0)
        # Right
        M.appendcons(1);
        r = M.getnumcon() - 1
        _put_X_ij_eq_coef(M, r, barR, _get_sel5(1, 3), 1.0)
        _put_X_ij_eq_coef(M, r, barR, _get_sel5(2, 4), 1.0)
        M.putaij(r, ycol, -float(ca))
        M.putconbound(r, mosek.boundkey.lo, 0.0, +0.0)

        # ---- (J-3) row0 mapping: [1, c_g, s_g, c'_g, s'_g]
        # c_g = E11*c + E12*s,  s_g = E21*c + E22*s  (local→global)
        # source edge면 현재는 start_* 사용, 그 외엔 (z_* → zp_*) 사용

        def _pin_X01_to_expr(bar, expr_c, expr_s, e, use_const=False):
            """X[0,1]=c_g, X[0,2]=s_g; expr_c/s는 '로컬' cos/sin의 선형식 또는 상수"""
            # X[0,1] = c_g
            M.appendcons(1);
            r = M.getnumcon() - 1
            _put_X_ij_eq_coef(M, r, bar, _get_sel5(0, 1), 1.0)
            if use_const:
                # rhs = E11*c0 + E12*s0
                M.putconbound(r, mosek.boundkey.fx, float(expr_c), float(expr_c))  # expr_c에 이미 E*곱한 값 넣음
            else:
                # X[0,1] - (E11*c + E12*s) = 0
                M.putaij(r, expr_c["c_col"], -E11)
                M.putaij(r, expr_c["s_col"], -E12)
                M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

            # X[0,2] = s_g
            M.appendcons(1);
            r = M.getnumcon() - 1
            _put_X_ij_eq_coef(M, r, bar, _get_sel5(0, 2), 1.0)
            if use_const:
                M.putconbound(r, mosek.boundkey.fx, float(expr_s), float(expr_s))
            else:
                M.putaij(r, expr_c["c_col"], -E21)
                M.putaij(r, expr_c["s_col"], -E22)
                M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        def _pin_X03_to_expr(bar, expr_c, expr_s, e, use_const=False):
            """X[0,3]=c'_g, X[0,4]=s'_g"""
            # X[0,3] = c'_g
            M.appendcons(1);
            r = M.getnumcon() - 1
            _put_X_ij_eq_coef(M, r, bar, _get_sel5(0, 3), 1.0)
            if use_const:
                M.putconbound(r, mosek.boundkey.fx, float(expr_c), float(expr_c))
            else:
                M.putaij(r, expr_c["c_col"], -E11)
                M.putaij(r, expr_c["s_col"], -E12)
                M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

            # X[0,4] = s'_g
            M.appendcons(1);
            r = M.getnumcon() - 1
            _put_X_ij_eq_coef(M, r, bar, _get_sel5(0, 4), 1.0)
            if use_const:
                M.putconbound(r, mosek.boundkey.fx, float(expr_s), float(expr_s))
            else:
                M.putaij(r, expr_c["c_col"], -E21)
                M.putaij(r, expr_c["s_col"], -E22)
                M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        if e in src_edges_set:
            # Left: X[0,1] = c0_g  (RHS는 나중에 set_parameters_optapi에서 채움)
            M.appendcons(1);
            r = M.getnumcon() - 1
            _put_X_ij_eq_coef(M, r, barL, _get_sel5(0, 1), 1.0)
            # 일단 0으로 고정; 나중에 set_parameters_optapi에서 갱신
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)
            meta["J_src_rows_L_X01"].append(r)

            # Left: X[0,2] = s0_g
            M.appendcons(1);
            r = M.getnumcon() - 1
            _put_X_ij_eq_coef(M, r, barL, _get_sel5(0, 2), 1.0)
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)
            meta["J_src_rows_L_X02"].append(r)

            # Right: X[0,1] = c0_g
            M.appendcons(1);
            r = M.getnumcon() - 1
            _put_X_ij_eq_coef(M, r, barR, _get_sel5(0, 1), 1.0)
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)
            meta["J_src_rows_R_X01"].append(r)

            # Right: X[0,2] = s0_g
            M.appendcons(1);
            r = M.getnumcon() - 1
            _put_X_ij_eq_coef(M, r, barR, _get_sel5(0, 2), 1.0)
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)
            meta["J_src_rows_R_X02"].append(r)

        else:
            # 일반 edge는 기존 구현대로 변수 계수(Ax=0)로 두면 됩니다.
            # X[0,1] - (E11*c + E12*s) = 0
            M.appendcons(1);
            r = M.getnumcon() - 1
            _put_X_ij_eq_coef(M, r, barL, _get_sel5(0, 1), 1.0)
            M.putaij(r, _ek_to_col(zL_sl.start, e, 3, 5), -E11)
            M.putaij(r, _ek_to_col(zL_sl.start, e, 4, 5), -E12)
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)
            # X[0,2] - (E21*c + E22*s) = 0
            M.appendcons(1);
            r = M.getnumcon() - 1
            _put_X_ij_eq_coef(M, r, barL, _get_sel5(0, 2), 1.0)
            M.putaij(r, _ek_to_col(zL_sl.start, e, 3, 5), -E21)
            M.putaij(r, _ek_to_col(zL_sl.start, e, 4, 5), -E22)
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

            # Right도 동일 패턴으로 (생략)
            # X[0,1] - (E11*c + E12*s) = 0
            M.appendcons(1);
            r = M.getnumcon() - 1
            _put_X_ij_eq_coef(M, r, barR, _get_sel5(0, 1), 1.0)
            M.putaij(r, _ek_to_col(zR_sl.start, e, 3, 5), -E11)
            M.putaij(r, _ek_to_col(zR_sl.start, e, 4, 5), -E12)
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)
            # X[0,2] - (E21*c + E22*s) = 0
            M.appendcons(1);
            r = M.getnumcon() - 1
            _put_X_ij_eq_coef(M, r, barR, _get_sel5(0, 2), 1.0)
            M.putaij(r, _ek_to_col(zR_sl.start, e, 3, 5), -E21)
            M.putaij(r, _ek_to_col(zR_sl.start, e, 4, 5), -E22)
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

    # =========================
    # (K) 7x7 SDP block (half-plane rotation constraints; global-axis aligned)
    # row0 = [1, x, y, c_g, s_g, x', y']
    # =========================

    n_edge = IDX["n_edge"]
    y_sl = V["y"]
    zL_sl, zR_sl = V["z_L"], V["z_R"]
    zpL_sl, zpR_sl = V["zp_L"], V["zp_R"]

    # 전역←로컬 변환계수 (E)
    E11 = float(x_ref[0]);
    E21 = float(x_ref[1])
    E12 = float(y_ref[0]);
    E22 = float(y_ref[1])

    # meta 보장 및 보조 저장
    if "meta" not in V: V["meta"] = {}
    meta = V["meta"]
    meta["E11"] = E11;
    meta["E12"] = E12
    meta["E21"] = E21;
    meta["E22"] = E22

    # 선택행렬(7x7) 캐시: ⟨Sel(i,j), X⟩ = X[i,j], lower-tri만
    if "sel7" not in meta:
        meta["sel7"] = {}

    def _get_sel7(i: int, j: int):
        key = (i, j) if i >= j else (j, i)  # lower-tri 보장
        if key in meta["sel7"]:
            return meta["sel7"][key]
        ii, jj = key
        subi = [ii];
        subj = [jj];
        val = [1.0]  # 단일 하삼각 좌표
        sidx = M.appendsparsesymmat(7, subi, subj, val)
        meta["sel7"][key] = sidx
        return sidx

    # 파라미터(상수)
    d_lat = 0.25
    src_edges_set = set(int(e) for e in src_edges)
    sink_edges_set = set(int(e) for e in sink_edges)

    # 소스/싱크에서 RHS를 나중에 채우기 위한 행 인덱스 저장 컨테이너
    meta.setdefault("K_src_rows_XL2_01", [])  # X_L2[0,1] = start_L.x
    meta.setdefault("K_src_rows_XL2_02", [])  # X_L2[0,2] = start_L.y
    meta.setdefault("K_src_rows_XL2_03", [])  # X_L2[0,3] = c_g(start_L)
    meta.setdefault("K_src_rows_XL2_04", [])  # X_L2[0,4] = s_g(start_L)

    meta.setdefault("K_src_rows_XR2_01", [])  # X_R2[0,1] = start_R.x
    meta.setdefault("K_src_rows_XR2_02", [])  # X_R2[0,2] = start_R.y
    meta.setdefault("K_src_rows_XR2_03", [])  # X_R2[0,3] = c_g(start_R)
    meta.setdefault("K_src_rows_XR2_04", [])  # X_R2[0,4] = s_g(start_R)

    meta.setdefault("K_sink_rows_XL2_05", [])  # X_L2[0,5] = target_R.x
    meta.setdefault("K_sink_rows_XL2_06", [])  # X_L2[0,6] = target_R.y
    meta.setdefault("K_sink_rows_XR2_05", [])  # X_R2[0,5] = target_L.x
    meta.setdefault("K_sink_rows_XR2_06", [])  # X_R2[0,6] = target_L.y

    # 편의: X[i,j] == const 또는 == 선형식 묶기
    def _eq_Xij_const(bar_idx, i, j, const):
        M.appendcons(1);
        r = M.getnumcon() - 1
        _put_X_ij_eq_coef(M, r, bar_idx, _get_sel7(i, j), 1.0)
        M.putconbound(r, mosek.boundkey.fx, float(const), float(const))
        return r

    def _eq_Xij_minus_lin0(bar_idx, i, j, terms):
        """
        X[i,j] - (sum_k a_k * x_col_k) = 0
        terms: list of (col_index, coef)
        """
        M.appendcons(1);
        r = M.getnumcon() - 1
        _put_X_ij_eq_coef(M, r, bar_idx, _get_sel7(i, j), 1.0)
        for col, a in terms:
            M.putaij(r, int(col), -float(a))
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)
        return r

    # w_perp = -s_g*(x' - x) + c_g*(y' - y)
    # = (-X[0,4]*X[0,5] + X[0,4]*X[0,1]) + (X[0,3]*X[0,6] - X[0,3]*X[0,2])
    # 하지만 여기선 X 원소 자체에 선형 조합으로 표현:
    #   w_perp ≡ (X[4,1] - X[4,5]) + (X[3,6] - X[3,2])
    # 주의: 이는 row0와의 곱 형태가 아니라, Fusion에서 설계한 lifting에서
    # 해당 원소가 저 조합을 의미하도록 정의된 것을 그대로 사용.
    def _add_ineq_wperp(bar_idx, sign, ycol, dlat):
        """
        sign=+1:  d_lat*y - w_perp <= 0  →  +d*y  - (X[4,1]-X[4,5]+X[3,6]-X[3,2]) <= 0
        sign=-1:  w_perp + d_lat*y <= 0  →  +d*y  + (X[4,1]-X[4,5]+X[3,6]-X[3,2]) <= 0
        """
        M.appendcons(1);
        r = M.getnumcon() - 1
        # + d*y
        M.putaij(r, ycol, float(dlat))
        # ± (X[4,1] - X[4,5] + X[3,6] - X[3,2])
        s = float(sign)
        _put_X_ij_eq_coef(M, r, bar_idx, _get_sel7(4, 1), -s)  # -(+/- X[4,1])
        _put_X_ij_eq_coef(M, r, bar_idx, _get_sel7(4, 5), s)  # -(-/+) X[4,5]
        _put_X_ij_eq_coef(M, r, bar_idx, _get_sel7(3, 6), -s)
        _put_X_ij_eq_coef(M, r, bar_idx, _get_sel7(3, 2), s)
        M.putconbound(r, mosek.boundkey.up, -0.0, 0.0)

    for e in range(n_edge):
        barL2 = V["X_L2"][e]
        barR2 = V["X_R2"][e]
        ycol = y_sl.start + e

        # ---- (K-0) X[0,0] == 1 ----
        _eq_Xij_const(barL2, 0, 0, 1.0)
        _eq_Xij_const(barR2, 0, 0, 1.0)

        # ---- (K-1) unit circle: X[3,3] + X[4,4] = 1 ----
        M.appendcons(1);
        r = M.getnumcon() - 1
        _put_X_ij_eq_coef(M, r, barL2, _get_sel7(3, 3), 1.0)
        _put_X_ij_eq_coef(M, r, barL2, _get_sel7(4, 4), 1.0)
        M.putconbound(r, mosek.boundkey.fx, 1.0, 1.0)

        M.appendcons(1);
        r = M.getnumcon() - 1
        _put_X_ij_eq_coef(M, r, barR2, _get_sel7(3, 3), 1.0)
        _put_X_ij_eq_coef(M, r, barR2, _get_sel7(4, 4), 1.0)
        M.putconbound(r, mosek.boundkey.fx, 1.0, 1.0)

        # ---- (K-2) row0 mapping ----
        if e in src_edges_set:
            # XL2 row0: [1, xL(start), yL(start), c_g(start_L), s_g(start_L), xR', yR']
            #   (0,1), (0,2), (0,3), (0,4)는 상수 → RHS로 이후 set_parameters에서 채움
            r = _eq_Xij_const(barL2, 0, 1, 0.0);
            meta["K_src_rows_XL2_01"].append(r)
            r = _eq_Xij_const(barL2, 0, 2, 0.0);
            meta["K_src_rows_XL2_02"].append(r)
            r = _eq_Xij_const(barL2, 0, 3, 0.0);
            meta["K_src_rows_XL2_03"].append(r)
            r = _eq_Xij_const(barL2, 0, 4, 0.0);
            meta["K_src_rows_XL2_04"].append(r)
            # (0,5) = zp_R.x , (0,6) = zp_R.y
            _eq_Xij_minus_lin0(barL2, 0, 5, [(_ek_to_col(zpR_sl.start, e, 0, 5), 1.0)])
            _eq_Xij_minus_lin0(barL2, 0, 6, [(_ek_to_col(zpR_sl.start, e, 1, 5), 1.0)])

            # XR2 row0: [1, xR(start), yR(start), c_g(start_R), s_g(start_R), xL', yL']
            r = _eq_Xij_const(barR2, 0, 1, 0.0);
            meta["K_src_rows_XR2_01"].append(r)
            r = _eq_Xij_const(barR2, 0, 2, 0.0);
            meta["K_src_rows_XR2_02"].append(r)
            r = _eq_Xij_const(barR2, 0, 3, 0.0);
            meta["K_src_rows_XR2_03"].append(r)
            r = _eq_Xij_const(barR2, 0, 4, 0.0);
            meta["K_src_rows_XR2_04"].append(r)
            _eq_Xij_minus_lin0(barR2, 0, 5, [(_ek_to_col(zpL_sl.start, e, 0, 5), 1.0)])
            _eq_Xij_minus_lin0(barR2, 0, 6, [(_ek_to_col(zpL_sl.start, e, 1, 5), 1.0)])

        elif e in sink_edges_set:
            # XL2 row0: [1, xL, yL, c_g(L), s_g(L), xR'(=target_R.x), yR'(=target_R.y)]
            _eq_Xij_minus_lin0(barL2, 0, 1, [(_ek_to_col(zL_sl.start, e, 0, 5), 1.0)])
            _eq_Xij_minus_lin0(barL2, 0, 2, [(_ek_to_col(zL_sl.start, e, 1, 5), 1.0)])
            # c_g = E11*c + E12*s; s_g = E21*c + E22*s
            _eq_Xij_minus_lin0(barL2, 0, 3, [(_ek_to_col(zL_sl.start, e, 3, 5), E11),
                                             (_ek_to_col(zL_sl.start, e, 4, 5), E12)])
            _eq_Xij_minus_lin0(barL2, 0, 4, [(_ek_to_col(zL_sl.start, e, 3, 5), E21),
                                             (_ek_to_col(zL_sl.start, e, 4, 5), E22)])
            # (0,5),(0,6)는 target_R 상수 → RHS는 set_parameters에서
            r = _eq_Xij_const(barL2, 0, 5, 0.0);
            meta["K_sink_rows_XL2_05"].append(r)
            r = _eq_Xij_const(barL2, 0, 6, 0.0);
            meta["K_sink_rows_XL2_06"].append(r)

            # XR2 row0: [1, xR, yR, c_g(R), s_g(R), xL'(=target_L.x), yL'(=target_L.y)]
            _eq_Xij_minus_lin0(barR2, 0, 1, [(_ek_to_col(zR_sl.start, e, 0, 5), 1.0)])
            _eq_Xij_minus_lin0(barR2, 0, 2, [(_ek_to_col(zR_sl.start, e, 1, 5), 1.0)])
            _eq_Xij_minus_lin0(barR2, 0, 3, [(_ek_to_col(zR_sl.start, e, 3, 5), E11),
                                             (_ek_to_col(zR_sl.start, e, 4, 5), E12)])
            _eq_Xij_minus_lin0(barR2, 0, 4, [(_ek_to_col(zR_sl.start, e, 3, 5), E21),
                                             (_ek_to_col(zR_sl.start, e, 4, 5), E22)])
            r = _eq_Xij_const(barR2, 0, 5, 0.0);
            meta["K_sink_rows_XR2_05"].append(r)
            r = _eq_Xij_const(barR2, 0, 6, 0.0);
            meta["K_sink_rows_XR2_06"].append(r)

        else:
            # 일반 edge
            # XL2 row0: [1, z_L.x, z_L.y, c_g(z_L), s_g(z_L), zp_R.x, zp_R.y]
            _eq_Xij_minus_lin0(barL2, 0, 1, [(_ek_to_col(zL_sl.start, e, 0, 5), 1.0)])
            _eq_Xij_minus_lin0(barL2, 0, 2, [(_ek_to_col(zL_sl.start, e, 1, 5), 1.0)])
            _eq_Xij_minus_lin0(barL2, 0, 3, [(_ek_to_col(zL_sl.start, e, 3, 5), E11),
                                             (_ek_to_col(zL_sl.start, e, 4, 5), E12)])
            _eq_Xij_minus_lin0(barL2, 0, 4, [(_ek_to_col(zL_sl.start, e, 3, 5), E21),
                                             (_ek_to_col(zL_sl.start, e, 4, 5), E22)])
            _eq_Xij_minus_lin0(barL2, 0, 5, [(_ek_to_col(zpR_sl.start, e, 0, 5), 1.0)])
            _eq_Xij_minus_lin0(barL2, 0, 6, [(_ek_to_col(zpR_sl.start, e, 1, 5), 1.0)])

            # XR2 row0: [1, z_R.x, z_R.y, c_g(z_R), s_g(z_R), zp_L.x, zp_L.y]
            _eq_Xij_minus_lin0(barR2, 0, 1, [(_ek_to_col(zR_sl.start, e, 0, 5), 1.0)])
            _eq_Xij_minus_lin0(barR2, 0, 2, [(_ek_to_col(zR_sl.start, e, 1, 5), 1.0)])
            _eq_Xij_minus_lin0(barR2, 0, 3, [(_ek_to_col(zR_sl.start, e, 3, 5), E11),
                                             (_ek_to_col(zR_sl.start, e, 4, 5), E12)])
            _eq_Xij_minus_lin0(barR2, 0, 4, [(_ek_to_col(zR_sl.start, e, 3, 5), E21),
                                             (_ek_to_col(zR_sl.start, e, 4, 5), E22)])
            _eq_Xij_minus_lin0(barR2, 0, 5, [(_ek_to_col(zpL_sl.start, e, 0, 5), 1.0)])
            _eq_Xij_minus_lin0(barR2, 0, 6, [(_ek_to_col(zpL_sl.start, e, 1, 5), 1.0)])

        # ---- (K-3) half-plane constraints per mode via w_perp ----
        to_v = int(flow_in_vertices[e])
        mode = mode_list[to_v]
        if mode == "SL":
            # left step: w_perp_L >= d_lat*y  →  d*y - w_perp_L <= 0
            _add_ineq_wperp(barR2, +1, ycol, d_lat)
        elif mode == "SR":
            # right step: w_perp_R <= -d_lat*y →  w_perp_R + d*y <= 0
            _add_ineq_wperp(barL2, -1, ycol, d_lat)
        elif mode == "D":
            _add_ineq_wperp(barR2, +1, ycol, d_lat)
            _add_ineq_wperp(barL2, -1, ycol, d_lat)
        else:
            pass

    # # =========================
    # # (K-extra) Full RLT tightening for 7x7 PSD blocks (Fusion apply_full_rlt_block)
    # # =========================
    #
    # def _bar_row_geq_const(bar_idx: int, sels: list[int], coefs: list[float], rhs: float):
    #     M.appendcons(1)
    #     r = M.getnumcon() - 1
    #     M.putbaraij(r, int(bar_idx), [int(s) for s in sels], [float(c) for c in coefs])
    #     M.putconbound(r, mosek.boundkey.lo, float(rhs), +0.0)  # >= rhs
    #     return r
    #
    # def _bar_row_leq_const(bar_idx: int, sels: list[int], coefs: list[float], rhs: float):
    #     M.appendcons(1)
    #     r = M.getnumcon() - 1
    #     M.putbaraij(r, int(bar_idx), [int(s) for s in sels], [float(c) for c in coefs])
    #     M.putconbound(r, mosek.boundkey.up, -0.0, float(rhs))  # <= rhs
    #     return r
    #
    # def _mccormick_pair_7(bar_idx: int, ii: int, jj: int, vL: float, vU: float, wL: float, wU: float):
    #     """
    #     McCormick for vw ≈ v*w
    #     v  = X[0,ii], w  = X[0,jj], vw = X[ii,jj]   (symmetric)
    #     All of them are bar-elements → linear constraints on bar entries.
    #     """
    #     # selections (use symmetric: X[0,ii] == X[ii,0])
    #     sel_v = _get_sel7(ii, 0)
    #     sel_w = _get_sel7(jj, 0)
    #     sel_vw = _get_sel7(max(ii, jj), min(ii, jj))
    #
    #     # 1) vw >= vL*w + wL*v - vL*wL
    #     # vw - vL*w - wL*v >= -vL*wL
    #     _bar_row_geq_const(bar_idx, [sel_vw, sel_w, sel_v], [1.0, -vL, -wL], -vL * wL)
    #
    #     # 2) vw >= vU*w + wU*v - vU*wU
    #     _bar_row_geq_const(bar_idx, [sel_vw, sel_w, sel_v], [1.0, -vU, -wU], -vU * wU)
    #
    #     # 3) vw <= vU*w + wL*v - vU*wL
    #     # vw - vU*w - wL*v <= -vU*wL
    #     _bar_row_leq_const(bar_idx, [sel_vw, sel_w, sel_v], [1.0, -vU, -wL], -vU * wL)
    #
    #     # 4) vw <= vL*w + wU*v - vL*wU
    #     _bar_row_leq_const(bar_idx, [sel_vw, sel_w, sel_v], [1.0, -vL, -wU], -vL * wU)
    #
    # def _apply_full_rlt_7(bar_idx: int, bounds6: list[tuple[float, float]]):
    #     """
    #     bounds6: [(xL,xU),(yL,yU),(cgL,cgU),(sgL,sgU),(xpL,xpU),(ypL,ypU)]
    #     Corresponding indices in 7x7 row0 are: 1,2,3,4,5,6
    #     """
    #     idxs = [1, 2, 3, 4, 5, 6]
    #     lows = [float(b[0]) for b in bounds6]
    #     ups = [float(b[1]) for b in bounds6]
    #
    #     # 모든 i<j 쌍에 대해 McCormick
    #     for a in range(6):
    #         for b in range(a + 1, 6):
    #             ii, jj = idxs[a], idxs[b]
    #             _mccormick_pair_7(bar_idx, ii, jj, lows[a], ups[a], lows[b], ups[b])
    #
    #     # diagonal cap: X[ii,ii] <= max(|L|,|U|)^2
    #     for a in range(6):
    #         ii = idxs[a]
    #         cap = max(abs(lows[a]), abs(ups[a])) ** 2
    #         sel_ii = _get_sel7(ii, ii)
    #         _bar_row_leq_const(bar_idx, [sel_ii], [1.0], cap)
    #
    # # ----- apply per-edge (like Fusion) -----
    # for e in range(n_edge):
    #     to_v = int(flow_in_vertices[e])
    #     mode = mode_list[to_v]
    #     if mode == "SL":
    #         _apply_full_rlt_7(V["X_L2"][e], BOUNDS7)
    #     elif mode == "SR":
    #         _apply_full_rlt_7(V["X_R2"][e], BOUNDS7)
    #     else:
    #         _apply_full_rlt_7(V["X_L2"][e], BOUNDS7)
    #         _apply_full_rlt_7(V["X_R2"][e], BOUNDS7)
    #
    # # =========================
    # # (L) Coupling tightness between 5x5 and 7x7 blocks (Fusion)
    # # =========================
    # def _eq_bar_minus_bar0(barA: int, selA: int, barB: int, selB: int):
    #     M.appendcons(1)
    #     r = M.getnumcon() - 1
    #     M.putbaraij(r, int(barA), [int(selA)], [1.0])
    #     M.putbaraij(r, int(barB), [int(selB)], [-1.0])
    #     M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)
    #     return r
    #
    # for e in range(n_edge):
    #     barXL = V["X_L"][e]
    #     barXR = V["X_R"][e]
    #     barXL2 = V["X_L2"][e]
    #     barXR2 = V["X_R2"][e]
    #
    #     # XL(0,1)=XL2(0,3), XL(0,2)=XL2(0,4)
    #     _eq_bar_minus_bar0(barXL, _get_sel5(1, 0), barXL2, _get_sel7(3, 0))
    #     _eq_bar_minus_bar0(barXL, _get_sel5(2, 0), barXL2, _get_sel7(4, 0))
    #     # XL(1,1)=XL2(3,3), XL(1,2)=XL2(3,4), XL(2,2)=XL2(4,4)
    #     _eq_bar_minus_bar0(barXL, _get_sel5(1, 1), barXL2, _get_sel7(3, 3))
    #     _eq_bar_minus_bar0(barXL, _get_sel5(2, 1), barXL2, _get_sel7(4, 3))
    #     _eq_bar_minus_bar0(barXL, _get_sel5(2, 2), barXL2, _get_sel7(4, 4))
    #
    #     # XR도 동일
    #     _eq_bar_minus_bar0(barXR, _get_sel5(1, 0), barXR2, _get_sel7(3, 0))
    #     _eq_bar_minus_bar0(barXR, _get_sel5(2, 0), barXR2, _get_sel7(4, 0))
    #     _eq_bar_minus_bar0(barXR, _get_sel5(1, 1), barXR2, _get_sel7(3, 3))
    #     _eq_bar_minus_bar0(barXR, _get_sel5(2, 1), barXR2, _get_sel7(4, 3))
    #     _eq_bar_minus_bar0(barXR, _get_sel5(2, 2), barXR2, _get_sel7(4, 4))



    # =========================
    # Objective block + mode-coupled cones (rho, rhoz)
    # =========================

    # 핸들/슬라이스
    y_sl   = V["y"]
    zL_sl  = V["z_L"];   zR_sl  = V["z_R"]
    zpL_sl = V["zp_L"];  zpR_sl = V["zp_R"]
    rhoL_sl = V["rho_L"]; rhoR_sl = V["rho_R"]
    rhozL_sl = V["rhoz_L"]; rhozR_sl = V["rhoz_R"]

    n_edge = IDX["n_edge"]

    # (H)에서 만들었던 보조변수들 재사용 가정:
    #  - dLx_sl, dLy_sl, dRx_sl, dRy_sl
    # 만약 네 코드에서 변수명이 다르면 같은 의미로 맞춰줘.


    # --- z축 차이 보조변수: hL = zp_L.z - z_L.z,  hR = zp_R.z - z_R.z ---
    hL_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")
    hR_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")

    # --- (패치) rho용 q-변수 생성: dLx/dLy 복사본 ---
    qLx_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")
    qLy_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")
    qRx_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")
    qRy_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")

    # d와 q를 등식으로 연결 (q = d). 등식은 cone 중복이 아님
    for e in range(n_edge):
        # qLx_e - dLx_e = 0
        M.appendcons(1);
        r = M.getnumcon() - 1
        M.putaij(r, qLx_sl.start + e, 1.0)
        M.putaij(r, dLx_sl.start + e, -1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # qLy_e - dLy_e = 0
        M.appendcons(1);
        r = M.getnumcon() - 1
        M.putaij(r, qLy_sl.start + e, 1.0)
        M.putaij(r, dLy_sl.start + e, -1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # qRx_e - dRx_e = 0
        M.appendcons(1);
        r = M.getnumcon() - 1
        M.putaij(r, qRx_sl.start + e, 1.0)
        M.putaij(r, dRx_sl.start + e, -1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # qRy_e - dRy_e = 0
        M.appendcons(1);
        r = M.getnumcon() - 1
        M.putaij(r, qRy_sl.start + e, 1.0)
        M.putaij(r, dRy_sl.start + e, -1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

    # 등식으로 hL/hR 연결
    for e in range(n_edge):
        # hL_e - (zpL_z - zL_z) = 0
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, hL_sl.start + e, 1.0)
        M.putaij(r, _ek_to_col(zpL_sl.start, e, 2, 5), -1.0)
        M.putaij(r, _ek_to_col(zL_sl.start,  e, 2, 5),  1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # hR_e - (zpR_z - zR_z) = 0
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, hR_sl.start + e, 1.0)
        M.putaij(r, _ek_to_col(zpR_sl.start, e, 2, 5), -1.0)
        M.putaij(r, _ek_to_col(zR_sl.start,  e, 2, 5),  1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

    # --- 모드별 제약 + SOC(rho) ---
    Mbig = 1.0  # indicator linking 상수

    for e in range(n_edge):
        m_to = mode_list[int(flow_in_vertices[e])]   # 네 코드와 동일: 'to' 모드 기준
        y_col = y_sl.start + e

        # delta 핸들 (이미 (H)에서 등식으로 연결됨)
        dLx_col = dLx_sl.start + e
        dLy_col = dLy_sl.start + e
        dRx_col = dRx_sl.start + e
        dRy_col = dRy_sl.start + e

        rhoL_col = rhoL_sl.start + e
        rhoR_col = rhoR_sl.start + e

        # --- 모드별 고정/활성 ---
        if m_to == "SL":
            # 오른발 고정: z_R(e,k) - zp_R(e,k) = 0, k in {0,1,3,4}
            for k in (0, 1, 3, 4):
                M.appendcons(1)
                r = M.getnumcon() - 1
                M.putaij(r, _ek_to_col(zR_sl.start,  e, k, 5),  1.0)
                M.putaij(r, _ek_to_col(zpR_sl.start, e, k, 5), -1.0)
                M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)
            # 왼발만 이동: (rho_L, dLx, dLy) ∈ Q
            # (수정) q-변수 사용
            M.appendcone(mosek.conetype.quad, 0.0, [rhoL_col, qLx_sl.start + e, qLy_sl.start + e])
            # rho_R = 0
            M.appendcons(1)
            r = M.getnumcon() - 1
            M.putaij(r, rhoR_col, 1.0)
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        elif m_to == "SR":
            # 왼발 고정
            for k in (0, 1, 3, 4):
                M.appendcons(1)
                r = M.getnumcon() - 1
                M.putaij(r, _ek_to_col(zL_sl.start,  e, k, 5),  1.0)
                M.putaij(r, _ek_to_col(zpL_sl.start, e, k, 5), -1.0)
                M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)
            # 오른발만 이동
            # (수정)
            M.appendcone(mosek.conetype.quad, 0.0, [rhoR_col, qRx_sl.start + e, qRy_sl.start + e])
            # rho_L = 0
            M.appendcons(1)
            r = M.getnumcon() - 1
            M.putaij(r, rhoL_col, 1.0)
            M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        elif m_to == "D":
            # 양발 이동

            M.appendcone(mosek.conetype.quad, 0.0, [rhoL_col, qLx_sl.start + e, qLy_sl.start + e])
            M.appendcone(mosek.conetype.quad, 0.0, [rhoR_col, qRx_sl.start + e, qRy_sl.start + e])

        # --- |h| <= rhoz  → (rhoz, h) ∈ Q^2 ---
        M.appendcone(mosek.conetype.quad, 0.0, [rhozL_sl.start + e, hL_sl.start + e])
        M.appendcone(mosek.conetype.quad, 0.0, [rhozR_sl.start + e, hR_sl.start + e])

        # --- indicator linking: rho_*, rhoz_* ≤ Mbig * y_e ---
        # rho_L - M*y ≤ 0
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, rhoL_col, 1.0)
        M.putaij(r, y_col,   -float(Mbig))
        M.putconbound(r, mosek.boundkey.up, -0.0, 0.0)

        # rho_R - M*y ≤ 0
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, rhoR_col, 1.0)
        M.putaij(r, y_col,   -float(Mbig))
        M.putconbound(r, mosek.boundkey.up, -0.0, 0.0)

        # rhoz_L - M*y ≤ 0
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, rhozL_sl.start + e, 1.0)
        M.putaij(r, y_col,              -float(Mbig))
        M.putconbound(r, mosek.boundkey.up, -0.0, 0.0)

        # rhoz_R - M*y ≤ 0
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, rhozR_sl.start + e, 1.0)
        M.putaij(r, y_col,              -float(Mbig))
        M.putconbound(r, mosek.boundkey.up, -0.0, 0.0)

    # ---- mode mask (wL, wR) 계산 (넘어온 mode_list/flow_*로 동일 로직) ----
    wL = np.zeros(n_edge, dtype=float)
    wR = np.zeros(n_edge, dtype=float)
    for e in range(n_edge):
        m_from = mode_list[int(flow_out_vertices[e])]
        m_to   = mode_list[int(flow_in_vertices[e])]
        if m_from == "SR" and m_to == "D":
            wR[e] = 1.5; wL[e] = 1.5
        elif m_from == "SL" and m_to == "D":
            wR[e] = 1.5; wL[e] = 1.5
        elif m_from == "SR" and m_to == "SL":
            wL[e] = 1.5
        elif m_from == "D"  and m_to == "SL":
            wL[e] = 1.5
        elif m_from == "SL" and m_to == "SR":
            wR[e] = 1.5
        elif m_from == "D"  and m_to == "SR":
            wR[e] = 1.5

    # ---- 선형 목적 계수 설정 ----
    obj_cols = []
    obj_coefs = []

    # xy terms: wL·rho_L + wR·rho_R
    for e in range(n_edge):
        if wL[e] != 0.0:
            obj_cols.append(rhoL_sl.start + e)
            obj_coefs.append(float(wL[e]))
        if wR[e] != 0.0:
            obj_cols.append(rhoR_sl.start + e)
            obj_coefs.append(float(wR[e]))

    # z terms: sum(rhoz_L) + sum(rhoz_R)
    for e in range(n_edge):
        obj_cols.append(rhozL_sl.start + e); obj_coefs.append(1.0)
        obj_cols.append(rhozR_sl.start + e); obj_coefs.append(1.0)

    # 적용
    if obj_cols:
        M.putclist(obj_cols, obj_coefs)
    # (기본이 minimize. 필요시: M.putobjsense(mosek.objsense.minimize))
    # =========================
    # Rotation-related cost (per-foot ref)
    # =========================

    # 전제: 아래 슬라이스들이 이미 존재함
    # y_sl, zL_sl, zR_sl, zpL_sl, zpR_sl
    # dLx_sl, dLy_sl, dRx_sl, dRy_sl  (H에서 생성)
    # 또한 _ek_to_col, _alloc_block_local 헬퍼가 위에서 정의되어 있다고 가정
    # (없다면 위 답변의 _alloc_block_local 정의를 재사용)

    lam_rot  = 0.1
    lam_circ = 1.0 * lam_rot   # (옵션; 여기서는 사용하지 않음)

    ref_scale = 0.1
    inv_scale = 1.0 / ref_scale

    n_edge = IDX["n_edge"]

    # 1) 회전 거리 t 변수 (>=0)
    tLrot_sl = _alloc_block_local(M, n_edge, bound_type="lower", lb=0.0)
    tRrot_sl = _alloc_block_local(M, n_edge, bound_type="lower", lb=0.0)

    # 2) u 보조변수: (cos - ref_c1), (sin - ref_c2)
    uL1_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")
    uL2_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")
    uR1_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")
    uR2_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")

    # 3) u = z_* - ref_c*  선형 등식 + SOCP (t, u1, u2)
    xr0, xr1 = float(x_ref[0]), float(x_ref[1])
    yr0, yr1 = float(y_ref[0]), float(y_ref[1])

    for e in range(n_edge):
        # --- 왼발: ref_c1 = inv_scale * (x_ref·[dLx,dLy]),  ref_c2 = inv_scale * (y_ref·[dLx,dLy])
        dLx_col = dLx_sl.start + e
        dLy_col = dLy_sl.start + e

        # uL1 = zL_cos - inv_scale*(xr0*dLx + xr1*dLy)
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, uL1_sl.start + e, 1.0)
        M.putaij(r, _ek_to_col(zL_sl.start, e, 3, 5), -1.0)  # - zL_cos
        # + inv_scale*(xr0*dLx + xr1*dLy)
        M.putaij(r, dLx_col,  inv_scale * xr0)
        M.putaij(r, dLy_col,  inv_scale * xr1)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # uL2 = zL_sin - inv_scale*(yr0*dLx + yr1*dLy)
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, uL2_sl.start + e, 1.0)
        M.putaij(r, _ek_to_col(zL_sl.start, e, 4, 5), -1.0)  # - zL_sin
        M.putaij(r, dLx_col,  inv_scale * yr0)
        M.putaij(r, dLy_col,  inv_scale * yr1)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # cone: (tLrot, uL1, uL2) ∈ Q
        M.appendcone(mosek.conetype.quad, 0.0,
                     [tLrot_sl.start + e, uL1_sl.start + e, uL2_sl.start + e])

        # --- 오른발: ref_c1 = inv_scale * (x_ref·[dRx,dRy]),  ref_c2 = inv_scale * (y_ref·[dRx,dRy])
        dRx_col = dRx_sl.start + e
        dRy_col = dRy_sl.start + e

        # uR1 = zR_cos - inv_scale*(xr0*dRx + xr1*dRy)
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, uR1_sl.start + e, 1.0)
        M.putaij(r, _ek_to_col(zR_sl.start, e, 3, 5), -1.0)  # - zR_cos
        M.putaij(r, dRx_col,  inv_scale * xr0)
        M.putaij(r, dRy_col,  inv_scale * xr1)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # uR2 = zR_sin - inv_scale*(yr0*dRx + yr1*dRy)
        M.appendcons(1)
        r = M.getnumcon() - 1
        M.putaij(r, uR2_sl.start + e, 1.0)
        M.putaij(r, _ek_to_col(zR_sl.start, e, 4, 5), -1.0)  # - zR_sin
        M.putaij(r, dRx_col,  inv_scale * yr0)
        M.putaij(r, dRy_col,  inv_scale * yr1)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # cone: (tRrot, uR1, uR2) ∈ Q
        M.appendcone(mosek.conetype.quad, 0.0,
                     [tRrot_sl.start + e, uR1_sl.start + e, uR2_sl.start + e])

    # 4) 목적함수: lam_rot * sum_e (tLrot_e + tRrot_e)
    obj_cols  = []
    obj_coefs = []
    for e in range(n_edge):
        obj_cols.append(tLrot_sl.start + e); obj_coefs.append(lam_rot)
        obj_cols.append(tRrot_sl.start + e); obj_coefs.append(lam_rot)
    if obj_cols:
        M.putclist(obj_cols, obj_coefs)

    # =========================
    # L2^2 균등화 항: lam_eq * sum(wL * tL) + lam_eq * sum(wR * tR)
    # with rotated QC: t >= rho^2
    # =========================
    lam_eq = 10.0
    gamma = 2.0  # 시작/끝 가중치 증폭

    rhoL_sl = V["rho_L"]
    rhoR_sl = V["rho_R"]

    # 1) 보조변수 생성
    tL_sl = _alloc_block_local(M, n_edge, bound_type="lower", lb=0.0)  # tL >= 0
    tR_sl = _alloc_block_local(M, n_edge, bound_type="lower", lb=0.0)  # tR >= 0
    rLcp_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")  # rho_L 복사본
    rRcp_sl = _alloc_block_local(M, n_edge, bound_type="unbounded")  # rho_R 복사본
    vL_sl = _alloc_block_local(M, n_edge, bound_type="lower", lb=0.0)  # 회전콘의 v(=0.5로 고정)
    vR_sl = _alloc_block_local(M, n_edge, bound_type="lower", lb=0.0)

    # v를 0.5에 고정 (각 변수는 한 콘에만 들어가야 하므로 edge마다 따로 고정)
    for e in range(n_edge):
        # 기존: M.putvarbound(vL_sl.start + e, ..., 0.5, 0.5)
        M.putvarbound(vL_sl.start + e, mosek.boundkey.fx, 1.0, 1.0)
        M.putvarbound(vR_sl.start + e, mosek.boundkey.fx, 1.0, 1.0)

    # rho 복사본 등식 rLcp = rho_L, rRcp = rho_R  (등식은 cone 겹침 아님)
    for e in range(n_edge):
        # rLcp_e - rhoL_e = 0
        M.appendcons(1);
        r = M.getnumcon() - 1
        M.putaij(r, rLcp_sl.start + e, 1.0)
        M.putaij(r, rhoL_sl.start + e, -1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

        # rRcp_e - rhoR_e = 0
        M.appendcons(1);
        r = M.getnumcon() - 1
        M.putaij(r, rRcp_sl.start + e, 1.0)
        M.putaij(r, rhoR_sl.start + e, -1.0)
        M.putconbound(r, mosek.boundkey.fx, 0.0, 0.0)

    # 2) 회전 2차콘: (t, v=0.5, rCopy) ∈ RQ → 2*t*0.5 >= rCopy^2 ⇒ t >= rCopy^2
    for e in range(n_edge):
        M.appendcone(mosek.conetype.rquad, 0.0,
                     [tL_sl.start + e, vL_sl.start + e, rLcp_sl.start + e])
        M.appendcone(mosek.conetype.rquad, 0.0,
                     [tR_sl.start + e, vR_sl.start + e, rRcp_sl.start + e])

    # 3) 시작/끝 구간 가중치 프로파일
    wProfile_L = np.ones(n_edge, dtype=float)
    wProfile_R = np.ones(n_edge, dtype=float)
    for e in range(n_edge):
        m_from = mode_list[int(flow_out_vertices[e])]
        m_to = mode_list[int(flow_in_vertices[e])]
        # 시작: D->SL/SR
        if m_from == "D" and m_to == "SL":
            wProfile_L[e] = gamma
        elif m_from == "D" and m_to == "SR":
            wProfile_R[e] = gamma
        # 끝: SL->D / SR->D
        elif m_from == "SL" and m_to == "D":
            wProfile_L[e] = gamma
        elif m_from == "SR" and m_to == "D":
            wProfile_R[e] = gamma

    # 4) 목적함수에 계수 추가 (기존 putclist들과 변수군이 겹치지 않으므로 안전)
    tL_cols = list(range(tL_sl.start, tL_sl.stop))
    tR_cols = list(range(tR_sl.start, tR_sl.stop))
    tL_coefs = [lam_eq * float(w) for w in wProfile_L.tolist()]
    tR_coefs = [lam_eq * float(w) for w in wProfile_R.tolist()]
    M.putclist(tL_cols, tL_coefs)
    M.putclist(tR_cols, tR_coefs)


    # =========================
    # Step-number penalty + finalize objective
    # =========================

    # (step number) 0.1 * sum(y)
    step_penalty = 1.9
    y_cols  = list(range(V["y"].start, V["y"].stop))
    y_coefs = [step_penalty] * len(y_cols)
    M.putclist(y_cols, y_coefs)          # y에만 새 계수 설정 (다른 변수 계수는 위에서 이미 putclist로 넣음)

    # 최소화 설정 (기본이 minimize지만 명시)
    M.putobjsense(mosek.objsense.minimize)

    # =========================
    # Return (Fusion과 동일 시그니처 유지)
    # =========================
    build_time = time.perf_counter() - t0

    # Optimizer API에선 parameter 타입이 없으므로 placeholder(None) 유지
    V = dict(
        y=V["y"], z_L=V["z_L"], z_R=V["z_R"], zp_L=V["zp_L"], zp_R=V["zp_R"],
        rho_L=V["rho_L"], rho_R=V["rho_R"], rhoz_L=V["rhoz_L"], rhoz_R=V["rhoz_R"],
        X_L=V["X_L"], X_R=V["X_R"], X_L2=V["X_L2"], X_R2=V["X_R2"],
        start_L=None, start_R=None, target_L=None, target_R=None,
        y_relax=None, M_scalar=None,
        meta=V.get("meta", {})
    )
    IDX = dict(n_edge=IDX["n_edge"], vertex_number=IDX["vertex_number"])
    PAR = dict(centroid=PAR.get("centroid", centroid))

    # task를 M으로 반환 (Fusion과 동일 변수명 유지)
    return M, V, IDX, PAR, build_time
