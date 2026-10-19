from __future__ import annotations
from typing import List, Iterable, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
import matplotlib.patches as patches

def compute_foot_rects(
    x: Iterable[float],
    y: Iterable[float],
    cosyaw: Iterable[float],
    sinyaw: Iterable[float],
    *,
    h: float = 0.20, w: float = 0.10,
    # === ref 좌표계 기반 옵션 ===
    startpos_L: Optional[np.ndarray] = None,
    startpos_R: Optional[np.ndarray] = None,
    forward_axis: str = "x",      # 로컬 forward 축: 'x' 또는 'y'
    invert_yaw: bool = False,     # 필요 시 sin 부호 반전
) -> List[np.ndarray]:
    """
    (cos,sin)은 'ref 좌표계' 기준이라 가정하고, ref 축을 월드 XY로 사상하여
    사각형의 방향을 맞춘다. startpos_*가 없으면 기존(+90° CCW) 방식으로 폴백.

    - 꼭짓점은 전방=+x 기준에서 0.5*[[-h,-w],[h,-w],[h,w],[-h,w]] 사용.
    - forward_axis='x'면 R(θ)[1,0], 'y'면 R(θ)[0,1]을 forward로 사용.
    """
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    c = np.asarray(cosyaw, float).reshape(-1)
    s = np.asarray(sinyaw, float).reshape(-1)
    if not (x.size == y.size == c.size == s.size):
        raise ValueError("x, y, cosyaw, sinyaw must have the same length")

    # (cos, sin) 정규화
    r = np.hypot(c, s)
    r[r == 0.0] = 1.0
    c /= r
    s /= r
    if invert_yaw:
        s = -s

    # 중심 원점 기준의 직사각형 꼭짓점(전방=+x 가정)
    corners = 0.5 * np.array(
        [[-h, -w],
         [ h, -w],
         [ h,  w],
         [-h,  w]], dtype=float
    )

    use_ref = (startpos_L is not None) and (startpos_R is not None)
    if use_ref:
        x_ref, y_ref = _ref_axes_from_startRL(startpos_L, startpos_R)

    rects: List[np.ndarray] = []
    for xi, yi, ci, si in zip(x, y, c, s):
        if use_ref:
            # ref 기준 forward 벡터: R(θ) e_f
            if forward_axis.lower() == "x":
                f_ref = np.array([ci, si], dtype=float)
            elif forward_axis.lower() == "y":
                # R(θ) @ [0,1] = [-sinθ, cosθ]
                f_ref = np.array([-si,  ci], dtype=float)
            else:
                raise ValueError("forward_axis must be 'x' or 'y'.")

            # 월드 XY로 사상: f_world = f_ref_x * x_ref + f_ref_y * y_ref
            f_world = f_ref[0] * x_ref + f_ref[1] * y_ref
            nf = np.linalg.norm(f_world) + 1e-12
            f_world /= nf  # 단위 벡터
            # 오른손좌표계: l = [-fy, fx]
            l_world = np.array([-f_world[1], f_world[0]], dtype=float)

            # corners의 +x를 f_world, +y를 l_world에 정렬
            R2 = np.column_stack([f_world, l_world])  # 2x2
        else:
            # ★ 폴백: 기존 규약 유지 (사각형 전방은 (cos,sin)에서 +90° CCW 회전)
            ci2, si2 = -si, ci  # (+90°): (cos, sin) -> (-sin, cos)
            R2 = np.array([[ci2, -si2],
                           [si2,  ci2]], dtype=float)

        poly = (R2 @ corners.T).T
        poly[:, 0] += xi
        poly[:, 1] += yi
        rects.append(poly)

    return rects

def compute_foot_rects_wfc(
    x: Iterable[float],
    y: Iterable[float],
    cosyaw: Iterable[float],
    sinyaw: Iterable[float],
    *,
    h: float = 0.20, w: float = 0.10,
    # === ref 좌표계 기반 옵션 ===
    startpos_L: Optional[np.ndarray] = None,
    startpos_R: Optional[np.ndarray] = None,
    forward_axis: str = "x",      # 로컬 forward 축: 'x' 또는 'y'
    invert_yaw: bool = False,     # 필요 시 sin 부호 반전
    # === 외부 ref 축 오버라이드(있으면 최우선) ===
    x_ref_override: Optional[np.ndarray] = None,
    y_ref_override: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    (cos,sin)은 'ref 좌표계' 기준이라 가정하고, ref 축을 월드 XY로 사상하여
    사각형의 방향을 맞춘다.

    ref 축 선택 우선순위:
      1) x_ref_override, y_ref_override (있으면 최우선)
      2) startpos_L/R로 내부에서 계산
      3) 둘 다 없으면 폴백(+90° CCW 회전 방식)
    """
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    c = np.asarray(cosyaw, float).reshape(-1)
    s = np.asarray(sinyaw, float).reshape(-1)
    if not (x.size == y.size == c.size == s.size):
        raise ValueError("x, y, cosyaw, sinyaw must have the same length")

    # (cos, sin) 정규화
    r = np.hypot(c, s)
    r[r == 0.0] = 1.0
    c /= r
    s /= r
    if invert_yaw:
        s = -s

    # 중심 원점 기준의 직사각형 꼭짓점(전방=+x 가정)
    corners = 0.5 * np.array(
        [[-h, -w],
         [ h, -w],
         [ h,  w],
         [-h,  w]], dtype=float
    )

    # --- ref 축 준비: override > startpos 기반 > 없음(폴백) ---
    use_override = (x_ref_override is not None) and (y_ref_override is not None)
    use_ref_by_start = (startpos_L is not None) and (startpos_R is not None)

    if use_override:
        x_ref = np.asarray(x_ref_override, float).reshape(2)
        y_ref = np.asarray(y_ref_override, float).reshape(2)
        x_ref /= (np.linalg.norm(x_ref) + 1e-12)
        y_ref /= (np.linalg.norm(y_ref) + 1e-12)
        # 우수좌표계(det>0) 보장
        if x_ref[0]*y_ref[1] - x_ref[1]*y_ref[0] < 0:
            x_ref = -x_ref
        use_ref = True
    elif use_ref_by_start:
        x_ref, y_ref = _ref_axes_from_startRL(startpos_L, startpos_R)
        use_ref = True
    else:
        use_ref = False  # 폴백

    rects: List[np.ndarray] = []
    for xi, yi, ci, si in zip(x, y, c, s):
        if use_ref:
            # ref 기준 forward 벡터 f_ref = R(θ) e_f
            if forward_axis.lower() == "x":
                f_ref = np.array([ci, si], dtype=float)       # R(θ)[1,0]
            elif forward_axis.lower() == "y":
                f_ref = np.array([-si,  ci], dtype=float)     # R(θ)[0,1]
            else:
                raise ValueError("forward_axis must be 'x' or 'y'.")

            # 월드 XY로 사상: f_world = f_ref_x * x_ref + f_ref_y * y_ref
            f_world = f_ref[0] * x_ref + f_ref[1] * y_ref
            f_world /= (np.linalg.norm(f_world) + 1e-12)
            # 왼쪽(left) 벡터: CCW 90°
            l_world = np.array([-f_world[1], f_world[0]], dtype=float)

            # corners의 +x를 f_world, +y를 l_world에 정렬
            R2 = np.column_stack([f_world, l_world])  # 2x2
        else:
            # ★ 폴백: ref 없을 때 기존 규약 유지 (+90° CCW 회전)
            ci2, si2 = -si, ci  # (cos,sin) -> (-sin,cos)
            R2 = np.array([[ci2, -si2],
                           [si2,  ci2]], dtype=float)

        poly = (R2 @ corners.T).T
        poly[:, 0] += xi
        poly[:, 1] += yi
        rects.append(poly)

    return rects



def _auto_limits(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, pad: float = 0.15
) -> Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]]:
    def lim(v):
        vmin, vmax = float(np.min(v)), float(np.max(v))
        if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
            margin = (vmax - vmin) * pad
            return (vmin - margin, vmax + margin)
        return (vmin - 0.5, vmax + 0.5)
    return lim(xs), lim(ys), lim(zs)


def _ref_axes_from_startRL(startpos_L: np.ndarray, startpos_R: np.ndarray):
    """R->L 벡터를 +y_ref로 두고, 오른손좌표계가 되도록 x_ref를 생성 (z-up)."""
    pL = np.asarray(startpos_L, float).reshape(-1)[:2]
    pR = np.asarray(startpos_R, float).reshape(-1)[:2]
    v = pL - pR  # R -> L
    n = np.linalg.norm(v) + 1e-12
    y_ref = v / n
    # x_ref × y_ref = +z 이도록 x_ref는 y_ref를 -90° 회전
    x_ref = np.array([ y_ref[1], -y_ref[0] ], dtype=float)
    return x_ref, y_ref

def _draw_yaw_arrows_xyplane_ref(ax, x, y, z, c, s, *,
                                 x_ref, y_ref,
                                 total_len=0.15, head_frac=0.35, width_frac=0.6,
                                 color="#000", lw=1.6,
                                 forward_axis="x", invert_yaw=False):
    """(cos,sin)이 ref 좌표계 기준일 때, ref 축(x_ref,y_ref)을 사용해 월드 XY 상에 화살표를 그림."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    c = np.asarray(c, float); s = np.asarray(s, float)
    if invert_yaw:
        s = -s

    # ref 기준 forward 벡터: R(θ) e_f
    if forward_axis.lower() == "x":
        f_ref_x = c
        f_ref_y = s
    elif forward_axis.lower() == "y":
        # R(θ) @ [0,1] = [-sinθ, cosθ]
        f_ref_x = -s
        f_ref_y =  c
    else:
        raise ValueError("forward_axis must be 'x' or 'y'.")

    # 월드 XY로 매핑: f_world = f_ref_x * x_ref + f_ref_y * y_ref
    dir_x = f_ref_x * x_ref[0] + f_ref_y * y_ref[0]
    dir_y = f_ref_x * x_ref[1] + f_ref_y * y_ref[1]

    # 정규화
    norm = np.sqrt(dir_x**2 + dir_y**2) + 1e-12
    dir_x /= norm; dir_y /= norm

    # 화살표 그리기
    for xi, yi, zi, dx, dy in zip(x, y, z, dir_x, dir_y):
        base = [xi, yi, zi]
        vec  = total_len * np.array([dx, dy])
        _draw_arrow_xyplane(ax, base, vec, color=color, lw=lw,
                            head_frac=head_frac, width_frac=width_frac)


def _draw_arrow_xyplane(ax, base, dir_xy, *,
                        total_len=0.18, head_frac=0.35, width_frac=0.6,
                        color="#111", lw=2.0):
    bx, by, bz = float(base[0]), float(base[1]), float(base[2])
    ux, uy = float(dir_xy[0]), float(dir_xy[1])
    n = np.hypot(ux, uy)
    if n == 0.0:
        return
    ux, uy = ux / n, uy / n

    head_len = total_len * head_frac
    shaft_len = max(0.0, total_len - head_len)
    head_w   = head_len * width_frac

    sx, sy = bx + shaft_len * ux, by + shaft_len * uy
    tx, ty = bx + total_len * ux, by + total_len * uy

    ax.plot([bx, sx], [by, sy], [bz, bz], color=color, linewidth=lw)
    px, py = -uy, ux
    v1 = [sx + (head_w/2)*px, sy + (head_w/2)*py, bz]
    v2 = [sx - (head_w/2)*px, sy - (head_w/2)*py, bz]
    tip = [tx, ty, bz]
    tri = np.array([v1, v2, tip], float)
    ax.add_collection3d(Poly3DCollection([tri], facecolors=color,
                                         edgecolors=color, linewidths=lw*0.6, alpha=1.0))


def plot_feet_3d(
    xL, yL, zL, cL, sL,
    xR, yR, zR, cR, sR,
    *,
    h: float = 0.20, w: float = 0.10,
    arrow_len: float = 0.15,
    left_color: str = "#F58518",
    right_color: str = "#4C78A8",
    face_alpha: float = 0.35,
    limits: Optional[Tuple[Tuple[float,float],Tuple[float,float],Tuple[float,float]]] = None,
    matlab_style: bool = True,
    show: bool = True,
    # ref 축 오버레이
    draw_ref_axes: bool = True,
    ref_origin: str = "R",       # "R" | "L" | "mid"
    ref_axis_len: float = 0.30,
    startpos_L: Optional[np.ndarray] = None,
    startpos_R: Optional[np.ndarray] = None,
    # === yaw 정의 일치 옵션 ===
    forward_axis: str = "x",
    invert_yaw: bool = False,
    # ===== 누적 플로팅 옵션 =====
    ax=None,                     # 외부에서 축을 넘겨주면 그 축에 누적
    accumulate: bool = False,    # True면 figure를 비우지 않고 덧그림
    label: Optional[str] = None, # 범례용 라벨(prefix)
    # ===== 애니메이션 =====
    animate: bool = False,       # True면 좌우 번갈아가며 한 스텝씩 추가
    interval: float = 0.5,       # 프레임 간 시간(초)
    first_is_left: Optional[bool] = None,  # True: L 먼저 / False: R 먼저 (relaxed_path[0]==1이면 True)
    # ====== Enter 캡처 ======
    capture_enter: bool = False,          # Enter 누르면 현재 프레임 캡처/정지
    return_capture: bool = True,          # 캡처 dict를 반환(아니면 fig, ax 반환)
):
    """
    (x,y,z,c,s)는 global ref 좌표계 기준.
    animate=True면 first_is_left에 맞춰 좌우 번갈아가며 한 스텝씩 추가로 그립니다.

    capture_enter=True & animate=True 일 때:
      - 애니메이션 중 Enter/Return 키를 누르면 즉시 정지하고 캡처 dict를 반환합니다.
      - 반환 dict 예:
        {
          "captured": True,
          "kL": int, "kR": int,
          "xL": float, "yL": float, "zL": float, "cL": float, "sL": float,
          "xR": float, "yR": float, "zR": float, "cR": float, "sR": float,
          "active_foot": "L" or "R",
        }
      - 캡처 없이 끝까지 진행되면 None을 반환합니다(return_capture=True일 때).
    """
    # ---------- 준비 ----------
    xL = np.asarray(xL, float); yL = np.asarray(yL, float); zL = np.asarray(zL, float)
    xR = np.asarray(xR, float); yR = np.asarray(yR, float); zR = np.asarray(zR, float)

    # --- figure/axis 준비 ---
    if ax is None:
        fig = plt.figure("GCS Footsteps (Live)")
        if not accumulate:
            fig.clf()
        if fig.axes:
            ax = fig.axes[0]
        else:
            ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    def _draw_once(xLk, yLk, zLk, cLk, sLk, xRk, yRk, zRk, cRk, sRk, highlight=None):
        # accumulate=False면 프레임마다 지우고 다시 그림
        if not accumulate:
            ax.cla()

        # Left
        L_rects = compute_foot_rects(
            xLk, yLk, cLk, sLk,
            h=h, w=w,
            startpos_L=startpos_L, startpos_R=startpos_R,
            forward_axis=forward_axis, invert_yaw=invert_yaw
        )
        L_polys3d = [np.c_[poly, np.full(4, zLk[i])] for i, poly in enumerate(L_rects)]
        ax.add_collection3d(Poly3DCollection(
            L_polys3d, facecolors=left_color, edgecolors="k", linewidths=0.8, alpha=face_alpha,
            label=(None)
        ))
        ax.scatter(xLk, yLk, zLk, s=14, c=left_color)

        # Right
        R_rects = compute_foot_rects(
            xRk, yRk, cRk, sRk,
            h=h, w=w,
            startpos_L=startpos_L, startpos_R=startpos_R,
            forward_axis=forward_axis, invert_yaw=invert_yaw
        )
        R_polys3d = [np.c_[poly, np.full(4, zRk[i])] for i, poly in enumerate(R_rects)]
        ax.add_collection3d(Poly3DCollection(
            R_polys3d, facecolors=right_color, edgecolors="k", linewidths=0.8, alpha=face_alpha,
            label=(None)
        ))
        ax.scatter(xRk, yRk, zRk, s=14, c=right_color)

        # Yaw arrows
        if (startpos_L is not None) and (startpos_R is not None):
            x_ref, y_ref = _ref_axes_from_startRL(startpos_L, startpos_R)
            _draw_yaw_arrows_xyplane_ref(ax, xLk, yLk, zLk, cLk, sLk,
                                         x_ref=x_ref, y_ref=y_ref,
                                         total_len=arrow_len, head_frac=0.35, width_frac=0.6,
                                         color=left_color, lw=1.6,
                                         forward_axis=forward_axis, invert_yaw=invert_yaw)
            _draw_yaw_arrows_xyplane_ref(ax, xRk, yRk, zRk, cRk, sRk,
                                         x_ref=x_ref, y_ref=y_ref,
                                         total_len=arrow_len, head_frac=0.35, width_frac=0.6,
                                         color=right_color, lw=1.6,
                                         forward_axis=forward_axis, invert_yaw=invert_yaw)

        # 강조(방금 디딘 발만 크게)
        if highlight == "L" and len(xLk):
            i = len(xLk) - 1
            ax.scatter([xLk[i]], [yLk[i]], [zLk[i]], s=40, c=left_color, edgecolor="k")
        elif highlight == "R" and len(xRk):
            i = len(xRk) - 1
            ax.scatter([xRk[i]], [yRk[i]], [zRk[i]], s=40, c=right_color, edgecolor="k")

        # ax.bar3d(0,0,0,5,0.1,0, "k", alpha = 0.5)
        # ax.bar3d(0, 4.9, 0, 5, 0.1, 0, "k", alpha=0.5)
        # ax.bar3d(0, 0, 0, 0.1, 5, 0, "k", alpha=0.5)
        # ax.bar3d(4.9, 0, 0, 0.1, 5, 0, "k", alpha=0.5)
        # ax.bar3d(4.0, 0.5, 0, 0.1, 4.5, 0, "k", alpha=0.5)

        # axes/view
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        if matlab_style:
            ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_zlim(-0.1, 1.0)
            ax.grid(True)
            ax.view_init(elev=90, azim=-90)  # top view
            ax.set_box_aspect([5, 5, 1.1])
        else:
            if limits is None:
                (xmin, xmax), (ymin, ymax), (zmin, zmax) = _auto_limits(
                    np.r_[xLk, xRk], np.r_[yLk, yRk], np.r_[zLk, zRk], pad=0.15
                )
                ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
                ax.set_zlim(min(0.0, zmin), max(0.0, zmax))
            else:
                ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
            ax.set_box_aspect([
                ax.get_xlim()[1] - ax.get_xlim()[0],
                ax.get_ylim()[1] - ax.get_ylim()[0],
                max(1e-6, ax.get_zlim()[1] - ax.get_zlim()[0]),
            ])

        # 범례(옵션)
        if label:
            prev = []
            leg = ax.get_legend()
            if leg is not None:
                prev = [t.get_text() for t in leg.get_texts()]
            labels = prev + [label]
            handles = [Line2D([], [], linestyle='None') for _ in labels]
            ax.legend(handles, labels, loc="upper right",
                      handlelength=0, handletextpad=0.4, framealpha=0.85)

        plt.tight_layout()

        # accumulate=True면 블로킹 없이 갱신
        if accumulate:
            fig.canvas.draw()
            fig.canvas.flush_events()

    # ---------- 애니메이션 ----------
    if animate:
        if first_is_left is None:
            # 안전 장치: 외부에서 꼭 넘겨줘야 하지만, None이면 왼발 먼저로 가정
            first_is_left = True

        # === 캡처 상태 ===
        _stop_flag = {"stop": False}
        _last_step = {"iL": 0, "iR": 0, "highlight": None}  # 마지막 그려진 인덱스/발

        def _on_key(ev):
            if capture_enter and ev.key in ("enter", "return"):
                _stop_flag["stop"] = True

        if capture_enter:
            cid = fig.canvas.mpl_connect("key_press_event", _on_key)

        plt.ion()
        NL, NR = len(xL), len(xR)
        iL = iR = 0
        turn_left = bool(first_is_left)

        while (iL < NL) or (iR < NR):
            highlight = None
            if turn_left:
                if iL < NL:
                    iL += 1
                    highlight = "L"
                elif iR < NR:
                    iR += 1
                    highlight = "R"
            else:
                if iR < NR:
                    iR += 1
                    highlight = "R"
                elif iL < NL:
                    iL += 1
                    highlight = "L"

            _draw_once(
                xL[:iL], yL[:iL], zL[:iL], cL[:iL], sL[:iL],
                xR[:iR], yR[:iR], zR[:iR], cR[:iR], sR[:iR],
                highlight=highlight
            )
            _last_step["iL"] = iL
            _last_step["iR"] = iR
            _last_step["highlight"] = highlight

            if not accumulate:
                fig.canvas.draw(); fig.canvas.flush_events()
            plt.pause(interval)

            # === Enter 캡처: 현재 프레임에서 정지하고 dict 반환
            if capture_enter and _stop_flag["stop"]:
                kL = max(_last_step["iL"] - 1, 0)
                kR = max(_last_step["iR"] - 1, 0)
                active = _last_step["highlight"] or ("L" if first_is_left else "R")

                cap = dict(
                    captured=True,
                    kL=kL, kR=kR,
                    xL=float(xL[kL]) if len(xL) else None,
                    yL=float(yL[kL]) if len(yL) else None,
                    zL=float(zL[kL]) if len(zL) else None,
                    cL=float(cL[kL]) if len(cL) else None,
                    sL=float(sL[kL]) if len(sL) else None,
                    xR=float(xR[kR]) if len(xR) else None,
                    yR=float(yR[kR]) if len(yR) else None,
                    zR=float(zR[kR]) if len(zR) else None,
                    cR=float(cR[kR]) if len(cR) else None,
                    sR=float(sR[kR]) if len(sR) else None,
                    active_foot=active,
                )

                if capture_enter:
                    try:
                        fig.canvas.mpl_disconnect(cid)
                    except Exception:
                        pass

                plt.ioff()
                if show:
                    plt.show()

                if return_capture:
                    return cap
                else:
                    return fig, ax

            turn_left = not turn_left

        # 애니메이션이 캡처 없이 정상 종료
        if capture_enter:
            try:
                fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass

        plt.ioff()
        if show:
            plt.show()
        if return_capture:
            return None
        return fig, ax

    # ---------- 정적(원래 동작) ----------
    _draw_once(xL, yL, zL, cL, sL, xR, yR, zR, cR, sR, highlight=None)

    if accumulate:
        fig.canvas.draw()
        fig.canvas.flush_events()
    elif show:
        plt.show()

    return fig, ax

def plot_feet_3d_VAE(
    xL, yL, zL, cL, sL,
    xR, yR, zR, cR, sR,
    *,
    h: float = 0.15, w: float = 0.07,
    arrow_len: float = 0.15,
    left_color: str = "#D62728",  # red-ish
    right_color: str = "#1F77B4",  # blue-ish
    face_alpha: float = 0.95,
    limits: Optional[Tuple[Tuple[float,float],Tuple[float,float],Tuple[float,float]]] = None,
    matlab_style: bool = True,
    show: bool = True,
    startpos_L: Optional[np.ndarray] = None,
    startpos_R: Optional[np.ndarray] = None,
    forward_axis: str = "x",
    invert_yaw: bool = False,
    ax=None,
    accumulate: bool = False,
    label: Optional[str] = None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # ---------- 준비 ----------
    xL = np.asarray(xL, float); yL = np.asarray(yL, float); zL = np.asarray(zL, float)
    xR = np.asarray(xR, float); yR = np.asarray(yR, float); zR = np.asarray(zR, float)
    cL = np.asarray(cL, float); sL = np.asarray(sL, float)
    cR = np.asarray(cR, float); sR = np.asarray(sR, float)

    # --- figure/axis 준비 ---
    if ax is None:
        fig = plt.figure("Generated stepping stone")
        if not accumulate:
            fig.clf()
        if fig.axes:
            ax = fig.axes[0]
        else:
            ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # =========================================================
    # (핵심) plot_stones_3d(view_top=True)가 만든 2D ax가 보이면
    #       자동으로 그 ax로 redirect (main 코드 수정 불필요)
    # =========================================================
    ax2d = getattr(fig, "_gcs_top2d_ax", None)
    if ax2d is not None and ax2d in fig.axes and ax2d.get_visible():
        ax = ax2d

    # 현재 ax가 3D인지 판별
    is3d = hasattr(ax, "add_collection3d")

    # ---------- 2D yaw arrow (간단 버전, 원하면 꺼도 됨) ----------
    def _draw_yaw_arrows_2d(ax, x, y, c, s, color):
        if (startpos_L is None) or (startpos_R is None):
            return
        # 기존에 쓰던 기준축 함수 재사용 (네 코드에 이미 있다고 가정)
        x_ref, y_ref = _ref_axes_from_startRL(startpos_L, startpos_R)

        c = np.asarray(c, float).copy()
        s = np.asarray(s, float).copy()
        if invert_yaw:
            s = -s

        # forward_axis에 따른 방향(대략 동일하게 맞춤)
        if forward_axis == "x":
            dx = arrow_len * (c * x_ref[0] + s * y_ref[0])
            dy = arrow_len * (c * x_ref[1] + s * y_ref[1])
        else:  # "y"
            dx = arrow_len * (s * x_ref[0] + c * y_ref[0])
            dy = arrow_len * (s * x_ref[1] + c * y_ref[1])

        head_w = 0.15 * w
        head_l = 0.25 * arrow_len
        for i in range(len(x)):
            ax.arrow(float(x[i]), float(y[i]),
                     float(dx[i]), float(dy[i]),
                     head_width=head_w, head_length=head_l,
                     fc=color, ec=color, linewidth=1.2,
                     length_includes_head=True)

    def _draw_once(xLk, yLk, zLk, cLk, sLk, xRk, yRk, zRk, cRk, sRk, highlight=None):
        if not accumulate:
            ax.cla()

        # =========================
        # 3D 모드 (기존 그대로)
        # =========================
        if is3d:
            # Left
            L_rects = compute_foot_rects(
                xLk, yLk, cLk, sLk,
                h=h, w=w,
                startpos_L=startpos_L, startpos_R=startpos_R,
                forward_axis=forward_axis, invert_yaw=invert_yaw
            )
            L_polys3d = [np.c_[poly, np.full(4, zLk[i])] for i, poly in enumerate(L_rects)]
            ax.add_collection3d(Poly3DCollection(
                L_polys3d, facecolors=left_color, edgecolors="k",
                linewidths=0.8, alpha=face_alpha, label=None
            ))
            ax.scatter(xLk, yLk, zLk, s=14, c=left_color)

            # Right
            R_rects = compute_foot_rects(
                xRk, yRk, cRk, sRk,
                h=h, w=w,
                startpos_L=startpos_L, startpos_R=startpos_R,
                forward_axis=forward_axis, invert_yaw=invert_yaw
            )
            R_polys3d = [np.c_[poly, np.full(4, zRk[i])] for i, poly in enumerate(R_rects)]
            ax.add_collection3d(Poly3DCollection(
                R_polys3d, facecolors=right_color, edgecolors="k",
                linewidths=0.8, alpha=face_alpha, label=None
            ))
            ax.scatter(xRk, yRk, zRk, s=14, c=right_color)

            # Yaw arrows(기존 함수 그대로)
            if (startpos_L is not None) and (startpos_R is not None):
                x_ref, y_ref = _ref_axes_from_startRL(startpos_L, startpos_R)
                _draw_yaw_arrows_xyplane_ref(ax, xLk, yLk, zLk, cLk, sLk,
                                             x_ref=x_ref, y_ref=y_ref,
                                             total_len=arrow_len, head_frac=0.35, width_frac=0.6,
                                             color=left_color, lw=1.6,
                                             forward_axis=forward_axis, invert_yaw=invert_yaw)
                _draw_yaw_arrows_xyplane_ref(ax, xRk, yRk, zRk, cRk, sRk,
                                             x_ref=x_ref, y_ref=y_ref,
                                             total_len=arrow_len, head_frac=0.35, width_frac=0.6,
                                             color=right_color, lw=1.6,
                                             forward_axis=forward_axis, invert_yaw=invert_yaw)

        # =========================
        # 2D 모드 (top-down)
        # =========================
        else:
            # Left
            L_rects = compute_foot_rects(
                xLk, yLk, cLk, sLk,
                h=h, w=w,
                startpos_L=startpos_L, startpos_R=startpos_R,
                forward_axis=forward_axis, invert_yaw=invert_yaw
            )
            for poly in L_rects:
                ax.add_patch(Polygon(poly, closed=True, facecolor=left_color,
                                     edgecolor="k", linewidth=0.8, alpha=face_alpha))
            ax.scatter(xLk, yLk, s=14, c=left_color)

            # Right
            R_rects = compute_foot_rects(
                xRk, yRk, cRk, sRk,
                h=h, w=w,
                startpos_L=startpos_L, startpos_R=startpos_R,
                forward_axis=forward_axis, invert_yaw=invert_yaw
            )
            for poly in R_rects:
                ax.add_patch(Polygon(poly, closed=True, facecolor=right_color,
                                     edgecolor="k", linewidth=0.8, alpha=face_alpha))
            ax.scatter(xRk, yRk, s=14, c=right_color)

            # 2D yaw arrows(원하면 주석 가능)
            _draw_yaw_arrows_2d(ax, xLk, yLk, cLk, sLk, left_color)
            _draw_yaw_arrows_2d(ax, xRk, yRk, cRk, sRk, right_color)

            ax.set_aspect("equal", adjustable="box")

            # limits 적용(2D일 때는 x/y만)
            if limits is not None:
                (xmin, xmax), (ymin, ymax), _ = limits
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

    # ---------- 정적(원래 동작) ----------
    _draw_once(xL, yL, zL, cL, sL, xR, yR, zR, cR, sR, highlight=None)

    if accumulate:
        fig.canvas.draw()
        fig.canvas.flush_events()
    elif show:
        plt.show()

    return fig, ax



def plot_feet_3d_wfc(
    xL, yL, zL, cL, sL,
    xR, yR, zR, cR, sR,
    *,
    # 발 크기/색
    h: float = 0.20, w: float = 0.10,
    left_color: str = "#F58518",
    right_color: str = "#4C78A8",
    face_alpha: float = 0.35,
    # ref 축(필수)
    x_ref_override: np.ndarray,
    y_ref_override: np.ndarray,
    # yaw 정의 옵션
    forward_axis: str = "x",
    invert_yaw: bool = False,
    # 축 범위
    world_size: tuple[float, float] | None = None,  # (W*tile, H*tile)
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] = (-0.1, 1.0),
    pad: float = 0.15,
    ax=None,
    show: bool = True,
):

    if ax is None:
        fig = plt.figure("GCS Footsteps"); fig.clf()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    """ref 기준 (c,s) + 외부 ref 축으로 한 번에 정적 플롯."""
    # --- 입력 정리 ---
    xL = np.asarray(xL, float); yL = np.asarray(yL, float); zL = np.asarray(zL, float)
    xR = np.asarray(xR, float); yR = np.asarray(yR, float); zR = np.asarray(zR, float)

    # --- ref 축 정규화 & 우수 보장 ---
    xr = np.asarray(x_ref_override, float).reshape(2)
    yr = np.asarray(y_ref_override, float).reshape(2)
    xr /= (np.linalg.norm(xr) + 1e-12)
    yr /= (np.linalg.norm(yr) + 1e-12)
    if xr[0]*yr[1] - xr[1]*yr[0] < 0:
        xr = -xr

    # --- figure/axis ---
    fig = plt.figure("GCS Footsteps")
    fig.clf()
    ax = fig.add_subplot(111, projection="3d")

    # --- 좌/우 폴리곤 + 포인트 ---
    L_rects = compute_foot_rects_wfc(
        xL, yL, cL, sL, h=h, w=w,
        forward_axis=forward_axis, invert_yaw=invert_yaw,
        x_ref_override=xr, y_ref_override=yr,
    )
    L_polys3d = [np.c_[poly, np.full(4, zL[i])] for i, poly in enumerate(L_rects)]
    ax.add_collection3d(Poly3DCollection(L_polys3d, facecolors=left_color,
                                         edgecolors="k", linewidths=0.8, alpha=face_alpha))
    ax.scatter(xL, yL, zL, s=14, c=left_color)

    R_rects = compute_foot_rects_wfc(
        xR, yR, cR, sR, h=h, w=w,
        forward_axis=forward_axis, invert_yaw=invert_yaw,
        x_ref_override=xr, y_ref_override=yr,
    )
    R_polys3d = [np.c_[poly, np.full(4, zR[i])] for i, poly in enumerate(R_rects)]
    ax.add_collection3d(Poly3DCollection(R_polys3d, facecolors=right_color,
                                         edgecolors="k", linewidths=0.8, alpha=face_alpha))
    ax.scatter(xR, yR, zR, s=14, c=right_color)

    # --- yaw 화살표(같은 ref 축 사용) ---
    _draw_yaw_arrows_xyplane_ref(ax, xL, yL, zL, cL, sL,
                                 x_ref=xr, y_ref=yr,
                                 total_len=0.15, head_frac=0.35, width_frac=0.6,
                                 color=left_color, lw=1.6,
                                 forward_axis=forward_axis, invert_yaw=invert_yaw)
    _draw_yaw_arrows_xyplane_ref(ax, xR, yR, zR, cR, sR,
                                 x_ref=xr, y_ref=yr,
                                 total_len=0.15, head_frac=0.35, width_frac=0.6,
                                 color=right_color, lw=1.6,
                                 forward_axis=forward_axis, invert_yaw=invert_yaw)

    # --- 축/뷰 ---
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    if world_size is not None:
        ax.set_xlim(0, float(world_size[0]))
        ax.set_ylim(0, float(world_size[1]))
    elif (xlim is not None) and (ylim is not None):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    else:
        xs = np.r_[xL, xR]; ys = np.r_[yL, yR]
        xmin, xmax = xs.min()-pad, xs.max()+pad
        ymin, ymax = ys.min()-pad, ys.max()+pad
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    ax.set_zlim(*zlim)
    ax.grid(True)
    ax.view_init(elev=90, azim=-90)  # top view

    # 균등 비율
    ax.set_box_aspect([
        ax.get_xlim()[1]-ax.get_xlim()[0],
        ax.get_ylim()[1]-ax.get_ylim()[0],
        max(1e-6, ax.get_zlim()[1]-ax.get_zlim()[0]),
    ])
    plt.tight_layout()

    if show: plt.show()
    return fig, ax

