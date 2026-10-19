# src/viz/kinematic.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# --- 회전 행렬 (roll, pitch만) ---
def rotX(q: float) -> np.ndarray:
    c, s = np.cos(q), np.sin(q)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]], float)

def rotY(q: float) -> np.ndarray:
    c, s = np.cos(q), np.sin(q)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], float)

# kinematic.py 상단 회전함수들 옆에 추가
def rotZ(q: float) -> np.ndarray:
    c, s = np.cos(q), np.sin(q)
    return np.array([[ c,-s, 0],
                     [ s, c, 0],
                     [ 0, 0, 1]], float)

# --- 2링크 무릎 위치(시각화용 간이 IK) ---
def _knee_point(hip: np.ndarray, foot: np.ndarray,
                l_thigh: float, l_shank: float,
                fwd_dir: np.ndarray,
                side_sign: float = +1.0,
                phi_min_deg: float = 15.0,   # 최소 굽힘각 [deg] (0°와 180° 모두로부터 떨어뜨림)
                bend_gain: float = 1.0,      # 굽힘 높이 증폭
                h_min: float = 0.0           # 최소 높이(m) 강제 (옵션)
                ) -> np.ndarray:
    v = foot - hip
    c = float(np.linalg.norm(v))
    if c < 1e-9:
        return hip.copy()

    vhat = v / c
    fwd = fwd_dir / max(1e-9, np.linalg.norm(fwd_dir))

    # 사지면(hip→foot 와 fwd가 만드는 평면) 내에서 굽힘
    plane_n = np.cross(vhat, fwd)
    n_norm = np.linalg.norm(plane_n)
    if n_norm < 1e-9:
        tmp = np.array([0, 0, 1.0])
        plane_n = np.cross(vhat, tmp)
        n_norm = np.linalg.norm(plane_n)
        if n_norm < 1e-9:
            tmp = np.array([0, 1.0, 0])
            plane_n = np.cross(vhat, tmp)
            n_norm = np.linalg.norm(plane_n)
    plane_n /= n_norm
    offset_dir = np.cross(plane_n, vhat)
    offset_dir /= max(1e-9, np.linalg.norm(offset_dir))

    a, b = float(l_thigh), float(l_shank)

    # 현재 무릎각 (삼각법)
    cos_phi = (a*a + b*b - c*c) / max(1e-9, 2.0*a*b)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = float(np.arccos(cos_phi))              # 0 ~ π (π에 가까우면 쭉 편 자세)

    # 0/π 모두로부터 떨어뜨리기 위한 "접힘 거리"
    phi_min = np.radians(phi_min_deg)
    phi_fold = min(phi, np.pi - phi)             # {0,π}와의 최소거리
    if phi_fold < phi_min:
        # 어느 쪽에 가까운지에 따라 목표각을 정함
        phi_target = (phi_min if phi <= np.pi/2 else np.pi - phi_min)
        # 해당 각을 만들 수 있는 c_req
        c_req = np.sqrt(max(a*a + b*b - 2*a*b*np.cos(phi_target), 1e-9))
        c_eff = min(c, c_req)  # 선분 밖으로 나가지 않도록
    else:
        c_eff = c

    # 분할거리/높이
    x = (a*a - b*b + c_eff*c_eff) / max(1e-9, 2.0*c_eff)
    x = float(np.clip(x, 0.0, c))
    h_sq = max(a*a - x*x, 0.0)
    h = float(np.sqrt(h_sq))
    h = max(h, h_min) * bend_gain

    return hip + x * vhat + (side_sign * h) * offset_dir



# --- 발 사각형 ---
def _foot_poly(ax, center: np.ndarray, R: np.ndarray,
               length: float = 0.2, width: float = 0.1, **plot_kw):
    L = length * 0.5; W = width * 0.5
    pts_local = np.array([
        [ L,  W, 0],
        [-L,  W, 0],
        [-L, -W, 0],
        [ L, -W, 0],
        [ L,  W, 0],
    ], float)
    pts = (R @ pts_local.T).T + center
    kw = {"linewidth": 2}
    kw |= plot_kw
    line, = ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], **kw)
    return line


# --- 메인: 스틱 로봇 그리기 ---
def draw_robot(ax3d,
               LF: np.ndarray, RF: np.ndarray,
               roll_pitch=(0.0, 0.0),
               *,
               # 새 인자들 ↓ (없으면 +y를 기본으로)
               yaw_body: float | None = None,
               yaw_L: float | None = None,
               yaw_R: float | None = None,
               z_pelvis: float = 0.0,
               hip_offset: float = 0.0,
               l_thigh: float = 0.42,
               l_shank: float = 0.42,
               foot_len: float = 0.2,
               foot_wid: float = 0.08,
               foot_color_L='r',
               foot_color_R='b',
               knee_phi_min_deg: float = 15.0,
               knee_bend_gain: float = 1.0,
               knee_h_min: float = 0.0,
               foot_alpha=1.0,
               line_width: float = 3.0,
               line_color: str = 'k'):
    """
    LF, RF: (3,) numpy (발끝 중심 좌표)
    roll_pitch: (roll, pitch) 라디안
    yaw_body: 몸통 yaw [rad] (없으면 두 발 평균 yaw; 그것도 없으면 +y(= +pi/2))
    yaw_L, yaw_R: 각 발 yaw [rad] (없으면 yaw_body 사용)
    """
    LF = np.asarray(LF, float).reshape(3)
    RF = np.asarray(RF, float).reshape(3)
    roll, pitch = map(float, roll_pitch)

    # --- COM & pelvis 위치
    COM = 0.5 * (LF + RF)
    pPEL = np.array([COM[0], COM[1], z_pelvis], float)

    # --- yaw 기본값/평균 처리
    default_yaw = np.pi * 0.5   # +y 방향
    def _mean_yaw(yaws):
        cs = np.array([np.cos(y) for y in yaws])
        sn = np.array([np.sin(y) for y in yaws])
        return float(np.arctan2(sn.sum(), cs.sum()))
    if yaw_body is None:
        known = []
        if yaw_L is not None: known.append(float(yaw_L))
        if yaw_R is not None: known.append(float(yaw_R))
        yaw_body = _mean_yaw(known) if known else default_yaw
    if yaw_L is None: yaw_L = yaw_body
    if yaw_R is None: yaw_R = yaw_body

    # --- 회전행렬
    Rpel   = rotZ(yaw_body) @ rotX(roll) @ rotY(-pitch)
    RfootL = rotZ(yaw_L)    @ rotX(roll) @ rotY(-pitch)
    RfootR = rotZ(yaw_R)    @ rotX(roll) @ rotY(-pitch)

    # --- 힙 좌/우 위치 (골반 프레임의 [0,±hip_offset,0]를 Rpel로 회전 후 pPEL로 이동)
    off_L = Rpel @ np.array([0.0, +hip_offset, 0.0])
    off_R = Rpel @ np.array([0.0, -hip_offset, 0.0])
    hipL = pPEL + off_L
    hipR = pPEL + off_R

    # --- 무릎 굽힘 평면 진행방향: 몸통 로컬 +x (앞)
    fwd_dir = Rpel @ np.array([1.0, 0.0, 0.0])

    # --- 무릎 (왼발 +1, 오른발 -1) : 같은 '앞쪽'으로 굽히게
    kneeL = _knee_point(hipL, LF, l_thigh, l_shank, fwd_dir=fwd_dir,
                        side_sign=+1.0,
                        phi_min_deg=knee_phi_min_deg,
                        bend_gain=knee_bend_gain,
                        h_min=knee_h_min)
    kneeR = _knee_point(hipR, RF, l_thigh, l_shank, fwd_dir=fwd_dir,
                        side_sign=+1.0,
                        phi_min_deg=knee_phi_min_deg,
                        bend_gain=knee_bend_gain,
                        h_min=knee_h_min)

    # --- 선분(스틱) 그리기
    segs = [
        (pPEL, hipL), (hipL, kneeL), (kneeL, LF),
        (pPEL, hipR), (hipR, kneeR), (kneeR, RF),
    ]

    lc = Line3DCollection([(a, b) for a, b in segs], linewidths=line_width, colors=[line_color])
    lc.set_zorder(2_000_000)
    ax3d.add_collection3d(lc)

    # --- 몸통 프레임(간단 박스)
    torso_box = np.array([[+0.08,+0.12,+0.08],
                          [-0.08,+0.12,+0.08],
                          [-0.08,-0.12,+0.08],
                          [+0.08,-0.12,+0.08],
                          [+0.08,+0.12,+0.08]], float)
    torso_pts = (Rpel @ torso_box.T).T + pPEL
    torso_line, = ax3d.plot(torso_pts[:,0], torso_pts[:,1], torso_pts[:,2], linewidth=2, color=line_color, zorder=1_000_000)

    # --- 발 외곽 (각자 yaw 반영)
    hL = _foot_poly(ax3d, LF, RfootL, length=foot_len, width=foot_wid,
                    color=foot_color_L, alpha=foot_alpha)
    hL.set_zorder(1_000_000)
    hR = _foot_poly(ax3d, RF, RfootR, length=foot_len, width=foot_wid,
                    color=foot_color_R, alpha=foot_alpha)
    hR.set_zorder(1_000_000)
    # --- COM/무릎 마커
    # scCOM = ax3d.scatter([COM[0]], [COM[1]], [COM[2]], s=30)
    # scKL  = ax3d.scatter([kneeL[0]], [kneeL[1]], [kneeL[2]], s=20)
    # scKR  = ax3d.scatter([kneeR[0]], [kneeR[1]], [kneeR[2]], s=20)

    # --- 핸들 모으기 & (있으면) 동적 컨테이너에 등록
    handles = [lc, torso_line, hL, hR]
    if hasattr(ax3d, "_kin_dynamic"):
        ax3d._kin_dynamic["collections"].append(lc)
        ax3d._kin_dynamic["lines"].extend([torso_line, hL, hR])
        # ax3d._kin_dynamic["scatters"].extend([scCOM, scKL, scKR])

    return handles


# --- DDR 출력과 바로 맞물리는 어댑터 ---
def draw_robot_from_ddr(ax3d,
                        xL, yL, zL, cL=None, sL=None,
                        xR=None, yR=None, zR=None, cR=None, sR=None,
                        roll: float = 0.0, pitch: float = 0.0,
                        **kwargs):
    """
    DDR (x,y,z,cos,sin) 포맷을 받아 발/몸통 yaw까지 반영해서 그린다.
    - 발 yaw: 각 발의 yaw = atan2(s, c)
    - 몸통 yaw: 두 발 yaw의 평균 방향 (둘 다 없으면 +y 기본)
    """
    LF = np.array([xL, yL, zL], float)
    RF = np.array([xR, yR, zR], float)

    yaw_L = float(np.arctan2(sL, cL)) if (cL is not None and sL is not None) else None
    yaw_R = float(np.arctan2(sR, cR)) if (cR is not None and sR is not None) else None

    # 몸통 yaw은 draw_robot 내부에서 평균/기본 처리하므로 여기선 None 전달 가능
    draw_robot(ax3d, LF, RF, (roll, pitch),
               yaw_body=None, yaw_L=yaw_L, yaw_R=yaw_R,
               **kwargs)


# --- 축 준비 유틸 ---
def prepare_3d_axes(fig=None, rect=111, lims=((-0.4,0.4),(-0.4,0.4),(-0.05,1.0))):
    if fig is None:
        fig = plt.figure(figsize=(7,5))
    ax3d = fig.add_subplot(rect, projection="3d")
    (xlim, ylim, zlim) = lims
    ax3d.set_xlim(*xlim); ax3d.set_ylim(*ylim); ax3d.set_zlim(*zlim)
    ax3d.set_box_aspect([1,1,1])
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    return ax3d
