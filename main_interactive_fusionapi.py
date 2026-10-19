from __future__ import annotations
import time
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.viz import plot_feet_3d
# ====== 당신의 기존 모듈들 ======
from src.experiments import list_experiments
from src.terrain_preset import load_terrain, list_terrains
from src.utils import tic, toc, expand_shorthand, postprocess_solution
from src.graph import build_graph, compute_centroids_from_vertices
from src.model import build_gcs_model, pack_relaxed_params_for_model, prepare_problem_params
from src.solver import (
    set_parameters, solve_relaxed, get_relaxed_y,
    run_trials_select_best, extract_solution
)
import math
import numpy as np

def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

def update_entry_goal(entry: dict, *, goal_center, goal_heading_deg: float) -> dict:
    """
    entry (축약 포맷)를 받아 goal.center, goal.heading_deg를 새 값으로 교체.
    foot_yaw_deg의 goal_*도 새 heading 기준으로 재계산.
    - start_L/R yaw는 항상 0 유지
    """
    entry2 = dict(entry)
    entry2["goal"] = dict(entry2.get("goal", {}))
    entry2["goal"]["center"] = [float(goal_center[0]), float(goal_center[1]), float(goal_center[2])]
    entry2["goal"]["heading_deg"] = float(goal_heading_deg)

    # foot yaw 규칙: start는 0, goal은 (goal_heading - start_heading)
    start_heading_deg = float(entry2["start"]["heading_deg"])
    delta = _wrap_deg(float(goal_heading_deg) - start_heading_deg)

    fy = dict(entry2.get("foot_yaw_deg", {}))
    fy.update({
        "start_L": 0.0, "start_R": 0.0,
        "goal_L":  float(delta), "goal_R": float(delta),
    })
    entry2["foot_yaw_deg"] = fy
    return entry2


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

def update_entry_start_from_last(entry: dict, last_pose: dict, stance_width: float = 0.25) -> dict:
    """
    entry: expand_shorthand '이전'의 축약 포맷( start.center/heading_deg, goal.center/heading_deg )
    last_pose: animator.last_pose (전역 yaw 포함)
    반환: start/foot_yaw_deg만 갱신된 새 entry (goal은 그대로 유지)
    규칙:
      - start.center = (last foot로부터) center 재계산
      - start.heading_deg = yaw_global(deg)
      - foot_yaw_deg.start_* = 0
      - foot_yaw_deg.goal_*  = wrap(goal_heading_deg - start.heading_deg)
    """
    # 전역 yaw
    if "yaw_global" in last_pose:
        heading = float(last_pose["yaw_global"])
        c_g, s_g = float(np.cos(heading)), float(np.sin(heading))
    elif "c_global" in last_pose and "s_global" in last_pose:
        c_g, s_g = float(last_pose["c_global"]), float(last_pose["s_global"])
        heading = float(np.arctan2(s_g, c_g))
    else:
        raise ValueError("last_pose에 yaw_global 또는 c_global/s_global 필요")

    heading_deg = float(np.degrees(heading))

    # 마지막 발에서 center/L/R 재계산
    px, py, pz = float(last_pose["x"]), float(last_pose["y"]), float(last_pose["z"])
    nL = np.array([-s_g, c_g], float)  # 왼쪽 법선
    foot = last_pose.get("foot")
    if foot == "R":
        pR = np.array([px, py], float)
        center = pR + 0.5 * stance_width * nL
        pL = pR + stance_width * nL
    elif foot == "L":
        pL = np.array([px, py], float)
        center = pL - 0.5 * stance_width * nL
        pR = pL - stance_width * nL
    else:
        raise ValueError("last_pose['foot'] must be 'L' or 'R'")

    # goal heading은 기존 entry의 goal 그대로 사용
    goal_heading_deg = float(entry["goal"]["heading_deg"])
    delta_deg = _wrap_deg(goal_heading_deg - heading_deg)

    # 기존 entry 복사 → start/foot_yaw_deg만 갱신 (goal은 고정)
    entry_next = dict(entry)
    entry_next["stance_width"] = float(stance_width)
    entry_next["start"] = {
        "center": [float(center[0]), float(center[1]), float(pz)],
        "heading_deg": float(heading_deg),
    }
    # foot yaw 규칙: start=0, goal=(goal - current)
    fy = dict(entry_next.get("foot_yaw_deg", {}))
    fy.update({
        "start_L": 0.0, "start_R": 0.0,
        "goal_L":  float(delta_deg), "goal_R": float(delta_deg),
    })
    entry_next["foot_yaw_deg"] = fy
    # goal은 entry["goal"] 그대로 둠
    return entry_next


def _wrap_deg(a):
    # [-180, 180) 로 정규화
    return (a + 180.0) % 360.0 - 180.0



def _wrap_deg(a: float) -> float:
    # [-180, 180)로 정규화
    return (a + 180.0) % 360.0 - 180.0

class StepAnimator:
    ...
    def stop(self):
        if not self.stopped:
            self.stopped = True
            try:
                self.timer.stop()
            except Exception:
                pass
            # 창 닫지 않음! (plt.close 금지)

    def dispose(self):
        """타이머/키 이벤트 정리 (figure는 유지)"""
        try:
            if self.timer:
                self.timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "cid") and self.cid is not None:
                self.ax.figure.canvas.mpl_disconnect(self.cid)
                self.cid = None
        except Exception:
            pass

def _infer_goal_heading_deg(exp_or_entry: dict) -> float:
    """
    goal heading_deg를 얻는 우선순위:
    1) entry["goal"]["heading_deg"]가 있으면 그것 사용
    2) targetpos_L/R(5D)가 있으면 cos,sin으로 계산
    3) targetpos_L/R(3D)만 있으면 R->L로 y축 만들고 x축=[y,-x]로 계산
    4) 없으면 0.0
    """
    # 1) goal.heading_deg 있는 경우
    g = exp_or_entry.get("goal")
    if g and "heading_deg" in g:
        return float(g["heading_deg"])

    # 2) 5D인 경우
    if "targetpos_L" in exp_or_entry and len(exp_or_entry["targetpos_L"]) >= 5:
        c, s = float(exp_or_entry["targetpos_L"][3]), float(exp_or_entry["targetpos_L"][4])
        return float(np.degrees(np.arctan2(s, c)))

    # 3) 3D 좌표만 있는 경우
    if "targetpos_L" in exp_or_entry and "targetpos_R" in exp_or_entry:
        pL = np.asarray(exp_or_entry["targetpos_L"], float).ravel()[:2]
        pR = np.asarray(exp_or_entry["targetpos_R"], float).ravel()[:2]
        v = pL - pR
        n = np.linalg.norm(v)
        if n > 1e-9:
            y_goal = v / n
            x_goal = np.array([y_goal[1], -y_goal[0]])  # 너의 convention(x_ref) 유지
            return float(np.degrees(np.arctan2(x_goal[1], x_goal[0])))

    # 4) fallback
    return 0.0

def build_next_entry_from_last(exp_or_entry: dict, last_pose: dict, stance_width: float = 0.25) -> dict:
    """
    - exp_or_entry: 기존 expand_shorthand 이전/이후 어떤 포맷이어도 OK.
      (goal heading은 위 보조함수로 유추)
    - last_pose: animator.last_pose (전역 기준 yaw 정보 포함)
    - 반환: expand_shorthand가 바로 먹는 '축약 포맷' entry
      {
        "start": {"center":[x,y,z], "heading_deg": ...},
        "goal":  {"center":[x,y,z], "heading_deg": ...},
        "stance_width": stance_width,
        "foot_yaw_deg": {"start_L":0, "start_R":0, "goal_L":Δ, "goal_R":Δ}
      }
    """
    # ---- 1) 전역 heading (현재 자세) ----
    if "yaw_global" in last_pose:
        heading = float(last_pose["yaw_global"])
        c_g, s_g = float(np.cos(heading)), float(np.sin(heading))
    elif "c_global" in last_pose and "s_global" in last_pose:
        c_g, s_g = float(last_pose["c_global"]), float(last_pose["s_global"])
        heading = float(np.arctan2(s_g, c_g))
    else:
        raise ValueError("last_pose에 yaw_global 또는 c_global/s_global가 필요합니다.")
    heading_deg = float(np.degrees(heading))

    # ---- 2) center/L/R 계산 ----
    px, py, pz = float(last_pose["x"]), float(last_pose["y"]), float(last_pose["z"])
    nL = np.array([-s_g, c_g], float)     # 왼쪽 법선

    foot = last_pose.get("foot")
    if foot == "R":
        pR = np.array([px, py], float)
        center = pR + 0.5 * stance_width * nL
        pL = pR + stance_width * nL
    elif foot == "L":
        pL = np.array([px, py], float)
        center = pL - 0.5 * stance_width * nL
        pR = pL - stance_width * nL
    else:
        raise ValueError("last_pose['foot'] must be 'L' or 'R'.")

    # ---- 3) goal heading 계산 ----
    goal_heading_deg = _infer_goal_heading_deg(exp_or_entry)

    # ---- 4) 축약 포맷 entry 구성 ----
    # goal.center는 기존 exp/entry에 있으면 그대로, 없으면 기존 targetpos의 평균으로 추정
    if "goal" in exp_or_entry and "center" in exp_or_entry["goal"]:
        goal_center = exp_or_entry["goal"]["center"]
    elif "targetpos_L" in exp_or_entry and "targetpos_R" in exp_or_entry:
        tpL = np.asarray(exp_or_entry["targetpos_L"], float).ravel()
        tpR = np.asarray(exp_or_entry["targetpos_R"], float).ravel()
        gz = float(tpL[2]) if tpL.size >= 3 else 0.0
        goal_center = [float((tpL[0]+tpR[0])/2), float((tpL[1]+tpR[1])/2), gz]
    else:
        # 정보가 전혀 없으면 현재 z를 재사용 (x,y는 유지)
        goal_center = [float(center[0]), float(center[1]), float(pz)]

    # foot_yaw_deg 규칙:
    # - start_L/R = 0
    # - goal_L/R  = wrap(goal_heading_deg - heading_deg)
    delta_deg = _wrap_deg(goal_heading_deg - heading_deg)

    entry_next = {
        "start": {"center": [float(center[0]), float(center[1]), float(pz)],
                  "heading_deg": float(heading_deg)},
        "goal":  {"center": [float(goal_center[0]), float(goal_center[1]), float(goal_center[2])],
                  "heading_deg": float(goal_heading_deg)},
        "stance_width": float(stance_width),
        "foot_yaw_deg": {
            "start_L": 0.0, "start_R": 0.0,
            "goal_L":  float(delta_deg), "goal_R": float(delta_deg),
        },
    }
    return entry_next


def build_next_exp_from_last(exp: dict, last_pose: dict, stance_width: float = 0.25) -> dict:
    """
    exp : expand_shorthand(exp_raw, stance_width) 이후의 dict
          (startpos_*, targetpos_* 는 5D: [x,y,z,cos,sin])
    last_pose : animator.last_pose (전역 기준 yaw_global 포함)
    stance_width : 보폭 (기본 0.25)

    반환: exp_next (다음 최적화 시작용)
      - startpos_L/R : center, heading으로부터 재구성
      - startyaw_L/R : 0 deg
      - targetyaw_L/R: goal_heading_deg - heading_deg (각도 wrap)
    """
    # ---- 1) 현재 전역 heading ----
    if "yaw_global" in last_pose:
        heading = float(last_pose["yaw_global"])
        c_g, s_g = float(np.cos(heading)), float(np.sin(heading))
    elif "c_global" in last_pose and "s_global" in last_pose:
        c_g, s_g = float(last_pose["c_global"]), float(last_pose["s_global"])
        heading = float(np.arctan2(s_g, c_g))
    else:
        raise ValueError("last_pose에 yaw_global 또는 c_global/s_global가 필요합니다.")

    heading_deg = float(np.degrees(heading))

    # ---- 2) center 계산 ----
    # 마지막 찍힌 발 좌표
    px, py, pz = float(last_pose["x"]), float(last_pose["y"]), float(last_pose["z"])
    # 진행방향 t, 왼쪽 법선 n
    t = np.array([c_g, s_g], dtype=float)
    n = np.array([-s_g, c_g], dtype=float)

    if last_pose.get("foot") == "R":
        pR = np.array([px, py], float)
        center = pR + 0.5 * stance_width * n
        pL = pR + stance_width * n
    elif last_pose.get("foot") == "L":
        pL = np.array([px, py], float)
        center = pL - 0.5 * stance_width * n
        pR = pL - stance_width * n
    else:
        # 마지막 발 정보를 신뢰하지 않고 center만 쓰고 싶다면:
        # center = np.array([px, py], float)
        # pL = center + 0.5 * stance_width * n
        # pR = center - 0.5 * stance_width * n
        raise ValueError("last_pose['foot'] must be 'L' or 'R'.")

    # ---- 3) 새 startpos_L/R 구성 (5D: x,y,z,cos,sin) ----
    startpos_L = np.array([pL[0], pL[1], pz, c_g, s_g], dtype=float)
    startpos_R = np.array([pR[0], pR[1], pz, c_g, s_g], dtype=float)

    # ---- 4) goal 측의 heading_deg (각 발별) ----
    # expand_shorthand 이후라면 targetpos_* 가 5D([x,y,z,c,s])
    gL_c, gL_s = float(exp["targetpos_L"][3]), float(exp["targetpos_L"][4])
    gR_c, gR_s = float(exp["targetpos_R"][3]), float(exp["targetpos_R"][4])
    goalL_deg = float(np.degrees(np.arctan2(gL_s, gL_c)))
    goalR_deg = float(np.degrees(np.arctan2(gR_s, gR_c)))

    # ---- 5) foot_yaw_deg 규칙 반영 ----
    # - start_L, start_R: 항상 0 deg
    # - goal_L, goal_R : goal_heading_deg - heading_deg (wrap)
    startyaw_L = 0.0
    startyaw_R = 0.0
    targetyaw_L = _wrap_deg(goalL_deg - heading_deg)
    targetyaw_R = _wrap_deg(goalR_deg - heading_deg)

    # ---- 6) exp_next 구성 ----
    exp_next = dict(exp)  # 얕은 복사 후 필요한 필드만 갱신
    exp_next.update({
        "startpos_L": startpos_L,
        "startpos_R": startpos_R,
        "startyaw_L": startyaw_L,
        "startyaw_R": startyaw_R,
        "targetyaw_L": targetyaw_L,
        "targetyaw_R": targetyaw_R,
        # 참고로 center도 필요하면 보관
        "start_center_xy": np.array([center[0], center[1]], float),
        "start_heading_deg": heading_deg,
    })
    return exp_next



def compute_new_start_from_last(last_pose: dict, stance_width: float = 0.25):
    """
    last_pose 예시:
    {
      'foot': 'R' or 'L',
      'x': ..., 'y': ..., 'z': ...,
      # 아래 중 하나 이상 존재한다고 가정
      'yaw_global': float (rad)  # 권장
      # 또는 'c_global': float, 's_global': float
      # (없다면 ValueError)
    }
    반환:
      startpos_L, startpos_R, center_xy, heading_deg
      - startpos_* : np.array([x, y, z, cos(yaw), sin(yaw)])  # 당신 파이프라인 5D 포맷
      - center_xy  : np.array([x_c, y_c])
      - heading_deg: float
    """
    if "yaw_global" in last_pose:
        yaw = float(last_pose["yaw_global"])
        c_g = float(np.cos(yaw))
        s_g = float(np.sin(yaw))
    elif ("c_global" in last_pose) and ("s_global" in last_pose):
        c_g = float(last_pose["c_global"])
        s_g = float(last_pose["s_global"])
        yaw = float(np.arctan2(s_g, c_g))
    else:
        raise ValueError("last_pose에 yaw_global 또는 c_global/s_global가 필요합니다.")

    # 진행방향(t), 왼쪽 법선(n)
    t = np.array([c_g, s_g], dtype=float)
    n = np.array([-s_g, c_g], dtype=float)

    foot = last_pose.get("foot", None)
    if foot not in ("L", "R"):
        raise ValueError("last_pose['foot'] must be 'L' or 'R'.")

    px = float(last_pose["x"]); py = float(last_pose["y"]); pz = float(last_pose["z"])

    if foot == "R":
        pR = np.array([px, py], dtype=float)
        pL = pR + stance_width * n
        center = pR + 0.5 * stance_width * n
    else:  # foot == "L"
        pL = np.array([px, py], dtype=float)
        pR = pL - stance_width * n
        center = pL - 0.5 * stance_width * n

    # 5D 포맷 (x, y, z, cos, sin) — 당신의 prepare_problem_params와 호환
    startpos_L = np.array([pL[0], pL[1], pz, c_g, s_g], dtype=float)
    startpos_R = np.array([pR[0], pR[1], pz, c_g, s_g], dtype=float)

    heading_deg = float(np.degrees(yaw))
    center_xy = np.array([center[0], center[1]], dtype=float)

    return startpos_L, startpos_R, center_xy, heading_deg

# ---- 실험 프리셋 로더(원문 유지) ----
def load_experiment(exp_name: str, experiments_file: str | None):
    path = Path(experiments_file or "configs/experiments.json")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if exp_name not in data:
        raise KeyError(f"Experiment '{exp_name}' not found in {path}")
    return data[exp_name]

# ================== 간단한 3D 발 사각형 드로잉 유틸 ==================
def make_foot_corners(x, y, z, c, s, length=0.20, width=0.10):
    """
    (x,y,z)를 중심, heading=(c,s) (cos,sin)로 하는 발 사각형 꼭짓점 4개 반환.
    길이(length): 전후, 너비(width): 좌우
    """
    hx, hy = length * 0.5, width * 0.5
    # 로컬 좌표(전/후, 좌/우): (±hx, ±hy)
    local = np.array([
        [+hx, +hy],  # 앞-좌
        [+hx, -hy],  # 앞-우
        [-hx, -hy],  # 뒤-우
        [-hx, +hy],  # 뒤-좌
    ])
    R = np.array([[c, -s],
                  [s,  c]])
    world2d = (R @ local.T).T + np.array([x, y])
    verts3d = [[world2d[i,0], world2d[i,1], z] for i in range(4)]
    return verts3d

def draw_foot(ax, x, y, z, c, s, color="#4C78A8", alpha=0.35, length=0.20, width=0.10, edgecolor="k"):
    verts = make_foot_corners(x, y, z, c, s, length, width)
    poly = Poly3DCollection([verts], alpha=alpha, facecolors=[color], edgecolors=[edgecolor])
    ax.add_collection3d(poly)
    # heading 화살표(앞쪽)
    tip = np.array([x, y, z]) + np.array([c, s, 0.0]) * (length * 0.6)
    ax.plot([x, tip[0]], [y, tip[1]], [z, tip[2]])
    return poly



class StepAnimator:
    """
    - 네가 만든 plot_feet_3d를 그대로 재사용해서 프레임마다 '현재까지의 스텝'을 그리고,
      Enter(또는 Space) 입력 시 그 순간 마지막으로 찍힌 발의 포즈를 반환용 dict로 보관한다.
    - 색/시점/화살표 등은 전부 plot_feet_3d 쪽 옵션(plot_kwargs)으로 제어.
    """
    def __init__(
        self,
        ax,
        proc,
        first_is_left: bool,
        *,
        interval: float = 0.05,
        plot_kwargs: dict | None = None,
        startpos_L=None,
        startpos_R=None,
        plot_fn=None,
        x_ref=None,y_ref=None,
        target_center=None
    ):
        """
        ax: 미리 만든 3D axis (누적/뷰를 유지)
        proc: postprocess_solution의 결과 (proc.L.x, proc.R.x 등)
        first_is_left: True면 L먼저, False면 R먼저
        interval: 초 단위 프레임 간격
        plot_kwargs: plot_feet_3d에 그대로 넘길 키워드(색/알파/arrow_len/forward_axis/invert_yaw 등)
        startpos_L/R: plot_feet_3d의 yaw/ref-axes 계산에 그대로 넘겨줌
        """
        self.ax = ax
        self.proc = proc
        self.first_is_left = bool(first_is_left)
        self.interval = float(interval)
        self.plot_kwargs = dict(plot_kwargs or {})
        self.plot_kwargs.setdefault("matlab_style", True)
        self.plot_kwargs.setdefault("face_alpha", 0.35)
        self.plot_kwargs.setdefault("left_color", "#F58518")
        self.plot_kwargs.setdefault("right_color", "#4C78A8")
        self.plot_kwargs.setdefault("arrow_len", 0.15)
        self.plot_kwargs.setdefault("forward_axis", "x")
        self.plot_kwargs.setdefault("invert_yaw", False)
        self.x_ref = np.asarray(x_ref, float).reshape(2) if x_ref is not None else None
        self.y_ref = np.asarray(y_ref, float).reshape(2) if y_ref is not None else None
        # 키 이벤트: Enter/Space 정지
        self.stopped = False
        self.cid = self.ax.figure.canvas.mpl_connect("key_press_event", self._on_key)
        # 목표점(센터) 저장: [x,y,z] (없어도 OK)
        self.target_center = None
        if target_center is not None:
            tc = np.asarray(target_center, float).ravel()
            if tc.size >= 3:
                self.target_center = tc[:3]
        # 인덱스/상태
        self.iL = 0
        self.iR = 0
        self.step_count = 0
        self.last_pose = None  # {"foot": "L"/"R", "x":..., "y":..., "z":..., "c":..., "s":..., "yaw_rad":..., "index": k}

        # 시작 포즈 전달(야우/참조축 그리기 용)
        self.startpos_L = startpos_L
        self.startpos_R = startpos_R

        # 타이머 구동
        self.timer = self.ax.figure.canvas.new_timer(interval=int(self.interval * 1000))
        self.timer.add_callback(self._on_timer)
        self.timer.start()

    # ---------- 이벤트/타이머 ----------
    def _on_key(self, ev):
        if ev.key in ("enter", " "):
            self.stop()

    def stop(self):
        if not self.stopped:
            self.stopped = True
            try:
                self.timer.stop()
            except Exception:
                pass
            # 필요 시 창을 닫고 싶다면 아래 주석 해제
            # plt.close(self.ax.figure)

    def _on_timer(self):
        if self.stopped:
            return
        has_more = self.step_once()
        self.ax.figure.canvas.draw_idle()
        if not has_more:
            self.stop()

    # ---------- 한 프레임(한 발) 업데이트 ----------
    def step_once(self) -> bool:
        # 다음 차례 발 결정
        turn_left = (self.first_is_left and (self.step_count % 2 == 0)) or \
                    ((not self.first_is_left) and (self.step_count % 2 == 1))

        NL = len(self.proc.L.x)
        NR = len(self.proc.R.x)

        # 실제로 찍을 발 선택(한쪽이 소진되면 반대발)
        if turn_left:
            if self.iL < NL:
                self.iL += 1
                active = "L"
            elif self.iR < NR:
                self.iR += 1
                active = "R"
            else:
                return False
        else:
            if self.iR < NR:
                self.iR += 1
                active = "R"
            elif self.iL < NL:
                self.iL += 1
                active = "L"
            else:
                return False

        # 현재까지의 히스토리 슬라이스
        xL = np.asarray(self.proc.L.x[:self.iL]); yL = np.asarray(self.proc.L.y[:self.iL]); zL = np.asarray(self.proc.L.z[:self.iL])
        cL = np.asarray(self.proc.L.c[:self.iL]); sL = np.asarray(self.proc.L.s[:self.iL])
        xR = np.asarray(self.proc.R.x[:self.iR]); yR = np.asarray(self.proc.R.y[:self.iR]); zR = np.asarray(self.proc.R.z[:self.iR])
        cR = np.asarray(self.proc.R.c[:self.iR]); sR = np.asarray(self.proc.R.s[:self.iR])

        # 네가 쓰던 스타일 그대로: plot_feet_3d 호출
        # - animate=False (여기서 수동으로 step 진행)
        # - accumulate=False (프레임마다 깔끔히 다시 그림)
        # - show=False (블로킹 X)
        plot_feet_3d(
            xL, yL, zL, cL, sL,
            xR, yR, zR, cR, sR,
            ax=self.ax,
            accumulate=False,
            animate=False,
            show=False,
            startpos_L=self.startpos_L,
            startpos_R=self.startpos_R,
            **self.plot_kwargs,
        )





        # 마지막으로 찍힌 발의 포즈 저장
        def _save_last_pose(foot, k, x, y, z, c_local, s_local):
            yaw_local = float(np.arctan2(s_local, c_local))

            # ✅ 글로벌 기준 yaw/cos/sin 계산: v_g = c*x_ref + s*y_ref
            if (self.x_ref is not None) and (self.y_ref is not None):
                vg = c_local * self.x_ref + s_local * self.y_ref
                c_global = float(vg[0])
                s_global = float(vg[1])
                yaw_global = float(np.arctan2(s_global, c_global))
            else:
                # x_ref/y_ref가 없으면 글로벌=로컬로 fallback
                c_global = float(c_local)
                s_global = float(s_local)
                yaw_global = float(yaw_local)

            self.last_pose = {
                "foot": foot, "index": k,
                "x": float(x), "y": float(y), "z": float(z),

                # 로컬 기준
                "c_local": float(c_local), "s_local": float(s_local),
                "yaw_local": yaw_local,

                # 글로벌 기준
                "c_global": c_global, "s_global": s_global,
                "yaw_global": yaw_global,
            }

        if active == "L" and self.iL > 0:
            k = self.iL - 1
            x, y, z = float(self.proc.L.x[k]), float(self.proc.L.y[k]), float(self.proc.L.z[k])
            c, s = float(self.proc.L.c[k]), float(self.proc.L.s[k])
            _save_last_pose("L", k, x, y, z, c, s)

        elif active == "R" and self.iR > 0:
            k = self.iR - 1
            x, y, z = float(self.proc.R.x[k]), float(self.proc.R.y[k]), float(self.proc.R.z[k])
            c, s = float(self.proc.R.c[k]), float(self.proc.R.s[k])
            _save_last_pose("R", k, x, y, z, c, s)

        self.step_count += 1
        return True



# ================== 메인 파이프라인 (계산은 main.py와 동일) ==================
def main():
    parser = argparse.ArgumentParser(description="GCS Footstep Planning (replanning with interactive pause)")
    # Start/Goal preset
    parser.add_argument("--experiment", type=str, default="test_01")
    parser.add_argument("--experiments-file", type=str, default=None)
    parser.add_argument("--list-experiments", action="store_true")
    # Terrain preset
    parser.add_argument("--terrain", type=str, default="default_flat_")
    parser.add_argument("--terrain-file", type=str, default=None)
    parser.add_argument("--list-terrains", action="store_true")
    # 하이퍼파라미터
    parser.add_argument("--copy_counts", type=int, nargs="+", default=[5, 5])
    parser.add_argument("--step_span", type=float, default=0.33)
    parser.add_argument("--alpha_deg", type=float, default=30.0)
    parser.add_argument("--M_large", type=float, default=1e3)
    parser.add_argument("--M_small", type=float, default=1e-6)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--stance-width", type=float, default=0.25)
    # 애니메이션
    parser.add_argument("--interval", type=float, default=0.05, help="seconds per step")
    parser.add_argument("--max-iters", type=int, default=2, help="최대 반복 횟수")
    parser.add_argument("--goal-tol", type=float, default=0.15, help="종료: center- goal 거리 임계값[m]")
    args = parser.parse_args()

    if args.list_experiments:
        for n in list_experiments(args.experiments_file):
            print("  -", n)
        return
    if args.list_terrains:
        for n in list_terrains(args.terrain_file):
            print("  -", n)
        return

    # 1) 최초 축약 포맷 로드
    entry_cur = load_experiment(args.experiment, args.experiments_file)   # ← shorthand entry (start/goal/stance_width..)
    if "stance_width" not in entry_cur:
        entry_cur["stance_width"] = float(args.stance_width)

    # goal 고정 (루프 내에서 유지/사용)
    goal_center = np.asarray(entry_cur["goal"]["center"], float).ravel()[:3]
    goal_heading_deg = float(entry_cur["goal"]["heading_deg"])


    # ===== 인터랙티브 애니메이션 (Enter로 멈춤) =====
    plt.ion()
    fig = plt.figure("Interactive replanning Footstep (Press Enter to replan)")
    ax = fig.add_subplot(111, projection="3d")

    for it in range(int(args.max_iters)):
        # 2) 축약→풀포맷
        exp = expand_shorthand(entry_cur, default_stance_width=args.stance_width)


        startpos_L = exp["startpos_L"]; startpos_R = exp["startpos_R"]
        targetpos_L = exp["targetpos_L"]; targetpos_R = exp["targetpos_R"]
        startyaw_L  = exp["startyaw_L"];  startyaw_R  = exp["startyaw_R"]
        targetyaw_L = exp["targetyaw_L"]; targetyaw_R = exp["targetyaw_R"]

        # ===== Terrain =====
        T = load_terrain(args.terrain, args.terrain_file)
        terrain_stones = T["stones"]
        copy_counts = T["copy_counts"] if T.get("copy_counts") is not None else list(args.copy_counts)

        # ===== Graph =====
        step_span = float(args.step_span)
        alpha = float(args.alpha_deg) * np.pi/180.0
        t_graph = tic()
        G = build_graph(
            startpos_L, startpos_R, targetpos_L, targetpos_R,
            step_span=step_span, copy_counts=copy_counts, edge_rule="general",
            terrain_stones=terrain_stones
        )
        v = G["v"]; edge = G["edge"]; mode_list = G["mode_list"]
        vertex_number = G["vertex_number"]; n_edge = G["n_edge"]
        toc(t_graph, "Making graph")

        # ===== Params & Model =====
        params_pose = prepare_problem_params(
            startpos_L, startpos_R, targetpos_L, targetpos_R,
            startyaw_L, startyaw_R, targetyaw_L, targetyaw_R
        )
        start_L = params_pose["start_L"]; start_R = params_pose["start_R"]
        target_L = params_pose["target_L"]; target_R = params_pose["target_R"]

        centroid_mat, base_param = compute_centroids_from_vertices(v)

        # local ref 축
        pL = np.asarray(startpos_L, float).reshape(-1)[:2]
        pR = np.asarray(startpos_R, float).reshape(-1)[:2]
        vRL = pL - pR
        if np.linalg.norm(vRL) < 1e-12:
            y_ref = np.array([0.0, 1.0])
        else:
            y_ref = vRL / np.linalg.norm(vRL)
        x_ref = np.array([y_ref[1], -y_ref[0]])

        t_build = tic()
        M, V, IDX, PAR, build_time = build_gcs_model(v, edge, step_span, alpha, mode_list, x_ref, y_ref,
                                                     v_names=[s["name"] for s in terrain_stones for _ in range(1)])
        toc(t_build, "making optimization problem")

        # ===== Relaxed solve =====
        params_relaxed = pack_relaxed_params_for_model(
            centroid_mat=centroid_mat, n_edge=n_edge, M_large=float(args.M_large),
            start_L=start_L, start_R=start_R, target_L=target_L, target_R=target_R,
        )
        params_relaxed["x_ref"] = x_ref; params_relaxed["y_ref"] = y_ref

        set_parameters(M, V, params_relaxed)
        sol_relaxed, obj1 = solve_relaxed(M, V, M_large=float(args.M_large), log_level=0)
        print(f"Cost (relaxed path): {obj1:.3f}")

        y_relaxed_val = get_relaxed_y(V)

        best_obj, best = run_trials_select_best(
            M, V, IDX,
            centroid_mat=centroid_mat,
            y_relaxed_val=y_relaxed_val,
            edge=edge,
            vertex_number=vertex_number,
            n_edge=n_edge,
            M_small=float(args.M_small),
            start_L=start_L, start_R=start_R,
            target_L=target_L, target_R=target_R,
            trials=int(args.trials),
            temp=getattr(args, "temp", 1.0),
            greedy_first=True,
        )

        relaxed_path = best["path"]
        ex_after = best["ex"]

        # ===== 후처리 =====
        proc = postprocess_solution(
            ex_after, edge, relaxed_path,
            startpos_L, startpos_R, targetpos_L, targetpos_R,
            target_L, target_R,
        )

        first_is_left = bool(relaxed_path[0] == 1)



        # 3) 애니메이션 (Enter로 멈춤)
        plot_kwargs = dict(
            left_color="#F58518", right_color="#4C78A8",
            face_alpha=0.35, matlab_style=True, arrow_len=0.15,
            forward_axis="x", invert_yaw=False, draw_ref_axes=True,
            ref_origin="R", ref_axis_len=0.30,
        )
        animator = StepAnimator(
            ax=ax,
            proc=proc,
            first_is_left=bool(relaxed_path[0] == 1),
            interval=float(args.interval),
            plot_kwargs=plot_kwargs,
            startpos_L=startpos_L, startpos_R=startpos_R,
            plot_fn=plot_feet_3d,
            x_ref=x_ref, y_ref=y_ref,  # ← 네가 main에서 계산하는 local ref 축
        )
        print(f"[Loop {it}] 창에 포커스를 두고 Enter(또는 Space)를 누르면 다음 반복으로 진행합니다.")
        while not animator.stopped:
            plt.pause(0.05)

        last = animator.last_pose
        print(f"[Loop {it}] last_pose:", last)

        # 4) 종료 조건: goal 근접
        #  - last_pose에서 center를 다시 계산(다음 단계에서도 필요)
        if last is None:
            print("[Loop] last_pose가 없어 루프 종료")
            break

        # center 재계산 (update 함수 내부 로직과 동일)
        yaw = float(last.get("yaw_global", math.atan2(last["s_global"], last["c_global"])))
        c_g, s_g = float(np.cos(yaw)), float(np.sin(yaw))
        nL = np.array([-s_g, c_g], float)
        px, py = float(last["x"]), float(last["y"])
        if last["foot"] == "R":
            center_xy = np.array([px, py], float) + 0.5 * entry_cur["stance_width"] * nL
        else:
            center_xy = np.array([px, py], float) - 0.5 * entry_cur["stance_width"] * nL

        dist_to_goal = np.linalg.norm(center_xy - goal_center[:2])
        print(f"[Loop {it}] dist_to_goal = {dist_to_goal:.3f} m")
        if dist_to_goal <= float(args.goal_tol):
            print("[Loop] 목표에 도달(또는 임계값 이하). 루프 종료.")
            break

        # 5) start만 갱신해서 다음 반복 준비 (goal은 그대로 유지)
        entry_cur = update_entry_start_from_last(entry_cur, last, stance_width=float(args.stance_width))
        # 두 번째 루프부터 goal 변경 예시
        if it == 0:  # 다음 반복(두 번째)부터 바꾸고 싶다면
            new_goal_center = [4.5, 2.2, entry_cur["goal"]["center"][2]]  # 원하는 z 유지
            new_goal_heading_deg = 0.0  # 원하는 헤딩(도)
            entry_cur = update_entry_goal(entry_cur,
                                          goal_center=new_goal_center,
                                          goal_heading_deg=new_goal_heading_deg)
        # 마지막 창 유지
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    result = main()
