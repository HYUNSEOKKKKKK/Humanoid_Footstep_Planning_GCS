# utils/expand_experiment.py
import math
from typing import Dict
from .feet import feet_from_center

def expand_shorthand(entry: Dict, default_stance_width: float) -> Dict:
    """
    입력(JSON, degree):
      {
        "start": {"center": [x,y,z], "heading_deg": 0.0},
        "goal":  {"center": [x,y,z], "heading_deg": 45.0},
        "stance_width": 0.25,
        "foot_yaw_deg": {  # 생략 가능, 없으면 모두 0
          "start_L": 0.0, "start_R": 0.0,
          "goal_L":  0.0, "goal_R":  0.0
        }
      }

    출력(dict):
      - 좌표: startpos_*, targetpos_*
      - 발 yaw (radian): startyaw_*, targetyaw_*
    """
    w = float(entry.get("stance_width", default_stance_width))

    # 좌표 (heading_deg는 degree로 들어오며, feet_from_center 안에서 radian 변환)
    sc = entry["start"]["center"];  shy = float(entry["start"]["heading_deg"])
    gc = entry["goal"]["center"];   ghy = float(entry["goal"]["heading_deg"])
    start_L, start_R = feet_from_center(sc, shy, w)
    goal_L,  goal_R  = feet_from_center(gc, ghy, w)

    # 발 자체 yaw (degree → radian 변환)
    fy = entry.get("foot_yaw_deg", {})
    sYL = math.radians(float(fy.get("start_L", shy)))  # ★ 0.0 → shy
    sYR = math.radians(float(fy.get("start_R", shy)))  # ★ 0.0 → shy
    gYL = math.radians(float(fy.get("goal_L", ghy)))  # ★ 0.0 → ghy
    gYR = math.radians(float(fy.get("goal_R", ghy)))  # ★ 0.0 → ghy

    return {
        "startpos_L": start_L,
        "startpos_R": start_R,
        "targetpos_L": goal_L,
        "targetpos_R": goal_R,
        "startyaw_L": sYL,
        "startyaw_R": sYR,
        "targetyaw_L": gYL,
        "targetyaw_R": gYR
    }
