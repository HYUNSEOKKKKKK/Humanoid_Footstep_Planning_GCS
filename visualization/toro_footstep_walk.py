"""
Footstep-plan-driven TORO walking visualization.
Run:
  conda activate mujoco_env && python toro_footstep_walk.py                          # save result/toro_footstep.mp4  (default)
  conda activate mujoco_env && python toro_footstep_walk.py --interactive            # open interactive viewer
  conda activate mujoco_env && python toro_footstep_walk.py --cam-elevation -20      # custom camera angle
"""
import numpy as np
import mujoco
import mujoco.viewer
import argparse, subprocess, os, tempfile, time, json, math
from PIL import Image
from toro_real_walk import TORO_FULL_XML, build_joint_map, set_joint, euler_to_quat

# ── Leg geometry constants ────────────────────────────────────────────────────
L1 = 0.42          # thigh length [m]
L2 = 0.42          # shank length [m]
# ankle joint → foot sole: 0.065 (LL_foot body offset from LL_ankle) + 0.112 (foot mesh bottom)
ANKLE_TO_SOLE = 0.177
FLOOR_Z  =  0.0    # world floor level (1m above MuJoCo floor plane at z=-1)
PELVIS_Z =  1.0    # nominal pelvis height
LEAN     = -0.04   # forward pitch of pelvis [rad]
STONE_SIZE_SCALE = 1.1  # multiply JSON stone width/height for visualization (1.0 = exact JSON size)

# ── Default footstep plan (derived from run_000.json, stone_1→origin) ────────
# First 'L' and first 'R' entries are initial foot positions (standing).
# All subsequent entries are actual steps in order.
# z = FLOOR_Z(-1.0) + json stone surface height
# DEFAULT_FOOTSTEPS = [  # run_005.json
#     ('R',   0.911,   2.285,   0.067,    0.00),
#     ('L',   0.980,   2.526,   0.067,    0.00),
#     ('R',   1.090,   2.263,   0.067,   -6.91),
#     ('L',   1.349,   2.482,  -0.205,    9.77),
#     ('R',   1.462,   2.234,  -0.205,   11.58),
#     ('L',   1.718,   2.441,  -0.067,   11.47),
#     ('R',   1.834,   2.205,  -0.067,   15.74),
#     ('L',   2.119,   2.409,   0.044,   16.65),
#     ('R',   2.323,   2.203,   0.044,   17.23),
#     ('L',   2.572,   2.414,   0.180,   16.04),
#     ('R',   2.859,   2.214,   0.039,  -10.96),
#     ('L',   2.973,   2.414,   0.039,    0.00),
#     ('R',   3.111,   2.146,   0.039,    0.00),
#     ('L',   3.115,   2.332,   0.039,    0.00),
# ]

# DEFAULT_FOOTSTEPS = [  # Ours run_015.json
#     ('R',   0.541,   2.462,   0.087,    0.00),
#     ('L',   0.595,   2.706,   0.087,    0.00),
#     ('R',   0.669,   2.479,   0.087,  -11.62),
#     ('L',   0.907,   2.735,   0.077,   22.16),
#     ('R',   1.040,   2.535,   0.077,   19.93),
#     ('L',   1.271,   2.797,   0.109,   22.07),
#     ('R',   1.515,   2.597,   0.109,   22.70),
#     ('L',   1.728,   2.875,   -0.038,   12.46),
#     ('R',   1.943,   2.675,   -0.038,    5.15),
#     ('L',   2.231,   2.875,   0.066,    5.15),
#     ('R',   2.458,   2.609,   0.066,   -0.36),
#     ('L',   2.745,   2.809,   0.097,    1.04),
#     ('R',   2.829,   2.531,   0.097,    0.00),
#     ('L',   2.870,   2.778,   0.097,    0.00),
# ]

# DEFAULT_FOOTSTEPS = [  # Graph based run_015.json
#     ('L',   0.600,   2.700,   0.087,  350.00),
#     ('R',   0.550,   2.500,   0.087,  350.00),
#     ('L',   0.650,   2.750,   0.087,   10.00),
#     ('R',   0.900,   2.500,   0.077,  350.00),
#     ('L',   1.000,   2.700,   0.077,   10.00),
#     ('R',   1.250,   2.550,   0.109,  350.00),
#     ('L',   1.500,   2.800,   0.109,   10.00),
#     ('R',   1.750,   2.550,   -0.038,  350.00),
#     ('L',   1.950,   2.800,   -0.038,   10.00),
#     ('R',   2.200,   2.550,   0.066,  350.00),
#     ('L',   2.450,   2.800,   0.066,   10.00),
#     ('R',   2.700,   2.600,   0.097,  340.00),
#     ('L',   2.800,   2.750,   0.097,  350.00),
#     ('R',   2.750,   2.550,   0.097,  350.00),
# ]
#
DEFAULT_FOOTSTEPS = [  # Optimization based  run_015.json
    ('L',    0.591,    2.692,    0.087,   -12.46),
    ('R',    0.544,    2.476,    0.087,   -12.46),
    ('L',    0.821,    2.690,    0.078,   -12.02),
    ('R',    0.875,    2.490,    0.078,   -12.02),
    ('L',    1.072,    2.703,    0.078,   -11.59),
    ('R',    1.207,    2.503,    0.108,   -11.59),
    ('L',    1.382,    2.733,    0.108,   -11.15),
    ('R',    1.529,    2.521,    0.108,   -11.15),
    ('L',    1.692,    2.764,    -0.039,   -10.72),
    ('R',    1.875,    2.535,    -0.039,   -10.72),
    ('L',    2.002,    2.794,    -0.039,   -10.28),
    ('R',    2.242,    2.539,    0.067,   -10.28),
    ('L',    2.436,    2.783,    0.067,    -9.85),
    ('R',    2.643,    2.505,    0.096,    -9.85),
    ('L',    2.869,    2.773,    0.097,    -9.41),
    ('R',    2.830,    2.536,    0.097,    -9.41),
]


# Default JSON path (run_000.json lives next to this script)
DEFAULT_JSON = os.path.join(os.path.dirname(__file__), "run_015.json")


# ── Analytical leg IK ─────────────────────────────────────────────────────────
def leg_ik(hip_world, foot_target_world, R_pelvis=None, hip_yaw_rad=0.0):
    """
    hip_world        : (3,) world position of hip JOINT (not pelvis).
    foot_target_world: (3,) world position of foot SOLE target.
    R_pelvis         : (3,3) pelvis body rotation matrix in world (data.xmat[pelvis_id].reshape(3,3)).
                       If provided, IK is solved in pelvis frame to correctly handle pelvis lean.
    hip_yaw_rad      : hip yaw joint angle [rad]. The downstream joints (abduct/flex/knee/ankle)
                       live in a frame rotated by hip_yaw around pelvis Z, so we apply the inverse
                       rotation before solving the sagittal IK.
    Returns (hip_abduct, hip_flex, knee_flex, ankle_flex) in radians.

    Derivation (sagittal 2-link IK, knee bends forward):
      c  = hip-to-ankle distance
      knee_flex = arccos((c²  − L1² − L2²) / (2·L1·L2))    [c=L1+L2 → 0, c small → bent]
      hip_flex  = atan2(d_fwd, d_down) + atan2(L2·sin(k), L1+L2·cos(k))
      ankle_flex = hip_flex − knee_flex                       [keeps foot horizontal]
    """
    ankle_tgt = np.array(foot_target_world, dtype=float)
    ankle_tgt[2] += ANKLE_TO_SOLE          # foot sole → ankle joint

    d_world = ankle_tgt - np.array(hip_world, dtype=float)

    # Transform to pelvis body frame so joint axes (1,0,0) and (0,-1,0) match world axes
    if R_pelvis is not None:
        d = R_pelvis.T @ d_world   # pelvis frame: corrects for forward lean etc.
    else:
        d = d_world

    # Undo hip_yaw rotation: abduct/flex/knee/ankle axes live in the yaw-rotated frame,
    # so we rotate the target vector by -hip_yaw around pelvis Z before solving.
    if abs(hip_yaw_rad) > 1e-9:
        cy, sy = np.cos(hip_yaw_rad), np.sin(hip_yaw_rad)
        # R(-θ) = R(θ)^T
        d = np.array([cy*d[0] + sy*d[1], -sy*d[0] + cy*d[1], d[2]])

    # 1. Hip abduction (rotation around pelvis X, in frontal YZ plane)
    hip_abduct = np.arctan2(d[1], -d[2])

    # 2. Sagittal 2-link IK
    d_down = np.sqrt(d[1]**2 + d[2]**2)   # frontal-plane component = "down" after abduction
    d_fwd  = d[0]
    c = float(np.sqrt(d_fwd**2 + d_down**2))
    c = np.clip(c, 1e-4, L1 + L2 - 1e-4)

    cos_knee  = (c**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    knee_flex = float(np.arccos(np.clip(cos_knee, -1.0 + 1e-7, 1.0 - 1e-7)))

    gamma    = np.arctan2(d_fwd, d_down)   # angle of hip→ankle from "straight down"
    delta    = np.arctan2(L2 * np.sin(knee_flex), L1 + L2 * np.cos(knee_flex))
    hip_flex = float(gamma + delta)         # knee bends toward +X (forward)

    # 3. Ankle flex: keep foot sole parallel to floor
    ankle_flex = float(hip_flex - knee_flex)

    # Clamp to MJCF joint limits
    hip_abduct = float(np.clip(hip_abduct, -0.50,  0.50))
    hip_flex   = float(np.clip(hip_flex,   -1.40,  1.00))
    knee_flex  = float(np.clip(knee_flex,   0.00,  2.40))
    ankle_flex = float(np.clip(ankle_flex, -0.60,  0.50))

    return hip_abduct, hip_flex, knee_flex, ankle_flex


# ── Trajectory helpers ────────────────────────────────────────────────────────
def smooth_step(s):
    """Hermite cubic: value 0→1 with zero velocity at both ends."""
    s = float(np.clip(s, 0.0, 1.0))
    return 3.0*s**2 - 2.0*s**3

def swing_trajectory(start, end, s, step_height=0.07):
    """
    Smooth XY interpolation + parabolic Z foot-lift arc.
    start, end: (3,) foot-sole positions (z can differ for stairs/terrain).
    Z arc: linear interpolation between start_z and end_z + step_height parabola.
    """
    a = smooth_step(s)
    x = start[0] + a * (end[0] - start[0])
    y = start[1] + a * (end[1] - start[1])
    # linear z baseline (handles height changes) + parabolic clearance on top
    z_base = start[2] + a * (end[2] - start[2])
    z = z_base + 4.0 * step_height * s * (1.0 - s)
    return np.array([x, y, z])


# ── Schedule builder ──────────────────────────────────────────────────────────
def build_schedule(footsteps, T_init=5.0, T_step=0.4, T_final=0.5):
    """
    Returns (schedule, total_duration).
    schedule: list of event dicts.
    """
    foot_pos = {}   # {side: np.array([x, y, z])}  — current foot position
    foot_yaw = {}   # {side: float}  — current foot yaw [rad]
    steps    = []

    for entry in footsteps:
        side, x, y = entry[0], entry[1], entry[2]
        z   = entry[3] if len(entry) >= 4 else FLOOR_Z
        yaw_deg = ((entry[4] + 180) % 360 - 180) if len(entry) >= 5 else 0.0
        yaw = float(np.radians(yaw_deg))
        pos = np.array([x, y, z])
        if side not in foot_pos:
            foot_pos[side] = pos
            foot_yaw[side] = yaw
        else:
            steps.append((side, pos, yaw))

    assert 'L' in foot_pos and 'R' in foot_pos, \
        "Footstep list must contain at least one 'L' and one 'R' entry."

    schedule = []
    t = 0.0

    # Initial double support
    schedule.append({
        'type': 'stance', 't_start': t, 't_end': t + T_init,
        'foot_L': foot_pos['L'].copy(), 'foot_R': foot_pos['R'].copy(),
        'yaw_L': foot_yaw['L'], 'yaw_R': foot_yaw['R'],
    })
    t += T_init

    for side, target, yaw_to in steps:
        other = 'R' if side == 'L' else 'L'
        schedule.append({
            'type':           'step',
            't_start':        t,
            't_end':          t + T_step,
            'swing':          side,
            'swing_from':     foot_pos[side].copy(),
            'swing_to':       target.copy(),
            'swing_yaw_from': foot_yaw[side],
            'swing_yaw_to':   yaw_to,
            'stance_pos':     foot_pos[other].copy(),
            'stance_yaw':     foot_yaw[other],
            'foot_L_pre':     foot_pos['L'].copy(),
            'foot_R_pre':     foot_pos['R'].copy(),
        })
        foot_pos[side] = target.copy()
        foot_yaw[side] = yaw_to
        t += T_step

    # Final double support
    schedule.append({
        'type': 'stance', 't_start': t, 't_end': t + T_final,
        'foot_L': foot_pos['L'].copy(), 'foot_R': foot_pos['R'].copy(),
        'yaw_L': foot_yaw['L'], 'yaw_R': foot_yaw['R'],
    })
    t += T_final

    return schedule, t


# ── Per-frame state ───────────────────────────────────────────────────────────
def get_frame_state(t, schedule, step_height=0.07):
    """Returns (foot_L, foot_R, yaw_L, yaw_R, pelvis_xy, arm_swing)."""
    # Find current event (default to last event if t is past the end)
    event = schedule[-1]
    for ev in schedule:
        if t < ev['t_end']:
            event = ev
            break

    if event['type'] == 'stance':
        foot_L    = event['foot_L'].copy()
        foot_R    = event['foot_R'].copy()
        yaw_L     = event['yaw_L']
        yaw_R     = event['yaw_R']
        pelvis_xy = (foot_L[:2] + foot_R[:2]) / 2.0
        arm_swing = 0.0

    else:  # step
        s = float(np.clip(
            (t - event['t_start']) / (event['t_end'] - event['t_start']),
            0.0, 1.0))

        sw_pos = swing_trajectory(event['swing_from'], event['swing_to'], s, step_height)
        st_pos = event['stance_pos'].copy()
        sw_yaw = event['swing_yaw_from'] + smooth_step(s) * (event['swing_yaw_to'] - event['swing_yaw_from'])
        st_yaw = event['stance_yaw']

        if event['swing'] == 'L':
            foot_L, foot_R = sw_pos, st_pos
            yaw_L,  yaw_R  = sw_yaw, st_yaw
        else:
            foot_L, foot_R = st_pos, sw_pos
            yaw_L,  yaw_R  = st_yaw, sw_yaw

        # Pelvis XY: lerp from pre-step midpoint → post-step midpoint
        prev_mid = (event['foot_L_pre'][:2] + event['foot_R_pre'][:2]) / 2.0
        if event['swing'] == 'L':
            next_mid = (event['swing_to'][:2] + event['stance_pos'][:2]) / 2.0
        else:
            next_mid = (event['stance_pos'][:2] + event['swing_to'][:2]) / 2.0
        pelvis_xy = prev_mid + smooth_step(s) * (next_mid - prev_mid)

        # Counter-swing arms (right arm forward when left foot swings)
        sign      = 1.0 if event['swing'] == 'L' else -1.0
        arm_swing = sign * 0.25 * np.sin(np.pi * s)

    return foot_L, foot_R, yaw_L, yaw_R, pelvis_xy, arm_swing


# ── Pose setter ───────────────────────────────────────────────────────────────
def set_footstep_pose(model, data, jmap, schedule, t, step_height=0.07):
    foot_L, foot_R, yaw_L, yaw_R, pelvis_xy, arm_swing = get_frame_state(t, schedule, step_height)

    # Pelvis height: driven by the LOWER foot (stance foot).
    # Using average causes over-extension when feet are at different heights
    # (e.g., one on elevated stone, one on flat floor) → IK clips at L1+L2 → foot floats.
    # Using min keeps pelvis low enough for the stance leg to always reach the floor.
    stance_z = min(foot_L[2], foot_R[2])
    pelvis_z = PELVIS_Z + (stance_z - FLOOR_Z)

    # Pelvis freejoint (qpos[0:3] = position, qpos[3:7] = quaternion w,x,y,z)
    data.qpos[0] = pelvis_xy[0]
    data.qpos[1] = pelvis_xy[1]
    data.qpos[2] = pelvis_z
    data.qpos[3:7] = euler_to_quat(0.0, LEAN, 0.0)

    # First kinematics pass → get hip joint world positions (accounting for lean)
    mujoco.mj_kinematics(model, data)
    bid_pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip_base")
    hid_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LL_hip")
    hid_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "RL_hip")
    R_pelvis = data.xmat[bid_pelvis].reshape(3, 3)
    hip_L = data.xpos[hid_L].copy()
    hip_R = data.xpos[hid_R].copy()

    # Analytical leg IK — pass hip_yaw so the sagittal IK solves in the yaw-rotated frame
    ha_L, hf_L, kf_L, af_L = leg_ik(hip_L, foot_L, R_pelvis, hip_yaw_rad=yaw_L)
    ha_R, hf_R, kf_R, af_R = leg_ik(hip_R, foot_R, R_pelvis, hip_yaw_rad=yaw_R)

    set_joint(data, jmap, 'LL_hip_yaw',    float(np.clip(yaw_L, -0.8, 0.8)))
    set_joint(data, jmap, 'LL_hip_abduct', ha_L)
    set_joint(data, jmap, 'LL_hip_flex',   hf_L)
    set_joint(data, jmap, 'LL_knee_flex',  kf_L)
    set_joint(data, jmap, 'LL_ankle_flex', af_L)

    set_joint(data, jmap, 'RL_hip_yaw',    float(np.clip(yaw_R, -0.8, 0.8)))
    set_joint(data, jmap, 'RL_hip_abduct', ha_R)
    set_joint(data, jmap, 'RL_hip_flex',   hf_R)
    set_joint(data, jmap, 'RL_knee_flex',  kf_R)
    set_joint(data, jmap, 'RL_ankle_flex', af_R)

    # Arm counter-swing (j2_right / j8_left = first shoulder DOF each arm)
    set_joint(data, jmap, 'j2_right',  arm_swing)
    set_joint(data, jmap, 'j8_left',  -arm_swing)

    # Elbow bend: fixed natural posture (j5_right / j11_left)
    set_joint(data, jmap, 'j5_right',  0.5)
    set_joint(data, jmap, 'j11_left',  0.5)

    # Waist yaw: follow average foot heading
    waist_yaw = 0.5 * (yaw_L + yaw_R)
    set_joint(data, jmap, 'j1_waist', float(np.clip(waist_yaw, -2.09, 2.09)))

    # Final kinematics pass
    mujoco.mj_kinematics(model, data)


# ── JSON terrain loader ───────────────────────────────────────────────────────
def load_run_json(json_path):
    """
    Parse run_000.json stones; translate stone_1 center to XY origin.
    Returns list of dicts: {name, cx, cy, cz, width, height, yaw_deg}.
    cx, cy: translated XY; cz: raw JSON surface height (add FLOOR_Z for MuJoCo z).
    """
    with open(json_path) as f:
        raw = json.load(f)
    stones = raw['terrain']['stones']
    ox, oy = stones[0]['center'][0], stones[0]['center'][1]
    result = []
    for s in stones:
        result.append({
            'name':    s['name'],
            'cx':      s['center'][0] - ox,
            'cy':      s['center'][1] - oy,
            'cz':      s['center'][2],
            'width':   s['width'],
            'height':  s['height'],
            'yaw_deg': s['yaw_deg'],
        })
    return result


def align_footsteps_to_stone1(footsteps, json_path, xy_scale=1.0):
    """
    Translate all footsteps so that the midpoint of the first two entries
    lands on stone_1 center from the JSON, then scale XY by xy_scale.
    xy_scale > 1 stretches the stride (e.g. 1.5 = data generated for 40cm max
    stride → rescaled to 60cm max stride).
    """
    stones = load_run_json(json_path)
    stone1_xy = np.array([stones[0]['cx'], stones[0]['cy']])  # = (0, 0)

    fp0, fp1 = footsteps[0], footsteps[1]
    mid_xy = np.array([(fp0[1] + fp1[1]) / 2.0,
                        (fp0[2] + fp1[2]) / 2.0])
    dx, dy = stone1_xy - mid_xy

    aligned = []
    for entry in footsteps:
        tx = (entry[1] + dx) * xy_scale
        ty = (entry[2] + dy) * xy_scale
        new_entry = (entry[0], tx, ty) + entry[3:]
        aligned.append(new_entry)
    return aligned


def make_footsteps_from_json(json_path):
    """
    Auto-generate DEFAULT_FOOTSTEPS from run JSON.
    stone_1 → both feet (initial stance, ±0.10 y-offset).
    stones 2-6 → alternating R, L, R, L, R at stone center.
    """
    stones = load_run_json(json_path)
    s1 = stones[0]
    z1 = FLOOR_Z + s1['cz']
    footsteps = [
        ('L', s1['cx'],  0.10, z1),
        ('R', s1['cx'], -0.10, z1),
    ]
    sides = ['R', 'L', 'R', 'L', 'R']
    for i, s in enumerate(stones[1:]):
        side = sides[i % len(sides)]
        fz = FLOOR_Z + s['cz']
        footsteps.append((side, s['cx'], s['cy'], fz))
    return footsteps


def build_mjcf_from_json_stones(json_path, stone_lz=0.025, xy_scale=1.0):
    """
    Build MJCF with stepping stones sized/oriented from run JSON.
    First stone: green. Last stone: red.
    Middle stones: grayscale by surface height (lower=darker, higher=lighter).
    xy_scale: scale stone cx/cy from stone_1 origin (matches align_footsteps_to_stone1).
    """
    stones = load_run_json(json_path)
    for s in stones:
        s['cx']     *= xy_scale
        s['cy']     *= xy_scale
        s['width']  *= xy_scale
        s['height'] *= xy_scale

    # Grayscale range for middle stones based on their heights
    mid_heights = [s['cz'] for s in stones[1:-1]]
    h_min = min(mid_heights) if mid_heights else 0.0
    h_max = max(mid_heights) if mid_heights else 1.0
    h_range = h_max - h_min if h_max > h_min else 1.0

    geoms = []
    for i, s in enumerate(stones):
        foot_z  = FLOOR_Z + s['cz']
        cz      = foot_z - stone_lz + 0.10
        yaw_rad = math.radians(s['yaw_deg'] + 90.0)
        half_h  = s['height'] / 2.0 * STONE_SIZE_SCALE
        half_w  = s['width']  / 2.0 * STONE_SIZE_SCALE

        if i == 0:
            color = "0.20 0.75 0.20 1"                    # start: green
        elif i == len(stones) - 1:
            color = "0.80 0.15 0.15 1"                    # end: red
        else:
            g = 0.35 + 0.45 * (s['cz'] - h_min) / h_range  # [0.35, 0.80]
            color = f"{g:.3f} {g:.3f} {g:.3f} 1"          # grayscale by height

        geoms.append(
            f'    <geom name="{s["name"]}" type="box" '
            f'pos="{s["cx"]:.4f} {s["cy"]:.4f} {cz:.4f}" '
            f'size="{half_h:.4f} {half_w:.4f} {stone_lz:.4f}" '
            f'euler="0 0 {yaw_rad:.4f}" '
            f'rgba="{color}" '
            f'contype="0" conaffinity="0"/>'
        )
    stones_block = '\n'.join(geoms)
    xml = TORO_FULL_XML.replace('  </worldbody>', f'{stones_block}\n  </worldbody>')
    # floor stays visible
    return xml


# ── Stepping-stone MJCF builder (simple per-footstep boxes) ──────────────────
def build_mjcf_with_stones(footsteps,
                            stone_lx=0.135, stone_ly=0.075, stone_lz=0.025):
    """
    Returns TORO_FULL_XML with flat box stepping-stones added to worldbody.
    One stone per entry in footsteps; top surface is flush with the foot's z.
    stone_lx, stone_ly, stone_lz: MuJoCo half-extents (full size = 2×)
      default → 0.27m × 0.15m × 0.05m  (foot is 0.20m × 0.08m)
    L stones: cool slate blue  |  R stones: warm sandstone
    """
    COLOR = {
        'L': "0.40 0.52 0.68 1",   # slate blue
        'R': "0.72 0.56 0.38 1",   # sandstone
    }
    geoms = []
    for i, entry in enumerate(footsteps):
        side = entry[0]
        x, y = float(entry[1]), float(entry[2])
        z    = float(entry[3]) if len(entry) >= 4 else FLOOR_Z
        cx, cy = x, y
        cz = z - stone_lz + 0.10   # raise stone 10 cm above foot target z
        geoms.append(
            f'    <geom name="stone_{i}" type="box" '
            f'pos="{cx:.4f} {cy:.4f} {cz:.4f}" '
            f'size="{stone_lx} {stone_ly} {stone_lz}" '
            f'rgba="{COLOR[side]}" '
            f'contype="0" conaffinity="0"/>'
        )
    stones_block = '\n'.join(geoms)
    xml = TORO_FULL_XML.replace('  </worldbody>', f'{stones_block}\n  </worldbody>')
    # floor stays visible
    return xml


# ── Footstep marker geoms ─────────────────────────────────────────────────────
def add_footstep_marker_geoms(xml, footsteps):
    """
    Inject invisible footstep marker geoms into the XML worldbody.
    Each footstep gets two geoms: a foot-outline box + a toe-direction box.
    Both start with alpha=0 so they can be revealed at runtime.
    Returns (new_xml, list_of_geom_names).
    """
    COLOR_L = (1.00, 0.0, 0.0)   # left: red
    COLOR_R = (0.0, 0.0, 1.00)   # right: blue
    geoms = []
    names = []

    for i, entry in enumerate(footsteps):
        side = entry[0]
        x, y = float(entry[1]), float(entry[2])
        z       = float(entry[3]) if len(entry) >= 4 else FLOOR_Z
        yaw_deg = ((entry[4] + 180) % 360 - 180) if len(entry) >= 5 else 0.0
        yaw_rad = float(np.radians(yaw_deg))
        r, g, b = COLOR_L if side == 'L' else COLOR_R
        mz = z + 0.10 + 0.005   # stone top = foot_z + 0.10, then 5 mm above

        # foot outline (flat rectangle)
        foot_name = f"mkfoot_{i}"
        geoms.append(
            f'    <geom name="{foot_name}" type="box" '
            f'pos="{x:.4f} {y:.4f} {mz:.4f}" '
            f'size="0.090 0.045 0.003" '
            f'euler="0 0 {yaw_rad:.4f}" '
            f'rgba="{r} {g} {b} 0" '
            f'contype="0" conaffinity="0"/>'
        )
        names.append(foot_name)

        # toe indicator – smaller box shifted forward in foot direction
        tx = x + 0.085 * math.cos(yaw_rad)
        ty = y + 0.085 * math.sin(yaw_rad)
        toe_name = f"mktoe_{i}"
        geoms.append(
            f'    <geom name="{toe_name}" type="box" '
            f'pos="{tx:.4f} {ty:.4f} {mz+0.003:.4f}" '
            f'size="0.020 0.035 0.005" '
            f'euler="0 0 {yaw_rad:.4f}" '
            f'rgba="{r} {g} {b} 0" '
            f'contype="0" conaffinity="0"/>'
        )
        names.append(toe_name)

    markers_block = '\n'.join(geoms)
    new_xml = xml.replace('  </worldbody>', f'{markers_block}\n  </worldbody>')
    return new_xml, names


# ── Verification helper ───────────────────────────────────────────────────────
def verify_foot_positions(model, data, schedule, n_check=20):
    """Print foot positioning error at stance moments (should be < 2 mm)."""
    jmap = build_joint_map(model)
    print("\nFoot positioning accuracy check:")
    print(f"  {'time':>6}  {'side':>5}  {'err_x':>8}  {'err_y':>8}  {'err_z':>8}")
    total, T = build_schedule(DEFAULT_FOOTSTEPS)

    # Check at end of each step (foot should be at target)
    for ev in schedule:
        if ev['type'] != 'step':
            continue
        t_check = ev['t_end'] - 0.001
        set_footstep_pose(model, data, jmap, schedule, t_check)
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                 "LL_foot" if ev['swing'] == 'L' else "RL_foot")
        foot_body_z = data.xpos[fid, 2]
        foot_sole_z = foot_body_z - 0.112  # foot mesh bottom
        # XY from body position (foot body is directly above sole center)
        ex = data.xpos[fid, 0] - ev['swing_to'][0]
        ey = data.xpos[fid, 1] - ev['swing_to'][1]
        ez = foot_sole_z - ev['swing_to'][2]   # compare against actual target z
        print(f"  {t_check:>6.2f}  {ev['swing']:>5}  {ex:>+8.4f}  {ey:>+8.4f}  {ez:>+8.4f}")
    print()


# ── Interactive viewer ────────────────────────────────────────────────────────
def run_interactive(footsteps=None, step_height=0.07, T_step=0.4, T_init=5.0, T_plan=0.1,
                    json_path=None, auto_footsteps=False, xy_scale=1.7,
                    cam_distance=8.0, cam_elevation=-15, cam_azimuth=120):
    if json_path and os.path.exists(json_path):
        xml = build_mjcf_from_json_stones(json_path, xy_scale=xy_scale)
        if auto_footsteps:
            footsteps = make_footsteps_from_json(json_path)
            print(f"Footsteps: auto from {json_path}")
        else:
            if footsteps is None:
                footsteps = DEFAULT_FOOTSTEPS
            footsteps = align_footsteps_to_stone1(footsteps, json_path, xy_scale=xy_scale)
            print(f"Footsteps: DEFAULT_FOOTSTEPS (aligned, scale={xy_scale})  |  Stones: {json_path}")
    else:
        if footsteps is None:
            footsteps = DEFAULT_FOOTSTEPS
        xml = build_mjcf_with_stones(footsteps)

    # Add invisible footstep markers; they appear all at once after T_init
    xml, marker_names = add_footstep_marker_geoms(xml, footsteps)

    model    = mujoco.MjModel.from_xml_string(xml)
    data     = mujoco.MjData(model)
    jmap     = build_joint_map(model)

    # Extend initial stand phase to cover both "thinking" (T_init) and "show plan" (T_plan)
    schedule, total = build_schedule(footsteps, T_init=T_init + T_plan, T_step=T_step)

    marker_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                   for n in marker_names]
    MARKER_ALPHA = 0.85
    markers_revealed = False

    print(f"Animation: {total:.2f}s | {len([e for e in schedule if e['type']=='step'])} steps "
          f"| think {T_init:.1f}s | plan {T_plan:.1f}s | walk {total - T_init - T_plan:.1f}s")
    set_footstep_pose(model, data, jmap, schedule, 0.0, step_height)

    # Camera: fixed lookat at center of stepping stone path
    xs = [e[1] for e in footsteps]
    ys = [e[2] for e in footsteps]
    cam_cx = (min(xs) + max(xs)) / 2.0
    cam_cy = (min(ys) + max(ys)) / 2.0
    dist   = cam_distance if cam_distance is not None else max(4.0, (max(xs) - min(xs)) * 1.7 + 3.0)

    # step_events[k] → footstep index k+2 (footsteps[0,1] are initial positions)
    step_events  = [ev for ev in schedule if ev['type'] == 'step']
    walk_start   = T_init + T_plan
    hidden_fps   = set()   # footstep indices whose markers have been hidden

    def hide_fp(fp_idx):
        for j in range(2):   # mkfoot and mktoe
            gid = marker_gids[fp_idx * 2 + j]
            if gid >= 0:
                model.geom_rgba[gid, 3] = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [cam_cx, cam_cy, 0.2]
        viewer.cam.distance   = dist
        viewer.cam.elevation  = cam_elevation
        viewer.cam.azimuth    = cam_azimuth

        t0 = time.perf_counter()
        while viewer.is_running():
            t = min(time.perf_counter() - t0, total)
            set_footstep_pose(model, data, jmap, schedule, t, step_height)

            # Phase 2: reveal all footstep markers at once when planning phase begins
            if not markers_revealed and t >= T_init:
                for gid in marker_gids:
                    if gid >= 0:
                        model.geom_rgba[gid, 3] = MARKER_ALPHA
                markers_revealed = True

            # Phase 3: hide markers as robot steps on them
            if markers_revealed and t >= walk_start:
                # initial position markers disappear when walk begins
                for fp_idx in (0, 1):
                    if fp_idx not in hidden_fps:
                        hide_fp(fp_idx)
                        hidden_fps.add(fp_idx)
                # step markers disappear when each step completes
                for k, ev in enumerate(step_events):
                    fp_idx = k + 2
                    if fp_idx not in hidden_fps and t >= ev['t_end']:
                        hide_fp(fp_idx)
                        hidden_fps.add(fp_idx)

            viewer.sync()
            time.sleep(1.0 / 60.0)


# ── Offline rendering ─────────────────────────────────────────────────────────
def run_render(footsteps=None, output="result/toro_footstep.mp4",
               fps=60, width=1280, height=720,
               step_height=0.07, T_step=0.4, T_init=5.0, T_plan=0.1,
               json_path=None, auto_footsteps=False, xy_scale=1.7,
               cam_distance=3.8, cam_elevation=-12, cam_azimuth=100):
    if json_path and os.path.exists(json_path):
        xml = build_mjcf_from_json_stones(json_path, xy_scale=xy_scale)
        if auto_footsteps:
            footsteps = make_footsteps_from_json(json_path)
            print(f"Footsteps: auto from {json_path}")
        else:
            if footsteps is None:
                footsteps = DEFAULT_FOOTSTEPS
            footsteps = align_footsteps_to_stone1(footsteps, json_path, xy_scale=xy_scale)
            print(f"Footsteps: DEFAULT_FOOTSTEPS (aligned, scale={xy_scale})  |  Stones: {json_path}")
    else:
        if footsteps is None:
            footsteps = DEFAULT_FOOTSTEPS
        xml = build_mjcf_with_stones(footsteps)

    xml, marker_names = add_footstep_marker_geoms(xml, footsteps)

    model    = mujoco.MjModel.from_xml_string(xml)
    data     = mujoco.MjData(model)
    jmap     = build_joint_map(model)
    schedule, total = build_schedule(footsteps, T_init=T_init + T_plan, T_step=T_step)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam      = mujoco.MjvCamera()

    marker_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                   for n in marker_names]
    MARKER_ALPHA  = 0.85
    markers_shown = False

    step_events = [ev for ev in schedule if ev['type'] == 'step']
    walk_start  = T_init + T_plan
    hidden_fps  = set()

    def hide_fp(fp_idx):
        for j in range(2):
            gid = marker_gids[fp_idx * 2 + j]
            if gid >= 0:
                model.geom_rgba[gid, 3] = 0.0

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    xs = [e[1] for e in footsteps]
    ys = [e[2] for e in footsteps]
    lookat_x = (min(xs) + max(xs)) / 2.0
    lookat_y = (min(ys) + max(ys)) / 2.0

    n      = int(total * fps)
    dt     = 1.0 / fps
    tmpdir = tempfile.mkdtemp()
    print(f"Rendering {n} frames ({total:.1f}s) → {output}")

    for i in range(n):
        t = i * dt

        if not markers_shown and t >= T_init:
            for gid in marker_gids:
                if gid >= 0:
                    model.geom_rgba[gid, 3] = MARKER_ALPHA
            markers_shown = True

        if markers_shown and t >= walk_start:
            for fp_idx in (0, 1):
                if fp_idx not in hidden_fps:
                    hide_fp(fp_idx)
                    hidden_fps.add(fp_idx)
            for k, ev in enumerate(step_events):
                fp_idx = k + 2
                if fp_idx not in hidden_fps and t >= ev['t_end']:
                    hide_fp(fp_idx)
                    hidden_fps.add(fp_idx)

        set_footstep_pose(model, data, jmap, schedule, t, step_height)

        cam.lookat[0] = lookat_x
        cam.lookat[1] = lookat_y
        cam.lookat[2] = 0.2
        cam.distance  = cam_distance
        cam.elevation = cam_elevation
        cam.azimuth   = cam_azimuth

        renderer.update_scene(data, camera=cam)
        Image.fromarray(renderer.render()).save(os.path.join(tmpdir, f"frame_{i:05d}.png"))
        if i % fps == 0:
            print(f"  {t:.1f}s / {total:.1f}s")

    renderer.close()
    cmd = ["ffmpeg", "-y", "-framerate", str(fps),
           "-i", os.path.join(tmpdir, "frame_%05d.png"),
           "-c:v", "libx264", "-preset", "slow", "-crf", "17",
           "-pix_fmt", "yuv420p", output]
    print("Encoding…")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Saved: {output}")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactive",  action="store_true", help="open interactive viewer instead of saving MP4")
    ap.add_argument("--output",      default="result/toro_footstep.mp4")
    ap.add_argument("--fps",         type=int,   default=60)
    ap.add_argument("--width",       type=int,   default=1280)
    ap.add_argument("--height",      type=int,   default=720)
    ap.add_argument("--step-height", type=float, default=0.07, dest="step_height")
    ap.add_argument("--T-init",      type=float, default=5.0,  dest="T_init",
                    help="seconds robot thinks before plan appears (default: 2.0)")
    ap.add_argument("--T-plan",      type=float, default=0.1,  dest="T_plan",
                    help="seconds footstep plan is visible before walk starts (default: 2.0)")
    ap.add_argument("--T-step",      type=float, default=0.4,  dest="T_step")
    ap.add_argument("--verify",      action="store_true", help="print foot positioning errors")
    ap.add_argument("--json",        default=DEFAULT_JSON,
                    help="path to run_*.json terrain file (default: run_000.json)")
    ap.add_argument("--no-json",        action="store_true",
                    help="ignore JSON entirely; use DEFAULT_FOOTSTEPS + simple boxes")
    ap.add_argument("--auto-footsteps", action="store_true", dest="auto_footsteps",
                    help="generate footsteps from JSON stone centers (default: use DEFAULT_FOOTSTEPS)")
    ap.add_argument("--xy-scale",       type=float, default=1.7, dest="xy_scale",
                    help="XY stride scale factor from stone_1 origin (default: 1.7 = VAE output → full-size robot)")
    ap.add_argument("--cam-distance",  type=float, default=None,  dest="cam_distance",
                    help="camera distance (default: auto for interactive, 3.8 for render)")
    ap.add_argument("--cam-elevation", type=float, default=None,  dest="cam_elevation",
                    help="camera elevation in degrees (default: -15 interactive, -12 render)")
    ap.add_argument("--cam-azimuth",   type=float, default=None,  dest="cam_azimuth",
                    help="camera azimuth in degrees (default: 100)")
    args = ap.parse_args()

    json_path = None if args.no_json else args.json

    cam_kw = {}
    if args.cam_distance  is not None: cam_kw['cam_distance']  = args.cam_distance
    if args.cam_elevation is not None: cam_kw['cam_elevation'] = args.cam_elevation
    if args.cam_azimuth   is not None: cam_kw['cam_azimuth']   = args.cam_azimuth

    if args.verify:
        fp  = make_footsteps_from_json(json_path) if (json_path and os.path.exists(json_path)) \
              else DEFAULT_FOOTSTEPS
        model    = mujoco.MjModel.from_xml_string(TORO_FULL_XML)
        data     = mujoco.MjData(model)
        schedule, _ = build_schedule(fp, T_step=args.T_step)
        verify_foot_positions(model, data, schedule)
    elif args.interactive:
        run_interactive(step_height=args.step_height, T_init=args.T_init, T_plan=args.T_plan,
                        T_step=args.T_step,
                        json_path=json_path, auto_footsteps=args.auto_footsteps,
                        xy_scale=args.xy_scale, **cam_kw)
    else:
        run_render(output=args.output, fps=args.fps,
                   width=args.width, height=args.height,
                   step_height=args.step_height, T_init=args.T_init, T_plan=args.T_plan,
                   T_step=args.T_step,
                   json_path=json_path, auto_footsteps=args.auto_footsteps,
                   xy_scale=args.xy_scale, **cam_kw)
