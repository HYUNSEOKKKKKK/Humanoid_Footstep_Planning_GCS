"""
DLR TORO humanoid walking visualization – real upper-body meshes + leg meshes.
Run:
  conda activate mujoco_env && python toro_real_walk.py            # interactive
  conda activate mujoco_env && python toro_real_walk.py --render   # save MP4
"""
import numpy as np
import mujoco
import mujoco.viewer
import argparse, subprocess, os, tempfile
from PIL import Image

# ── Absolute mesh directory ──────────────────────────────────────────────────
MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "model/toro/meshes/obj/")

# ── Full TORO MJCF (upper body real meshes + leg meshes) ────────────────────
# Coordinate system (same as original toro.xml):
#   - pelvis (hip_base) origin at z = 0 world
#   - floor at z = -1 world
#   - legs reach z ≈ -0.95 (foot bottom)
# Leg geometry (from footstep_planner dimensions):
#   hip offset y = ±0.115 m,  hip z = -0.05 m from pelvis
#   thigh = 0.42 m,  shank = 0.42 m
# ─────────────────────────────────────────────────────────────────────────────
TORO_FULL_XML = f"""
<mujoco model="DLR_TORO_walk">
  <compiler angle="radian" meshdir="{MESH_DIR}" autolimits="true"/>

  <visual>
    <headlight diffuse="0.65 0.65 0.65" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.22 0.3 1"/>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <!-- upper body -->
    <mesh name="upper_hip"            file="upper_hip.stl.obj"/>
    <mesh name="trunk"                file="trunk.stl.obj"/>
    <mesh name="kuka_link1"           file="kuka-link1.stl.obj"/>
    <mesh name="va_r_shoulder_front"  file="va_right_shoulder_front.stl.obj"/>
    <mesh name="va_r_upper_arm"       file="va_right_upper_arm.stl.obj"/>
    <mesh name="va_r_lower_elbow"     file="va_right_lower_elbow.stl.obj"/>
    <mesh name="va_r_lower_arm"       file="va_right_lower_arm.stl.obj"/>
    <mesh name="wrist"                file="wrist.stl.obj"/>
    <mesh name="wrist_cap"            file="wrist_cap.stl.obj"/>
    <mesh name="va_l_shoulder_front"  file="va_left_shoulder_front.stl.obj"/>
    <mesh name="va_l_upper_arm"       file="va_left_upper_arm.stl.obj"/>
    <mesh name="va_l_lower_elbow"     file="va_left_lower_elbow.stl.obj"/>
    <mesh name="va_l_lower_arm"       file="va_left_lower_arm.stl.obj"/>
    <mesh name="neck"                 file="neck.stl.obj"/>
    <mesh name="head"                 file="head.stl.obj"/>
    <!-- legs -->
    <mesh name="left_lower_hip"   file="left_lower_hip.stl.obj"/>
    <mesh name="right_lower_hip"  file="right_lower_hip.stl.obj"/>
    <mesh name="left_calf"        file="left_calf.stl.obj"/>
    <mesh name="right_calf"       file="right_calf.stl.obj"/>
    <mesh name="ankle"            file="ankle.stl.obj"/>
    <mesh name="foot"             file="foot.stl.obj"/>

    <!-- floor material -->
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.22 0.28 0.35" rgb2="0.15 0.20 0.27"
             markrgb="0.7 0.7 0.7" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="6 6" reflectance="0.15"/>
  </asset>

  <default>
    <joint limited="true" damping="1" armature="0.05"/>
    <geom contype="0" conaffinity="0"/>
  </default>

  <worldbody>
    <light pos="2 0 4" dir="-0.3 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <light pos="-2 2 3" dir="0.3 -0.2 -1" directional="true" diffuse="0.3 0.3 0.35"/>
    <geom name="floor" pos="0 0 -1" size="0 0 0.05" type="plane"
          material="groundplane" contype="0" conaffinity="0"/>

    <!-- ── PELVIS (floating base) ── -->
    <body name="hip_base" pos="0 0 0">
      <freejoint name="root"/>
      <geom type="mesh" mesh="upper_hip" rgba="1 1 1 1" contype="0" conaffinity="0"/>

      <!-- ── TRUNK (waist joint) ── -->
      <body name="trunk">
        <joint name="j1_waist" pos="0 0 0" axis="0 0 1" range="-2.0944 2.0944"/>
        <geom type="mesh" mesh="trunk"   rgba="0.18 0.30 0.82 1" contype="0" conaffinity="0"/>
        <geom pos="0.0549 0 0.3136" type="mesh" mesh="neck" rgba="1 1 1 1" contype="0" conaffinity="0"/>
        <geom pos="0.0199 0 0.4006" type="mesh" mesh="head" rgba="1 1 1 1" contype="0" conaffinity="0"/>

        <!-- ── RIGHT ARM ── -->
        <body name="ra_link1" pos="0 -0.259056 0.355225" quat="0.866025 0.5 0 0">
          <joint name="j2_right" pos="0 0 0" axis="0 0 1" range="-2.0944 2.0944"/>
          <geom quat="0.707106 2e-06 3e-07 0.707107" type="mesh" mesh="kuka_link1"
                rgba="0.18 0.30 0.82 1" contype="0" conaffinity="0"/>
          <body name="ra_link2" pos="0.05395 0 0" quat="0.707107 0 0 0.707107">
            <joint name="j3_right" pos="0 0 0" axis="0 1 0" range="-0.2618 3.1416"/>
            <geom quat="0.612372 -0.353553 0.353553 -0.612372" type="mesh"
                  mesh="va_r_shoulder_front" rgba="1 1 1 1" contype="0" conaffinity="0"/>
            <body name="ra_link3" quat="-0.5 0 0.866025 0">
              <joint name="j4_right" pos="0 0 0" axis="0 0 1" range="-1.8326 1.8326"/>
              <geom quat="-3.6e-12 -0.707107 0.707107 3.6e-12" type="mesh"
                    mesh="va_r_upper_arm" rgba="0.18 0.30 0.82 1" contype="0" conaffinity="0"/>
              <body name="ra_link4" pos="0 -0.03 0.3129">
                <joint name="j5_right" pos="0 0 0" axis="1 0 0" range="0 2.5831"/>
                <geom quat="-3.6e-12 -0.707107 0.707107 3.6e-12" type="mesh"
                      mesh="va_r_lower_elbow" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                <body name="ra_link5" pos="-0.0115 0.03 0.3059">
                  <joint name="j6_right" pos="0 0 0" axis="0 0 1" range="-2.5307 2.0595"/>
                  <geom quat="-3.6e-12 0.707109 0.707104 -3.6e-12" type="mesh"
                        mesh="va_r_lower_arm" rgba="0.18 0.30 0.82 1" contype="0" conaffinity="0"/>
                  <body name="ra_link6" pos="0 -0.014 0.056">
                    <joint name="j7_right" pos="0 0 0" axis="0 1 0" range="-1.8326 1.8326"/>
                    <geom pos="0 0 -0.056" quat="-3.6e-12 -0.707107 0.707107 3.6e-12"
                          type="mesh" mesh="wrist"     rgba="1 1 1 1" contype="0" conaffinity="0"/>
                    <geom pos="0 0 -0.056" quat="-3.6e-12 -0.707107 0.707107 3.6e-12"
                          type="mesh" mesh="wrist_cap" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>

        <!-- ── LEFT ARM ── -->
        <body name="la_link1" pos="0 0.259056 0.355225" quat="0.5 0.866025 0 0">
          <joint name="j8_left" pos="0 0 0" axis="0 0 1" range="-2.0944 2.0944"/>
          <geom quat="2e-06 0.707106 -0.707107 -3e-07" type="mesh" mesh="kuka_link1"
                rgba="0.18 0.30 0.82 1" contype="0" conaffinity="0"/>
          <body name="la_link2" pos="0.05395 0 0" quat="0.965926 -0.258819 0 0">
            <joint name="j9_left" pos="0 0 0" axis="1 0 0" range="-0.2618 3.1416"/>
            <geom quat="0.707107 -0.707107 0 0" type="mesh"
                  mesh="va_l_shoulder_front" rgba="1 1 1 1" contype="0" conaffinity="0"/>
            <body name="la_link3">
              <joint name="j10_left" pos="0 0 0" axis="0 1 0" range="-1.8326 1.8326"/>
              <geom quat="0.707107 -0.707107 0 0" type="mesh"
                    mesh="va_l_upper_arm" rgba="0.18 0.30 0.82 1" contype="0" conaffinity="0"/>
              <body name="la_link4" pos="0.03 -0.3128 0">
                <joint name="j11_left" pos="0 0 0" axis="0 0 1" range="0 2.5831"/>
                <geom quat="0.707107 -0.707107 0 0" type="mesh"
                      mesh="va_l_lower_elbow" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                <body name="la_link5" pos="-0.03 -0.3058 0.0115">
                  <joint name="j12_left" pos="0 0 0" axis="0 1 0" range="-2.5307 2.0595"/>
                  <geom quat="3e-06 -3e-06 -0.707107 -0.707107" type="mesh"
                        mesh="va_l_lower_arm" rgba="0.18 0.30 0.82 1" contype="0" conaffinity="0"/>
                  <body name="la_link6" pos="0.014 -0.056 0">
                    <joint name="j13_left" pos="0 0 0" axis="1 0 0" range="-1.8326 1.8326"/>
                    <geom pos="0 0.056 0" quat="0.707107 -0.707107 0 0"
                          type="mesh" mesh="wrist"     rgba="1 1 1 1" contype="0" conaffinity="0"/>
                    <geom pos="0 0.056 0" quat="0.707107 -0.707107 0 0"
                          type="mesh" mesh="wrist_cap" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body><!-- trunk -->

      <!-- ════════════════════════════════════════════════════
           LEFT LEG
           Hip joint at (0, +0.115, -0.05) from pelvis.
           Thigh = 0.42 m,  Shank = 0.42 m
      ════════════════════════════════════════════════════ -->
      <body name="LL_hip" pos="0 0.115 -0.05">
        <joint name="LL_hip_yaw"    axis="0 0 1"  range="-0.8 0.8"/>
        <joint name="LL_hip_abduct" axis="1 0 0"  range="-0.5 0.5"/>
        <joint name="LL_hip_flex"   axis="0 -1 0" range="-1.4 1.0"/>
        <!-- hip joint visual: left_lower_hip mesh (two ball bearings) -->
        <geom type="mesh" mesh="left_lower_hip" rgba="0.18 0.30 0.82 1"
              contype="0" conaffinity="0"/>
        <!-- thigh capsule bridging hip → knee -->
        <geom type="capsule" fromto="0 0 -0.10 0 0 -0.42" size="0.048"
              rgba="0.18 0.30 0.82 1" contype="0" conaffinity="0"/>

        <body name="LL_knee" pos="0 0 -0.42">
          <joint name="LL_knee_flex" axis="0 1 0" range="0 2.4"/>
          <!-- lower leg: left_calf mesh (vertical tube, ±0.27m around knee) -->
          <geom type="mesh" mesh="left_calf" rgba="1 1 1 1"
                contype="0" conaffinity="0"/>

          <body name="LL_ankle" pos="0 0 -0.42">
            <joint name="LL_ankle_flex" axis="0 1 0" range="-0.6 0.5"/>
            <geom type="mesh" mesh="ankle" rgba="0.18 0.30 0.82 1"
                  contype="0" conaffinity="0"/>
            <body name="LL_foot" pos="0 0 -0.065">
              <geom type="mesh" mesh="foot" rgba="1 1 1 1"
                    contype="0" conaffinity="0"/>
            </body>
          </body>
        </body>
      </body><!-- LL_hip -->

      <!-- ════════════════════════════════════════════════════
           RIGHT LEG (mirror in Y)
      ════════════════════════════════════════════════════ -->
      <body name="RL_hip" pos="0 -0.115 -0.05">
        <joint name="RL_hip_yaw"    axis="0 0 1"  range="-0.8 0.8"/>
        <joint name="RL_hip_abduct" axis="1 0 0"  range="-0.5 0.5"/>
        <joint name="RL_hip_flex"   axis="0 -1 0" range="-1.4 1.0"/>
        <geom type="mesh" mesh="right_lower_hip" rgba="0.18 0.30 0.82 1"
              contype="0" conaffinity="0"/>
        <geom type="capsule" fromto="0 0 -0.10 0 0 -0.42" size="0.048"
              rgba="0.18 0.30 0.82 1" contype="0" conaffinity="0"/>

        <body name="RL_knee" pos="0 0 -0.42">
          <joint name="RL_knee_flex" axis="0 1 0" range="0 2.4"/>
          <geom type="mesh" mesh="right_calf" rgba="1 1 1 1"
                contype="0" conaffinity="0"/>

          <body name="RL_ankle" pos="0 0 -0.42">
            <joint name="RL_ankle_flex" axis="0 1 0" range="-0.6 0.5"/>
            <geom type="mesh" mesh="ankle" rgba="0.18 0.30 0.82 1"
                  contype="0" conaffinity="0"/>
            <body name="RL_foot" pos="0 0 -0.065">
              <geom type="mesh" mesh="foot" rgba="1 1 1 1"
                    contype="0" conaffinity="0"/>
            </body>
          </body>
        </body>
      </body><!-- RL_hip -->

    </body><!-- hip_base -->
  </worldbody>
</mujoco>
"""


# ── Quaternion helper ────────────────────────────────────────────────────────
def euler_to_quat(roll, pitch, yaw):
    """ZYX Euler → MuJoCo quaternion [w, x, y, z]."""
    cr, cp, cy = np.cos(roll/2), np.cos(pitch/2), np.cos(yaw/2)
    sr, sp, sy = np.sin(roll/2), np.sin(pitch/2), np.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


# ── Build joint-address lookup ───────────────────────────────────────────────
def build_joint_map(model):
    jmap = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            jmap[name] = model.jnt_qposadr[i]
    return jmap


def set_joint(data, jmap, name, val):
    if name in jmap:
        data.qpos[jmap[name]] = val


# ── Kinematic gait ───────────────────────────────────────────────────────────
def gait_angles(t, T=1.0):
    phi = 2.0 * np.pi * t / T   # full cycle

    def leg(p):
        hip_flex   =  0.38 * np.sin(p)
        knee_flex  =  0.52 * np.clip(np.sin(p + 0.35), 0, 1)
        ankle_flex = -0.18 * np.sin(p - 0.2)
        hip_abduct = -0.04 * np.cos(p)
        return hip_flex, knee_flex, ankle_flex, hip_abduct

    hfL, kfL, afL, haL = leg(phi)
    hfR, kfR, afR, haR = leg(phi + np.pi)

    # Arm swing: j2_right / j8_left oscillate (shoulder rotation)
    # These are the first shoulder DOFs; axis="0 0 1" in rotated frame
    # → a small oscillation gives visible arm swing
    arm_swing = 0.30 * np.sin(phi)

    return dict(
        LL_hip_flex=hfL, LL_knee_flex=kfL, LL_ankle_flex=afL, LL_hip_abduct=haL,
        RL_hip_flex=hfR, RL_knee_flex=kfR, RL_ankle_flex=afR, RL_hip_abduct=haR,
        j2_right= arm_swing,   # right arm swings with left leg (counter)
        j8_left =-arm_swing,   # left arm swings with right leg
    )


FLOOR_Z = -1.0   # floor level in world coordinates

def set_pose(model, data, jmap, t, speed=0.35, T=1.0, lean=-0.04):
    phi = 2.0 * np.pi * t / T

    # ── Leg & arm joint angles ──────────────────────────────────────────────
    for name, val in gait_angles(t, T).items():
        set_joint(data, jmap, name, val)

    # ── Pelvis (freejoint) ──────────────────────────────────────────────────
    px = speed * t
    py = 0.02 * np.sin(phi)              # tiny lateral sway
    pz = 0.0 - 0.012 * np.cos(2*phi)    # vertical bob (around nominal z=0)

    roll  =  0.02 * np.sin(phi)
    pitch =  lean
    yaw   =  0.03 * np.sin(phi + 0.3)

    data.qpos[0] = px
    data.qpos[1] = py
    data.qpos[2] = pz
    data.qpos[3:7] = euler_to_quat(roll, pitch, yaw)

    # ── First kinematics pass ───────────────────────────────────────────────
    mujoco.mj_kinematics(model, data)

    # ── Adjust pelvis z so feet stay at floor level ─────────────────────────
    # foot bottom ≈ body_z + mesh z-min. foot mesh z-min ≈ -0.112; ankle z-min ≈ 0.
    # LL_foot body offset from LL_ankle = -0.065; ankle from LL_knee = -0.42; ...
    # Simpler: measure actual foot body z and correct.
    fid_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LL_foot")
    fid_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "RL_foot")
    FOOT_MESH_ZMIN = -0.112   # foot mesh extends 0.112 m below foot body origin
    bot_L = data.xpos[fid_L, 2] + FOOT_MESH_ZMIN
    bot_R = data.xpos[fid_R, 2] + FOOT_MESH_ZMIN
    lowest = min(bot_L, bot_R)

    # We want lowest ≈ FLOOR_Z  (= -1.0)
    delta = lowest - FLOOR_Z
    data.qpos[2] -= delta          # shift pelvis to bring foot to floor

    # ── Final kinematics pass ───────────────────────────────────────────────
    mujoco.mj_kinematics(model, data)


# ── Interactive viewer ───────────────────────────────────────────────────────
def run_interactive():
    import time
    model = mujoco.MjModel.from_xml_string(TORO_FULL_XML)
    data  = mujoco.MjData(model)
    jmap  = build_joint_map(model)

    print("Joints:", list(jmap.keys()))
    set_pose(model, data, jmap, 0.0)

    print("MuJoCo viewer (close window to quit)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.0]
        viewer.cam.distance  = 4.0
        viewer.cam.elevation = -12
        viewer.cam.azimuth   = 100

        t0 = time.perf_counter()
        while viewer.is_running():
            t = time.perf_counter() - t0
            set_pose(model, data, jmap, t, speed=0.35, T=1.0)
            viewer.cam.lookat[0] = data.qpos[0]
            viewer.cam.lookat[2] = data.qpos[2] + 0.2   # track mid-torso
            viewer.sync()
            time.sleep(1.0 / 60.0)


# ── Offline rendering ────────────────────────────────────────────────────────
def run_render(output="toro_real_walk.mp4", duration=7.0, fps=60,
               width=1280, height=720, speed=0.35, T=1.0):
    model    = mujoco.MjModel.from_xml_string(TORO_FULL_XML)
    data     = mujoco.MjData(model)
    jmap     = build_joint_map(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam      = mujoco.MjvCamera()

    n = int(duration * fps)
    dt = 1.0 / fps
    tmpdir = tempfile.mkdtemp()
    print(f"Rendering {n} frames → {output}")

    for i in range(n):
        t = i * dt
        set_pose(model, data, jmap, t, speed=speed, T=T)

        # side view following robot
        cam.lookat[0] = data.qpos[0]
        cam.lookat[1] = 0.0
        cam.lookat[2] = data.qpos[2] + 0.15  # aim at mid-torso
        cam.distance  = 3.8
        cam.elevation = -12
        cam.azimuth   = 100          # side view

        renderer.update_scene(data, camera=cam)
        Image.fromarray(renderer.render()).save(
            os.path.join(tmpdir, f"frame_{i:05d}.png"))

        if i % fps == 0:
            print(f"  {t:.1f}s / {duration:.1f}s")

    renderer.close()

    cmd = ["ffmpeg", "-y", "-framerate", str(fps),
           "-i", os.path.join(tmpdir, "frame_%05d.png"),
           "-c:v", "libx264", "-preset", "slow", "-crf", "17",
           "-pix_fmt", "yuv420p", output]
    print("Encoding…")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Saved: {output}")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render",   action="store_true")
    ap.add_argument("--output",   default="toro_real_walk.mp4")
    ap.add_argument("--duration", type=float, default=7.0)
    ap.add_argument("--fps",      type=int,   default=60)
    ap.add_argument("--width",    type=int,   default=1280)
    ap.add_argument("--height",   type=int,   default=720)
    args = ap.parse_args()

    if args.render:
        run_render(output=args.output, duration=args.duration,
                   fps=args.fps, width=args.width, height=args.height)
    else:
        run_interactive()
