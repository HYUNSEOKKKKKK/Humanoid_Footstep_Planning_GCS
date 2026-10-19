"""
TORO humanoid walking visualization in MuJoCo (kinematic, no physics).
Run:  conda activate mujoco_env && python toro_walk.py
      conda activate mujoco_env && python toro_walk.py --render  # save video
"""
import numpy as np
import mujoco
import mujoco.viewer
import argparse
import subprocess
import os
import tempfile

# ──────────────────────────────────────────────────────────────────────────────
# MJCF model (TORO-proportioned humanoid)
# Dimensions from footstep_planner:
#   pelvis_z=0.90, hip_offset=0.12 (lateral), thigh=0.42, shank=0.42, foot 0.20x0.08
# ──────────────────────────────────────────────────────────────────────────────
TORO_XML = """
<mujoco model="toro">
  <compiler angle="radian" autolimits="true"/>

  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4" specular="0.05 0.05 0.05"/>
    <rgba haze="0.2 0.25 0.3 1"/>
    <global offwidth="1920" offheight="1080" elevation="-15" azimuth="0"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1="0.82 0.82 0.82" rgb2="0.72 0.72 0.72" width="512" height="512"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="6 6" reflectance="0.1"/>
    <material name="body_mat"  rgba="0.38 0.48 0.58 1"/>
    <material name="dark_mat"  rgba="0.22 0.30 0.38 1"/>
    <material name="mid_mat"   rgba="0.45 0.55 0.65 1"/>
  </asset>

  <worldbody>
    <!-- Ground -->
    <geom name="floor" type="plane" size="6 6 0.1" material="floor_mat"
          contype="0" conaffinity="0"/>

    <!-- ── PELVIS (root) ── -->
    <body name="pelvis" pos="0 0 0.90">
      <freejoint name="root"/>
      <geom type="box" size="0.115 0.09 0.09" material="body_mat"/>

      <!-- ── TORSO ── -->
      <body name="torso" pos="0 0 0.17">
        <geom type="box" size="0.12 0.09 0.17" material="body_mat"/>

        <!-- ── HEAD / NECK ── -->
        <body name="head" pos="0 0 0.32">
          <geom type="capsule" fromto="0 0 -0.07 0 0 0.10"
                size="0.085" material="mid_mat"/>
        </body>

        <!-- ── LEFT SHOULDER / ARM ── -->
        <body name="upper_arm_L" pos="0 0.20 0.12">
          <!-- shoulder_L: positive angle = forward swing (arm goes toward +x) -->
          <joint name="shoulder_L" type="hinge" axis="0 -1 0"
                 range="-1.4 1.4" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.28"
                size="0.038" material="body_mat"/>
          <body name="lower_arm_L" pos="0 0 -0.28">
            <joint name="elbow_L" type="hinge" axis="0 1 0"
                   range="-2.2 0" damping="0.1"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.26"
                  size="0.032" material="dark_mat"/>
          </body>
        </body>

        <!-- ── RIGHT SHOULDER / ARM ── -->
        <body name="upper_arm_R" pos="0 -0.20 0.12">
          <joint name="shoulder_R" type="hinge" axis="0 -1 0"
                 range="-1.4 1.4" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.28"
                size="0.038" material="body_mat"/>
          <body name="lower_arm_R" pos="0 0 -0.28">
            <joint name="elbow_R" type="hinge" axis="0 1 0"
                   range="-2.2 0" damping="0.1"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.26"
                  size="0.032" material="dark_mat"/>
          </body>
        </body>
      </body>

      <!-- ── LEFT LEG ── -->
      <!-- hip at y=+0.12, z=-0.05 relative to pelvis center -->
      <body name="thigh_L" pos="0 0.12 -0.05">
        <!-- hip_abduct_L: positive = abduction (leg moves left = +y) -->
        <joint name="hip_abduct_L" type="hinge" axis="1 0 0"
               range="-0.5 0.5" damping="0.1"/>
        <!-- hip_flex_L: positive = forward swing -->
        <joint name="hip_flex_L"   type="hinge" axis="0 -1 0"
               range="-1.4 1.0" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.42"
              size="0.05" material="body_mat"/>
        <body name="shank_L" pos="0 0 -0.42">
          <!-- knee_L: positive = flexion (bends backward) -->
          <joint name="knee_L" type="hinge" axis="0 1 0"
                 range="0 2.4" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.42"
                size="0.045" material="dark_mat"/>
          <body name="foot_L" pos="0 0 -0.42">
            <!-- ankle_L: positive = dorsiflexion (toes up) -->
            <joint name="ankle_L" type="hinge" axis="0 1 0"
                   range="-0.7 0.5" damping="0.05"/>
            <!-- foot geom: extends 0.14 forward, 0.06 back from ankle, 2.5cm thick -->
            <geom type="box" size="0.10 0.04 0.025"
                  pos="0.04 0 -0.03" material="dark_mat"/>
          </body>
        </body>
      </body>

      <!-- ── RIGHT LEG ── -->
      <body name="thigh_R" pos="0 -0.12 -0.05">
        <joint name="hip_abduct_R" type="hinge" axis="1 0 0"
               range="-0.5 0.5" damping="0.1"/>
        <joint name="hip_flex_R"   type="hinge" axis="0 -1 0"
               range="-1.4 1.0" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.42"
              size="0.05" material="body_mat"/>
        <body name="shank_R" pos="0 0 -0.42">
          <joint name="knee_R" type="hinge" axis="0 1 0"
                 range="0 2.4" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.42"
                size="0.045" material="dark_mat"/>
          <body name="foot_R" pos="0 0 -0.42">
            <joint name="ankle_R" type="hinge" axis="0 1 0"
                   range="-0.7 0.5" damping="0.05"/>
            <geom type="box" size="0.10 0.04 0.025"
                  pos="0.04 0 -0.03" material="dark_mat"/>
          </body>
        </body>
      </body>

    </body>
  </worldbody>
</mujoco>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Quaternion helper
# ──────────────────────────────────────────────────────────────────────────────
def euler_to_quat(roll, pitch, yaw):
    """ZYX Euler → MuJoCo quaternion [w, x, y, z]."""
    cr, cp, cy = np.cos(roll/2), np.cos(pitch/2), np.cos(yaw/2)
    sr, sp, sy = np.sin(roll/2), np.sin(pitch/2), np.sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])


# ──────────────────────────────────────────────────────────────────────────────
# Kinematic gait generator
# ──────────────────────────────────────────────────────────────────────────────
def gait_joint_angles(t, T=1.0):
    """
    Sinusoidal gait for TORO.
    Returns dict of joint_name → angle (radians).
    T: stride period in seconds.
    """
    phi = 2.0 * np.pi * t / T   # full gait cycle

    # Each leg is half a cycle apart
    phi_L = phi
    phi_R = phi + np.pi

    def leg(p):
        # hip flexion: ±22°, positive = forward
        hip_flex   = 0.38 * np.sin(p)
        # knee: bends only during swing (leg moving forward)
        # knee range is [0, 2.4] so angle must stay ≥ 0
        knee_flex  = 0.50 * np.clip(np.sin(p + 0.3), 0, 1)
        # ankle: dorsiflexion leads the swing
        ankle      = -0.18 * np.sin(p - 0.2)
        # small hip abduction for lateral balance
        hip_abduct = -0.05 * np.cos(p)
        return hip_flex, knee_flex, ankle, hip_abduct

    hf_L, kf_L, ank_L, ha_L = leg(phi_L)
    hf_R, kf_R, ank_R, ha_R = leg(phi_R)

    # Arms swing counter-phase to ipsilateral leg
    sh_L =  0.45 * np.sin(phi_R)   # left arm swings with right leg
    sh_R =  0.45 * np.sin(phi_L)
    el_L = -0.35 - 0.12 * np.cos(phi_L)  # always slightly bent
    el_R = -0.35 - 0.12 * np.cos(phi_R)

    return {
        "hip_abduct_L": ha_L, "hip_flex_L": hf_L,
        "knee_L":        kf_L, "ankle_L":   ank_L,
        "hip_abduct_R": ha_R, "hip_flex_R": hf_R,
        "knee_R":        kf_R, "ankle_R":   ank_R,
        "shoulder_L": sh_L, "elbow_L": el_L,
        "shoulder_R": sh_R, "elbow_R": el_R,
    }


def set_kinematic_pose(model, data, t,
                       walking_speed=0.4,  # m/s forward
                       T=1.0,              # gait period
                       forward_tilt=-0.04, # rad, slight forward lean
                       ):
    """Set full kinematic pose at time t; adjusts pelvis z to keep feet on ground."""
    phi = 2.0 * np.pi * t / T

    # ── Joint angles ──────────────────────────────────────────────────────────
    angles = gait_joint_angles(t, T)
    for name, val in angles.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = val

    # ── Pelvis 6-DOF ─────────────────────────────────────────────────────────
    pelvis_x     = walking_speed * t
    pelvis_y     = 0.025 * np.sin(phi)          # lateral sway
    pelvis_z_nom = 0.90 - 0.012 * np.cos(2*phi) # vertical bob

    roll  =  0.025 * np.sin(phi)
    pitch =  forward_tilt
    yaw   =  0.04  * np.sin(phi + 0.3)

    q = euler_to_quat(roll, pitch, yaw)

    data.qpos[0] = pelvis_x
    data.qpos[1] = pelvis_y
    data.qpos[2] = pelvis_z_nom
    data.qpos[3:7] = q

    # ── Run kinematics once to get actual foot positions ──────────────────────
    mujoco.mj_kinematics(model, data)

    # ── Adjust pelvis z so the lowest foot just touches ground ───────────────
    fid_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    fid_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")

    # foot geom bottom ≈ body_z + geom_pos_z - geom_size[2]
    # geom pos="0.04 0 -0.03", size[2]=0.025 → bottom at body_z - 0.055
    foot_bottom_L = data.xpos[fid_L, 2] - 0.055
    foot_bottom_R = data.xpos[fid_R, 2] - 0.055
    lowest = min(foot_bottom_L, foot_bottom_R)

    if lowest < 0.0:
        data.qpos[2] -= lowest   # lift pelvis so feet stay at ground
    elif lowest > 0.005:
        data.qpos[2] -= (lowest - 0.002)  # don't float too high

    # ── Final kinematics pass ─────────────────────────────────────────────────
    mujoco.mj_kinematics(model, data)


# ──────────────────────────────────────────────────────────────────────────────
# Camera follow helper
# ──────────────────────────────────────────────────────────────────────────────
def update_camera(viewer_or_renderer, pelvis_x, pelvis_z,
                  distance=3.5, elevation=-15, azimuth=-30):
    """Smoothly follow the robot."""
    cam = viewer_or_renderer.cam if hasattr(viewer_or_renderer, 'cam') else viewer_or_renderer
    cam.lookat[0] = pelvis_x
    cam.lookat[1] = 0.0
    cam.lookat[2] = pelvis_z
    cam.distance  = distance
    cam.elevation = elevation
    cam.azimuth   = azimuth


# ──────────────────────────────────────────────────────────────────────────────
# Interactive viewer mode
# ──────────────────────────────────────────────────────────────────────────────
def run_interactive():
    model = mujoco.MjModel.from_xml_string(TORO_XML)
    data  = mujoco.MjData(model)

    # warmup
    set_kinematic_pose(model, data, 0.0)

    print("MuJoCo viewer 실행 중 (창 닫으면 종료)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = 3.5
        viewer.cam.elevation = -15
        viewer.cam.azimuth   = -30
        viewer.cam.lookat[:] = [0, 0, 0.90]

        import time
        t_start = time.perf_counter()
        T_GAIT  = 1.0
        SPEED   = 0.4   # m/s

        while viewer.is_running():
            t = time.perf_counter() - t_start
            set_kinematic_pose(model, data, t, walking_speed=SPEED, T=T_GAIT)

            # follow camera
            pelvis_x = data.qpos[0]
            viewer.cam.lookat[0] = pelvis_x
            viewer.cam.lookat[2] = data.qpos[2]

            viewer.sync()
            time.sleep(1.0 / 60.0)


# ──────────────────────────────────────────────────────────────────────────────
# Offline rendering → MP4
# ──────────────────────────────────────────────────────────────────────────────
def make_camera(lookat_x=0.0, lookat_z=0.90,
                distance=3.5, elevation=-14, azimuth=-35):
    """Create and configure a MjvCamera."""
    cam = mujoco.MjvCamera()
    cam.lookat[0] = lookat_x
    cam.lookat[1] = 0.0
    cam.lookat[2] = lookat_z
    cam.distance  = distance
    cam.elevation = elevation
    cam.azimuth   = azimuth
    return cam


def run_render(output_path="toro_walk.mp4",
               duration=6.0,
               fps=60,
               width=1280,
               height=720,
               T_GAIT=1.0,
               SPEED=0.4):
    model    = mujoco.MjModel.from_xml_string(TORO_XML)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam      = mujoco.MjvCamera()

    n_frames = int(duration * fps)
    dt       = 1.0 / fps

    tmpdir = tempfile.mkdtemp()
    print(f"Rendering {n_frames} frames → {output_path}")

    for i in range(n_frames):
        t = i * dt
        set_kinematic_pose(model, data, t, walking_speed=SPEED, T=T_GAIT)

        cam.lookat[0] = data.qpos[0]
        cam.lookat[1] = 0.0
        cam.lookat[2] = 0.85
        cam.distance  = 3.2
        cam.elevation = -12
        cam.azimuth   = 100   # side view, slightly behind

        renderer.update_scene(data, camera=cam)
        frame = renderer.render()

        img_path = os.path.join(tmpdir, f"frame_{i:05d}.png")
        from PIL import Image
        Image.fromarray(frame).save(img_path)

        if i % fps == 0:
            print(f"  {t:.1f}s / {duration:.1f}s")

    renderer.close()

    # Compile with ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(tmpdir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    print("FFmpeg encoding...")
    subprocess.run(cmd, check=True)
    print(f"Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true",
                    help="Render to MP4 instead of interactive viewer")
    ap.add_argument("--output", default="toro_walk.mp4")
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width",  type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    if args.render:
        run_render(output_path=args.output,
                   duration=args.duration,
                   fps=args.fps,
                   width=args.width,
                   height=args.height)
    else:
        run_interactive()
