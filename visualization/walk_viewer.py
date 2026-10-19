"""
Humanoid walking visualization using MuJoCo + dm_control.
실행: conda activate mujoco_env && python walk_viewer.py
"""
import numpy as np
import mujoco
import mujoco.viewer
from dm_control import suite


def main():
    # dm_control humanoid walk 환경 로드
    env = suite.load(domain_name="humanoid", task_name="walk", task_kwargs={"random": 42})
    physics = env.physics

    print("모델 정보:")
    print(f"  관절 수: {physics.model.njnt}")
    print(f"  액추에이터 수: {physics.model.nu}")
    print(f"  바디 수: {physics.model.nbody}")
    print("\nMuJoCo 뷰어 창을 띄웁니다. 창을 닫으면 종료됩니다.")

    # 초기 reset
    time_step = env.reset()

    with mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

        step = 0
        while viewer.is_running():
            # 사인파 기반 간단한 걷기 동작 (키네마틱 데모용)
            t = step * 0.002
            action_spec = env.action_spec()
            action_dim = action_spec.shape[0]

            # 각 관절에 주기적인 사인파 액션 — 그럴듯한 걷기 모션
            action = np.zeros(action_dim)
            for i in range(action_dim):
                phase = 2 * np.pi * i / action_dim
                action[i] = 0.4 * np.sin(2 * t + phase)

            time_step = env.step(action)

            # 넘어지면 reset
            if time_step.last():
                time_step = env.reset()

            viewer.sync()
            step += 1


if __name__ == "__main__":
    main()
