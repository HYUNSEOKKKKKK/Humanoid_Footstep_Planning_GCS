from __future__ import annotations
import matplotlib.pyplot as plt
from src.run_main_DDR_sliding_TORO import run_main_DDR_sliding_TORO
from src.viz.kinematic import prepare_3d_axes
from collections import deque
from matplotlib.animation import FuncAnimation

import time

def timed_call(fn, *args, label="elapsed", **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Total time: {dt_ms:.2f} ms")
    return out


def main():

    fig = plt.figure("Pump game Footstep Planner", figsize=(6,3))
    mgr = plt.get_current_fig_manager()
    try:
        mgr.window.move(-100, -100)  # QtAgg (PyQt/PySide)
    except Exception:
        try:
            mgr.window.wm_geometry("+100+120")  # TkAgg
        except Exception:
            try:
                w, h = fig.get_size_inches() * fig.dpi
                mgr.window.setGeometry(0, 0, int(w), int(h))
            except Exception:
                pass

    ax2d = fig.add_subplot(121)
    ax3d = prepare_3d_axes(fig, rect=122)
    ax2d.set_position([0.05, 0.1, 0.4, 0.8])
    ax3d.set_position([0.5, 0.05, 0.48, 0.9])

    # 키→라벨 매핑
    key_to_label = {
        '0':'All', '1':'DL','2':'D','3':'DR','4':'L','5':'C','6':'R','7':'UL','8':'U','9':'UR'
    }

    # ===== 윈도우 길이(7)로 변경 =====
    WINDOW_LEN = 7

    # 초기 윈도우
    # ['3', '1', '9','7','3','5','9','1']
    label_window = deque(
        ['All','UR','UL','DR','C','UR','DL'],

        maxlen=WINDOW_LEN
    )

    # # 초기 윈도우(전부 'All'로 프리필)
    # label_window = deque(['All'] * WINDOW_LEN, maxlen=WINDOW_LEN)


    # === 오토플레이 시퀀스 (원하면 None으로 두고 키보드로 조작)
    sequence_keys = None

    # TORO - 100bpm (1.2sec)
    # sequence_keys = ['9', '7', '0', '0', '0', '0', '0', '0']                 # safety RLRL mode 10steps
    # sequence_keys = ['9', '7', '5', '7', '9', '5', '3', '1', '9', '7', '5', '1', '3', '5', '9', '1', '5', '7', '3', '5',
    #                  '9', '7', '3', '1', '9', '5', '3', '1', '3', '7', '9', '7',
    #                  '9','7','9','7','3','1','3','1','5','7','9','7','3','5','3',
    #                  '0', '0', '0', '0', '0', '0']
    # sequence_keys = ['9', '7', '5', '7', '9', '5', '3', '1', '9', '7', '5', '1', '3', '5', '9', '1', '5', '7', '3',
    #                  '0', '0', '0', '0', '0', '0']

# ######################################## seconde experiment preparing #########################################
#     # 초기 윈도우
#     # ['3', '1', '5','7','3','5','9','1']
#     label_window = deque(
#         ['All','C','UL','DR','C','UR','DL'],
#
#         maxlen=WINDOW_LEN
#     )
#     sequence_keys = ['9', '7', '0', '0', '0', '0', '0', '0']
#     # sequence_keys = ['9', '7', '0', '0', '0', '0', '0', '0']



    # 상태
    prevL = prevR = None
    prev_mover = None
    prev_y0_hint = None

    # 첫 프레임 실행 (list로 변환해 함수에 전달)
    prevL, prevR, prev_mover, prev_y0_hint = timed_call(
        run_main_DDR_sliding_TORO, list(label_window),
        prevL, prevR, prev_mover, prev_y0_hint,
        ax2d=ax2d, ax3d=ax3d, label="[first] run_main_DDR_sliding_TORO"
    )
    fig = plt.gcf()

    # === 키보드 모드 (sequence_keys가 None일 때만 활성)
    # 키보드 모드에서도 동일하게 4개 언패킹
    def on_key(event):
        nonlocal label_window, prevL, prevR, prev_mover, prev_y0_hint
        if event.key == 'q':
            plt.close(event.canvas.figure);
            return
        if event.key in key_to_label:
            label_window.append(key_to_label[event.key])
            prevL, prevR, prev_mover, prev_y0_hint = timed_call(
                run_main_DDR_sliding_TORO, list(label_window),
                prevL, prevR, prev_mover, prev_y0_hint,
                ax2d=ax2d, ax3d=ax3d, label="[key] run_main_DDR_sliding_TORO"
            )
        else:
            print(f'wrong button "{event.key}". [0-9] enter.')

    # === 오토플레이 모드 (고정 주기 스케줄러: 드리프트 0, 지연 보정) ===
    if sequence_keys is not None:
        seq_iter = iter(sequence_keys)

        STEP_PERIOD_S = 1.2  # 1.2초(= 100 bpm) 고정 주기
        # STEP_PERIOD_S = step_interval_ms / 1000.0  # ms를 쓰고 싶다면 이렇게

        # 스케줄 상태
        k = 0
        t0 = time.perf_counter()  # 기준 시작 시각
        active_timer = None
        stopped = False

        def schedule_next():
            """다음 tick을 절대시간(deadline) 기준으로 예약"""
            nonlocal active_timer, k
            if stopped:
                return
            deadline = t0 + (k + 1) * STEP_PERIOD_S
            now = time.perf_counter()
            delay = max(0.0, deadline - now)  # 늦었으면 0으로 즉시 실행(캐치업)
            # matplotlib 백엔드 타이머 (single-shot)
            active_timer = fig.canvas.new_timer(interval=int(delay * 1000))
            active_timer.single_shot = True
            active_timer.add_callback(tick)
            active_timer.start()

        def tick():
            """한 스텝 진행 (플래닝 시간과 무관하게 다음은 절대시간으로 예약)"""
            nonlocal k, stopped, label_window, prevL, prevR, prev_mover, prev_y0_hint

            # 키 꺼내기
            try:
                key = next(seq_iter)
            except StopIteration:
                stopped = True
                return

            if key not in key_to_label:
                print(f'ignored invalid key "{key}"')
            else:
                label = key_to_label[key]
                label_window.append(label)

                # === 여기서 플래닝 수행 ===
                prevL, prevR, prev_mover, prev_y0_hint = timed_call(
                    run_main_DDR_sliding_TORO, list(label_window),
                    prevL, prevR, prev_mover, prev_y0_hint,
                    ax2d=ax2d, ax3d=ax3d, label=f"[auto] step {k}"
                )

            k += 1
            schedule_next()  # 다음 호출을 절대시간 기준으로 예약

        # 첫 번 예약(지금 시각을 기준으로 STEP_PERIOD_S 뒤에 1스텝)
        schedule_next()
        plt.show()
    else:
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()



if __name__ == "__main__":
    main()
