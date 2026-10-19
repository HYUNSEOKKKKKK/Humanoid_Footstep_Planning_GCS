from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

_prev_first_label = None

# 👇 모듈 상수로 빼두기 (plot_stone_map과 공유)
STONE_MAP = {
    'DL': dict(center=(-0.20, -0.20), width=0.20, height=0.20),
    'D':  dict(center=(0.0, -0.20),  width=0.20, height=0.20),
    'DR': dict(center=(0.20, -0.20), width=0.20, height=0.20),
    'L':  dict(center=(-0.20, 0.0),  width=0.20, height=0.20),
    'C':  dict(center=(0.0, 0.0),    width=0.20, height=0.20),
    'R':  dict(center=(0.20, 0.0),   width=0.20, height=0.20),
    'UL': dict(center=(-0.20, 0.20), width=0.20, height=0.20),
    'U':  dict(center=(0.0, 0.20),   width=0.20, height=0.20),
    'UR': dict(center=(0.20, 0.20),  width=0.20, height=0.20),
}

def plot_stone_map(ax=None, show_labels=False, show_grid=False,
                   window_labels=None, last_key=None,
                   bg_image=None):
    global _prev_first_label
    if ax is None:
        ax = plt.gca()

    if bg_image is not None:
        img = plt.imread(bg_image)
        img = np.flipud(img)
        ax.imshow(img, extent=(-0.3, 0.3, -0.3, 0.3), origin='lower', zorder=0)

    # 돌판 색/라벨만 그림 (하이라이트는 여기서 그리지 않음!)
    for k, st in STONE_MAP.items():
        cx, cy = st['center']; w, h = st['width'], st['height']
        if show_labels:
            ax.text(cx, cy, k, ha='center', va='center', fontsize=10, fontweight='bold', zorder=6)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.4, 0.4); ax.set_ylim(-0.4, 0.4)

    ax.set_title('DLR TORO')

    _prev_first_label = window_labels[0] if (window_labels and len(window_labels) > 0) else None

# ✨ 항상 맨 위에 1개 하이라이트만 유지/업데이트
def upsert_highlight(ax, label_key, *, zorder=1_000_000):
    if label_key not in STONE_MAP:
        if hasattr(ax, "_hl_rect") and ax._hl_rect:
            ax._hl_rect.set_visible(False)
        return

    st = STONE_MAP[label_key]
    cx, cy = st['center']; w, h = st['width'], st['height']
    x0, y0 = cx - w/2, cy - h/2

    if not hasattr(ax, "_hl_rect") or ax._hl_rect is None:
        rect = plt.Rectangle((x0, y0), w, h, fill=False,
                             edgecolor='w', linewidth=3.0, zorder=zorder)
        rect.set_path_effects([
            pe.Stroke(linewidth=10, foreground='k', alpha=1.6),
            pe.Normal(),
        ])
        ax.add_patch(rect)
        ax._hl_rect = rect
    else:
        r = ax._hl_rect
        r.set_xy((x0, y0)); r.set_width(w); r.set_height(h)
        r.set_zorder(zorder); r.set_visible(True)

def compute_foot_rects(x, y, c, s, h=0.2, w=0.1):
    """MATLAB compute_foot_rects와 동일."""
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    c = np.asarray(c).reshape(-1)
    s = np.asarray(s).reshape(-1)
    n = x.size
    rects = []
    base = 0.5 * np.array([[-h, -w], [h, -w], [h, w], [-h, w]], float)  # (4,2)
    for i in range(n):
        norm = max(1e-9, float(np.hypot(c[i], s[i])))
        cc, ss = c[i] / norm, s[i] / norm
        R = np.array([[cc, -ss], [ss, cc]], float)
        rot = (R @ base.T).T
        rot[:, 0] += x[i]; rot[:, 1] += y[i]
        rects.append(rot)
    return rects

def draw_heading_arrow(ax, x, y, c, s,
                       length=0.12, color='k', alpha=1.0,
                       zorder=6, linewidth=1.6, mutation_scale=18):
    """
    (x,y)에서 [c,s] 방향으로 length 길이의 화살표를 그림.
    c,s는 보통 [cos(yaw), sin(yaw)].
    """
    n = float(np.hypot(c, s))
    if n < 1e-9:
        return
    ux, uy = c / n, s / n
    ax.annotate(
        '', xy=(x + length * ux, y + length * uy), xytext=(x, y),
        arrowprops=dict(arrowstyle='-|>', color=color, lw=linewidth,
                        mutation_scale=mutation_scale, alpha=alpha),
        zorder=zorder
    )

