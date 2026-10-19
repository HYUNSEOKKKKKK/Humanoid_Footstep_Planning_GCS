import numpy as np

def rect_A_b_from_vertices(v3x4):
    xs, ys = v3x4[0, :], v3x4[1, :]
    hx, hy = float(np.max(np.abs(xs))), float(np.max(np.abs(ys)))
    A = np.array([[ 1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    b = np.array([hx, hx, hy, hy], dtype=float)
    return A, b