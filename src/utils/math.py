from __future__ import annotations
import numpy as np

def normalize_cos_sin(vec5):
    vec5 = np.array(vec5, dtype=float).copy()
    c, s = vec5[3], vec5[4]
    r = np.hypot(c, s)
    if r == 0:
        vec5[3], vec5[4] = 1.0, 0.0
    else:
        vec5[3], vec5[4] = c / r, s / r
    vec5[3] = np.clip(vec5[3], -1.0, 1.0)
    vec5[4] = np.clip(vec5[4], -1.0, 1.0)
    return vec5


