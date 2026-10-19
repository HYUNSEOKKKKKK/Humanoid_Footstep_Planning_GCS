from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np


@dataclass(frozen=True)
class PiecewiseLinearTrig:
    phi: np.ndarray        # (L+1,)
    g_sin: np.ndarray      # (L,)
    h_sin: np.ndarray      # (L,)
    g_cos: np.ndarray      # (L,)
    h_cos: np.ndarray      # (L,)


def build_piecewise_linear_sin_cos(num_segments: int,
                                  theta_min: float = -math.pi,
                                  theta_max: float = math.pi) -> PiecewiseLinearTrig:
    """
    Build chord-based piecewise-linear approximations for sin(theta) and cos(theta).

    Paper constraint form (7)-(8):
        S_{l,j} => [phi_l <= theta_j <= phi_{l+1},  s_j = g_l * theta_j + h_l]
        C_{l,j} => [phi_l <= theta_j <= phi_{l+1},  c_j = g_l * theta_j + h_l]

    NOTE (paper): The choice of g,h matters; approximation should not overestimate reachable space.
    Here we use simple chords; for more conservatism, increase `num_segments` and/or slightly
    shrink the reachability radii d1,d2 in config.
    """
    if num_segments < 2:
        raise ValueError("num_segments must be >= 2")

    phi = np.linspace(theta_min, theta_max, num_segments + 1)
    g_sin = np.zeros(num_segments)
    h_sin = np.zeros(num_segments)
    g_cos = np.zeros(num_segments)
    h_cos = np.zeros(num_segments)

    for l in range(num_segments):
        a = float(phi[l])
        b = float(phi[l + 1])
        sa, sb = math.sin(a), math.sin(b)
        ca, cb = math.cos(a), math.cos(b)

        g_s = (sb - sa) / (b - a)
        h_s = sa - g_s * a
        g_c = (cb - ca) / (b - a)
        h_c = ca - g_c * a

        g_sin[l] = g_s
        h_sin[l] = h_s
        g_cos[l] = g_c
        h_cos[l] = h_c

    return PiecewiseLinearTrig(phi=phi, g_sin=g_sin, h_sin=h_sin, g_cos=g_cos, h_cos=h_cos)
