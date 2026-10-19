from __future__ import annotations
"""Small helpers wrapping MOSEK Fusion primitives.
Keep MOSEK dependency isolated to the solver layer.
"""
from typing import Iterable
import numpy as np
from mosek.fusion import Domain, Expr, Matrix, Model


def eq0(M: Model, expr):
    M.constraint(expr, Domain.equalsTo(0.0))

def le0(M: Model, expr):
    M.constraint(expr, Domain.lessThan(0.0))

def ge0(M: Model, expr):
    M.constraint(expr, Domain.greaterThan(0.0))

def cone2(M: Model, t, *vec):
    """\|\|vec\|\|_2 <= t  → [t, vec...] ∈ QCone"""
    M.constraint(Expr.vstack(t, *vec), Domain.inQCone())

def repeat_scalar(scalar_expr, n: int):
    """Return column vector [s, s, ..., s]^T with n copies."""
    return Expr.mul(Matrix.dense(np.ones((n, 1))), scalar_expr)

def sum_expr(exprs: Iterable):
    s = None
    for e in exprs:
        s = e if s is None else Expr.add(s, e)
    return s if s is not None else Expr.constTerm(0.0)

def sum_rows_col(var2d, rows, col):
    """Sum var2d[rows, col] where rows are 0-based indices."""
    rows = list(map(int, rows))
    if not rows:
        return Expr.constTerm(0.0)
    stack = Expr.vstack(*[var2d.index(r, int(col)) for r in rows])
    return Expr.sum(stack)

def sum_pick_1d(var1d, idx_list):
    """Sum var1d[idx_list] for 0-based indices."""
    idx = np.asarray(idx_list, dtype=np.int32)
    if idx.size == 0:
        return Expr.constTerm(0.0)
    return Expr.sum(var1d.pick(idx))