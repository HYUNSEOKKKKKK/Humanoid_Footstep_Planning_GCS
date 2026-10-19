from __future__ import annotations
import time

def tic():
    return time.perf_counter()

def toc(t0, label: str | None = None):
    dt = time.perf_counter() - t0
    if label:
        print(f"{label}: {dt:.3f} sec")
    return dt
