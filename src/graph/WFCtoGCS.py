"""
WFCtoGCS — bridge to turn a compact/pruned graph (Gk) from Wave Function Collapse (WFC)
into the "stones" structure that your GCS pipeline already consumes (same shape as
load_terrain(...)["stones"]).

Usage (inside your main):

    from src.graph.WFCtoGCS import build_terrain_from_Gk, wfc_to_terrain_stones

    # ... you already have grid, Gk, node_info, and optionally start/goal ...
    T = build_terrain_from_Gk(
        grid, Gk, node_info,
        tile_size=1.0, y_up=True,
        floor_z={"floor0": 0.0, "floor1": 0.2},
        include_stairs=False,
        stair_z_mode="mid",
        add_initial_goal=True,
        start=start, goal=goal,   # if you have them
        start_is_grid=True,       # set False if start/goal are world coords already
        init_goal_size=0.6,
        name="from_graph"
    )
    terrain_stones = T["stones"]

Notes
-----
- No JSON required. This module just returns a Python dict with keys {name, stones, copy_counts}.
- Coordinates: plotting uses image-style y↓, whereas optimization usually uses y↑.
  Set y_up=True to get standard math/world coordinates (recommended for GCS).
- Stairs: include/exclude per your pipeline. If included, represented as 1×1 tiles.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional


def build_terrain_from_WFC_outputs(
    *,
    grid, Gk, node_info, start, goal,
    tile_size: float = 1.0,
    y_up: bool = True,
    floor_z: dict = {"floor0": 0.0, "floor1": 0.2},
    include_stairs: bool = False,
    stair_z_mode: str = "mid",
    add_initial_goal: bool = True,
    start_is_grid: bool = True,
    init_goal_size: float = 0.6,
    name: str = "from_graph",
):
    return build_terrain_from_Gk(
        grid=grid,
        Gk=Gk,
        node_info=node_info,
        tile_size=tile_size,
        y_up=y_up,
        floor_z=floor_z,
        include_stairs=include_stairs,
        stair_z_mode=stair_z_mode,
        add_initial_goal=add_initial_goal,
        start=start, goal=goal,
        start_is_grid=start_is_grid,
        init_goal_size=init_goal_size,
        name=name,
    )


def build_terrain_via_WFC_function(
    *,
    tile_size: float = 1.0,
    y_up: bool = True,
    floor_z: dict = {"floor0": 0.0, "floor1": 0.2},
    include_stairs: bool = False,
    stair_z_mode: str = "mid",
    add_initial_goal: bool = True,
    start_is_grid: bool = True,
    init_goal_size: float = 0.6,
    name: str = "from_graph",
    # 아래는 WFC 쪽에서 grid/G/start/goal을 얻는 너의 준비 코드에 맞춰 필요한 인자들 추가 가능
    # 예: seed=None, size=(H,W), ...
    prepared_outputs: dict | None = None,
):
    """
    (A) 이미 wavefunctioncollapse.py에서 만든 출력 dict를 prepared_outputs로 넘기거나,
    (B) 나중에 이 함수 안에서 wavefunctioncollapse의 생성 함수를 직접 호출하도록 확장 가능.
    """
    if prepared_outputs is None:
        raise ValueError("prepared_outputs dict를 넘겨주세요: {'grid','Gk','node_info','start','goal'}")

    return build_terrain_from_WFC_outputs(
        grid=prepared_outputs["grid"],
        Gk=prepared_outputs["Gk"],
        node_info=prepared_outputs["node_info"],
        start=prepared_outputs["start"],
        goal=prepared_outputs["goal"],
        tile_size=tile_size, y_up=y_up, floor_z=floor_z,
        include_stairs=include_stairs, stair_z_mode=stair_z_mode,
        add_initial_goal=add_initial_goal, start_is_grid=start_is_grid,
        init_goal_size=init_goal_size, name=name,
    )


def _grid_to_world_center(r: float, c: float, H: int, *, tile: float = 1.0, y_up: bool = True) -> Tuple[float, float]:
    """Convert a cell *center* at (r, c) in grid index to world (x, y).

    Grid convention: r increases downward, c increases to the right.
    If y_up=True, y increases upward in world coords.
    """
    x = (c + 0.5) * tile
    if y_up:
        y = (H - (r + 0.5)) * tile
    else:
        y = (r + 0.5) * tile
    return float(x), float(y)


def _rect_center_world(r0: int, c0: int, h: int, w: int, H: int, *, tile: float, y_up: bool) -> Tuple[float, float]:
    """World-space center of an axis-aligned rectangle spanning rows [r0, r0+h) and cols [c0, c0+w)."""
    rc = r0 + 0.5 * h
    cc = c0 + 0.5 * w
    return _grid_to_world_center(rc, cc, H, tile=tile, y_up=y_up)


def _stair_z_value(mode: str, floor_z: Dict[str, float]) -> float:
    if mode == "lower":
        return float(floor_z.get("floor0", 0.0))
    if mode == "upper":
        return float(floor_z.get("floor1", 0.0))
    # default: mid
    return 0.5 * (float(floor_z.get("floor0", 0.0)) + float(floor_z.get("floor1", 0.2)))


def build_terrain_from_Gk(
    grid: List[List[str]],
    Gk: Dict[int, List[int]],
    node_info: Dict[int, Dict[str, Any]],
    *,
    tile_size: float = 1.0,
    y_up: bool = True,
    floor_z: Dict[str, float] | None = None,
    include_stairs: bool = False,
    stair_z_mode: str = "mid",  # "mid" | "lower" | "upper"
    add_initial_goal: bool = False,
    start: Optional[Tuple[float, float]] = None,  # grid or world depending on start_is_grid
    goal: Optional[Tuple[float, float]] = None,
    start_is_grid: bool = True,
    init_goal_size: float = 0.6,
    name: str = "from_graph",
) -> Dict[str, Any]:
    """
    Build a terrain preset dict: {"name": name, "stones": [...], "copy_counts": None}
    from a pruned compact graph (Gk) and node_info produced by the WFC pipeline.

    Each rectangle node becomes one rectangular stone (width = w*tile, height = h*tile).
    Stairs can be optionally converted into 1×1 stones.

    start/goal can be injected as small stones for convenience.
    - If start_is_grid=True, (start, goal) are (r, c) in grid index. Otherwise treat as world (x, y).
    """
    if floor_z is None:
        floor_z = {"floor0": 0.0, "floor1": 0.2}

    H = len(grid)
    W = len(grid[0]) if H else 0

    # ensure node set includes neighbors referenced in Gk
    nodes: set[int] = set(Gk.keys())
    for u, nbrs in Gk.items():
        for v in nbrs:
            nodes.add(v)

    stones: List[Dict[str, Any]] = []

    # 1) rectangles → stones
    for nid in sorted(nodes):
        meta = node_info.get(nid)
        if not meta:
            continue
        if meta.get("type") == "rect":
            r0, c0 = int(meta["r0"]), int(meta["c0"])
            h, w   = int(meta["h"]),  int(meta["w"])  # grid cells
            cx, cy = _rect_center_world(r0, c0, h, w, H, tile=tile_size, y_up=y_up)
            floor  = str(meta.get("floor", "floor0"))
            z      = float(floor_z.get(floor, 0.0))
            stones.append({
                "name":   f"rect_{nid}_{floor}",
                "center": [cx, cy, z],
                "width":  float(w * tile_size),
                "height": float(h * tile_size),
            })
        elif meta.get("type") == "stair" and include_stairs:
            r, c = int(meta["r"]), int(meta["c"])  # single cell
            cx, cy = _grid_to_world_center(r, c, H, tile=tile_size, y_up=y_up)
            z = _stair_z_value(stair_z_mode, floor_z)
            stones.append({
                "name":   f"stair_{nid}",
                "center": [cx, cy, z],
                "width":  float(1.0 * tile_size),
                "height": float(1.0 * tile_size),
            })
        else:
            # unknown type or stair excluded
            continue

    # 2) optional initial/goal stones
    if add_initial_goal:
        if start is not None:
            if start_is_grid:
                sx, sy = _grid_to_world_center(start[0], start[1], H, tile=tile_size, y_up=y_up)
            else:
                sx, sy = float(start[0]), float(start[1])
            stones.append({
                "name":   "initial",
                "center": [sx, sy, float(floor_z.get("floor0", 0.0))],
                "width":  float(init_goal_size),
                "height": float(init_goal_size),
            })
        if goal is not None:
            if start_is_grid:
                gx, gy = _grid_to_world_center(goal[0], goal[1], H, tile=tile_size, y_up=y_up)
            else:
                gx, gy = float(goal[0]), float(goal[1])
            stones.append({
                "name":   "goal",
                "center": [gx, gy, float(floor_z.get("floor0", 0.0))],
                "width":  float(init_goal_size),
                "height": float(init_goal_size),
            })

    return {"name": name, "stones": stones, "copy_counts": None}


def wfc_to_terrain_stones(
    grid: List[List[str]],
    Gk: Dict[int, List[int]],
    node_info: Dict[int, Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, Any]]:
    """Lightweight wrapper to directly get just the stones list.

    Example:
        terrain_stones = wfc_to_terrain_stones(grid, Gk, node_info, tile_size=1.0, y_up=True)
    """
    T = build_terrain_from_Gk(grid, Gk, node_info, **kwargs)
    return T["stones"]


__all__ = [
    "build_terrain_from_Gk",
    "wfc_to_terrain_stones",
]
