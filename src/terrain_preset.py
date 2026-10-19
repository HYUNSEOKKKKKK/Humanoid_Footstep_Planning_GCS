# src/terrain_presets.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import numpy as np

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "terrain.json"

DEFAULT_TERRAINS: Dict[str, Dict[str, Any]] = {
    "default_flat": {
        "stones": [
            {"name": "initial", "center": [2.5, 0.5, 0.0], "width": 5.0, "height": 1.0},
            {"name": "goal",    "center": [2.5, 1.5, 0.0], "width": 5.0, "height": 1.0},
        ],
        "copy_counts": [5, 5],
    }
}

def _load_json(path: Optional[str | Path]) -> Optional[Dict[str, Any]]:
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def list_terrains(path: Optional[str | Path] = None) -> List[str]:
    data = _load_json(path) or DEFAULT_TERRAINS
    return list(data.keys())

def load_terrain(name: str, path: Optional[str | Path] = None) -> Dict[str, Any]:
    data = _load_json(path) or DEFAULT_TERRAINS

    keys = list(data.keys())
    low_map = {k.lower(): k for k in keys}
    name_l = name.lower()
    match = None
    if name_l in low_map:
        match = low_map[name_l]
    else:
        pref = [k for k in keys if k.lower().startswith(name_l)]
        if len(pref) == 1:
            match = pref[0]
        elif len(pref) > 1:
            raise ValueError(f"Ambiguous terrain name '{name}'. Candidates: {pref}")
    if match is None:
        raise KeyError(f"Terrain '{name}' not found. Available: {keys}")

    item = data[match]
    stones = item["stones"]
    # 기본 검증/정규화
    out_stones: List[Dict[str, Any]] = []
    for i, st in enumerate(stones):
        name_i = st.get("name", f"stone_{i+1}")
        center = np.asarray(st["center"], dtype=float).reshape(-1).tolist()
        if len(center) == 2:
            center = [center[0], center[1], 0.0]
        width  = float(st["width"])
        height = float(st["height"])
        out_stones.append({"name": name_i, "center": center, "width": width, "height": height})

    copy_counts = item.get("copy_counts", None)
    if copy_counts is not None:
        copy_counts = [int(x) for x in copy_counts]
        if len(copy_counts) != len(out_stones):
            raise ValueError(f"copy_counts length ({len(copy_counts)}) must match stones ({len(out_stones)})")

    return {"name": match, "stones": out_stones, "copy_counts": copy_counts}


def load_terrain_yaw(name: str, path: Optional[str | Path] = None) -> Dict[str, Any]:
    data = _load_json(path) or DEFAULT_TERRAINS

    keys = list(data.keys())
    low_map = {k.lower(): k for k in keys}
    name_l = name.lower()
    match = None
    if name_l in low_map:
        match = low_map[name_l]
    else:
        pref = [k for k in keys if k.lower().startswith(name_l)]
        if len(pref) == 1:
            match = pref[0]
        elif len(pref) > 1:
            raise ValueError(f"Ambiguous terrain name '{name}'. Candidates: {pref}")
    if match is None:
        raise KeyError(f"Terrain '{name}' not found. Available: {keys}")

    item = data[match]
    stones = item["stones"]

    out_stones: List[Dict[str, Any]] = []
    for i, st in enumerate(stones):
        name_i = st.get("name", f"stone_{i+1}")

        center = np.asarray(st["center"], dtype=float).reshape(-1).tolist()
        if len(center) == 2:
            center = [center[0], center[1], 0.0]

        width  = float(st["width"])
        height = float(st["height"])

        # --- yaw 처리: yaw_deg 우선, 없으면 yaw_rad → deg 변환 ---
        yaw_deg = st.get("yaw_deg", None)
        if yaw_deg is None and "yaw_rad" in st:
            yaw_deg = float(np.degrees(st["yaw_rad"]))

        stone_out = {
            "name":   name_i,
            "center": center,
            "width":  width,
            "height": height,
        }
        if yaw_deg is not None:
            stone_out["yaw_deg"] = float(yaw_deg)  # 일관되게 deg로 유지

        out_stones.append(stone_out)

    copy_counts = item.get("copy_counts", None)
    if copy_counts is not None:
        copy_counts = [int(x) for x in copy_counts]
        if len(copy_counts) != len(out_stones):
            raise ValueError(f"copy_counts length ({len(copy_counts)}) must match stones ({len(out_stones)})")

    return {"name": match, "stones": out_stones, "copy_counts": copy_counts}
