from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import numpy as np

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiments.json"

def _to_rad(deg: float) -> float:
    import numpy as _np
    return float(_np.deg2rad(deg))

def list_experiments(path: Optional[str | Path] = None) -> List[str]:
    p = Path(path) if path else _DEFAULT_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.keys())

def load_experiment(name: str,
                    path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    experiments.json에서 name에 해당하는 프리셋 로드.
    - 이름은 대소문자 무시, prefix 매칭 지원
    반환: numpy 배열/라디안으로 정리된 dict
    """
    p = Path(path) if path else _DEFAULT_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # case-insensitive + prefix 매칭
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
            raise ValueError(f"Ambiguous experiment name '{name}'. Candidates: {pref}")
    if match is None:
        raise KeyError(f"Experiment '{name}' not found. Available: {keys}")

    item = data[match]

    def arr(key): return np.array(item[key], dtype=float)
    def take_deg(key): return _to_rad(float(item[key]))

    out = dict(
        startpos_L=arr("startpos_L"),
        startpos_R=arr("startpos_R"),
        targetpos_L=arr("targetpos_L"),
        targetpos_R=arr("targetpos_R"),
        startyaw_L=take_deg("startyaw_deg_L"),
        startyaw_R=take_deg("startyaw_deg_R"),
        targetyaw_L=take_deg("targetyaw_deg_L"),
        targetyaw_R=take_deg("targetyaw_deg_R"),
        name=match,
    )
    return out
