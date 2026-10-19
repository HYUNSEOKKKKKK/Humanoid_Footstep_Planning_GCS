"""Verify preserved terrain, result, checkpoint, and paper files (standard library only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT / "paper/artifact_manifest.json").read_text())
    errors = []
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.is_file():
            errors.append(f"missing: {name}")
        elif path.stat().st_size != expected["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            errors.append(f"changed: {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(manifest['files'])} preserved artifacts.")


if __name__ == "__main__":
    main()
