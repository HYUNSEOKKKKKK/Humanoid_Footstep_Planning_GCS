"""Generate new terrain JSONs using a shipped CVAE, without retraining it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from terrain_generation.main import CVAE, Step6, generate_sequence, seed_all, vae_seq_to_terrain_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "terrain_generation/checkpoints/cvae_mapgen.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/terrains")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--stones", type=int, default=6)
    parser.add_argument("--alpha", type=float, choices=(0.0, 0.5, 1.0), default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.stones < 2:
        parser.error("--samples must be positive and --stones must be at least 2")
    if args.output_dir.resolve().is_relative_to((ROOT / "data").resolve()):
        parser.error("data/ contains preserved reference inputs; select an output directory outside data/")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    targets = [args.output_dir / f"run_{i:03d}.json" for i in range(args.samples)]
    if not args.overwrite and any(p.exists() for p in targets + [args.output_dir / "manifest.json"]):
        parser.error("output already exists; choose another directory or pass --overwrite")

    model = CVAE(cond_dim=12, x_dim=6, z_dim=8)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial = (Step6(0, 0, 0, 0, 0, 0), Step6(0.30, 0, 0, 0, 0, 0))
    with torch.no_grad():
        for i, path in enumerate(targets):
            seed_all(args.seed + i)
            sequence = generate_sequence(model, *initial, steps=args.stones, a=args.alpha, device=args.device)
            vae_seq_to_terrain_json(sequence, center0=(0, 2.5, 0), size0=(0.2, 0.4),
                                    yaw0_deg=0.0, terrain_key="terrain", json_path=path, use_from=2)
            print(f"Saved {path}")
    manifest = {
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "base_seed": args.seed, "alpha": args.alpha, "stones": args.stones,
        "samples": args.samples, "device": args.device, "torch_version": torch.__version__,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in targets},
        "note": "New samples. The preserved paper benchmark is in data/terrains/alpha/.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
