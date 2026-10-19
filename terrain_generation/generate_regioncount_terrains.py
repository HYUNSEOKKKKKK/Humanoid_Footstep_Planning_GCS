"""Region-count (N=3,4,5,6 stones) terrain generator for the copy_counts scalability study.

Trains the CVAE exactly once with a fixed, recorded seed and reuses that single
checkpoint to sample 30 terrains per stone count N in {3,4,5,6} (stone count is
controlled purely by how many autoregressive steps are generated after training,
so one trained model can serve every N without confounding "which trained model"
with "how many regions").

Output: vae_terrains_regioncount/N{3,4,5,6}/run_{000..029}.json, plus a manifest.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import (  # noqa: E402
    CVAE,
    Step6,
    SyntheticStepDeltaDataset,
    generate_sequence,
    seed_all,
    train_cvae,
    vae_seq_to_terrain_json,
)

DATA_LEN = 40000
SEED = 123
BATCH_SIZE = 512
EPOCHS = 50
LR = 2e-4
COND_DIM = 12
X_DIM = 6
Z_DIM = 8

YAW0_DEG = 0.0
START_CENTER = (0.0, 2.5, 0.0)
START_SIZE = (0.2, 0.4)

SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_PATH = SCRIPT_DIR / "checkpoints/cvae_mapgen_regioncount.pt"
OUT_ROOT = SCRIPT_DIR.parent / "outputs/region_count"
REGION_COUNTS = (3, 4, 5)
NUM_SAMPLES = 100


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate region-count samples using the preserved CVAE; no training.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR.parent / "outputs/region_count")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--region-counts", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or min(args.region_counts) < 2:
        parser.error("samples must be positive and region counts at least 2")
    if args.output_dir.resolve().is_relative_to((SCRIPT_DIR.parent / "data").resolve()):
        parser.error("select an output outside preserved data/")
    checkpoint = SCRIPT_DIR / "checkpoints/cvae_mapgen_regioncount.pt"
    model = CVAE(cond_dim=12, x_dim=6, z_dim=8)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    model.eval()
    targets = [args.output_dir / f"N{n}" / f"run_{i:03d}.json" for n in args.region_counts for i in range(args.samples)]
    if not args.overwrite and any(p.exists() for p in targets + [args.output_dir / "manifest.json"]):
        parser.error("output already exists; choose another directory or pass --overwrite")
    initial = (Step6(0, 0, 0, 0, 0, 0), Step6(0.30, 0, 0, 0, 0, 0))
    with torch.no_grad():
        for n in args.region_counts:
            for i in range(args.samples):
                seed_all(args.seed + i)
                path = args.output_dir / f"N{n}" / f"run_{i:03d}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                seq = generate_sequence(model, *initial, steps=n, a=0.0, device="cpu")
                vae_seq_to_terrain_json(seq, center0=(0, 2.5, 0), size0=(0.2, 0.4), yaw0_deg=0, terrain_key="terrain", json_path=path, use_from=2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps({"checkpoint_sha256": _file_sha256(checkpoint), "base_seed": args.seed, "region_counts": args.region_counts, "samples": args.samples}, indent=2)+"\n")
    print(f"Saved {len(targets)} terrains to {args.output_dir}")


if __name__ == "__main__":
    main()
