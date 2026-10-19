"""Train a new CVAE; the reference checkpoints are never overwritten."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from terrain_generation.main import CVAE, SyntheticStepDeltaDataset, seed_all, train_cvae

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/checkpoints/cvae_mapgen.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to((ROOT / "terrain_generation/checkpoints").resolve()) or args.output.resolve().is_relative_to((ROOT / "data").resolve()):
        parser.error("choose an output outside the reference data/checkpoint directories")
    if args.output.exists() and not args.overwrite:
        parser.error("output already exists; select another path or pass --overwrite")
    if args.epochs < 1 or args.samples < 10 or args.batch_size < 1:
        parser.error("epochs/batch-size must be positive and samples must be at least 10")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    seed_all(args.seed)
    dataset = SyntheticStepDeltaDataset(length=args.samples, seed=args.seed)
    n_train = int(0.9 * len(dataset))
    train, valid = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    model = CVAE(cond_dim=12, x_dim=6, z_dim=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train_cvae(model, DataLoader(train, batch_size=args.batch_size, shuffle=True),
               DataLoader(valid, batch_size=args.batch_size), epochs=args.epochs,
               lr=2e-4, device=args.device, ckpt_path=str(args.output))
    print(f"Saved new checkpoint: {args.output}")


if __name__ == "__main__":
    main()
