"""Checks for the portable input/plan/generation workflow; no solver license required."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReproductionChecks(unittest.TestCase):
    def run_python(self, *args, ok=True):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def test_preserved_artifacts(self):
        self.run_python("-m", "experiments.verify_artifacts")

    def test_benchmark_coverage_and_schema(self):
        for alpha in ("0.0", "0.5", "1.0"):
            paths = sorted((ROOT / "data/terrains/alpha" / alpha).glob("run_*.json"))
            self.assertEqual([p.name for p in paths], [f"run_{i:03d}.json" for i in range(100)])
            for path in paths:
                stones = json.loads(path.read_text())["terrain"]["stones"]
                self.assertEqual(len(stones), 4, str(path))
                self.assertTrue(all(len(s["center"]) == 3 and s["width"] > 0 and s["height"] > 0 for s in stones))

    def test_sampling_is_repeatable_and_preserves_checkpoint(self):
        checkpoint = ROOT / "terrain_generation/checkpoints/cvae_mapgen.pt"
        original = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a", "b"):
                self.run_python("terrain_generation/generate.py", "--samples", 1, "--stones", 4,
                                "--seed", 123, "--output-dir", root / name)
            self.assertEqual((root / "a/run_000.json").read_bytes(), (root / "b/run_000.json").read_bytes())
            before = (root / "a/run_000.json").read_bytes()
            self.run_python("terrain_generation/generate.py", "--samples", 1, "--output-dir", root / "a", ok=False)
            self.assertEqual(before, (root / "a/run_000.json").read_bytes())
        self.assertEqual(original, hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    def test_reference_outputs_cannot_be_overwritten_by_sampling(self):
        self.run_python("terrain_generation/generate.py", "--output-dir", ROOT / "data/terrains/alpha/0.0", "--overwrite", ok=False)
        self.run_python("terrain_generation/train.py", "--output", ROOT / "terrain_generation/checkpoints/cvae_mapgen.pt", "--overwrite", ok=False)

    def test_example_plan_loads_in_toro_model(self):
        self.run_python("visualization/replay_plan.py", "--plan", "examples/gcs_plan.json", "--check")

    def test_robot_udp_is_opt_in(self):
        import os
        from unittest.mock import patch
        from src.utils.udp import send_udp
        with patch.dict(os.environ, {}, clear=True), patch("src.utils.udp.socket.socket") as socket:
            self.assertEqual(send_udp(b"test", "127.0.0.1", 30322), 0)
            socket.assert_not_called()


if __name__ == "__main__":
    unittest.main()
