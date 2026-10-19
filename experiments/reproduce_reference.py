"""Recompute Table II and Pump deadline statistics from preserved raw results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reference")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT / "data") or output.is_relative_to(ROOT / "paper"):
        parser.error("select an output outside preserved data/ and paper/")
    if (output / "table_ii.csv").exists() and not args.overwrite:
        parser.error("output already exists; choose another directory or pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((ROOT / "data/reference/gcs/Com10_3_extended/raw_results.csv").open()))
    configs = [("Nominal sparse", "nominal", "25.66"), ("Dense mode-compatible", "edge_dense_mode_compatible", "32.66")]
    table = []
    for label, config, expected in configs:
        values = [float(row["delta_relax_percent"]) for row in rows
                  if row["configuration"] == config and row["success"].lower() in ("true", "1")]
        median = statistics.median(values)
        if f"{median:.2f}" != expected:
            raise RuntimeError(f"Table II mismatch: {config}: {median:.2f} != {expected}")
        table.append({"edge_rule": label, "successful_trials": len(values), "median_gap_percent": median,
                      "paper_rounded_value": expected})
    with (output / "table_ii.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    subprocess.run([sys.executable, "-m", "experiments.pump_deadlines", "summarize",
                    str(ROOT / "data/reference/gcs/Com10_2/pump_planner_cycle_results.csv"),
                    "--output-dir", str(output / "pump_deadlines")], cwd=ROOT, check=True)

    # Preserve all recorded cost pairs, including startup windows. The final
    # paper's exact warmup/selection rule has not been established from the logs.
    pairs = []
    number = r"([-+0-9.eE]+)"
    for window in range(3, 10):
        text = (ROOT / f"data/reference/pump_logs/{window}.txt").read_text()
        relaxed = re.findall(r"\[relaxed\] Cost\s*=\s*" + number, text)
        fixed = re.findall(r"\[fixed\] Cost\s*=\s*" + number, text)
        if len(relaxed) != len(fixed):
            raise RuntimeError(f"Unpaired cost records in {window}.txt")
        for i, (r, f) in enumerate(zip(relaxed, fixed)):
            r, f = float(r), float(f)
            pairs.append({"log_window_label": window, "record_index": i, "C_relax": r, "C_round": f,
                          "gap_percent_raw": 100 * (f - r) / r if r else 0.0})
    with (output / "pump_cost_pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for window in range(3, 10):
        group = [row for row in pairs if row["log_window_label"] == window]
        ax.scatter([row["C_relax"] for row in group], [max(0, row["gap_percent_raw"]) for row in group],
                   s=12, alpha=0.65, label=f"log {window}")
    ax.set(xlabel="Relaxed objective", ylabel="Relaxation gap (%)", title="Preserved Pump logs: all recorded cost pairs")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "pump_cost_pairs.png", dpi=180)
    plt.close(fig)
    (output / "report.json").write_text(json.dumps({"table_ii": table, "pump_cost_records": len(pairs),
        "note": "Table II matches the final paper. Cost-pair plot includes startup records; no exact Figure 8 selection is claimed."}, indent=2) + "\n")
    print("Table II reproduced: 25.66%, 32.66%")
    print(f"Saved reference summaries and {len(pairs)} Pump cost pairs under {output}")


if __name__ == "__main__":
    main()
