"""Create a simple experiment report from metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def make_report(metrics_csv: str | Path, output: str | Path, experiment_name: str) -> None:
    metrics_path = Path(metrics_csv)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")
    df = pd.read_csv(metrics_path)
    val_df = df[df["split"] == "val"].sort_values("epoch")
    last_row = val_df.iloc[-1] if not val_df.empty else None
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"# Report: {experiment_name}\n\n")
        if last_row is None:
            f.write("No validation metrics available.\n")
        else:
            f.write(
                f"Final epoch: {int(last_row['epoch'])}\n\n"
                f"* Val Loss: {last_row['loss']:.4f}\n"
                f"* Val Accuracy: {last_row['acc']:.4f}\n"
                f"* Val CE: {last_row['ce']:.4f}\n"
                f"* Val KD: {last_row['kd']:.4f}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown report from metrics")
    parser.add_argument("--metrics", required=True, help="Path to metrics.csv")
    parser.add_argument("--output", required=True, help="Output markdown path")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    args = parser.parse_args()
    make_report(args.metrics, args.output, args.experiment)


if __name__ == "__main__":  # pragma: no cover
    main()
