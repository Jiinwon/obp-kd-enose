"""Parse GNINA outputs into a prior table."""
import json
from pathlib import Path

def make_prior_table(input_dir: str, output_path: str) -> None:
    """Placeholder parser that writes an empty prior table.

    Args:
        input_dir: Directory containing GNINA JSON outputs.
        output_path: Path to write the combined prior JSON.
    """
    table = {}
    Path(output_path).write_text(json.dumps(table))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Directory with GNINA outputs")
    parser.add_argument("output", help="Path to save prior table")
    args = parser.parse_args()
    make_prior_table(args.input, args.output)
