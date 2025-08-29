"""Parse GNINA docking outputs and build class-wise prior vectors.

The script scans docking/outputs/* directories for GNINA JSON logs and
aggregates metrics into a compact prior table suitable for model training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml

FEATURES = [
    "affinity",
    "cnnscore",
    "hbond_count",
    "hydrophobic_contacts",
    "pocket_vol",
    "pocket_hydrophobicity",
    "topk_mean",
]


def parse_gnina_json(path: Path) -> Dict[str, float]:
    """Extract selected metrics from a GNINA JSON log."""
    data = json.loads(path.read_text()) if path.exists() else {}
    return {k: float(data.get(k, 0.0)) for k in FEATURES}


def build_prior(inputs: Path, vocs: List[str], obps: List[str]) -> Dict[str, List[float]]:
    """Aggregate metrics into class-wise feature vectors.

    For each VOC, metrics are averaged over all OBPs.
    """
    table: Dict[str, List[float]] = {}
    for voc in vocs:
        accum = {k: [] for k in FEATURES}
        for obp in obps:
            j = inputs / voc / obp / "result.json"
            metrics = parse_gnina_json(j)
            for k in FEATURES:
                accum[k].append(metrics[k])
        # average across OBPs
        vec = [sum(accum[k]) / len(accum[k]) if accum[k] else 0.0 for k in FEATURES]
        table[voc] = vec
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True, help="Directory with GNINA outputs")
    parser.add_argument("--voc_list", type=Path, required=True, help="YAML containing voc list")
    parser.add_argument("--obp_list", type=Path, required=True, help="YAML containing obp list")
    parser.add_argument("--out", type=Path, required=True, help="Output prior.json path")
    args = parser.parse_args()

    config_vocs = yaml.safe_load(args.voc_list.read_text())
    config_obps = yaml.safe_load(args.obp_list.read_text())
    vocs = config_vocs.get("vocs", [])
    obps = config_obps.get("obps", [])

    table = build_prior(args.inputs, vocs, obps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
