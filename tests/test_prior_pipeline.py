from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.prior_pipeline import build_label_to_prior, load_prior_table, make_prior_table


def test_prior_generation(tmp_path):
    logs_dir = tmp_path / "docking" / "logs"
    logs_dir.mkdir(parents=True)
    log_path = logs_dir / "sample.csv"
    df = pd.DataFrame(
        {
            "ligand": ["ETHANOL", "ETHANOL", "METHANE"],
            "obp": ["OBP1", "OBP1", "OBP2"],
            "affinity": [-5.0, -6.0, -3.0],
            "rmsd": [1.2, 1.0, 1.5],
        }
    )
    df.to_csv(log_path, index=False)
    cfg_path = tmp_path / "prior.yaml"
    cfg_path.write_text(
        """
obp_list: configs/prior/obp_list.yaml
voc_map: configs/prior/voc_map.yaml
input:
  glob: "docking/logs/**/*.csv"
  format: auto
  required_cols: [obp, ligand, affinity]
metrics:
  order: [min_affinity, p25_affinity, median_affinity, p75_affinity, mean_affinity, std_affinity, best_rmsd, top3_mean_affinity]
  flip_sign: [affinity]
scaling:
  method: zscore
  eps: 1.0e-6
  fillna: mean
defaults:
  fill: mean
output:
  table_csv: "{table}"
  qc_json: "{qc}"
""".format(table=tmp_path / "prior.csv", qc=tmp_path / "qc.json"),
        encoding="utf-8",
    )
    make_prior_table(cfg_path)
    table = load_prior_table(tmp_path / "prior.csv")
    assert table.prior_dim == 56
    vectors = build_label_to_prior(["ETHANOL", "UNKNOWN"], table, "configs/prior/voc_map.yaml", default="mean")
    assert vectors.shape == (2, table.prior_dim)
    assert not np.isnan(vectors).any()
