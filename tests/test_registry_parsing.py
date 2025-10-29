from pathlib import Path
import tempfile
import textwrap

from src.datasets.base_enose import load_registry


def test_load_registry_minimal():
    content = textwrap.dedent("""    toy:
      root: "data/toy"
      files: {pattern: "*.csv"}
      mapping:
        time: "time"
        sensor_cols: ["s0","s1"]
        gas: "gas"
        conc: "ppm"
        temp: null
        humid: null
        stage: null
        batch_id: null
        device_id: 0
        session_id: "file_id"
        sampling_rate: 10.0
      splits:
        train: {}
        val: {}
        test: {}
    """)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "reg.yaml"
        p.write_text(content, encoding="utf-8")
        reg = load_registry(p)
        assert "toy" in reg
        assert reg["toy"]["mapping"]["sensor_cols"] == ["s0","s1"]
