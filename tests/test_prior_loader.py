import json

from src.features.prior_loader import load_priors


def test_load_priors(tmp_path):
    p = tmp_path / "prior.json"
    p.write_text(json.dumps({"class": [1.0, 2.0, 3.0]}))
    matrix, classes = load_priors(p)
    assert classes == ["class"]
    # matrix can be numpy array or torch tensor
    assert getattr(matrix, "shape", None) == (1, 3)
