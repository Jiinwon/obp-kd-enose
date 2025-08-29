import json

from src.features.prior_loader import load_priors


def test_load_priors(tmp_path):
    p = tmp_path / "prior.json"
    p.write_text(json.dumps({"class": [1, 2, 3]}))
    priors = load_priors(p)
    assert "class" in priors
