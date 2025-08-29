import numpy as np

from src.features.preprocessing import warmup_cut


def test_warmup_cut():
    data = np.arange(10)
    out = warmup_cut(data, n=2)
    assert len(out) == 8
    assert out[0] == 2
