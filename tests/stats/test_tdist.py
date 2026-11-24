"""stats/tdist.py: the Student-t upper tail against scipy (the dev-only reference), edge cases and errors."""

import math

import pytest
from scipy import stats as sps

from nig_timenet_torch.stats.tdist import t_sf


@pytest.mark.parametrize("df", [1.0, 1.5, 3.0, 20.0, 400.0])
@pytest.mark.parametrize("t", [-8.0, -3.1, -0.4, 0.0, 0.7, 2.5, 8.0, 40.0])
def test_t_sf_matches_scipy(t, df):
    assert t_sf(t, df) == pytest.approx(float(sps.t.sf(t, df)), abs=1e-11, rel=1e-9)


def test_t_sf_edges():
    assert t_sf(0.0, 5) == 0.5
    assert t_sf(float("inf"), 5) == 0.0 and t_sf(float("-inf"), 5) == 1.0
    assert math.isnan(t_sf(float("nan"), 5))
    assert t_sf(1.3, 7) + t_sf(-1.3, 7) == pytest.approx(1.0, abs=1e-14)
    with pytest.raises(ValueError, match="df"):
        t_sf(1.0, 0)
