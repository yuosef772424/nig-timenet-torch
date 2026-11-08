"""Exact running-maximum moments (Spitzer) against brute-force Monte Carlo."""

import numpy as np
import pytest

from nig_timenet_torch.data_io.walk_max import max_mgf, max_moments, normal_cdf

N_MC = 200_000


def _mc(mu, s, n, seed=0):
    steps = mu + s * np.random.default_rng(seed).standard_normal((N_MC, n))
    return np.maximum(np.cumsum(steps, axis=1).max(axis=1), 0.0)


@pytest.mark.parametrize("mu,s,n", [(0.0, 1.0, 96), (0.05, 1.0, 96), (-0.1, 0.5, 10), (0.3, 1.0, 1), (0.0, 2.0, 2)])
def test_mean_and_sd_match_monte_carlo(mu, s, n):
    m = _mc(mu, s, n)
    mean, var = max_moments(mu, s, n)
    se = m.std() / np.sqrt(N_MC)
    assert abs(mean[0] - m.mean()) < 4 * se
    assert np.sqrt(var[0]) == pytest.approx(m.std(), rel=0.01)


@pytest.mark.parametrize("theta", [1.0, 2.0, -1.0, -2.0])
def test_mgf_matches_monte_carlo(theta):
    mu, s, n = 0.0005, 0.003, 96
    e = np.exp(theta * _mc(mu, s, n, seed=1))
    assert max_mgf(mu, s, n, theta)[0] == pytest.approx(e.mean(), abs=4 * e.std() / np.sqrt(N_MC))


def test_mgf_derivative_at_zero_is_the_mean():
    mu, s, n, h = np.array([0.0, 0.01, -0.02]), np.array([0.01, 0.02, 0.01]), 48, 1e-4
    slope = (max_mgf(mu, s, n, h) - max_mgf(mu, s, n, -h)) / (2 * h)
    np.testing.assert_allclose(slope, max_moments(mu, s, n)[0], rtol=1e-5)


def test_one_step_is_the_positive_part_of_a_normal():
    mean, var = max_moments(0.0, 1.0, 1)
    assert mean[0] == pytest.approx(1 / np.sqrt(2 * np.pi)) and var[0] == pytest.approx(0.5 - 1 / (2 * np.pi))


def test_normal_cdf_tails_and_inputs():
    assert normal_cdf(0.0) == 0.5 and normal_cdf(-40.0) == 0.0 and 0 < normal_cdf(-30.0) < 1e-190
    with pytest.raises(ValueError):
        max_moments(0.0, 0.0, 5)
    with pytest.raises(ValueError):
        max_moments(0.0, 1.0, 0)
