"""stats/nig.py ``nig_uncertainty_bounded``: the TF evaluation's recomputed epistemic / aleatoric, capped."""

import numpy as np

from nig_timenet_torch.core.constants import NIG_ALPHA_DEN_MIN
from nig_timenet_torch.stats.nig import NIG_UNC_MAX, nig_predictive_std, nig_uncertainty_bounded


def test_the_formula_and_the_predictive_std_relation():
    nu, alpha, beta = np.array([0.5, 2.0]), np.array([3.0, 1.5]), np.array([0.04, 0.09])
    epi, ale, clipped = nig_uncertainty_bounded(nu, alpha, beta)
    np.testing.assert_allclose(ale, np.sqrt(beta / (alpha - 1)))
    np.testing.assert_allclose(epi, np.sqrt(beta / (nu * (alpha - 1))))
    assert not clipped.any()
    np.testing.assert_allclose(np.sqrt(epi ** 2 + ale ** 2), np.sqrt(beta / (alpha - 1) * (1 + 1 / nu)))
    np.testing.assert_allclose(nig_predictive_std(nu, alpha, beta), np.sqrt(epi ** 2 + ale ** 2))


def test_the_alpha_floor_the_cap_and_non_finite_values():
    nu, alpha = np.array([1.0, 1.0, 1.0, np.nan]), np.array([1.0, 3.0, 3.0, 3.0])
    beta = np.array([4.0, 1e6, 1.0, 1.0])
    epi, ale, clipped = nig_uncertainty_bounded(nu, alpha, beta)
    assert ale[0] == np.sqrt(4.0 / NIG_ALPHA_DEN_MIN) == NIG_UNC_MAX                           # alpha = 1: the floor
    assert ale[1] == NIG_UNC_MAX
    assert clipped.tolist() == [False, True, False, True]                                      # row 0 lands ON the cap
    assert epi[3] == NIG_UNC_MAX and ale[3] == np.sqrt(1.0 / 2.0)
    epi2, ale2, clipped2 = nig_uncertainty_bounded(nu, alpha, beta, unc_max=None)               # no cap
    assert not clipped2.any() and ale2[1] > NIG_UNC_MAX
