"""NIG predictive distribution in numpy (T4): the predictive std and central interval of a Normal-Inverse-Gamma head,
computed from the saved ``nu / alpha / beta`` columns of a PredictionTable (the model's own ``_epistemic`` /
``_aleatoric`` outputs are capped at ``unc_max`` and so cannot be used for a coverage check).

Student-t marginal (Amini et al.): df = 2 alpha, scale^2 = beta (1 + nu) / (nu alpha), variance = beta (1 + nu) /
(nu (alpha - 1)) = aleatoric^2 + epistemic^2 (the same alpha - 1 floor as the model layer, NIG_ALPHA_DEN_MIN).
"""

from statistics import NormalDist

import numpy as np

from nig_timenet_torch.core.constants import NIG_ALPHA_DEN_MIN

__all__ = ["student_t_quantile", "nig_predictive_std", "nig_interval_halfwidth"]


def student_t_quantile(p: float, df: np.ndarray) -> np.ndarray:
    """Quantile of Student's t (Abramowitz-Stegun 26.7.5, four terms in 1/df). Accurate to ~2e-3 for df >= 4 and
    p in [0.5, 0.99]; the NIG head guarantees df = 2 alpha >= 4 by default (alpha_min = 2). Raises below df 3."""
    df = np.asarray(df, dtype=np.float64)
    if not 0.5 < p < 1.0:
        raise ValueError("p must be in (0.5, 1)")
    if (df < 3.0).any():
        raise ValueError("student_t_quantile needs df >= 3 (alpha >= 1.5)")
    z = NormalDist().inv_cdf(p)
    g1 = (z**3 + z) / 4
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96
    g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384
    g4 = (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / 92160
    return z + g1 / df + g2 / df**2 + g3 / df**3 + g4 / df**4


def nig_predictive_std(nu: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """sqrt(beta (1 + nu) / (nu max(alpha - 1, NIG_ALPHA_DEN_MIN))): the total predictive std, in target units."""
    nu, alpha, beta = (np.asarray(a, dtype=np.float64) for a in (nu, alpha, beta))
    return np.sqrt(beta * (1.0 + nu) / (nu * np.maximum(alpha - 1.0, NIG_ALPHA_DEN_MIN)))


def nig_interval_halfwidth(nu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, level: float) -> np.ndarray:
    """Half width of the central ``level`` (e.g. 0.8) predictive interval: t_{(1+level)/2}(2 alpha) * scale."""
    nu, alpha, beta = (np.asarray(a, dtype=np.float64) for a in (nu, alpha, beta))
    scale = np.sqrt(beta * (1.0 + nu) / (nu * alpha))
    return student_t_quantile((1.0 + level) / 2.0, 2.0 * alpha) * scale
