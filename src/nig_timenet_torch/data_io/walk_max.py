"""PURPOSE: Exact moments of the running maximum M_n = max(0, S_1..S_n) of a Gaussian random walk with iid N(mu, s^2)
    steps (Spitzer's identity), vectorised over rows: mean, variance and E[exp(theta M_n)].
TAGS: random walk, running maximum, spitzer identity, exact moments, moment generating function, oracle, synthetic
PITFALLS: the maximum INCLUDES S_0 = 0 (a bar's high includes its open); the steps must be iid over the n steps (the E1
    generator keeps drift and sd constant over a row's horizon for this reason); the minimum is minus the maximum of
    the walk with drift -mu; the MGF series costs O(n^2) per row.

Spitzer's identity: sum_n t^n E[exp(i l M_n)] = exp(sum_k t^k / k E[exp(i l S_k^+)]). Expanding it in l gives
``E[M_n] = sum_{k<=n} E[S_k^+] / k`` and ``E[M_n^2] = sum_{k<=n} E[(S_k^+)^2] / k + sum_{j+l<=n} c_j c_l`` with
``c_k = E[S_k^+] / k``; with real ``theta`` it gives ``E[exp(theta M_n)] = sum_{m<=n} f_m``, where ``f`` are the power
series coefficients of ``exp(sum_k t^k / k (E[exp(theta S_k^+)] - 1))``
(recurrence ``f_m = sum_{k<=m} d_k f_{m-k} / m``).
``S_k ~ N(k mu, k s^2)`` makes every term closed form. Pure numpy (the normal cdf is ``math.erfc``).
"""

import math

import numpy as np

__all__ = ["normal_cdf", "max_moments", "max_mgf"]

_ERFC = np.frompyfunc(math.erfc, 1, 1)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def normal_cdf(z) -> np.ndarray:
    """Phi(z), accurate in both tails (``0.5 erfc(-z / sqrt 2)``)."""
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * np.asarray(_ERFC(-z / math.sqrt(2.0)), dtype=np.float64)


def _steps(mu, s, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, s = np.broadcast_arrays(np.atleast_1d(np.asarray(mu, np.float64)), np.atleast_1d(np.asarray(s, np.float64)))
    if mu.ndim != 1 or (s <= 0).any() or not (np.isfinite(mu).all() and np.isfinite(s).all()):
        raise ValueError("mu and s must be finite 1-D arrays (or scalars) with s > 0")
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    k = np.arange(1, n + 1, dtype=np.float64)
    return mu[:, None] * k, s[:, None] * np.sqrt(k), k


def max_moments(mu, s, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(E[M_n], Var[M_n]) per row for steps N(mu, s^2); ``mu`` / ``s`` are scalars or 1-D arrays (broadcast)."""
    mk, sk, k = _steps(mu, s, n)
    z = mk / sk
    cdf, pdf = normal_cdf(z), np.exp(-0.5 * z * z) * _INV_SQRT_2PI
    pos1 = mk * cdf + sk * pdf                              # E[S_k^+]
    pos2 = (mk * mk + sk * sk) * cdf + mk * sk * pdf        # E[(S_k^+)^2]
    c = pos1 / k
    mean = c.sum(axis=1)
    cross = (c[:, : n - 1] * np.cumsum(c, axis=1)[:, n - 2::-1]).sum(axis=1) if n > 1 else np.zeros_like(mean)
    second = (pos2 / k).sum(axis=1) + cross
    return mean, np.maximum(second - mean * mean, 0.0)


def max_mgf(mu, s, n: int, theta: float) -> np.ndarray:
    """E[exp(theta M_n)] per row for steps N(mu, s^2) (any real theta; M_n >= 0 is bounded below)."""
    mk, sk, _ = _steps(mu, s, n)
    var = sk * sk
    psi = normal_cdf(-mk / sk) + np.exp(theta * mk + 0.5 * theta * theta * var) * normal_cdf((mk + theta * var) / sk)
    d = psi - 1.0
    f = np.zeros((d.shape[0], n + 1))
    f[:, 0] = 1.0
    for m in range(1, n + 1):
        f[:, m] = (d[:, :m] * f[:, m - 1::-1]).sum(axis=1) / m
    return f.sum(axis=1)
