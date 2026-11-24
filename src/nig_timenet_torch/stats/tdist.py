"""PURPOSE: The Student-t upper tail P(T > t) in pure python (regularised incomplete beta by a continued fraction),
    so p-values need no scipy at run time.
TAGS: student t, survival function, p value, incomplete beta, continued fraction, one sided test
PITFALLS: ``t_sf`` takes scalars (a t statistic and its degrees of freedom), not arrays; accurate to ~1e-12 against
    ``scipy.stats.t.sf`` (the test compares); a non-positive ``df`` raises; the two-sided p-value is ``2 * t_sf(|t|)``.
"""

import math

__all__ = ["t_sf"]

_EPS, _TINY, _MAX_ITER = 1e-14, 1e-300, 400


def _beta_cf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz, Numerical Recipes betacf)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _TINY else _TINY)
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > _TINY else _TINY
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > _TINY else _TINY
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def _beta_inc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with ``df`` degrees of freedom (NaN in, NaN out)."""
    if not df > 0:
        raise ValueError(f"df must be > 0, got {df!r}")
    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * _beta_inc(0.5 * df, 0.5, df / (df + t * t))      # P(T > |t|)
    return tail if t > 0 else 1.0 - tail
