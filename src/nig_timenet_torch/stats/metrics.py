"""The single owner of every metric this project computes (C7; BUILD_PLAN.md P21). Pure numpy — no torch.

Shared NaN policy: rows where ANY input is NaN are dropped first (pairwise); a metric that is undefined on what is
left returns ``float("nan")`` instead of raising. Ties are always handled by average ranks.
"""

import math

import numpy as np

__all__ = ["auc", "rank_ic", "pearson", "spearman", "wilson_interval", "pinball_loss", "ece", "ece_bins", "mae",
           "average_ranks", "confidence_error_spearman", "confidence_error_gap", "width_below_share",
           "interval_coverage"]


def _clean(*arrays: np.ndarray) -> list[np.ndarray]:
    arrays = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays]
    n = {a.shape[0] for a in arrays}
    if len(n) != 1:
        raise ValueError(f"length mismatch: {[a.shape[0] for a in arrays]}")
    keep = np.ones(arrays[0].shape[0], dtype=bool)
    for a in arrays:
        keep &= ~np.isnan(a)
    return [a[keep] for a in arrays]


def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks; tied values all get the mean of their ranks. NaN: caller must drop them first."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(x.shape[0], dtype=np.float64)
    boundaries = np.flatnonzero(np.diff(xs) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [x.shape[0]]])
    for s, e in zip(starts, ends, strict=True):
        ranks[order[s:e]] = 0.5 * (s + e - 1) + 1.0
    return ranks


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC (Mann-Whitney U with average ranks: a tied pos/neg pair counts 1/2). NaN if one class is empty."""
    y, s = _clean(y_true, y_score)
    pos = y > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = average_ranks(s)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r. NaN if fewer than 2 rows or either side is constant."""
    a, b = _clean(a, b)
    if a.shape[0] < 2:
        return float("nan")
    da, db = a - a.mean(), b - b.mean()
    den = math.sqrt(float((da * da).sum()) * float((db * db).sum()))
    return float((da * db).sum() / den) if den > 0 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho = Pearson on average ranks (ties averaged). NaN as pearson."""
    a, b = _clean(a, b)
    if a.shape[0] < 2:
        return float("nan")
    return pearson(average_ranks(a), average_ranks(b))


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray | None = None) -> float:
    """Rank IC. With ``groups`` (e.g. timestamps): mean over groups of the within-group Spearman (cross-sectional);
    groups where it is NaN (< 2 rows or constant) are skipped; NaN if no group is defined. Without: plain Spearman."""
    if groups is None:
        return spearman(y_true, y_pred)
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    g = np.asarray(groups).reshape(-1)
    if not (y.shape == p.shape == g.shape):
        raise ValueError("y_true, y_pred and groups must have the same length")
    order = np.argsort(g, kind="stable")  # one sort instead of a boolean mask per group (O(n) vs O(n * groups))
    _, starts = np.unique(g[order], return_index=True)
    ics = [spearman(y[ix], p[ix]) for ix in np.split(order, starts[1:])]
    ics = [v for v in ics if not math.isnan(v)]
    return float(np.mean(ics)) if ics else float("nan")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. (nan, nan) if n == 0."""
    if n <= 0:
        return float("nan"), float("nan")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, n], got {successes}/{n}")
    p = successes / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """mean(max(q (y - p), (q - 1)(y - p))). NaN if no rows."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    y, p = _clean(y_true, y_pred)
    if y.shape[0] == 0:
        return float("nan")
    e = y - p
    return float(np.maximum(quantile * e, (quantile - 1.0) * e).mean())


def ece_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """(counts, mean_label, mean_prob) per equal-width bin on [0, 1]; prob == 1.0 falls in the last bin, probs are
    clipped to [0, 1]; empty bins have NaN means."""
    y, p = _clean(y_true, y_prob)
    p = np.clip(p, 0.0, 1.0)
    idx = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_y = np.bincount(idx, weights=y, minlength=n_bins) / counts
        mean_p = np.bincount(idx, weights=p, minlength=n_bins) / counts
    return counts, mean_y, mean_p


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, sum_b (n_b / N) |mean_label_b - mean_prob_b| over non-empty bins. NaN if no rows."""
    counts, mean_y, mean_p = ece_bins(y_true, y_prob, n_bins)
    total = counts.sum()
    if total == 0:
        return float("nan")
    nz = counts > 0
    return float((counts[nz] / total * np.abs(mean_y[nz] - mean_p[nz])).sum())


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """mean |y - p|. NaN if no rows."""
    y, p = _clean(y_true, y_pred)
    return float(np.abs(y - p).mean()) if y.shape[0] else float("nan")


def _confidence_bins(abs_error, confidence, groups, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """(bin, normalised error) per row. Within each group (coin) the confidence is cut into ``n_bins`` equal-count
    bins by average rank (bin 0 = least confident; a constant confidence puts the whole group in ONE middle bin) and
    |error| is divided by the group's mean |error|, so a coin's own error scale cannot fake a pattern. Groups with
    fewer than ``n_bins`` rows or a zero mean error are skipped."""
    e, c = _clean(abs_error, confidence)
    g = np.asarray(groups).reshape(-1)
    if g.shape[0] != np.asarray(abs_error).reshape(-1).shape[0]:
        raise ValueError("abs_error, confidence and groups must have the same length")
    keep = ~(np.isnan(np.asarray(abs_error, dtype=np.float64).reshape(-1))
             | np.isnan(np.asarray(confidence, dtype=np.float64).reshape(-1)))
    g = g[keep]
    bins, norm = [], []
    for k in np.unique(g):
        m = g == k
        if m.sum() < n_bins or e[m].mean() <= 0:
            continue
        r = average_ranks(c[m])
        bins.append(np.minimum(((r - 0.5) / m.sum() * n_bins).astype(np.int64), n_bins - 1))
        norm.append(e[m] / e[m].mean())
    if not bins:
        return np.empty(0, dtype=np.int64), np.empty(0)
    return np.concatenate(bins), np.concatenate(norm)


def _bin_means(bins: np.ndarray, norm: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(bins, minlength=n_bins)
    sums = np.bincount(bins, weights=norm, minlength=n_bins)
    nz = counts > 0
    return np.flatnonzero(nz).astype(np.float64), sums[nz] / counts[nz]


def confidence_error_spearman(abs_error: np.ndarray, confidence: np.ndarray, groups: np.ndarray,
                              n_bins: int = 10) -> float:
    """T-conf (ROADMAP Targets): Spearman of the within-group confidence bin vs the mean normalised |error| of that
    bin. A meaningful confidence gives <= -0.8 (more confident = smaller error). A confidence that carries no order
    (constant), a result with fewer than 2 non-empty bins, or equal error in every bin scores exactly 0.0 (no
    evidence), not NaN. NaN if no
    group is usable."""
    bins, norm = _confidence_bins(abs_error, confidence, groups, n_bins)
    if bins.shape[0] == 0:
        return float("nan")
    x, y = _bin_means(bins, norm, n_bins)
    return 0.0 if x.shape[0] < 2 or np.ptp(y) == 0 else spearman(x, y)


def confidence_error_gap(abs_error: np.ndarray, confidence: np.ndarray, groups: np.ndarray, n_bins: int = 10,
                         n_boot: int = 200, seed: int = 0) -> tuple[float, float, float]:
    """(gap, lo, hi): mean normalised |error| of the least-confident bin minus that of the most-confident bin
    (positive = confidence helps), with a 95 % percentile bootstrap interval over rows (iid: it ignores time
    dependence, so it is optimistic). (nan, nan, nan) if no group is usable; a constant confidence gives (0, 0, 0)."""
    bins, norm = _confidence_bins(abs_error, confidence, groups, n_bins)
    if bins.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")

    def gap(b, v):
        lo_m, hi_m = (v[b == 0], v[b == n_bins - 1])
        return float(lo_m.mean() - hi_m.mean()) if lo_m.size and hi_m.size else 0.0

    rng = np.random.default_rng(seed)
    n = bins.shape[0]
    boots = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        boots.append(gap(bins[ix], norm[ix]))
    return gap(bins, norm), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def width_below_share(width: np.ndarray, range_: np.ndarray) -> float:
    """T-sharp part 1: share of rows with width < range_. TRIVIAL: a zero width scores 1.0 for any model, so this is
    never read without ``interval_coverage``. NaN if no rows."""
    w, r = _clean(width, range_)
    return float((w < r).mean()) if w.shape[0] else float("nan")


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of rows with lower <= y <= upper. TRIVIAL: an infinite interval scores 1.0, a zero-width one ~0.
    NaN if no rows."""
    y, lo, hi = _clean(y_true, lower, upper)
    return float(((y >= lo) & (y <= hi)).mean()) if y.shape[0] else float("nan")
