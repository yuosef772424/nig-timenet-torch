"""The single owner of every metric this project computes (C7; BUILD_PLAN.md P21). Pure numpy — no torch.

PURPOSE: Every metric of the project in one module (AUC, rank IC and partial rank IC, R^2, calibration, intervals,
    sharpness, confidence-error gap, block-bootstrap CI), each with its trivial baseline stated and tested (rule C1).
TAGS: metrics, auc, rank ic, calibration, coverage, bootstrap, trivial baseline
PITFALLS: NaN rows are dropped pairwise and an undefined metric returns nan, never 0; bootstrap CIs resample
    rows or blocks only, so serial dependence between blocks is not covered.

Shared NaN policy: rows where ANY input is NaN are dropped first (pairwise); a metric that is undefined on what is
left returns ``float("nan")`` instead of raising. Ties are always handled by average ranks.
"""

import math

import numpy as np

__all__ = ["auc", "rank_ic", "pearson", "spearman", "wilson_interval", "pinball_loss", "ece", "ece_bins", "mae",
           "average_ranks", "r2", "confidence_error_spearman", "confidence_error_gap", "width_below_share",
           "interval_coverage", "block_bootstrap_ci", "partial_rank_ic"]


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


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination 1 - SS_res / SS_tot (on the rows kept after the NaN policy). NaN if no rows or if
    y is constant (SS_tot = 0). Trivial baseline: predicting mean(y) scores exactly 0."""
    y, p = _clean(y_true, y_pred)
    ss = float(((y - y.mean()) ** 2).sum()) if y.shape[0] else 0.0
    return 1.0 - float(((y - p) ** 2).sum()) / ss if ss > 0 else float("nan")


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


def block_bootstrap_ci(values: np.ndarray, blocks: np.ndarray, level: float = 0.95, n_boot: int = 500,
                       seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi): the mean of ``values`` with a percentile CI from resampling whole BLOCKS (rows sharing a
    block id, e.g. the same calendar day) with replacement, so rows that move together (coins on one day) are not
    counted as independent. TRIVIAL: zero-mean noise has a CI around 0 (it contains 0 at the stated rate); a
    constant series has a zero-width CI. NaN values are dropped; fewer than 2 blocks -> (mean, nan, nan); no rows ->
    all nan. This ignores serial dependence BETWEEN blocks (use blocks at least as long as the dependence)."""
    v, b = np.asarray(values, dtype=np.float64).reshape(-1), np.asarray(blocks).reshape(-1)
    if v.shape[0] != b.shape[0]:
        raise ValueError(f"length mismatch: {v.shape[0]} values, {b.shape[0]} blocks")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    keep = ~np.isnan(v)
    v, b = v[keep], b[keep]
    if v.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    _, inv = np.unique(b, return_inverse=True)
    sums, counts = np.bincount(inv, weights=v), np.bincount(inv).astype(np.float64)
    mean = float(v.mean())
    if sums.shape[0] < 2:
        return mean, float("nan"), float("nan")
    ix = np.random.default_rng(seed).integers(0, sums.shape[0], (n_boot, sums.shape[0]))
    boots = sums[ix].sum(axis=1) / counts[ix].sum(axis=1)
    tail = 100.0 * (1.0 - level) / 2.0
    return mean, float(np.percentile(boots, tail)), float(np.percentile(boots, 100.0 - tail))


def _residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a`` minus its least-squares line on ``b`` (both centred)."""
    a, b = a - a.mean(), b - b.mean()
    bb = float(b @ b)
    return a - (float(a @ b) / bb) * b if bb > 0 else a


def partial_rank_ic(y_true: np.ndarray, y_pred: np.ndarray, baseline: np.ndarray, groups: np.ndarray) -> float:
    """Rank IC of ``y_pred`` NET of ``baseline``: per group (e.g. timestamp) the Pearson correlation of the ranks of
    ``y_true`` and of ``y_pred``, each with its linear dependence on the ranks of ``baseline`` removed (a partial
    Spearman); the mean over groups with >= 4 rows. TRIVIAL: the baseline plus noise scores ~0 and a prediction with
    the baseline's own ranks is undefined (NaN), so only information beyond the baseline counts (rule C1: the
    volatility baseline of the high / low heads). NaN rows dropped; NaN if no group is defined."""
    y, p, b = (np.asarray(a, dtype=np.float64).reshape(-1) for a in (y_true, y_pred, baseline))
    g = np.asarray(groups).reshape(-1)
    if not (y.shape == p.shape == b.shape == g.shape):
        raise ValueError("y_true, y_pred, baseline and groups must have the same length")
    keep = ~(np.isnan(y) | np.isnan(p) | np.isnan(b))
    y, p, b, g = y[keep], p[keep], b[keep], g[keep]
    order = np.argsort(g, kind="stable")
    _, starts = np.unique(g[order], return_index=True)
    vals = []
    for ix in np.split(order, starts[1:]):
        if ix.shape[0] < 4:
            continue
        rb = average_ranks(b[ix])
        v = pearson(_residual(average_ranks(y[ix]), rb), _residual(average_ranks(p[ix]), rb))
        if not math.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")
