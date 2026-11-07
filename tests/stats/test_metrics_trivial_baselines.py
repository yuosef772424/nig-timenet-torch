"""Rule C1 (docs/RULES.md): every metric states what a TRIVIAL model scores on it, and the number is tested here.

One ``test_trivial_baseline_<metric>`` per name in ``stats.metrics.__all__`` (the coverage test below fails when a
metric is added without one). "Trivial" = a constant score, the best constant, a shuffled score, or an in-bar
position style predictor that knows nothing about the target. A metric whose trivial baseline already scores
"perfectly" is not evidence on its own (see ece: a constant at the base rate scores 0).
"""

import numpy as np
import pytest

from nig_timenet_torch.stats import metrics

N = 4000
NAN = float("nan")


@pytest.fixture
def rng():
    """A fresh generator per test, so a test's numbers do not depend on which tests ran before it."""
    return np.random.default_rng(20261102)


def test_every_metric_has_a_trivial_baseline_test():
    missing = [m for m in metrics.__all__ if f"test_trivial_baseline_{m}" not in globals()]
    assert not missing, f"stats/metrics.py metrics without a trivial-baseline test (rule C1): {missing}"


def test_trivial_baseline_auc(rng):
    y = (rng.random(N) < 0.3).astype(float)
    assert metrics.auc(y, np.zeros(N)) == 0.5, "a constant score is exactly chance"
    assert metrics.auc(y, rng.random(N)) == pytest.approx(0.5, abs=0.03), "a shuffled score is chance"
    assert metrics.auc(y, y) == 1.0 and metrics.auc(y, -y) == 0.0


def test_trivial_baseline_rank_ic(rng):
    y, groups = rng.normal(size=N), np.repeat(np.arange(N // 20), 20)
    assert np.isnan(metrics.rank_ic(y, np.zeros(N), groups)), "a constant prediction has no rank IC (undefined, not 0)"
    shuffled = metrics.rank_ic(y, rng.normal(size=N), groups)
    assert abs(shuffled) < 0.03, "a shuffled prediction has rank IC ~ 0 (std of the mean over 200 groups ~ 0.016)"
    assert metrics.rank_ic(y, y, groups) == pytest.approx(1.0)


def test_trivial_baseline_pearson(rng):
    y = rng.normal(size=N)
    assert np.isnan(metrics.pearson(y, np.zeros(N)))
    assert abs(metrics.pearson(y, rng.normal(size=N))) < 0.06  # 3.8 sigma at n=4000
    assert metrics.pearson(y, y) == pytest.approx(1.0)


def test_trivial_baseline_spearman(rng):
    y = rng.normal(size=N)
    assert np.isnan(metrics.spearman(y, np.zeros(N)))
    assert abs(metrics.spearman(y, rng.normal(size=N))) < 0.06
    assert metrics.spearman(y, np.exp(y)) == pytest.approx(1.0), "monotone transform keeps the rank order"


def test_trivial_baseline_wilson_interval(rng):
    """A coin-flip classifier (success rate 1/2) has an interval that contains 0.5: not evidence of skill."""
    lo, hi = metrics.wilson_interval(N // 2, N)
    assert lo < 0.5 < hi
    assert all(np.isnan(v) for v in metrics.wilson_interval(0, 0))
    lo, hi = metrics.wilson_interval(int(0.6 * N), N)
    assert lo > 0.5, "a real 60 % hit rate at n=4000 excludes chance"


def test_trivial_baseline_pinball_loss(rng):
    """The best constant is the empirical q-quantile; any model must beat THAT, not zero."""
    y = rng.normal(size=N)
    for q in (0.1, 0.5, 0.9):
        best_constant = metrics.pinball_loss(y, np.full(N, np.quantile(y, q)), q)
        assert best_constant > 0.05
        for c in (-1.0, 0.0, 1.0, float(y.mean())):
            assert metrics.pinball_loss(y, np.full(N, c), q) >= best_constant - 1e-9
    assert metrics.pinball_loss(y, y, 0.5) == 0.0


def test_trivial_baseline_ece(rng):
    """A constant at the base rate is perfectly calibrated (ECE ~ 0): ECE alone is never evidence of skill."""
    y = (rng.random(N) < 0.3).astype(float)
    assert metrics.ece(y, np.full(N, y.mean())) < 1e-9
    assert metrics.ece(y, np.full(N, 0.9)) > 0.5, "a confidently wrong constant is far from calibrated"
    assert np.isnan(metrics.ece(np.array([]), np.array([])))


def test_trivial_baseline_ece_bins(rng):
    """A constant probability lands in exactly one bin (all other bins empty -> NaN means)."""
    y = (rng.random(N) < 0.3).astype(float)
    counts, mean_y, mean_p = metrics.ece_bins(y, np.full(N, 0.35))
    assert counts.sum() == N and (counts > 0).sum() == 1
    assert np.isnan(mean_y[counts == 0]).all() and np.isnan(mean_p[counts == 0]).all()


def test_trivial_baseline_mae(rng):
    """The best constant for MAE is the median; predicting 0 or the mean scores worse (skewed y)."""
    y = rng.exponential(size=N)  # skewed: median != mean
    best = metrics.mae(y, np.full(N, np.median(y)))
    assert best > 0.1
    assert metrics.mae(y, np.full(N, y.mean())) > best
    assert metrics.mae(y, np.zeros(N)) > best
    assert metrics.mae(y, y) == 0.0


def test_trivial_baseline_average_ranks(rng):
    """Not a skill metric (a helper): a constant column has every rank equal to the mean rank (n + 1) / 2."""
    assert (metrics.average_ranks(np.zeros(7)) == 4.0).all()


def _coin_errors(rng, n_coins=8, per=400):
    """Coins with very different error scales; the true error shrinks with a hidden confidence."""
    groups = np.repeat(np.arange(n_coins), per)
    scale = np.repeat(10.0 ** np.arange(n_coins), per)
    conf = rng.random(n_coins * per)
    return groups, conf, scale * (1.5 - conf)


def test_trivial_baseline_confidence_error_spearman(rng):
    groups, conf, err = _coin_errors(rng)
    assert metrics.confidence_error_spearman(err, np.zeros(len(err)), groups) == 0.0, "constant confidence: 0"
    assert metrics.confidence_error_spearman(err, conf, groups) <= -0.8, "a real confidence passes T-conf"
    assert metrics.confidence_error_spearman(err, -conf, groups) >= 0.8, "an inverted one is caught too"
    # a shuffled confidence: rho over 10 bin means has sd ~0.33, so <= -0.8 happens by chance in well under 3 % of draws
    # (hence T-conf also demands the gap CI above 0); a coin with a huge error scale must not move that rate
    scale = np.where(groups == groups.max(), 1e6, 1.0) * rng.random(len(groups))
    null = [metrics.confidence_error_spearman(scale, rng.random(len(scale)), groups) for _ in range(100)]
    assert np.mean(np.asarray(null) <= -0.8) < 0.03 and abs(np.mean(null)) < 0.1


def test_trivial_baseline_confidence_error_gap(rng):
    groups, conf, err = _coin_errors(rng)
    assert metrics.confidence_error_gap(err, np.zeros(len(err)), groups) == (0.0, 0.0, 0.0)
    gap, lo, hi = metrics.confidence_error_gap(err, rng.random(len(err)), groups)
    assert lo <= 0.0 <= hi, "a shuffled confidence has a gap CI that contains 0"
    gap, lo, hi = metrics.confidence_error_gap(err, conf, groups)
    assert gap > 0.5 and lo > 0.0, "a real confidence has a positive gap above its CI"


def test_trivial_baseline_width_below_share(rng):
    r = rng.random(N) + 0.1
    assert metrics.width_below_share(np.zeros(N), r) == 1.0, "zero width scores perfectly: never read alone"
    assert metrics.width_below_share(r * 2, r) == 0.0
    assert np.isnan(metrics.width_below_share([], []))


def test_trivial_baseline_interval_coverage(rng):
    y = rng.normal(size=N)
    assert metrics.interval_coverage(y, np.full(N, 0.0), np.full(N, 0.0)) == 0.0, "a zero-width interval covers ~0"
    assert metrics.interval_coverage(y, np.full(N, -np.inf), np.full(N, np.inf)) == 1.0, "infinite covers all"
    z = 1.2815515655446004  # exact 80 % two-sided normal quantile
    assert metrics.interval_coverage(y, np.full(N, -z), np.full(N, z)) == pytest.approx(0.8, abs=0.02)


def test_trivial_baseline_block_bootstrap_ci(rng):
    days = np.repeat(np.arange(N // 20), 20)
    noise = rng.normal(size=N)
    mean, lo, hi = metrics.block_bootstrap_ci(noise, days)
    assert lo < 0.0 < hi, "zero-mean noise: the 95 % CI contains 0"
    assert metrics.block_bootstrap_ci(np.full(N, 0.3), days)[1:] == pytest.approx((0.3, 0.3)), "a constant has no width"
    # rows of one day move together: the iid CI (one row per block) is far too narrow for them
    daily = np.repeat(rng.normal(size=N // 20), 20) + 0.1 * rng.normal(size=N)
    wide = metrics.block_bootstrap_ci(daily, days)
    iid = metrics.block_bootstrap_ci(daily, np.arange(N))
    assert (wide[2] - wide[1]) > 2.0 * (iid[2] - iid[1]), "dependence-aware CI is wider than the iid one"
    assert np.isnan(metrics.block_bootstrap_ci(noise, np.zeros(N))[1]), "one block: no interval"
    assert metrics.block_bootstrap_ci(noise, days, level=0.999)[2] > hi, "a higher level widens the interval"
    assert np.isnan(metrics.block_bootstrap_ci(np.full(3, np.nan), np.arange(3))[0])
