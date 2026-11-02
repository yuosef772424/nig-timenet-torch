"""P21: stats/metrics.py — cross-checks against sklearn/scipy and the documented tie/NaN policies."""

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

from nig_timenet_torch.stats import metrics as m

RNG = np.random.default_rng(0)


@pytest.mark.parametrize("ties", [False, True])
def test_auc_matches_sklearn(ties):
    for _ in range(20):
        n = int(RNG.integers(5, 300))
        y = RNG.integers(0, 2, n)
        s = RNG.integers(0, 4, n).astype(float) if ties else RNG.normal(size=n)
        if y.min() == y.max():
            continue
        assert m.auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auc_nan_policy_documented():
    assert math.isnan(m.auc(np.ones(5), np.arange(5.0)))
    assert "NaN if one class is empty" in m.auc.__doc__
    y, s = np.array([0, 1, 0, 1, np.nan]), np.array([0.1, 0.9, np.nan, 0.2, 0.3])
    assert m.auc(y, s) == pytest.approx(roc_auc_score([0, 1, 1], [0.1, 0.9, 0.2]))  # NaN rows dropped


def test_pearson_spearman_match_scipy():
    a, b = RNG.normal(size=200), RNG.integers(0, 5, 200).astype(float)
    assert m.pearson(a, b) == pytest.approx(sps.pearsonr(a, b)[0], abs=1e-12)
    assert m.spearman(a, b) == pytest.approx(sps.spearmanr(a, b)[0], abs=1e-12)
    assert math.isnan(m.pearson(np.ones(4), a[:4]))


def test_rank_ic_per_timestamp_matches_manual_groupby():
    g = np.repeat(np.arange(6), 10)
    y, p = RNG.normal(size=60), RNG.normal(size=60)
    manual = np.mean([sps.spearmanr(y[g == k], p[g == k])[0] for k in range(6)])
    assert m.rank_ic(y, p, g) == pytest.approx(manual, abs=1e-12)
    assert m.rank_ic(y, p) == pytest.approx(sps.spearmanr(y, p)[0], abs=1e-12)
    g1 = g.copy()
    g1[0] = 99  # a singleton group is undefined and skipped
    assert not math.isnan(m.rank_ic(y, p, g1))


def test_wilson_interval():
    lo, hi = m.wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4) and hi == pytest.approx(0.5962, abs=1e-4)
    assert all(math.isnan(v) for v in m.wilson_interval(0, 0))


def test_pinball_and_mae():
    y, p = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0])
    assert m.pinball_loss(y, p, 0.5) == pytest.approx(0.5 * m.mae(y, p))
    assert m.pinball_loss(y, p, 0.9) == pytest.approx((0.1 * 1 + 0 + 0.9 * 1) / 3)
    assert math.isnan(m.mae(np.array([]), np.array([])))


def test_ece_bins_sum_to_total_count():
    y, p = RNG.integers(0, 2, 500), RNG.uniform(size=500)
    p[:3] = [0.0, 1.0, 0.1]
    counts, _, _ = m.ece_bins(y, p, n_bins=10)
    assert counts.sum() == 500 and counts.shape == (10,)
    assert 0.0 <= m.ece(y, p) <= 1.0
    assert m.ece(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0


def test_every_function_documents_its_policy_and_no_torch_import():
    src = Path(m.__file__).read_text()
    imports = [n for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name for n in imports for a in n.names} | {n.module for n in imports if isinstance(n, ast.ImportFrom)}
    assert not {x for x in names if x and x.split(".")[0] in ("torch", "nig_timenet_torch")}
    for name in m.__all__:
        doc = getattr(m, name).__doc__ or ""
        assert "NaN" in doc or "nan" in doc, f"{name} docstring must state its NaN policy"
