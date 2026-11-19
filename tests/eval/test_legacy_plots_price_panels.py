"""eval/legacy_plots.py::price_panels / improve_vs_naive: TF price-level panels with naive baseline and lag scan."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval import legacy_plots


def _table(pred_kind="noisy", with_close=True):
    rng = np.random.default_rng(1)
    rows = []
    for c, n in (("AAA", 40), ("BBB", 25)):
        price = 100 + np.cumsum(rng.standard_normal(n + 1))          # price[t] = last close of row t
        last = price[:-1]
        y = price[1:] / last - 1
        mu = {"perfect": y, "naive": np.zeros(n)}.get(pred_kind, 0.5 * y)
        t = pd.Timestamp("2026-01-01") + pd.to_timedelta(4 * np.arange(n), unit="h")
        d = pd.DataFrame({"coin": c, "end_time": t, "mu": mu, "sigma": 0.02, "y_ret": y,
                          "pred_high": mu + 0.01, "fut_high": y + 0.01, "pred_low": mu - 0.01, "fut_low": y - 0.01})
        if with_close:
            d["last_close"] = last
        rows.append(d)
    return pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=0)   # unsorted on purpose


def test_improve_vs_naive_known_series():
    a = np.array([1.0, 2.0, 4.0, 3.0])
    assert legacy_plots.improve_vs_naive(a, a) == pytest.approx(100.0)
    assert legacy_plots.improve_vs_naive(a, np.r_[np.nan, a[:-1]]) == pytest.approx(0.0)
    assert legacy_plots.improve_vs_naive(a, a + 5) < 0


def test_price_panels_writes_one_png_per_coin(tmp_path):
    paths = legacy_plots.price_panels(_table(), ["AAA", "BBB"], tmp_path / "f", title="T")
    assert [p.name for p in paths] == ["T_price_panels_AAA.png", "T_price_panels_BBB.png"]
    assert all(p.is_file() and p.stat().st_size > 1000 for p in paths)


def test_price_series_and_improve_on_a_table():
    t = _table("perfect")
    d = t[t["coin"] == "AAA"].sort_values("end_time")
    actual, pred = legacy_plots._price_series(d, "close")
    np.testing.assert_allclose(pred, actual)
    assert legacy_plots.improve_vs_naive(actual, pred) == pytest.approx(100.0)
    d0 = _table("naive")
    d0 = d0[d0["coin"] == "AAA"].sort_values("end_time")
    a0, p0 = legacy_plots._price_series(d0, "close")
    assert np.allclose(p0[1:], a0[:-1])                              # pred = last close = previous actual


def test_lag_scan_peaks_at_zero_for_a_perfect_prediction():
    a = np.cumsum(np.random.default_rng(0).standard_normal(60))
    lags = legacy_plots._lag_corr(a, a, 5)
    assert sorted(lags) == list(range(-5, 6)) and lags[0] == pytest.approx(1.0)


def test_errors(tmp_path):
    with pytest.raises(ValueError, match="last_close.*re-run the battery"):
        legacy_plots.price_panels(_table(with_close=False), ["AAA"], tmp_path)
    with pytest.raises(ValueError, match="ZZZ.*not in the table"):
        legacy_plots.price_panels(_table(), ["ZZZ"], tmp_path)
