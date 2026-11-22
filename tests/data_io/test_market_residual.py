"""data/market_residual.py (owner task 2026-11-19, method 3): the market is the equal-weight mean move, beta uses
only past bars and finds a planted beta, the residual label removes beta x the market move over the label span and still
decodes, and the run steps read the 10 channels."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.data import market_residual as mr
from nig_timenet_torch.data_io.artifact import load_artifact
from nig_timenet_torch.data_io.artifact_writer import write_artifact
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.data_io.synthetic_panel import PanelSyntheticSettings, build_panel_synthetic
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.run_common import check_decode
from nig_timenet_torch.workflow.settings import RunSettings

from .test_indicator_channels import _fetch_from_artifact

STEP = pd.Timedelta("4h").value


def test_market_is_the_equal_weight_mean_and_needs_enough_coins():
    idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    closes = {f"C{i}": pd.Series(np.exp(np.arange(4) * 0.01 * (i + 1)), index=idx) for i in range(mr.MARKET_MIN)}
    first, mkt, moves = mr.market_panel(closes, STEP)
    assert np.isnan(mkt[0])                                                  # no previous close
    np.testing.assert_allclose(mkt[1:], 0.01 * np.mean(np.arange(1, mr.MARKET_MIN + 1)))
    _, mkt2, _ = mr.market_panel(dict(list(closes.items())[:mr.MARKET_MIN - 1]), STEP)
    assert np.isnan(mkt2).all()


def test_rolling_beta_finds_a_planted_beta_and_uses_no_later_bar():
    rng = np.random.default_rng(0)
    m = 0.01 * rng.standard_normal(400)
    r = 2.0 * m + 0.002 * rng.standard_normal(400)
    b = mr.rolling_beta(r, m)
    assert np.isnan(b[mr.BETA_MIN - 2]) and abs(b[-1] - 2.0) < 0.1
    r2 = r.copy()
    r2[300:] *= -5
    np.testing.assert_array_equal(mr.rolling_beta(r2, m)[:300], b[:300])


def test_residual_artifact_label_channels_and_run(tmp_path):
    src = tmp_path / "e4.dataset"
    write_artifact(build_panel_synthetic(PanelSyntheticSettings(n_coins=6, n_bars=600)), src, source_commit="0" * 40)
    a = load_artifact(src).dataset
    base = _fetch_from_artifact(a)
    rec = mr.add_market_residual(src, tmp_path / "e4_m.dataset", candles=lambda n, s, e: base(n, "4h", s, e))
    b = load_artifact(tmp_path / "e4_m.dataset").dataset
    assert list(b["feature_order"]) == list(mr.CHANNELS)
    assert rec["rows_with_channels"] > 0.5 * rec["rows"]
    assert rec["label_var_residual"] < rec["label_var_own"]                  # the planted market factor is removed
    np.testing.assert_array_equal(np.asarray(b["y_high_reg"]), np.asarray(a["y_high_reg"]))
    check_decode(b, ("close", "high", "low"))
    beta = np.asarray(b["X_4h"])[..., mr.CHANNELS.index("beta")]
    filled = beta[np.abs(np.asarray(b["X_4h"])).sum(axis=(1, 2)) > 0]
    assert filled.min() >= mr.BETA_CLIP[0] and filled.max() <= mr.BETA_CLIP[1]
    s = RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(name=str(tmp_path / "e4_m.dataset"),
                                                                         local_dir=""))
    s = updated(s, {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu",
                    "evaluation.battery.enabled": False})
    r = steps.prepare(s)
    assert r.x["train"].shape[2] == len(mr.CHANNELS)
    steps.train(steps.build(r))
    assert r.result.epochs_done == 1 and np.isfinite(r.result.best_value)


def test_archive_path_creates_the_cache_folder_and_reuses_it(tmp_path):
    src = tmp_path / "e4.dataset"
    write_artifact(build_panel_synthetic(PanelSyntheticSettings(n_coins=5, n_bars=300)), src, source_commit="0" * 40)
    base = _fetch_from_artifact(load_artifact(src).dataset)
    calls = []

    def fetch(symbol, tf, start, end):
        calls.append(symbol)
        return base(symbol, tf, start, end)

    cache = tmp_path / "new" / "cache"
    r1 = mr.add_market_residual(src, tmp_path / "a.dataset", cache=cache, fetch=fetch)
    n = len(calls)
    assert cache.is_dir() and len(list(cache.glob("*.parquet"))) == n
    r2 = mr.add_market_residual(src, tmp_path / "b.dataset", cache=cache, fetch=fetch)
    assert len(calls) == n and r1["fingerprint"] == r2["fingerprint"]                # second run reads the cache only


def test_rank_gauss_label_is_the_normal_score_of_the_rank_inside_the_timestamp():
    y = np.array([0.03, -0.01, 0.20, 0.00, 0.05])
    ts = np.array([1, 1, 1, 2, 2])
    z = mr.rank_gauss_label(y, ts)
    sd = float(np.std(y))
    q = 0.6744897501960817                                       # inverse normal CDF of 0.75
    np.testing.assert_allclose(z, [0.0, -q * sd, q * sd, -0.4307272992954576 * sd, 0.4307272992954576 * sd])
    assert mr.rank_gauss_label(np.array([0.1]), np.array([7]))[0] == 0.0          # one row: the median


def test_close_label_option_default_unchanged_and_unknown_raises(tmp_path):
    src = tmp_path / "e4.dataset"
    write_artifact(build_panel_synthetic(PanelSyntheticSettings(n_coins=6, n_bars=400)), src, source_commit="0" * 40)
    base = _fetch_from_artifact(load_artifact(src).dataset)

    def candles(n, s, e):
        return base(n, "4h", s, e)

    r0 = mr.add_market_residual(src, tmp_path / "d.dataset", candles=candles)
    r1 = mr.add_market_residual(src, tmp_path / "r.dataset", candles=candles, label="residual")
    assert r0["fingerprint"] == r1["fingerprint"] and r0["close_label"] == "residual"
    mr.add_market_residual(src, tmp_path / "g.dataset", candles=candles, label="cs_rank_gauss")
    d, g = (load_artifact(tmp_path / n).dataset for n in ("d.dataset", "g.dataset"))
    ts = np.asarray(d["last_candles"])[:, 3]
    np.testing.assert_allclose(np.asarray(g["y_close_reg"]),
                               mr.rank_gauss_label(np.asarray(d["y_close_reg"], dtype=np.float64), ts), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(g["X_4h"]), np.asarray(d["X_4h"]))      # inputs unchanged
    check_decode(g, ("close", "high", "low"))
    with pytest.raises(ValueError, match="label must be one of"):
        mr.add_market_residual(src, tmp_path / "x.dataset", candles=candles, label="nope")


def test_beta_window_option_shorter_window_past_only_and_default_unchanged(tmp_path):
    rng = np.random.default_rng(1)
    m = 0.01 * rng.standard_normal(300)
    r = np.where(np.arange(300) < 150, 0.5, 2.0) * m + 0.001 * rng.standard_normal(300)
    np.testing.assert_array_equal(mr.rolling_beta(r, m, mr.BETA_BARS), mr.rolling_beta(r, m))
    b60 = mr.rolling_beta(r, m, 60)
    assert np.isnan(b60[28]) and np.isfinite(b60[29])                   # at least half the window when shorter
    assert abs(b60[-1] - 2.0) < 0.1 and abs(mr.rolling_beta(r, m)[-1] - 2.0) > 0.1   # the short window adapts
    r2 = r.copy()
    r2[200:] *= -3
    np.testing.assert_array_equal(mr.rolling_beta(r2, m, 60)[:200], b60[:200])
    src = tmp_path / "e4.dataset"
    write_artifact(build_panel_synthetic(PanelSyntheticSettings(n_coins=5, n_bars=300)), src, source_commit="0" * 40)
    base = _fetch_from_artifact(load_artifact(src).dataset)
    rec = mr.add_market_residual(src, tmp_path / "b.dataset", candles=lambda n, s, e: base(n, "4h", s, e),
                                 beta_bars=60)
    assert rec["beta_bars"] == 60 and rec["beta_min"] == 30
    with pytest.raises(ValueError, match="beta_bars"):
        mr.add_market_residual(src, tmp_path / "x.dataset", candles=lambda n, s, e: base(n, "4h", s, e), beta_bars=1)
