"""data_io/btc_relative.py (owner task 2026-11-19): relative moves are the coin's minus BTC's of the same bar, the
ratio indicators read coin / BTC, no later bar changes a window, BTC's own rows get zero relative channels, and the
derived artifact keeps rows / labels and is read by the run steps with 11 channels."""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.data.artifact_writer import write_artifact
from nig_timenet_torch.data_io import btc_relative as br
from nig_timenet_torch.data_io.artifact import load_artifact
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.data_io.synthetic_panel import PanelSyntheticSettings, build_panel_synthetic
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import RunSettings

from .test_indicator_channels import _fetch_from_artifact


def _candles(close, start="2023-01-01"):
    close = np.asarray(close, dtype=np.float64)
    idx = pd.date_range(start, periods=len(close), freq="4h", tz="UTC")
    return pd.DataFrame({"high": close * 1.002, "low": close * 0.998, "close": close,
                         "quote_volume": np.full(len(close), 1e6)}, index=idx)


def _x_of(c):
    prev = c["close"].shift(1)
    return np.stack([np.log(c[k] / prev).to_numpy() for k in ("high", "low", "close")], axis=1)


def _windows(c, btc, i, window=32, is_btc=False):
    x = _x_of(c)[i - window + 1:i + 1][None]
    return br.coin_channels(c, btc, np.array([c.index[i].value]), "4h", window, x, is_btc=is_btc)


def test_relative_moves_subtract_btc_and_ratio_indicators_read_the_ratio():
    rng = np.random.default_rng(0)
    btc = _candles(np.exp(np.cumsum(0.01 * rng.standard_normal(200))))
    coin = _candles(btc["close"].to_numpy() * np.exp(np.linspace(0.0, 0.5, 200)))   # coin / BTC rises steadily
    w, ok = _windows(coin, btc, 199)
    assert ok.tolist() == [True] and w.shape == (1, 32, len(br.CHANNELS))
    rel_close = w[0, :, br.CHANNELS.index("rel_close")]
    np.testing.assert_allclose(rel_close, 0.5 / 199, atol=1e-6)                       # the ratio's own log move
    assert w[0, -1, br.CHANNELS.index("ratio_rsi_14")] > 0.99
    assert w[0, -1, br.CHANNELS.index("ratio_stoch_k_14_3")] > 0.5
    np.testing.assert_allclose(w[0, :, br.CHANNELS.index("btc_close")], _x_of(btc)[168:200, 2], atol=1e-6)


def test_btc_rows_get_zero_relative_channels_and_a_flat_ratio():
    btc = _candles(np.exp(np.linspace(0.0, 1.0, 200)))
    w, ok = _windows(btc, btc, 199, is_btc=True)
    assert ok[0] and not w[0, :, :3].any()
    assert abs(w[0, -1, br.CHANNELS.index("ratio_rsi_14")]) < 1e-9                    # constant ratio: RSI 50
    assert w[0, -1, br.CHANNELS.index("btc_rsi_14")] > 0.99


def test_no_lookahead_and_missing_btc_bar_gives_zeros():
    rng = np.random.default_rng(1)
    btc = _candles(np.exp(np.cumsum(0.01 * rng.standard_normal(300))))
    coin = _candles(np.exp(np.cumsum(0.02 * rng.standard_normal(300))))
    w1, _ = _windows(coin, btc, 250)
    btc2, coin2 = btc.copy(), coin.copy()
    btc2.iloc[251:, :] *= 2.0
    coin2.iloc[251:, :] *= 3.0
    w2, _ = _windows(coin2, btc2, 250)
    np.testing.assert_array_equal(w1, w2)
    w3, ok = _windows(coin, btc.drop(btc.index[240]), 250)
    assert ok.tolist() == [False] and not w3.any()


def test_derived_artifact_keeps_rows_and_the_run_reads_eleven_channels(tmp_path):
    src = tmp_path / "e4.dataset"
    write_artifact(build_panel_synthetic(PanelSyntheticSettings(n_coins=4, n_bars=400)), src, source_commit="0" * 40)
    a = load_artifact(src).dataset
    base = _fetch_from_artifact(a)
    first = a["asset_bounds"][0]["name"]

    def fetch(symbol, tf, start, end):
        return base(first if symbol == br.BTC else symbol, tf, start, end)

    rec = br.add_btc_relative(src, tmp_path / "e4_b.dataset", fetch=fetch, workers=2)
    b = load_artifact(tmp_path / "e4_b.dataset").dataset
    assert list(b["feature_order"]) == list(br.CHANNELS) and not rec["btc_in_artifact"]
    np.testing.assert_array_equal(np.asarray(a["last_candles"]), np.asarray(b["last_candles"]))
    xb = np.asarray(b["X_4h"])
    s, e = int(a["asset_bounds"][0]["start"]), int(a["asset_bounds"][0]["end"])
    filled = np.abs(xb[s:e]).sum(axis=(1, 2)) > 0
    assert filled.any() and np.abs(xb[s:e][filled][..., :3]).max() < 1e-5               # the "BTC" coin vs itself
    assert 0.0 < rec["missing_share"] < 0.5
    s_ = RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(name=str(tmp_path / "e4_b.dataset"),
                                                                          local_dir=""))
    s_ = updated(s_, {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu",
                      "evaluation.battery.enabled": False})
    r = steps.prepare(s_)
    assert r.x["train"].shape[2] == len(br.CHANNELS)
    steps.train(steps.build(r))
    assert r.result.epochs_done == 1 and np.isfinite(r.result.best_value)
