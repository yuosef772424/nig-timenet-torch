"""data_io/indicator_channels.py (owner task 2026-11-19): RSI / stochastic / relative volume per bar use no later bar,
take known values on simple paths, a window with a missing bar or a close that disagrees with the artifact is zeros
and counted, and the derived artifact keeps the source's rows / labels and is read by the run steps."""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data import indicator_channels as ic
from nig_timenet_torch.data_io.artifact import load_artifact
from nig_timenet_torch.data_io.artifact_writer import write_artifact
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.data_io.synthetic_panel import PanelSyntheticSettings, build_panel_synthetic
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import RunSettings

STEP = pd.Timedelta("4h")


def _candles(close, vol=None, start="2023-01-01"):
    close = np.asarray(close, dtype=np.float64)
    idx = pd.date_range(start, periods=len(close), freq="4h", tz="UTC")
    vol = np.full(len(close), 1e6) if vol is None else np.asarray(vol, dtype=np.float64)
    return pd.DataFrame({"high": close * 1.001, "low": close * 0.999, "close": close, "quote_volume": vol}, index=idx)


def _end(c, i):
    return np.array([c.index[i].value], dtype=np.int64)


def test_rising_path_has_high_rsi_and_stochastic_and_constant_volume_is_zero():
    c = _candles(np.exp(np.linspace(0.0, 1.0, 200)))
    w, ok, mis = ic.indicator_windows(c, _end(c, 199), "4h", 32)
    assert ok.tolist() == [True] and not mis.any() and w.shape == (1, 32, 4)
    assert w[0, -1, 0] > 0.99                     # RSI ~ 100
    assert w[0, -1, 1] > 0.5 and w[0, -1, 2] > 0.5  # close near the top of its 14-bar range
    assert abs(w[0, -1, 3]) < 1e-6                 # volume equal to its own mean


def test_volume_spike_is_positive_and_bounded():
    vol = np.full(200, 1e6)
    vol[199] = 1e9
    c = _candles(np.exp(np.linspace(0.0, 1.0, 200)), vol)
    w, ok, _ = ic.indicator_windows(c, _end(c, 199), "4h", 32)
    assert ok[0] and 5.0 >= w[0, -1, 3] > 4.0 and abs(w[0, -2, 3]) < 1e-6


def test_no_lookahead_a_later_bar_never_changes_an_earlier_window():
    rng = np.random.default_rng(0)
    close = np.exp(np.cumsum(0.02 * rng.standard_normal(300)))
    c = _candles(close)
    w1, _, _ = ic.indicator_windows(c, _end(c, 250), "4h", 32)
    c2 = c.copy()
    c2.iloc[251:, :] *= 3.0
    w2, _, _ = ic.indicator_windows(c2, _end(c2, 250), "4h", 32)
    np.testing.assert_array_equal(w1, w2)


def test_missing_bar_warmup_and_close_mismatch_give_zeros():
    c = _candles(np.exp(np.linspace(0.0, 1.0, 200)))
    w, ok, _ = ic.indicator_windows(c, _end(c, 40), "4h", 32)           # window starts at bar 9: no warm-up
    assert ok.tolist() == [False] and not w.any()
    gap = c.drop(c.index[190])
    w, ok, _ = ic.indicator_windows(gap, _end(c, 199), "4h", 32)
    assert ok.tolist() == [False] and not w.any()
    x_close = np.zeros((1, 32))                                          # the artifact says "flat", candles rise
    w, ok, mis = ic.indicator_windows(c, _end(c, 199), "4h", 32, x_close)
    assert mis.tolist() == [True] and ok.tolist() == [False] and not w.any()


def _fetch_from_artifact(ds):
    """Candles that agree with the artifact's own windows (close / high / low moves), random volume."""
    x, ts, bounds = np.asarray(ds["X_4h"]), np.asarray(ds["last_candles"])[:, TS_COL].astype(np.int64), {}
    window = x.shape[1]
    for b in ds["asset_bounds"]:
        bounds[b["name"]] = (int(b["start"]), int(b["end"]))

    def fetch(symbol, tf, start, end):
        s, e = bounds[symbol]
        moves = {}
        for r in range(s, e):
            for j in range(window):
                moves[int(ts[r]) - (window - 1 - j) * STEP.value] = x[r, j]
        times = sorted(moves)
        m = np.stack([moves[t] for t in times])
        prev = np.exp(np.concatenate([[0.0], np.cumsum(m[:-1, 2])]))
        close = prev * np.exp(m[:, 2])
        vol = np.random.default_rng(1).uniform(1e5, 1e6, len(times))
        return pd.DataFrame({"high": prev * np.exp(m[:, 0]), "low": prev * np.exp(m[:, 1]), "close": close,
                             "quote_volume": vol}, index=pd.to_datetime(times, utc=True))

    return fetch


def test_derived_artifact_keeps_rows_and_the_run_reads_seven_channels(tmp_path):
    src = tmp_path / "e4.dataset"
    write_artifact(build_panel_synthetic(PanelSyntheticSettings(n_coins=4, n_bars=400)), src, source_commit="0" * 40)
    a = load_artifact(src).dataset
    rec = ic.add_indicator_channels(src, tmp_path / "e4_i.dataset", fetch=_fetch_from_artifact(a), workers=2)
    b = load_artifact(tmp_path / "e4_i.dataset").dataset
    assert list(b["feature_order"]) == [*a["feature_order"], *ic.CHANNELS]
    np.testing.assert_array_equal(np.asarray(b["X_4h"])[..., :3], np.asarray(a["X_4h"]))
    np.testing.assert_array_equal(np.asarray(a["last_candles"]), np.asarray(b["last_candles"]))
    assert rec["close_mismatch_rows"] == 0 and 0.0 < rec["missing_share"] < 0.5
    new = np.asarray(b["X_4h"])[..., 3:]
    assert np.abs(new[..., :3]).max() <= 1.0 and np.abs(new).max() > 0
    s = RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(name=str(tmp_path / "e4_i.dataset"),
                                                                         local_dir=""))
    s = updated(s, {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu",
                    "evaluation.battery.enabled": False})
    r = steps.prepare(s)
    assert r.x["train"].shape[2] == 7
    steps.train(steps.build(r))
    assert r.result.epochs_done == 1 and np.isfinite(r.result.best_value)
