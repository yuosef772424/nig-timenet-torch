"""data/lower_timeframe.py (owner task 2026-11-18): the fine window of a row ends where its base bar closes, the
channels are the artifact's own, a gap invalidates a row (zeros, counted), the derived artifact keeps the source's
rows / labels and loads as a contract artifact; the run steps read the two frames side by side only when asked."""

import json

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data import lower_timeframe as lt
from nig_timenet_torch.data_io.artifact import load_artifact
from nig_timenet_torch.data_io.artifact_writer import write_artifact
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.data_io.synthetic_panel import PanelSyntheticSettings, build_panel_synthetic
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import RunSettings


def _candles(start: str, n: int, seed: int = 0, tf: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * np.exp(np.abs(rng.normal(0, 0.005, n)))
    low = close * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    idx = pd.date_range(start, periods=n, freq=tf, tz="UTC")
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)


def test_window_ends_where_the_base_bar_closes_with_the_artifact_channels():
    c = _candles("2024-01-01", 200)
    t = pd.Timestamp("2024-01-04 08:00", tz="UTC")                   # a 4h bar OPEN time; it closes at 12:00
    x, ok = lt.log_move_windows(c, np.array([t.value]), "1h", 32, "4h")
    assert ok.tolist() == [True] and x.shape == (1, 32, 3) and x.dtype == np.float32
    last, prev = c.loc[t + pd.Timedelta("3h")], c.loc[t + pd.Timedelta("2h"), "close"]   # the 11:00 bar
    want = np.log([last["high"] / prev, last["low"] / prev, last["close"] / prev])
    np.testing.assert_allclose(x[0, -1], want, rtol=1e-6)
    first, before = c.loc[t + pd.Timedelta("4h") - pd.Timedelta("32h")], c.loc[t - pd.Timedelta("29h"), "close"]
    np.testing.assert_allclose(x[0, 0, 2], np.log(first["close"] / before), rtol=1e-6)


def test_a_gap_or_too_little_history_gives_zeros_and_invalid():
    c = _candles("2024-01-01", 200)
    t_ok, t_early = pd.Timestamp("2024-01-05", tz="UTC"), pd.Timestamp("2024-01-01 20:00", tz="UTC")
    gap = c.drop(t_ok + pd.Timedelta("1h"))
    x, ok = lt.log_move_windows(gap, np.array([t_ok.value, t_early.value]), "1h", 32, "4h")
    assert ok.tolist() == [False, False] and not x.any()
    x, ok = lt.log_move_windows(c, np.array([t_ok.value, t_early.value]), "1h", 32, "4h")
    assert ok.tolist() == [True, False]


@pytest.fixture(scope="module")
def two_frames(tmp_path_factory):
    root = tmp_path_factory.mktemp("lt")
    src = root / "e4.dataset"
    write_artifact(build_panel_synthetic(PanelSyntheticSettings(n_coins=6, n_bars=400)), src, source_commit="0" * 40)

    def fetch(symbol, tf, start, end):                                # deterministic fake archive, one gap per coin
        c = _candles(str(start.floor("D")), int((end - start.floor("D")) / pd.Timedelta(tf)) + 48,
                     seed=sum(map(ord, symbol)))
        return c.drop(c.index[400])
    rec = lt.add_timeframe(src, root / "e4_1h.dataset", tf="1h", window=32, cache=root / "cache", workers=2,
                           fetch=fetch)
    return src, root / "e4_1h.dataset", rec


def test_derived_artifact_keeps_rows_and_labels_and_loads(two_frames):
    src, dst, rec = two_frames
    a, b = load_artifact(src).dataset, load_artifact(dst).dataset
    assert b["timeframes"] == ["4h", "1h"] and b["window_sizes"]["1h"] == 32
    assert b["X_1h"].shape == (*b["X_4h"].shape[:2], 3)
    for k in ("X_4h", "last_candles", "y_close_reg", "y_high_reg", "y_low_reg"):
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))
    assert [x["name"] for x in a["asset_bounds"]] == [x["name"] for x in b["asset_bounds"]]
    assert rec["fingerprint"] != rec["source_fingerprint"] and 0 < rec["missing_rows"] < rec["rows"]
    assert json.loads((dst / lt.DERIVATION_JSON).read_text())["added"] == "X_1h"
    ts = np.asarray(b["last_candles"])[:, TS_COL]
    assert (np.abs(np.asarray(b["X_1h"])).sum(axis=(1, 2)) > 0).sum() == rec["rows"] - rec["missing_rows"] > 0
    assert len(ts) == rec["rows"]


def test_run_steps_read_both_frames_only_when_asked(two_frames, tmp_path):
    _, dst, _ = two_frames
    s = RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(name=str(dst), local_dir=""))
    s = updated(s, {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu", "split.cross_section": True,
                    "evaluation.battery.enabled": False, "train.selection_metric": "val_rank_ic_reg"})
    with pytest.raises(ValueError, match="single-timeframe"):
        steps.prepare(s)
    with pytest.raises(ValueError, match="no \\['15m'\\]"):
        steps.prepare(updated(s, {"data.input_timeframes": ("4h", "15m")}))
    r = steps.prepare(updated(s, {"data.input_timeframes": ("4h", "1h")}))
    assert r.x["train"].shape[1:] == (32, 7)                          # 3 + 3 channels + the timestamp channel
    np.testing.assert_array_equal(r.x["train"][:, :, 3:6].numpy(),
                                  np.asarray(r.dataset["X_1h"])[r.idx["train"]].astype(np.float32))
    steps.train(steps.build(r))
    assert r.result.epochs_done == 1 and np.isfinite(r.result.best_value)
