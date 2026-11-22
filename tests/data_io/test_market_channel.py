"""data/market_channel.py (owner task 2026-11-19): every bar of a window gets the series' k-day log growth of the
PREVIOUS UTC day (never the bar's own day), missing history gives 0 and is counted, the derived artifact keeps the
source's rows / labels and loads, and the run steps read the extra channel."""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data import market_channel as mc
from nig_timenet_torch.data_io.artifact import load_artifact
from nig_timenet_torch.data_io.artifact_writer import write_artifact
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.data_io.synthetic_panel import PanelSyntheticSettings, build_panel_synthetic
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import RunSettings


def _series(start="2022-11-01", days=200):
    idx = pd.date_range(start, periods=days, freq="D", tz="UTC")
    return pd.Series(np.exp(np.linspace(0.0, 1.0, days)) * 1e9, index=idx)       # log growth 1 / (days-1) a day


def test_each_bar_reads_the_previous_day_never_its_own():
    s = _series()
    g = mc.daily_growth(s, 7)
    t = pd.Timestamp("2023-02-10 20:00", tz="UTC")                                 # last bar opens 20:00
    vals, ok = mc.bar_values(g, np.array([t.value]), "4h", 32)
    assert ok.tolist() == [True] and vals.shape == (1, 32)
    assert vals[0, -1] == np.float32(g[pd.Timestamp("2023-02-09", tz="UTC")])     # the day before the bar's day
    first_bar = t - pd.Timedelta("4h") * 31                                         # 2023-02-05 16:00
    assert vals[0, 0] == np.float32(g[first_bar.floor("D") - pd.Timedelta("1D")])
    # a jump on the bar's own day is not seen by that bar
    s2 = s.copy()
    s2[pd.Timestamp("2023-02-10", tz="UTC")] *= 10
    v2, _ = mc.bar_values(mc.daily_growth(s2, 7), np.array([t.value]), "4h", 32)
    assert v2[0, -1] == vals[0, -1]


def test_missing_history_gives_zero_and_invalid():
    g = mc.daily_growth(_series(start="2023-02-01", days=30), 7)
    t = pd.Timestamp("2023-02-05", tz="UTC")
    vals, ok = mc.bar_values(g, np.array([t.value]), "4h", 32)
    assert ok.tolist() == [False] and vals[0, 0] == 0.0


def test_derived_artifact_keeps_rows_and_the_run_reads_four_channels(tmp_path):
    src = tmp_path / "e4.dataset"
    write_artifact(build_panel_synthetic(PanelSyntheticSettings(n_coins=6, n_bars=400)), src, source_commit="0" * 40)
    rec = mc.add_market_channel(src, tmp_path / "e4_m.dataset", _series(start="2022-12-01", days=200),
                                name="usdt_supply_g7", days=7)
    a, b = load_artifact(src).dataset, load_artifact(tmp_path / "e4_m.dataset").dataset
    assert list(b["feature_order"]) == [*a["feature_order"], "usdt_supply_g7"]
    np.testing.assert_array_equal(np.asarray(b["X_4h"])[..., :3], np.asarray(a["X_4h"]))
    np.testing.assert_array_equal(np.asarray(a["last_candles"]), np.asarray(b["last_candles"]))
    assert rec["fingerprint"] != rec["source_fingerprint"] and rec["missing_share"] < 0.5
    ts = np.asarray(b["last_candles"])[:, TS_COL]
    assert len(ts) == rec["rows"]
    s = RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(name=str(tmp_path / "e4_m.dataset"),
                                                                         local_dir=""))
    s = updated(s, {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu", "split.cross_section": True,
                    "evaluation.battery.enabled": False, "train.selection_metric": "val_rank_ic_reg",
                    "model.overrides": {**s.model.overrides, "cross_input": "relative"}})
    r = steps.prepare(s)
    assert r.x["train"].shape[2] == 5                                               # 4 channels + timestamp channel
    steps.train(steps.build(r))
    assert r.result.epochs_done == 1 and np.isfinite(r.result.best_value)
