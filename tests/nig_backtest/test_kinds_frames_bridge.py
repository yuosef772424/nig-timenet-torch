"""nig_backtest: output kinds (absolute kinds decode to a predicted move, lossy kinds refuse 'move' exits), candles
finer than the model's frame (hold as a duration, ATR on the model's frame, the finer path orders target and stop),
the archive reader on a fake archive, and the bridge to nig_timenet_torch on the fixture artifact (the model run on one
coin, the artifact's own candles)."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

import nig_backtest as bt
from nig_backtest.engine import hold_bars
from nig_backtest.prices import vision_klines
from tests.data_io.helpers import art  # noqa: F401  (fixture)

T0 = pd.Timestamp("2026-01-01")
H1, H4 = pd.Timedelta(hours=1), pd.Timedelta(hours=4)


def _bars(closes, step=H1, highs=None, lows=None):
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + 0.1 if highs is None else np.asarray(highs, float)
    lo = np.minimum(o, c) - 0.1 if lows is None else np.asarray(lows, float)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=T0 + step * np.arange(len(c)))


def _sig(scores, kind="score", t=T0 + 40 * H1, **extra):
    frame = pd.DataFrame({"coin": "A", "time": [t] * len(scores) if len(scores) > 1 else [t], "score": scores, **extra})
    return bt.SignalSet(frame=frame, kind=kind, horizon="8h")


def test_kinds_decode_direction_strength_and_the_predicted_move():
    rows = pd.DataFrame({"coin": "A", "time": T0, "score": [0.02, -0.01], "ref_price": [100.0, 100.0]})
    d = bt.decode(rows, "return")
    assert list(d["dir"]) == [1, -1] and list(d["pred_move"]) == [0.02, -0.01]
    assert bt.decode(rows.assign(score=[np.log(1.02), 0.0]), "log_return")["pred_move"].iloc[0] == pytest.approx(0.02)
    assert list(bt.decode(rows.assign(score=[103.0, 99.0]), "price")["pred_move"]) == pytest.approx([0.03, -0.01])
    r = bt.decode(pd.DataFrame({"coin": list("ABC"), "time": T0, "score": [0.9, 0.5, 0.1]}), "rank")
    assert list(r["dir"]) == [1, 0, -1] and r["pred_move"].isna().all()       # relative to the time's median
    p = bt.decode(rows.assign(score=[0.7, 0.4]), "probability")
    assert list(p["dir"]) == [1, -1] and list(p["strength"]) == pytest.approx([0.2, 0.1])
    with pytest.raises(ValueError, match="class"):
        bt.decode(rows.assign(score=[0.5, 1.0]), "class")
    with pytest.raises(ValueError, match="ref_price"):
        bt.SignalSet(frame=rows.drop(columns="ref_price"), kind="price")
    assert bt.is_absolute("return") and not bt.is_absolute("rank")


def test_move_exits_need_an_absolute_output():
    frames = {"A": _bars(100 + 0.01 * np.arange(80))}
    s = bt.strategy_from_dict({"exit": {"take_profit": 1.0, "take_profit_unit": "move", "max_bars": 8}})
    with pytest.raises(ValueError, match="absolute output kind"):
        bt.run_backtest(_sig([0.6], kind="probability"), bt.FramePrices(frames), s)
    tr = bt.run_backtest(_sig([0.004], kind="return"), bt.FramePrices(frames), s).trades.iloc[0]
    assert tr["target_frac"] == pytest.approx(0.004) and tr["pred_move"] == pytest.approx(0.004)


def test_a_4h_model_on_1h_candles_holds_the_same_time():
    frames = {"A": _bars(100 + np.arange(80.0))}
    hold = bt.strategy_from_dict({"exit": {"hold": "8h"}})
    assert hold_bars(hold, frames["A"]) == 8
    assert hold_bars(bt.strategy_from_dict({"exit": {"hold": "horizon"}}), frames["A"], "8h") == 8
    with pytest.raises(ValueError, match="whole number"):
        hold_bars(bt.strategy_from_dict({"exit": {"hold": "90min"}}), frames["A"])
    tr = bt.run_backtest(_sig([1.0]), bt.FramePrices(frames),
                         bt.strategy_from_dict({"exit": {"hold": "horizon"}})).trades.iloc[0]
    assert tr["bars_held"] == 8 and tr["held"] == 8 * H1 and tr["exit_price"] == 100 + 47


def test_atr_on_the_model_frame_and_the_finer_path_orders_target_and_stop():
    n = 120
    closes = np.full(n, 100.0)
    highs, lows = closes + 0.25, closes - 0.25
    frames = {"A": _bars(closes, highs=highs, lows=lows)}
    s = bt.strategy_from_dict({"exit": {"hold": "4h", "take_profit": 1.0, "atr_timeframe": "4h"}})
    tr = bt.run_backtest(_sig([1.0], t=T0 + 80 * H1), bt.FramePrices(frames), s).trades.iloc[0]
    assert tr["atr_entry"] == pytest.approx(0.5)            # the 4h bars: range 0.5, the 1h bars too here
    highs2, lows2 = highs.copy(), lows.copy()
    highs2[81], lows2[83] = 103.0, 97.0                       # inside ONE 4h bar: +3 at hour 1, -3 at hour 3
    rule = {"exit": {"hold": "4h", "take_profit": 0.02, "take_profit_unit": "pct", "stop_loss": 0.02,
                     "stop_loss_unit": "pct"}}
    fine = bt.run_backtest(_sig([1.0], t=T0 + 80 * H1), bt.FramePrices({"A": _bars(closes, highs=highs2, lows=lows2)}),
                           bt.strategy_from_dict(rule)).trades.iloc[0]
    assert fine["exit_reason"] == "take_profit"               # 1h candles see the target first
    coarse = frames["A"].resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    coarse.loc[T0 + 80 * H1, ["high", "low"]] = [103.0, 97.0]
    tr4 = bt.run_backtest(_sig([1.0], t=T0 + 80 * H1), bt.FramePrices({"A": coarse}),
                          bt.strategy_from_dict({**rule, "exit": {**rule["exit"], "hold": None, "max_bars": 1}}))
    assert tr4.trades.iloc[0]["exit_reason"] == "stop_loss"   # one 4h bar touches both: stop first


def test_vision_klines_on_a_fake_archive():
    class Fake:
        def get_bytes(self, url):
            if "/monthly/" not in url:
                return None
            t = pd.date_range("2026-03-01", "2026-03-31 23:00", freq="h")
            rows = "\n".join(f"{int(x.value // 10**6)},1,2,0.5,1.5,10,0,15,3,0,0,0" for x in t)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("k.csv", "open_time,open\n" + rows)
            return buf.getvalue() if "2026-03" in url else None
    f = vision_klines("AUSDT", "1h", "2026-03-10", "2026-03-11", Fake())
    assert len(f) == 24 and f.index[0] == pd.Timestamp("2026-03-10") and f["close"].iloc[0] == 1.5


def test_bridge_runs_the_model_on_one_coin_and_reads_the_artifact_candles(art, tmp_path):  # noqa: F811
    from nig_backtest.bridges import nig_timenet_torch as bridge
    from nig_timenet_torch.core.config import updated
    from nig_timenet_torch.data_io.loading import LoaderSettings
    from nig_timenet_torch.workflow import run as steps
    from nig_timenet_torch.workflow.settings import RunSettings

    s = updated(RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(name=str(art))),
                {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu"})
    run = steps.train(steps.build(steps.prepare(s)))
    full, meta = bridge.signals_from_run(run, "test")
    assert meta["kind"] == "return" and pd.Timedelta(meta["horizon"]) > pd.Timedelta(0)
    coin = full["coin"].iloc[0]
    n_test = len(run.idx["test"])
    one, _ = bridge.signals_from_run(run, "test", coins=[coin])
    assert set(one["coin"]) == {coin} and len(run.idx["test"]) == n_test             # the run is left as it was
    ref = full[full["coin"] == coin].reset_index(drop=True)
    np.testing.assert_allclose(one["score"], ref["score"], rtol=1e-5)               # same model, same rows
    assert "confidence" in one and (one["time"] == ref["time"]).all()
    prices = bridge.artifact_prices(run.dataset, coins=[coin])
    bars = prices.bars(coin)
    assert {"open", "high", "low", "close"} <= set(bars.columns) and bars.index.is_monotonic_increasing
    step = int(run.dataset["stride"]) * pd.Timedelta(run.dataset["base_tf"])
    assert bt.prices.bar_interval(bars) == step               # one candle per window: the stride's spacing
    res = bt.run_backtest(bridge.signal_set(one, meta), prices,
                          bt.strategy_from_dict({"exit": {"max_bars": 2, "take_profit": 1.0,
                                                          "take_profit_unit": "move"}}))
    assert len(res.trades) > 0 and set(res.trades["exit_reason"]) <= {"take_profit", "stop_loss", "time", "data_end"}
    with pytest.raises(ValueError, match="no test rows"):
        bridge.signals_from_run(run, "test", coins=["NOPE"])
