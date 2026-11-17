"""nig_backtest: periods (train / val / test from dates or the stage; train never merged, val + test on request), the
VAL -> TEST protocol (switchable), cost modes, book admission (per coin, max open, strongest first), the quantile side
rule of a ranked output, missing candles (reported, or a stop), the compressed price store and the alignment check
against the last candle / target kept in the training data (and the bridge's reference on the fixture artifact)."""

import numpy as np
import pandas as pd
import pytest

import nig_backtest as bt
from nig_backtest.align import check_alignment
from tests.data_io.helpers import art  # noqa: F401  (fixture)

T0 = pd.Timestamp("2026-01-01")
H4 = pd.Timedelta(hours=4)


def _bars(closes, step=H4):
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999, "close": c},
                        index=T0 + step * np.arange(len(c)))


def _trend(n=300, drift=0.002, seed=0):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(drift + 0.01 * rng.standard_normal(n)))


def _signals(coins=("A", "B"), every=3, n=290):
    rows = [(c, T0 + i * H4, 1.0) for c in coins for i in range(5, n, every)]
    return pd.DataFrame(rows, columns=["coin", "time", "score"])


PER = bt.Periods("2026-01-20", "2026-02-10")


def test_periods_from_dates_train_never_merged_and_val_test_merge():
    frames = {c: _bars(_trend(seed=i)) for i, c in enumerate("AB")}
    res = bt.run_backtest(bt.SignalSet(frame=_signals(), periods=PER), bt.FramePrices(frames),
                          bt.strategy_from_dict({"exit": {"max_bars": 2}}))
    assert set(res.trades["period"]) == {"train", "val", "test"}
    assert (res.trades.loc[res.trades["period"] == "train", "time"] < pd.Timestamp("2026-01-20")).all()
    sep = bt.by_period(res.trades, 0.02)
    merged = bt.by_period(res.trades, 0.02, merge_val_test=True)
    assert list(sep) == ["train", "val", "test"] and list(merged) == ["train", "val", "test", "val+test"]
    assert merged["val+test"]["trades"] == sep["val"]["trades"] + sep["test"]["trades"]
    assert "train+val" not in merged and merged["train"]["trades"] == sep["train"]["trades"]
    stage = _signals().assign(stage="val")                     # a stage column wins over the dates
    assert set(bt.SignalSet(frame=stage, periods=PER).read()["period"]) == {"val"}
    with pytest.raises(ValueError):
        bt.Periods("2026-02-10", "2026-01-20")


def test_protocol_hides_test_of_the_variants_not_chosen_on_val():
    frames = {c: _bars(_trend(seed=i)) for i, c in enumerate("AB")}
    sig = bt.SignalSet(frame=_signals(), periods=PER)
    base = bt.strategy_from_dict({"exit": {"max_bars": 2}})
    table, runs = bt.ladder(sig, bt.FramePrices(frames), base, [("hold 6", {"exit.max_bars": 6}),
                                                                ("short", {"entry.side": "short"})])
    chosen = table.attrs["chosen"]
    val = table[table["period"] == "val"].set_index("variant")["final_equity"].astype(float)
    assert chosen == val.idxmax()
    test = table[table["period"] == "test"].set_index("variant")["final_equity"]
    assert test[chosen] != "hidden" and all(test[v] == "hidden" for v in test.index if v != chosen)
    assert (table[table["period"] == "train"]["final_equity"] != "hidden").all()
    off, _ = bt.ladder(sig, bt.FramePrices(frames), base, [("hold 6", {"exit.max_bars": 6})],
                       bt.Protocol(enabled=False))
    assert not (off == "hidden").any().any() and off.attrs["protocol"].startswith("off")
    with pytest.raises(ValueError, match="no 'val' period"):
        bt.compare(bt.SignalSet(frame=_signals()), bt.FramePrices(frames), {"a": base})


def test_cost_modes():
    frames = {"A": _bars(100 + np.arange(40.0))}
    sig = bt.SignalSet(frame=_signals(("A",), every=10, n=30))
    def run(costs):
        return bt.run_backtest(sig, bt.FramePrices(frames), bt.strategy_from_dict({"costs": costs})).trades
    none, fixed = run({"mode": "none"}), run({"mode": "fixed", "fee_per_side": 0.001})
    move = run({"mode": "move", "fee_per_side": 0.001, "move_share": 0.1})
    assert (none["net"] == none["gross"]).all() and np.allclose(fixed["cost"], 0.002)
    assert np.allclose(move["cost"], 0.002 + 0.1 * move["gross"].abs())
    with pytest.raises(KeyError, match="cost_mode"):
        bt.strategy_from_dict({"costs": {"mode": "percent"}})


def test_admission_strongest_first_and_the_quantile_side_rule():
    frames = {c: _bars(100 + np.arange(40.0)) for c in "ABCD"}
    t = T0 + 10 * H4
    sig = pd.DataFrame({"coin": list("ABCD"), "time": t, "score": [0.1, 0.4, -0.3, 0.2]})
    s = bt.strategy_from_dict({"book": {"max_open": 2, "admit": "strongest"}})
    res = bt.run_backtest(bt.SignalSet(frame=sig), bt.FramePrices(frames), s)
    assert sorted(res.trades["coin"]) == ["B", "C"] and res.skipped["overlap"] == 2
    q = bt.strategy_from_dict({"entry": {"side": "quantile", "side_params": {"q": 0.25}}})
    res = bt.run_backtest(bt.SignalSet(frame=sig, kind="rank"), bt.FramePrices(frames), q)
    sides = dict(zip(res.trades["coin"], res.trades["side"], strict=True))
    assert sides == {"B": 1.0, "C": -1.0}                       # top / bottom quarter of the cross-section


def test_missing_candles_are_reported_or_stop_the_run(tmp_path):
    frames = {"A": _bars(100 + np.arange(40.0))}
    sig = bt.SignalSet(frame=_signals(("A", "GONE"), every=10, n=30))
    res = bt.run_backtest(sig, bt.FramePrices(frames), bt.strategy_from_dict({}))
    assert res.skipped["no_prices"] == 3 and res.skipped["coins_without_prices"] == ["GONE"]
    bt.write_report(res, tmp_path, plots=False)
    assert "WARNING" in (tmp_path / "summary.md").read_text()
    with pytest.raises(ValueError, match="max_missing_share"):
        bt.run_backtest(sig, bt.FramePrices(frames), bt.strategy_from_dict({"max_missing_share": 0.1}))


def test_store_is_built_once_and_read_once(tmp_path):
    frames = {c: _bars(_trend(n=50, seed=i)) for i, c in enumerate("ABC")}
    meta = bt.build_store(bt.FramePrices(frames), ["A", "B", "C", "NOPE"], tmp_path / "s.parquet")
    assert meta["coins"] == ["A", "B", "C"] and meta["missing"] == ["NOPE"] and meta["interval"] == str(H4)
    assert bt.store_info(tmp_path / "s.parquet")["rows"] == 150
    px = bt.StorePrices(tmp_path / "s.parquet", coins=["B"])
    got = px.bars("B")
    assert (got.index == frames["B"].index).all() and np.allclose(got[["open", "high", "low", "close"]],
                                                                   frames["B"][["open", "high", "low", "close"]])
    with pytest.raises(KeyError):
        px.bars("A")


def _reference(bars, idx, h):
    rows = []
    for i in idx:
        rows.append({"coin": "A", "time": bars.index[i], "last_high": bars["high"].iloc[i],
                     "last_low": bars["low"].iloc[i], "last_close": bars["close"].iloc[i],
                     "future_close": bars["close"].iloc[i + h],
                     "future_high_max": bars["high"].iloc[i + 1:i + h + 1].max(),
                     "future_low_min": bars["low"].iloc[i + 1:i + h + 1].min()})
    return pd.DataFrame(rows)


def test_alignment_finds_the_matching_shift_and_works_on_finer_candles():
    fine = _bars(_trend(n=400, seed=3), step=pd.Timedelta(hours=1))
    coarse = fine.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    ref = _reference(coarse, range(5, 80, 3), 2)
    same = check_alignment(bt.FramePrices({"A": coarse}), ref, "8h", "4h")
    assert same["best_shift"] == 0 and same["per_shift"][0]["all"] == 1.0
    finer = check_alignment(bt.FramePrices({"A": fine}), ref, "8h", "4h")       # 1h candles, resampled to 4h
    assert finer["per_shift"][0]["all"] == 1.0
    late = ref.assign(time=ref["time"] - H4)                     # a source one bar late
    assert check_alignment(bt.FramePrices({"A": coarse}), late, "8h", "4h")["best_shift"] == 1


def test_bridge_alignment_reference_and_periods_on_the_fixture(art):  # noqa: F811
    from nig_backtest.bridges import nig_timenet_torch as bridge
    from nig_timenet_torch.data_io import npy_dir
    ds = npy_dir.load(art)
    ref = bridge.alignment_reference(ds)
    coin = ref["coin"].iloc[0]
    assert set(bt.align.REFERENCE_COLUMNS) <= set(ref.columns)
    res = check_alignment(bridge.artifact_prices(ds), ref[ref["coin"] == coin], bridge.horizon_of(ds),
                          str(int(ds["stride"]) * pd.Timedelta(ds["base_tf"])), shifts=(0,))
    assert res["per_shift"][0]["last_close"] == 1.0          # the artifact's own candles carry its own last close
