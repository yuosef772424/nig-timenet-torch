"""The backtest package on hand-made candles with known answers: entry after the signal, time exit after N bars,
take-profit / stop in ATR and % (gap fills, stop first), one trade per coin, the open-position cap, the contract
(unknown columns, optional features absent / switched off), custom rules by name, the explorer, the rule ladder,
parquet batches, the training labels as an oracle signal, the CLI and the battery adapter."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nig_backtest as bt
from nig_backtest.__main__ import main
from nig_backtest._base import UnknownNameError

T0 = pd.Timestamp("2026-01-01")
H4 = pd.Timedelta(hours=4)


def _bars(closes, rng=0.0, opens=None, highs=None, lows=None):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]] if opens is None else np.asarray(opens, float)
    h = np.maximum(o, c) + rng if highs is None else np.asarray(highs, float)
    lo = np.minimum(o, c) - rng if lows is None else np.asarray(lows, float)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1.0},
                        index=T0 + H4 * np.arange(len(c)))


def _signals(rows):
    return pd.DataFrame(rows, columns=["coin", "time", "score", *(["sigma"] if len(rows[0]) > 3 else [])])


def _run(frames, rows, **strategy):
    s = bt.strategy_from_dict({"costs": {"fee_per_side": 0.0}, **strategy})
    return bt.run_backtest(bt.SignalSet(frame=_signals(rows)), bt.FramePrices(frames), s)


def test_time_exit_after_n_bars_enters_at_the_next_open():
    closes = 100 + np.arange(40.0)                      # +1 per bar, open = previous close
    res = _run({"A": _bars(closes)}, [("A", T0 + 20 * H4, 1.0)], exit={"max_bars": 3})
    tr = res.trades.iloc[0]
    assert tr["entry_time"] == T0 + 20 * H4 and tr["entry_price"] == closes[19]     # the bar opening at the signal
    assert tr["exit_reason"] == "time" and tr["bars_held"] == 3 and tr["exit_price"] == closes[22]
    assert tr["exit_time"] == T0 + 23 * H4 and tr["gross"] == pytest.approx(closes[22] / closes[19] - 1)
    short = _run({"A": _bars(closes)}, [("A", T0 + 20 * H4, -1.0)], exit={"max_bars": 3}).trades.iloc[0]
    assert short["gross"] == pytest.approx(-(closes[22] / closes[19] - 1)) and not short["win"]


def test_signal_close_entry_uses_the_close_at_the_signal_time():
    closes = 100 + np.arange(40.0)
    tr = _run({"A": _bars(closes)}, [("A", T0 + 20 * H4, 1.0)],
              entry={"entry_price": "signal_close"}, exit={"max_bars": 2}).trades.iloc[0]
    assert tr["entry_price"] == closes[19] and tr["exit_price"] == closes[21]


def test_take_profit_in_atr_units_and_a_gap_fills_at_the_open():
    closes = np.full(40, 100.0)
    closes[22:] = 103.0                                  # +3 at bar 22; ATR of the flat part = 2 x rng = 1.0
    res = _run({"A": _bars(closes, rng=0.5)}, [("A", T0 + 20 * H4, 1.0)],
               exit={"max_bars": 5, "take_profit": 2.0, "take_profit_unit": "atr"})
    tr = res.trades.iloc[0]
    assert tr["atr_entry"] == pytest.approx(1.0) and tr["target_frac"] == pytest.approx(0.02)
    assert tr["exit_reason"] == "take_profit" and tr["bars_held"] == 3
    assert tr["exit_price"] == pytest.approx(102.0)      # the bar opens at 100 and reaches 103.5: filled at the level
    gap = np.full(40, 100.0)
    gap[22:] = 110.0
    opens = np.r_[gap[0], gap[:-1]]
    opens[22] = 110.0                                    # gap open above the target
    tr = _run({"A": _bars(gap, rng=0.5, opens=opens)}, [("A", T0 + 20 * H4, 1.0)],
              exit={"max_bars": 5, "take_profit": 2.0}).trades.iloc[0]
    assert tr["exit_price"] == pytest.approx(110.0)


def test_stop_loss_in_pct_and_the_stop_counts_first_when_one_bar_touches_both():
    closes = np.full(40, 100.0)
    highs, lows = closes + 0.1, closes - 0.1
    highs[21], lows[21] = 106.0, 94.0                    # one bar through +5 % and -5 %
    frames = {"A": _bars(closes, highs=highs, lows=lows)}
    rule = {"max_bars": 5, "take_profit": 0.05, "take_profit_unit": "pct", "stop_loss": 0.05, "stop_loss_unit": "pct"}
    tr = _run(frames, [("A", T0 + 20 * H4, 1.0)], exit=rule).trades.iloc[0]
    assert tr["exit_reason"] == "stop_loss" and tr["gross"] == pytest.approx(-0.05)
    tr = _run(frames, [("A", T0 + 20 * H4, 1.0)], exit={**rule, "both_hit": "target_first"}).trades.iloc[0]
    assert tr["exit_reason"] == "take_profit" and tr["gross"] == pytest.approx(0.05)


def test_data_end_one_per_coin_and_the_open_cap():
    closes = 100 + np.arange(30.0)
    frames = {"A": _bars(closes), "B": _bars(closes)}
    rows = [("A", T0 + 20 * H4, 1.0), ("A", T0 + 21 * H4, 1.0), ("A", T0 + 28 * H4, 1.0), ("B", T0 + 20 * H4, 1.0)]
    res = _run(frames, rows, exit={"max_bars": 3})
    assert res.skipped["overlap"] == 1                   # A at bar 21 opens while A from bar 20 is held
    assert set(res.trades["exit_reason"]) == {"time", "data_end"}
    assert len(_run(frames, rows, exit={"max_bars": 3}, book={"max_open": 1}).trades) == 2   # B opens with A
    res2 = _run(frames, rows, book={"per_coin": None}, exit={"max_bars": 3})
    assert len(res2.trades) == 4 and res2.skipped["overlap"] == 0
    res3 = _run(frames, rows, book={"per_coin": 2}, exit={"max_bars": 3})    # two A trades may overlap
    assert len(res3.trades) == 4


def test_contract_and_optional_features():
    good = _signals([("A", T0, 0.1, 0.02)])
    with pytest.raises(ValueError, match="unknown"):
        bt.SignalSet(frame=good.assign(mystery=1.0))
    assert "x_note" in bt.SignalSet(frame=good.assign(x_note=1.0)).read().columns
    s = bt.strategy_from_dict({"entry": {"score_over_sigma": 1.0}})
    frames = {"A": _bars(100 + np.arange(30.0))}
    bt.run_backtest(bt.SignalSet(frame=good), bt.FramePrices(frames), s)              # sigma present: runs
    with pytest.raises(ValueError, match="needs the features"):
        bt.run_backtest(bt.SignalSet(frame=good, use=()), bt.FramePrices(frames), s)  # sigma switched off
    with pytest.raises(ValueError, match="confidence"):
        bt.run_backtest(bt.SignalSet(frame=good), bt.FramePrices(frames),
                        bt.strategy_from_dict({"entry": {"min_confidence": 0.5}}))   # absent
    with pytest.raises(ValueError, match="absent"):
        bt.SignalSet(frame=good, use=("confidence",))
    with pytest.raises(ValueError):
        bt.strategy_from_dict({"exit": {"take_profit_unit": "ticks"}})
    with pytest.raises(ValueError, match="unknown key"):
        bt.strategy_from_dict({"exit": {"takeprofit": 1}})


def test_entry_rules_trace_top_share_per_stage_and_custom_filters():
    rng = np.random.default_rng(0)
    n = 200
    sig = pd.DataFrame({"coin": "A", "time": T0 + H4 * np.arange(n), "score": rng.normal(size=n),
                        "stage": np.where(np.arange(n) < 100, "val", "test"), "x_window_vol": rng.random(n)})
    s = bt.strategy_from_dict({"entry": {"top_share": 0.1, "filters": ["min_column"],
                                         "filter_params": {"min_column": {"min": 0.5}}}})
    picked = bt.rules.select_entries(bt.SignalSet(frame=sig).read(), s)
    assert (picked["x_window_vol"] >= 0.5).all()
    kept = int((sig["x_window_vol"] >= 0.5).sum())
    assert abs(len(picked) - 0.1 * kept) <= 2 and set(picked["stage"]) == {"val", "test"}
    assert [k for k, _ in picked.attrs["rule_trace"]] == ["signals", "side", "filter:min_column", "top_share"]
    with pytest.raises(UnknownNameError):
        bt.strategy_from_dict({"entry": {"filters": ["no_such_rule"]}})


@pytest.mark.slow
def test_the_training_label_as_a_signal_is_an_oracle_and_the_explorer_filters_it(tmp_path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(1)
    frames, rows = {}, []
    for coin in ("A", "B", "C"):
        closes = 100 * np.exp(np.cumsum(0.02 * rng.standard_normal(300)))
        frames[coin] = _bars(closes)
        for i in range(10, 290, 5):                       # label = the 2-bar move after the next open
            move = closes[i + 1] / closes[i - 1] - 1
            rows.append((coin, T0 + i * H4, move, move, "train"))
    sig = pd.DataFrame(rows, columns=["coin", "time", "score", "y_true", "stage"])
    res = bt.run_backtest(bt.SignalSet(frame=sig), bt.FramePrices(frames),
                          bt.strategy_from_dict({"costs": {"fee_per_side": 0.0}, "exit": {"max_bars": 2}}))
    book = bt.TradeBook(res.trades)
    assert len(book) == len(rows) and book.trades["direction_right"].all() and book.trades["label_right"].all()
    assert len(book.where(coins="A")) == len(rows) // 3 and len(book.where(outcome="loss")) == 0
    assert len(book.where(exit_reason="time", side="short", period="train")) == int((sig["score"] < 0).sum())
    assert set(book.by("coin")["coin"]) == {"A", "B", "C"}
    path = book.plot(0, bt.FramePrices(frames), tmp_path)
    assert path.is_file()
    periods = bt.write_report(res, tmp_path / "rep", prices=bt.FramePrices(frames), gallery=2, batch_coins=2)
    assert list(periods) == ["train"] and periods["train"]["final_equity"] > 1
    assert (tmp_path / "rep" / "summary.md").is_file()
    for batch in ("batch_000", "batch_001"):              # coins A, B | C: the charts are drawn per batch
        assert len(list((tmp_path / "rep" / batch / "worst").glob("*.png"))) == 2
        assert (tmp_path / "rep" / batch / "equity.png").is_file()


def test_ladder_compare_and_unknown_change_keys():
    closes = 100 + np.cumsum(np.r_[0, np.tile([1.0, -0.5], 60)])
    frames = {"A": _bars(closes, rng=0.3)}
    sig = bt.SignalSet(frame=_signals([("A", T0 + i * H4, 1.0) for i in range(20, 100, 4)]))
    base = bt.strategy_from_dict({"exit": {"max_bars": 3}})
    off = bt.Protocol(enabled=False)
    table, runs = bt.ladder(sig, bt.FramePrices(frames), base,
                            [("tp", {"exit.take_profit": 0.5}), ("stop", {"exit.stop_loss": 0.5})], off)
    assert list(table["variant"]) == ["base", "tp", "stop"] and table.attrs["variants_tried"] == 3
    assert set(table["period"]) == {"all"} and table.attrs["protocol"].startswith("off")
    assert runs["stop"].strategy.exit.take_profit == 0.5 and runs["stop"].strategy.exit.stop_loss == 0.5
    loo, _ = bt.leave_one_out(sig, bt.FramePrices(frames), runs["stop"].strategy, [("stop", {"exit.stop_loss": None})],
                              off)
    assert list(loo["variant"]) == ["full", "without stop"]
    with pytest.raises(KeyError):
        bt.apply_changes(base, {"exit.target": 1})


def test_parquet_signals_in_batches_and_the_cli(tmp_path):
    closes = 100 + np.arange(60.0)
    root = tmp_path / "candles"
    root.mkdir()
    for coin in ("A", "B", "C"):
        _bars(closes).to_parquet(root / f"{coin}.parquet")
    sig = pd.DataFrame({"coin": np.repeat(["A", "B", "C"], 5), "time": np.tile(T0 + H4 * np.arange(10, 60, 10), 3),
                        "score": 1.0, "stage": "test"})
    path = tmp_path / "s.parquet"
    sig.to_parquet(path)
    ss = bt.SignalSet(path=str(path), stages=("test",))
    assert [c for c, _ in ss.batches(size=2)] == [["A", "B"], ["C"]] and len(ss.read(["B"])) == 5
    (tmp_path / "s.json").write_text(json.dumps({"name": "t", "exit": {"max_bars": 2}}))
    main(["run", "--signals", str(path), "--prices", f"files:{root}:{{coin}}.parquet", "--strategy",
          str(tmp_path / "s.json"), "--out", str(tmp_path / "o"), "--gallery", "0"])
    assert len(pd.read_parquet(tmp_path / "o" / "trades.parquet")) == 15
    (tmp_path / "steps.json").write_text(json.dumps([["tp", {"exit.take_profit": 0.01}]]))
    main(["ladder", "--signals", str(path), "--prices", f"files:{root}:{{coin}}.parquet", "--strategy",
          str(tmp_path / "s.json"), "--steps", str(tmp_path / "steps.json"), "--out", str(tmp_path / "l"),
          "--no-protocol"])
    assert "tp" in (tmp_path / "l" / "comparison.md").read_text()


def test_battery_adapter_maps_the_columns():
    table = pd.DataFrame({"mu": [0.01, -0.02], "sigma": [0.02, 0.03], "p_dir": [0.6, 0.3], "y_ret": [0.01, 0.0],
                          "coin": ["A", "B"], "end_time": [T0, T0], "pred_high": [0.03, 0.01],
                          "pred_low": [-0.01, -0.04]})
    from nig_backtest.bridges.nig_timenet_torch import from_battery_table
    sig = from_battery_table(table, stage="test")
    assert list(sig["score"]) == [0.01, -0.02] and list(sig["p_up"]) == [0.6, 0.3] and set(sig["stage"]) == {"test"}
    assert (sig["pred_high"] >= 0).all() and (sig["pred_low"] <= 0).all()


def _imports(path, top_level_only: bool):
    tree = ast.parse(path.read_text())
    nodes = tree.body if top_level_only else list(ast.walk(tree))
    out = set()
    for n in nodes:
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module.split(".")[0])
        elif isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
    return out


def test_the_backtest_is_a_separate_package():
    """nig_backtest moves to any project as a folder: no module imports nig_timenet_torch except the bridges, and
    those only inside their functions; nig_timenet_torch never imports nig_backtest."""
    src = Path(bt.__file__).parent
    for path in src.rglob("*.py"):
        bridge = "bridges" in path.relative_to(src).parts
        assert "nig_timenet_torch" not in _imports(path, top_level_only=bridge), path
    host = src.parent / "nig_timenet_torch"
    assert not [p for p in host.rglob("*.py") if "nig_backtest" in _imports(p, top_level_only=False)]
