"""nig_backtest/app.py: the pure functions behind the web page (signal filters, controls -> strategy, a run whose
TEST is hidden until asked, trade filters, figures) and the page itself (built, not launched; needs gradio)."""

import numpy as np
import pandas as pd
import pytest

import nig_backtest as bt
from nig_backtest import app

T0 = pd.Timestamp("2026-01-01")
H4 = pd.Timedelta(hours=4)


def _bars(seed):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(400)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998, "close": c},
                        index=T0 + H4 * np.arange(400))


@pytest.fixture()
def setup():
    rng = np.random.default_rng(1)
    frames = {c: _bars(i) for i, c in enumerate(("AAA", "BBB", "CCC"))}
    rows = [(c, T0 + i * H4, float(rng.normal()), 0.02) for c in frames for i in range(5, 380, 2)]
    sig = bt.SignalSet(frame=pd.DataFrame(rows, columns=["coin", "time", "score", "sigma"]), horizon="8h",
                       periods=bt.Periods("2026-01-20", "2026-02-10"))
    return sig, bt.FramePrices(frames)


def test_filter_signals(setup):
    sig, _ = setup
    rows = sig.read()
    f = app.filter_signals(rows, coins=["AAA"], periods=["val"], side="short", min_strength=0.5,
                           start="2026-01-25", end="2026-02-05")
    assert len(f) and set(f["coin"]) == {"AAA"} and set(f["period"]) == {"val"} and (f["dir"] < 0).all()
    assert (f["strength"] >= 0.5).all() and f["strength"].is_monotonic_decreasing
    assert len(app.filter_signals(rows)) == len(rows)


def test_controls_to_strategy():
    s = app.controls_to_strategy({})
    assert s.entry.top_share == 0.1 and s.exit.hold == "horizon" and s.exit.take_profit is None
    assert s.book.per_coin == 1 and s.book.max_open is None
    q = app.controls_to_strategy({"side": "quantile", "quantile_q": 0.3, "top_share": 1.0, "take_profit": 1.5,
                                  "hold": "", "max_bars": 4, "per_coin": 0, "max_open": 5,
                                  "filters": ["min_column"], "filter_params": '{"min_column": {"min": 0.1}}'})
    assert q.entry.side_params == {"q": 0.3} and q.entry.top_share is None and q.exit.take_profit == 1.5
    assert q.exit.hold is None and q.exit.max_bars == 4 and q.book.per_coin is None and q.book.max_open == 5
    assert q.entry.filter_params == {"min_column": {"min": 0.1}}


def test_run_view_hides_test_until_asked(setup):
    sig, px = setup
    hidden = app.run_view(sig, px, {"take_profit": 1.0})
    test_row = hidden["summary"].set_index("period").loc["test"]
    assert test_row["trades"] == "hidden" and "test" not in set(hidden["trades"]["period"])
    assert "TEST is hidden" in hidden["message"] and hidden["figure"] is not None
    shown = app.run_view(sig, px, {"take_profit": 1.0}, show_test=True, merge_val_test=True)
    rows = shown["summary"].set_index("period")
    assert rows.loc["test", "trades"] != "hidden" and "val+test" in rows.index
    assert "test" in set(shown["trades"]["period"])


def test_filter_trades_and_figures(setup):
    sig, px = setup
    trades = app.run_view(sig, px, {}, show_test=True)["trades"]
    worst = app.filter_trades(trades, coins=["BBB"], outcome="loss", period=["val"], order="worst")
    assert set(worst["coin"]) == {"BBB"} and (worst["net"] <= 0).all() and worst["net"].is_monotonic_increasing
    assert app.filter_trades(pd.DataFrame()).empty
    fig = app.signal_figure(sig.read().iloc[10], px, "8h")
    assert fig.axes and "realised over" in fig.axes[0].get_title()
    assert bt.TradeBook(trades).figure(0, px).axes


def test_the_page_builds(setup):
    pytest.importorskip("gradio")
    sig, px = setup
    page = app.build_app(sig, px, title="test")
    assert page is not None and hasattr(page, "launch")
