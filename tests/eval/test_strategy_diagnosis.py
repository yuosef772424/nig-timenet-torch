"""eval/strategy_diagnosis.py on trades with known answers."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval import strategy_diagnosis as sd


def trades():
    days = np.repeat(pd.date_range("2026-01-01", periods=10).values.astype("datetime64[D]"), 10)
    pnl = np.tile([0.01] * 8 + [-0.05, -0.02], 10)
    return pd.DataFrame({"pnl": pnl, "hit": (pnl > 0).astype(float), "side": -1.0,
                         "coin": np.tile([f"C{i}" for i in range(10)], 10), "day": days,
                         "mae": np.abs(np.minimum(pnl, 0)) + 0.001, "pred_move": 0.02,
                         "real_move": np.where(pnl > 0, 0.02, 0.005)})


def test_payoff_and_break_even():
    p = sd.payoff(trades())
    assert p["win"] == pytest.approx(0.8) and p["avg_win"] == pytest.approx(0.01)
    assert p["avg_loss"] == pytest.approx(-0.035) and p["breakeven_win"] == pytest.approx(0.035 / 0.045)
    assert p["hits"]["n"] == 80 and p["misses"]["pts"] == pytest.approx(-70.0)
    assert p["worst1_share_of_losses"] == pytest.approx(0.05 / 0.7)


def test_days_market_link_and_calibration():
    tr = trades()
    days = np.unique(tr["day"])
    market = pd.Series(np.linspace(-0.01, 0.01, 10), index=days)
    tr["pnl"] = tr["pnl"] + np.repeat(2 * market.values, 10)  # strategy day mean = 2 x market + const
    d = sd.diagnose(tr, market, -market)
    assert d["market"]["slope"] == pytest.approx(2.0) and d["market"]["corr"] == pytest.approx(1.0)
    dy = d["days"]
    assert dy["n_days"] == 10 and dy["worst"][0]["day"] == str(days[0])[:10]
    assert dy["without_both"] == pytest.approx(dy["total_pts"] - sum(r["total_pts"] for r in dy["worst"] + dy["best"]))
    c = d["calibration"]
    assert c["reached_full"] == pytest.approx(0.8) and c["reached_half"] == pytest.approx(0.8)
    assert "break-even" in sd.render("x", d)
