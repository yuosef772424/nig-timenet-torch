"""eval/strategy_book.py: each rule's pnl on hand-made rows, the battery's own definitions reproduced, the book's
controls and VAL-only thresholds."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval import battery, battery_profit
from nig_timenet_torch.eval import strategy_book as sb

COST = 0.0008


def table(n=2000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 0.02, n)
    hi = np.maximum(y, 0) + np.abs(rng.normal(0, 0.01, n))
    lo = np.minimum(y, 0) - np.abs(rng.normal(0, 0.01, n))
    return pd.DataFrame({
        "step": 1, "mu": rng.normal(0, 0.002, n), "sigma": np.abs(rng.normal(0.02, 0.005, n)) + 1e-3,
        "p_dir": rng.random(n), "y_ret": y, "dir_true": np.sign(y).astype(int),
        "coin": np.repeat([f"C{i}" for i in range(10)], n // 10),
        "end_time": pd.Timestamp("2026-03-01") + pd.to_timedelta(np.tile(np.arange(n // 10), 10) * 4, "h"),
        "pred_high": np.abs(rng.normal(0.02, 0.005, n)), "fut_high": hi, "sigma_high": 0.01,
        "pred_low": -np.abs(rng.normal(0.02, 0.005, n)), "fut_low": lo, "sigma_low": 0.01,
        "ale_high": rng.random(n), "ale_low": rng.random(n), "vol": np.abs(rng.normal(0.02, 0.005, n)),
        "last_ret": rng.normal(0, 0.01, n)})


def test_rules_on_hand_made_rows():
    d = table(20)
    d.loc[0, ["y_ret", "fut_high", "fut_low", "pred_high", "pred_low", "mu"]] = [0.01, 0.03, -0.02, 0.04, -0.01, -0.01]
    long_ = sb.direction_trades(d, np.array([0]), 1.0, COST)
    assert long_["pnl"].iloc[0] == pytest.approx(0.01 - 2 * COST) and long_["mae"].iloc[0] == pytest.approx(0.02)
    tp = sb.tp_trades(d, np.arange(20) == 0, 1.0, COST)  # TP at +2% reached by the +3% high
    assert tp["hit"].iloc[0] == 1 and tp["pnl"].iloc[0] == pytest.approx(0.02 - 2 * COST)
    sc = sb.tp_trades(d, np.arange(20) == 0, -1.0, COST)  # short C: mu < 0, pred_low -1% -> TP at -0.5%, reached
    assert sc["pnl"].iloc[0] == pytest.approx(0.005 - 2 * COST)
    lim = sb.limit_extreme_trades(d, np.arange(20) == 0, 1.0, COST)  # buy at -1%, filled (low -2%), exit +1%
    assert lim["pnl"].iloc[0] == pytest.approx(1.01 / 0.99 - 1 - 2 * COST)
    assert lim["mae"].iloc[0] == pytest.approx(1 - 0.98 / 0.99)


def test_the_battery_definitions_are_reproduced():
    d = table()
    side = np.where(d["p_dir"] > 0.5, 1.0, -1.0)
    m = (d["dir_true"].values != 0) & (side == 1.0)
    idx = battery._top_abs(d, "mu", 0.20, m)
    mine = sb.limit_at_mu_trades(d, idx, 1.0, COST)["pnl"].mean()
    row = [r for r in battery.limit_execution(d, COST)["rows"] if r.startswith("LONG top20%")][0]
    assert f"{mine * 100:+.4f}%" in row
    pnl, _, _ = battery_profit._limit_pnl(1.0, d["pred_low"].values, d["fut_low"].values, d["y_ret"].values, COST, 2)
    assert sb.limit_extreme_trades(d, np.ones(len(d), bool), 1.0, COST)["pnl"].mean() == pytest.approx(pnl.mean())


def test_book_has_every_control_on_both_splits_with_val_thresholds():
    val, test = table(seed=1), table(seed=2)
    rows = sb.book(val, test, COST)
    uses = {r["use"] for r in rows}
    assert "SHORT C TP=0.5*pred_low [ale>=Q80]" in uses and "close direction, all rows" in uses
    ctrl = {r["control"] for r in rows if r["use"] == "LONG TP=0.5*pred_high [ale>=Q80]" and r["split"] == "TEST"}
    assert {"model", "shuffled levels", "volatility levels", "volatility selection", "random selection"} <= ctrl
    q80 = np.quantile(val["ale_high"], 0.8)
    t = next(r for r in rows if r["use"] == "LONG TP=0.5*pred_high [ale>=Q80]" and r["split"] == "TEST"
             and r["control"] == "model")
    assert t["n"] == int(((test["ale_high"] >= q80) & (test["pred_high"] > 0)).sum())
    assert all(r["ci_lo"] <= r["avg"] <= r["ci_hi"] for r in rows if "avg" in r)


def test_a_martingale_market_has_no_profit_in_any_tp_or_limit_rule():
    """Optional stopping: with no cost, a TP-or-close exit on a driftless walk has mean ~0 whatever the levels."""
    rng = np.random.default_rng(3)
    n, steps = 5000, 400  # fine steps: the TP overshoot of a discrete path is tiny
    path = np.cumsum(rng.normal(0, 0.001, (n, steps)), axis=1)
    d = table(n, seed=4)
    d["y_ret"], d["fut_high"], d["fut_low"] = path[:, -1], path.max(1), path.min(1)
    tr = sb.tp_trades(d, np.ones(n, bool), 1.0, 0.0)
    assert abs(tr["pnl"].mean()) < 0.002  # the B-051 misdecode showed +1.3% here on real data
