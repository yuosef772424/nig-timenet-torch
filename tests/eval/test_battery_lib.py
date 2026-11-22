"""eval/battery_lib.py and eval/day_blocks.py: the one implementation of cost, direction, top-|x| selection, the
multiple-testing level and the day blocks; guards (rule B3) that no module re-types them."""

import ast

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval import battery_lib as bl
from nig_timenet_torch.eval import day_blocks as db
from nig_timenet_torch.workflow.settings import BatterySettings
from tests import scan


def test_net_pnl_is_gross_minus_one_round_trip():
    assert bl.net_pnl(0.01, 0.0008) == pytest.approx(0.01 - 2 * 0.0008)
    assert np.allclose(bl.net_pnl(np.array([0.0, 0.01]), 0.001), [-0.002, 0.008])
    assert bl.round_trip(0.0008) == bl.SIDES * 0.0008


def test_bonferroni_level_divides_alpha_by_the_number_of_tests():
    assert bl.bonferroni_level(1) == pytest.approx(0.95)
    assert bl.bonferroni_level(5) == pytest.approx(1.0 - 0.05 / 5)
    assert bl.bonferroni_level(4, alpha=0.1) == pytest.approx(0.975)
    with pytest.raises(ValueError):
        bl.bonferroni_level(0)


def test_trade_side_is_the_close_class_head():
    df = pd.DataFrame({"p_dir": [0.9, 0.5, 0.1, 0.5000001]})
    assert bl.trade_side(df).tolist() == [1.0, -1.0, -1.0, 1.0]


def test_top_abs_is_stable_on_ties_and_respects_the_mask():
    values = np.array([0.5, -0.5, 0.2, 0.5, -0.5, 0.1])
    assert bl.top_abs(values, 0.5).tolist() == [0, 1, 3]                  # ties keep row order
    assert bl.top_abs(values, 0.5, np.array([0, 0, 1, 1, 1, 1], bool)).tolist() == [3, 4]
    assert len(bl.top_abs(values, 0.0)) == 1                              # at least one row


def test_trade_stats_and_the_row_text_read_the_same_minimum():
    assert bl.trade_stats(np.ones(bl.MIN_TRADES - 1)) is None
    assert bl.trade_stats(np.arange(bl.MIN_TRADES))["n"] == bl.MIN_TRADES
    assert bl.format_row("x", None).endswith(f"(n<{bl.MIN_TRADES})")


def test_limit_at_extreme_fills_only_when_the_extreme_reaches_the_order():
    pred = np.array([-0.01, -0.01, 0.02])                                 # a buy at -1 %, a buy at -1 %, not a long
    fut = np.array([-0.02, -0.005, -0.03])                                # the low reaches the first order only
    close = np.array([0.01, 0.0, 0.0])
    pnl, fill, valid = bl.limit_at_extreme(1.0, pred, fut, close, 0.0008)
    assert fill.tolist() == [True, False, False] and valid.tolist() == [True, True, False]
    assert pnl.tolist() == pytest.approx([1.01 / 0.99 - 1 - 2 * 0.0008])


def test_day_blocks_one_day_per_block_and_five_day_blocks():
    t = pd.to_datetime(["2026-01-01 03:00", "2026-01-01 23:00", "2026-01-02 01:00", "2026-01-08 00:00"])
    days = db.calendar_day(t)
    assert days.dtype == np.dtype("datetime64[D]") and days[0] == days[1] != days[2]
    assert db.day_blocks(days, 1).tolist() == [0, 0, 1, 7]
    assert db.day_blocks(days, db.DAY_BLOCK_DAYS).tolist() == [0, 0, 0, 1]
    assert db.day_blocks(days[:0], 1).tolist() == []
    ns = t.values.astype("datetime64[ns]").astype(np.int64)
    assert (db.calendar_day(ns) == days).all()                            # integers are nanoseconds (the gates)


def test_the_cost_default_has_one_home():
    """Rule B3: the run settings read the one default (``core/costs.py``, re-exported by battery_lib), and no float
    0.0008 is typed anywhere else in the package (a new place that needs the cost per side imports it)."""
    assert BatterySettings().cost_per_side == bl.DEFAULT_COST_PER_SIDE == 0.0008
    found = [f"{rel}:{n.lineno}" for rel, (_, tree) in scan.sources().items() if rel != "core/costs.py"
             for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)
             and n.value == bl.DEFAULT_COST_PER_SIDE]
    assert not found, "0.0008 typed instead of core.costs.DEFAULT_COST_PER_SIDE: " + ", ".join(found)
    text = [rel for rel, (src, _) in scan.sources().items() if rel != "core/costs.py" and "0.0008" in src]
    assert not text, "0.0008 written in " + ", ".join(text)


def test_the_multiple_testing_level_and_day_constants_are_defined_once():
    """The level ``1 - 0.05 / k`` and the 5-day block are not re-derived: no module but battery_lib assigns ALPHA /
    MIN_TRADES / SIDES and none but day_blocks assigns DAY_BLOCK_DAYS."""
    homes = {"ALPHA": "eval/battery_lib.py", "MIN_TRADES": "eval/battery_lib.py", "SIDES": "eval/battery_lib.py",
             "DAY_BLOCK_DAYS": "eval/day_blocks.py"}
    found = []
    for rel, (_, tree) in scan.sources().items():
        for n in ast.walk(tree):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
            found += [f"{rel}:{n.lineno}: {t.id}" for t in targets
                      if isinstance(t, ast.Name) and homes.get(t.id, rel) != rel]
    assert not found, found


def test_take_profit_at_atr_targets_k_atr_on_the_trade_side_and_falls_back_to_the_close():
    """Owner 2026-11-22: a take-profit at k x ATR (log units per bar) on the trade's side, else exit at the close."""
    import pandas as pd
    atr = np.array([0.01, 0.01, 0.02, np.nan])
    df = pd.DataFrame({"atr": atr, "y_ret": [0.001, -0.004, 0.002, 0.003],
                       "fut_high": [0.0102, 0.004, 0.015, 0.01], "fut_low": [-0.001, -0.0105, -0.015, -0.01]})
    idx = np.arange(4)
    lo = bl.take_profit_at_atr(df, idx, 1.0, 1.0, 0.0)
    assert lo["path"] and lo["level"][0] == pytest.approx(np.expm1(0.01))
    assert lo["hit"].tolist() == [True, False, False, False]            # row 3 has no ATR: no target
    assert lo["pnl"].tolist() == pytest.approx([np.expm1(0.01), -0.004, 0.002, 0.003])
    sh = bl.take_profit_at_atr(df, idx, -1.0, 1.0, 0.0008)
    assert sh["hit"].tolist() == [False, True, False, False] and sh["target"].tolist() == [True, True, True, False]
    assert sh["pnl"][1] == pytest.approx(-np.expm1(-0.01) - 2 * 0.0008)   # the short's gain at its target
    close_only = bl.take_profit_at_atr(df.drop(columns=["fut_high", "fut_low"]), idx, 1.0, 0.5, 0.0)
    assert not close_only["path"] and not close_only["hit"].any()          # no close reaches half an ATR here


def test_battery_settings_n8_switches_default_to_the_old_behaviour_and_refuse_unknown_values():
    b = BatterySettings()
    assert (b.verdict_trim, b.drop_flat, b.account_books) == ("best", True, 0)
    with pytest.raises(ValueError, match="verdict_trim"):
        BatterySettings(verdict_trim="worst")
    with pytest.raises(ValueError, match="account_books"):
        BatterySettings(account_books=-1)
