"""PURPOSE: The effect of each strategy decision on the portfolio: several strategies side by side, a ladder that adds
    one rule at a time to a base, and leave-one-out (each rule removed from the full strategy), on the same signals.
TAGS: rule impact, strategy comparison, ablation, ladder, leave one out, multiple testing, portfolio effect, backtest
PITFALLS: every variant runs on the SAME SignalSet and PriceSource (candles are cached, so the variants cost little
    more than one run); a change is a dotted key of the strategy (``"exit.take_profit": 1.5``); the table counts the
    variants tried: a best variant picked from many is a hypothesis for unseen data, not a result.
"""

import dataclasses

import pandas as pd

from nig_backtest.contract import SignalSet
from nig_backtest.engine import Backtest, run_backtest
from nig_backtest.portfolio import summary
from nig_backtest.prices import PriceSource
from nig_backtest.rules import Strategy, strategy_from_dict, strategy_to_dict

__all__ = ["apply_changes", "compare", "ladder", "leave_one_out", "ROW_KEYS"]

ROW_KEYS = ("trades", "final_equity", "max_drawdown", "positive_days")


def apply_changes(base: Strategy, changes: dict, name: str | None = None) -> Strategy:
    """``base`` with dotted-key changes (``{"entry.top_share": 0.1}``); unknown keys raise."""
    d = strategy_to_dict(base)
    for key, value in changes.items():
        obj, node = base, d
        *path, leaf = key.split(".")
        for p in path:
            if not dataclasses.is_dataclass(obj) or p not in {f.name for f in dataclasses.fields(obj)}:
                raise KeyError(f"unknown strategy key {key!r}")
            obj, node = getattr(obj, p), node.setdefault(p, {})
        if not dataclasses.is_dataclass(obj) or leaf not in {f.name for f in dataclasses.fields(obj)}:
            raise KeyError(f"unknown strategy key {key!r}")
        node[leaf] = value
    if name is not None:
        d["name"] = name
    return strategy_from_dict(d)


def _row(name: str, bt: Backtest) -> dict:
    s = summary(bt.trades, bt.strategy.book.risk_per_trade, bt.strategy.book.max_open)
    pt = s["per_trade"]
    return {"variant": name, **{k: s[k] for k in ROW_KEYS}, "win_rate": pt.get("win_rate"),
            "mean_net": pt.get("mean"), "sharpe_per_trade": pt.get("sharpe"), "profit_factor": pt.get("profit_factor")}


def compare(signals: SignalSet, prices: PriceSource, strategies: dict) -> tuple[pd.DataFrame, dict]:
    """(table, {name: Backtest}) for named strategies on the same signals and candles."""
    runs = {name: run_backtest(signals, prices, s) for name, s in strategies.items()}
    table = pd.DataFrame([_row(n, bt) for n, bt in runs.items()])
    table.attrs["variants_tried"] = len(runs)
    return table, runs


def ladder(signals: SignalSet, prices: PriceSource, base: Strategy, steps: list) -> tuple[pd.DataFrame, dict]:
    """Add one decision at a time: ``steps`` = [(label, {dotted key: value}), ...]; row i = base + steps[:i]."""
    variants, current = {"base": base}, base
    for label, changes in steps:
        current = apply_changes(current, changes, name=label)
        variants[label] = current
    table, runs = compare(signals, prices, variants)
    table.insert(1, "added", ["-"] + [", ".join(f"{k}={v}" for k, v in ch.items()) for _, ch in steps])
    return table, runs


def leave_one_out(signals: SignalSet, prices: PriceSource, full: Strategy, removals: list) -> tuple[pd.DataFrame,
                                                                                                    dict]:
    """The full strategy, then the full strategy with ONE decision undone: ``removals`` = [(label, {key: value
    that switches it off}), ...]: what each rule contributes when all the others stay."""
    variants = {"full": full}
    for label, changes in removals:
        variants[f"without {label}"] = apply_changes(full, changes, name=f"without {label}")
    return compare(signals, prices, variants)
