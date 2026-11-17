"""PURPOSE: The effect of each strategy decision on the portfolio, per period: strategies side by side, a ladder that
    adds one rule at a time, leave-one-out, under a switchable protocol: choose on VAL, judge the chosen one on TEST.
TAGS: rule impact, strategy comparison, ablation, ladder, leave one out, protocol, choose on val, judge on test,
    multiple testing, periods
PITFALLS: every variant runs on the SAME signals and candles (cached). With the protocol on (default), the TEST
    numbers of the variants NOT chosen on VAL are hidden, so nobody picks a rule by looking at TEST; turning it off
    shows everything (exploration) and the table says so. Train is reported on its own (in-sample) and never merged.
    The table counts the variants tried.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nig_backtest.contract import SignalSet
from nig_backtest.engine import Backtest, run_backtest
from nig_backtest.periods import TEST, VAL, VAL_TEST
from nig_backtest.portfolio import by_period
from nig_backtest.prices import PriceSource
from nig_backtest.rules import Strategy, strategy_from_dict, strategy_to_dict

__all__ = ["Protocol", "apply_changes", "compare", "ladder", "leave_one_out", "ROW_KEYS"]

ROW_KEYS = ("trades", "final_equity", "max_drawdown", "positive_days", "mean_cost")


@dataclass(frozen=True)
class Protocol:
    """``enabled``: choose the variant with the best ``metric`` on ``select_on`` and show ``judge_on`` only for it;
    off = every number of every variant is shown (the table is then marked exploratory)."""
    enabled: bool = True
    select_on: str = VAL
    judge_on: str = TEST
    metric: str = "final_equity"
    merge_val_test: bool = False


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


def _rows(name: str, bt: Backtest, merge: bool) -> list[dict]:
    out = []
    for period, s in by_period(bt.trades, bt.strategy.book.risk_per_trade, merge).items():
        pt = s["per_trade"]
        out.append({"variant": name, "period": period, **{k: s[k] for k in ROW_KEYS},
                    "win_rate": pt.get("win_rate"), "mean_net": pt.get("mean"), "sharpe_per_trade": pt.get("sharpe"),
                    "profit_factor": pt.get("profit_factor")})
    return out


def _apply_protocol(table: pd.DataFrame, protocol: Protocol) -> pd.DataFrame:
    if not protocol.enabled:
        table.attrs["protocol"] = "off: every period of every variant is shown (exploratory)"
        return table
    sel = table[table["period"] == protocol.select_on]
    if sel.empty:
        raise ValueError(f"protocol: no {protocol.select_on!r} period in the signals (give stages or split dates, or "
                         f"switch the protocol off)")
    chosen = str(sel.loc[sel[protocol.metric].astype(float).idxmax(), "variant"])
    judged = table["period"].isin([protocol.judge_on, VAL_TEST]) if protocol.judge_on == TEST else \
        table["period"] == protocol.judge_on
    hide = judged & (table["variant"] != chosen)
    cols = [c for c in table.columns if c not in ("variant", "period")]
    table = table.astype({c: object for c in cols})
    table.loc[hide, cols] = "hidden"
    table["chosen"] = np.where(table["variant"] == chosen, "<- chosen on " + protocol.select_on, "")
    table.attrs["protocol"] = (f"on: chosen by {protocol.metric} on {protocol.select_on}; {protocol.judge_on} shown "
                               f"for the chosen variant only")
    table.attrs["chosen"] = chosen
    return table


def compare(signals: SignalSet, prices: PriceSource, strategies: dict, protocol: Protocol | None = None
            ) -> tuple[pd.DataFrame, dict]:
    """(table, {name: Backtest}): one row per variant x period, under ``protocol``."""
    protocol = protocol or Protocol()
    runs = {name: run_backtest(signals, prices, s) for name, s in strategies.items()}
    table = pd.DataFrame([r for n, bt in runs.items() for r in _rows(n, bt, protocol.merge_val_test)])
    table = _apply_protocol(table, protocol)
    table.attrs["variants_tried"] = len(runs)
    return table, runs


def ladder(signals: SignalSet, prices: PriceSource, base: Strategy, steps: list, protocol: Protocol | None = None
           ) -> tuple[pd.DataFrame, dict]:
    """Add one decision at a time: ``steps`` = [(label, {dotted key: value}), ...]; variant i = base + steps[:i]."""
    variants, current = {"base": base}, base
    for label, changes in steps:
        current = apply_changes(current, changes, name=label)
        variants[label] = current
    table, runs = compare(signals, prices, variants, protocol)
    added = {"base": "-", **{lab: ", ".join(f"{k}={v}" for k, v in ch.items()) for lab, ch in steps}}
    table.insert(1, "added", table["variant"].map(added))
    return table, runs


def leave_one_out(signals: SignalSet, prices: PriceSource, full: Strategy, removals: list,
                  protocol: Protocol | None = None) -> tuple[pd.DataFrame, dict]:
    """The full strategy, then the full strategy with ONE decision undone: ``removals`` = [(label, {key: value that
    switches it off}), ...]."""
    variants = {"full": full}
    for label, changes in removals:
        variants[f"without {label}"] = apply_changes(full, changes, name=f"without {label}")
    return compare(signals, prices, variants, protocol)
