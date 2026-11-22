"""Shared helpers: cost, direction, top-k, trade stats, section shaping of the battery

PURPOSE: Shared helpers of the battery: ONE implementation of cost, direction, top-|x| selection, trade stats,
    market and limit-order PnL, the multiple-testing level and section shaping.
TAGS: battery helpers, trade stats, sharpe, profit factor, section, cost, round trip, bonferroni
PITFALLS: cost is PER SIDE (round_trip = 2x; net_pnl pays it once per trade); the direction of a trade is p_dir
    (trade_side), never sign(mu); trade_stats returns None below MIN_TRADES; Sharpe is per trade; top_abs ranks by
    |values| with a STABLE sort (ties keep row order), so the side comes from the mask; limit_at_mu sets a target
    only when sign(mu) == side and checks it on the realised extreme (fut_high / fut_low) when the table has it,
    else on the close (a lower bound of the fill); limit_at_extreme fills only when the realised extreme reaches
    the order.
SOURCE: ported from feat/ar-stream 7c9736b, f052cf6 (owner's work)

Split out of battery.py so the core and the classic sections share ONE implementation (rule C2). The constants here
(``SIDES``, ``ALPHA``, ``MIN_TRADES``, ``DEFAULT_COST_PER_SIDE``) are the only definition of cost per side, the
significance level and the minimum trade count: eval, workflow and the notebooks read them, nothing re-types them.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SIDES", "ALPHA", "MIN_TRADES", "DEFAULT_COST_PER_SIDE", "round_trip", "net_pnl", "bonferroni_level",
           "trade_side", "limit_at_mu", "limit_at_extreme", "trade_stats", "scored_rows", "top_abs", "direction_pnl",
           "section", "format_row", "HEADER"]


#: a round trip pays the cost on entry AND exit: ``cost`` everywhere in the battery is the cost PER SIDE
SIDES = 2
#: the family-wise significance level the confidence intervals are widened from (Bonferroni, ``bonferroni_level``)
ALPHA = 0.05
#: a row with fewer trades than this is not scored (``trade_stats`` returns None, the row prints n<MIN_TRADES)
MIN_TRADES = 50
#: the cost per side every default of the battery, the reports and the run settings starts from (0.08 % per side)
DEFAULT_COST_PER_SIDE = 0.0008


def round_trip(cost: float) -> float:
    """What one trade (entry + exit) pays when each side costs ``cost``."""
    return SIDES * cost


def net_pnl(gross, cost: float):
    """``gross`` (a trade's return before cost) minus one round trip at ``cost`` per side."""
    return gross - round_trip(cost)


def bonferroni_level(n_tests: int, alpha: float = ALPHA) -> float:
    """Confidence level ``1 - alpha / n_tests`` of an interval that is one of ``n_tests`` looks at the data."""
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1")
    return 1.0 - alpha / n_tests


def trade_side(df) -> np.ndarray:
    """+1 (long) where the close class head says up (p_dir > 0.5), else -1: the direction of every trade."""
    return np.where(df["p_dir"] > 0.5, 1.0, -1.0)


def trade_stats(pnl: np.ndarray, min_n: int = MIN_TRADES) -> dict | None:
    """n, win rate, average, Sharpe (per-trade, unannualized) and profit factor - None below min_n."""
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) < min_n:
        return None
    s = pnl.std(ddof=1)
    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    return {"n": int(len(pnl)), "win": float((pnl > 0).mean()), "avg": float(pnl.mean()),
            "sharpe": float(pnl.mean() / s) if s > 0 else 0.0,
            "pf": float(gross_win / gross_loss) if gross_loss > 1e-12 else float("inf")}


def scored_rows(df) -> "np.ndarray":
    """The non-flat rows: a flat truth teaches nothing about direction."""
    return (df["dir_true"] != 0).values


def top_abs(values: np.ndarray, pct: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Indices (into ``values``) of the strongest ``pct`` fraction by |values|, optionally inside a mask. The sort is
    stable: rows tied on |values| are taken in row order."""
    base = np.where(mask if mask is not None else np.ones(len(values), dtype=bool))[0]
    strength = np.abs(np.asarray(values)[base])
    k = max(1, int(len(base) * pct))
    return base[np.argsort(-strength, kind="stable")[:k]]


def direction_pnl(df, idx: np.ndarray, cost: float) -> np.ndarray:
    """Trade the model's direction (p_dir) on the true return, minus the round-trip cost (``cost`` per side)."""
    return net_pnl(trade_side(df)[idx] * df["y_ret"].values[idx], cost)


def limit_at_mu(df, idx: np.ndarray, side, cost: float) -> dict:
    """Market entry at the last close with a take-profit AT the predicted close move ``mu``, else exit at the close.

    The target exists only on the trade's side (``sign(mu) == side``). A row whose mu points against the side
    (p_dir long with mu below zero) has no target and exits at the close; B-053: the old rule paid such a long
    ``+|mu|`` when its price fell. The target is reached when the realised favourable extreme over the horizon
    touches it (``fut_high >= mu`` long, ``fut_low <= mu`` short). Without those columns the close stands in for
    the extreme, which only misses fills (a lower bound). Returns ``pnl`` (after the round trip), ``hit``,
    ``target`` (the row had a target) and ``path`` (True when the extreme columns were used)."""
    idx = np.asarray(idx)
    side = np.broadcast_to(np.asarray(side, dtype=np.float64), (len(idx),))
    mu, y = df["mu"].values[idx], df["y_ret"].values[idx]
    path = {"fut_high", "fut_low"} <= set(df.columns)
    reach = np.where(side > 0, df["fut_high"].values[idx], df["fut_low"].values[idx]) if path else y
    target = np.greater(side * mu, 0.0)
    hit = target & (side * reach >= side * mu)
    pnl = net_pnl(np.where(hit, side * mu, side * y), cost)
    return {"pnl": pnl, "hit": hit, "target": target, "path": path}


def limit_at_extreme(side: float, pred, fut_extreme, close_ret, cost: float):
    """PnL of a limit order at ``pred`` (a return vs the last close) that fills when the realised extreme reaches it,
    exit at the close; unfilled orders are no trade. side +1: buy at pred_low (< 0), -1: short at pred_high (> 0).
    Returns ``(pnl, fill, valid)``: ``fill`` / ``valid`` are row masks over the inputs, ``pnl`` has one value per
    filled row (after the round trip)."""
    valid = (pred < 0) if side > 0 else (pred > 0)
    fill = valid & ((fut_extreme <= pred) if side > 0 else (fut_extreme >= pred))
    entry, exit_ = 1.0 + pred[fill], 1.0 + close_ret[fill]
    return net_pnl(side * (exit_ - entry) / entry, cost), fill, valid


def section(title: str, rows: list[str], note: str = "") -> dict:
    return {"title": title, "rows": rows, "note": note}


def format_row(label: str, s: dict | None) -> str:
    if s is None:
        return f"{label:<44}(n<{MIN_TRADES})"
    return (f"{label:<44}{s['n']:>8,}{s['win'] * 100:>7.2f}%{s['avg'] * 100:>+9.4f}%"
            f"{s['sharpe']:>+9.3f}{s['pf']:>7.2f}")


HEADER = f"{'strategy':<44}{'n':>8}{'win%':>8}{'avg%':>10}{'Sharpe':>9}{'PF':>7}"
