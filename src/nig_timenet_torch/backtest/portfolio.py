"""PURPOSE: What a backtest's trades do to a portfolio: the open-position cap, the equity and drawdown when each
    trade risks a share of the capital (booked at its exit), the daily book and the summary tables.
TAGS: portfolio, equity curve, drawdown, max open positions, daily pnl, tearsheet, summary by group, backtest
PITFALLS: a trade is booked at its EXIT time with the capital it was given at its ENTRY (risk_per_trade x the equity
    then), so open trades never compound on each other; ``max_open`` skips later entries (it does not close earlier
    trades); the tearsheet is ``eval.legacy_reports.tearsheet`` (one implementation).
"""

import heapq

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.legacy_reports import tearsheet

__all__ = ["cap_open", "equity", "daily", "summary", "by_group"]


def cap_open(trades: pd.DataFrame, max_open: int | None) -> pd.DataFrame:
    """Drop the trades that would open while ``max_open`` trades are already open (entry-time order)."""
    if max_open is None or trades.empty:
        return trades
    order = trades.sort_values(["entry_time", "coin"], kind="stable")
    open_exits: list = []
    keep = []
    for i, (t_in, t_out) in zip(order.index, zip(order["entry_time"], order["exit_time"], strict=True), strict=True):
        while open_exits and open_exits[0] <= t_in:
            heapq.heappop(open_exits)
        if len(open_exits) < max_open:
            heapq.heappush(open_exits, t_out)
            keep.append(i)
    return trades.loc[keep]


def equity(trades: pd.DataFrame, risk: float, start: float = 1.0) -> pd.DataFrame:
    """One row per trade in exit order: the capital it was given (``risk`` x the equity at its entry), its pnl and the
    equity / drawdown after it."""
    if trades.empty:
        return pd.DataFrame(columns=["exit_time", "coin", "stake", "pnl", "equity", "drawdown"])
    events = sorted([(t, 1, i) for i, t in zip(trades.index, trades["exit_time"], strict=True)]
                    + [(t, 0, i) for i, t in zip(trades.index, trades["entry_time"], strict=True)])
    eq, stake, rows = start, {}, []
    net = trades["net"].clip(lower=-1.0)
    for t, kind, i in events:
        if kind == 0:
            stake[i] = risk * eq
        else:
            pnl = stake.pop(i) * float(net.loc[i])
            eq += pnl
            rows.append((t, trades.at[i, "coin"], risk, pnl, eq))
    out = pd.DataFrame(rows, columns=["exit_time", "coin", "stake", "pnl", "equity"])
    out["drawdown"] = out["equity"] / np.maximum.accumulate(out["equity"].to_numpy()) - 1.0
    return out


def daily(curve: pd.DataFrame, start: float = 1.0) -> pd.DataFrame:
    """The equity at the end of each UTC day (days without an exit carry the last value) and the day's return."""
    if curve.empty:
        return pd.DataFrame(columns=["equity", "ret", "drawdown"])
    eq = curve.set_index(pd.to_datetime(curve["exit_time"]).dt.floor("D"))["equity"].groupby(level=0).last()
    eq = eq.reindex(pd.date_range(eq.index.min(), eq.index.max(), freq="D")).ffill()
    out = pd.DataFrame({"equity": eq})
    out["ret"] = out["equity"].pct_change().fillna(out["equity"].iloc[0] / start - 1.0)
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    out.index.name = "day"
    return out


def by_group(trades: pd.DataFrame, key) -> pd.DataFrame:
    """Trades, win rate, mean net / gross, sum of net, share hitting the target / the stop, per group of ``key`` (a
    column name or an array)."""
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(key, observed=True)
    out = pd.DataFrame({"trades": g.size(), "win_rate": g["win"].mean(), "direction_right": g["direction_right"].mean(),
                        "mean_net": g["net"].mean(), "mean_gross": g["gross"].mean(), "sum_net": g["net"].sum(),
                        "take_profit": g["exit_reason"].apply(lambda s: float((s == "take_profit").mean())),
                        "stop_loss": g["exit_reason"].apply(lambda s: float((s == "stop_loss").mean())),
                        "mean_bars": g["bars_held"].mean()})
    return out.reset_index()


def summary(trades: pd.DataFrame, risk: float, max_open: int | None = None) -> dict:
    """The portfolio numbers of a trade list: per-trade tearsheet, final equity / max drawdown of the book, days,
    share of positive days, and the tables by exit reason, side and stage."""
    traded = cap_open(trades, max_open)
    curve = equity(traded, risk)
    days = daily(curve)
    out = {"trades": int(len(traded)), "skipped_max_open": int(len(trades) - len(traded)),
           "per_trade": tearsheet(traded["net"].to_numpy()) if len(traded) else {"n": 0},
           "final_equity": float(curve["equity"].iloc[-1]) if len(curve) else 1.0,
           "max_drawdown": float(curve["drawdown"].min()) if len(curve) else 0.0,
           "days": int(len(days)), "positive_days": float((days["ret"] > 0).mean()) if len(days) else float("nan"),
           "by_exit": by_group(traded, "exit_reason"), "by_side": by_group(traded, pd.Series(np.where(
               traded["side"] > 0, "long", "short"), index=traded.index, name="side"))}
    if "stage" in traded:
        out["by_stage"] = by_group(traded, "stage")
    out["curve"], out["daily"] = curve, days
    return out
