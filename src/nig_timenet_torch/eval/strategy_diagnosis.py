"""PURPOSE: Deep diagnosis of one strategy's trades: payoff vs break-even win rate, hits vs misses, tail losses,
    adverse excursion, worst trades, day concentration, link to the market day, calibration of the predicted move.
TAGS: strategy diagnosis, break even win rate, payoff, tail losses, adverse excursion, day concentration, market beta,
    calibration
PITFALLS: amounts are in %-points of one unit per trade (sum of per-trade returns x 100), not a compounded equity;
    the market day must be the same side as the strategy (short-all for a short rule); a row whose sign differs
    between VAL and TEST is not evidence, whatever this report says.

``diagnose(trades, market_day)`` returns the numbers of the four sections of the owner's reference report (2026-11-08),
``render(title, d)`` the text. ``market_day`` maps a day to the mean return of trading EVERY row of that day on the
strategy's side, gross of cost (the 'short-all' / 'long-all' day).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nig_timenet_torch.stats import metrics

__all__ = ["payoff", "days", "market_link", "calibration", "diagnose", "render"]


def payoff(tr: pd.DataFrame, n_worst: int = 5) -> dict:
    """Section [1]: why a high win rate can lose."""
    p = tr["pnl"].values
    win, loss = p[p > 0], p[p <= 0]
    aw, al = float(win.mean()) if len(win) else 0.0, float(loss.mean()) if len(loss) else 0.0
    be = abs(al) / (aw + abs(al)) if (aw + abs(al)) > 0 else float("nan")
    hit = tr["hit"].values
    has_hit = np.isfinite(hit).any()
    losses = np.sort(loss)  # most negative first
    tot_loss = float(-losses.sum())

    def tail(f):
        k = max(1, int(np.ceil(f * len(p))))
        return float(-losses[:k].sum() / tot_loss) if tot_loss > 0 else float("nan")
    mae = tr["mae"].values
    worst = tr.nsmallest(n_worst, "pnl")
    out = {"n": int(len(p)), "win": float((p > 0).mean()), "avg": float(p.mean()), "total_pts": float(p.sum() * 100),
           "avg_win": aw, "avg_loss": al, "payoff": aw / abs(al) if al else float("inf"), "breakeven_win": be,
           "quantiles": {f"p{q}": float(np.quantile(p, q / 100)) for q in (1, 5, 25, 50, 75, 95)},
           "worst1_share_of_losses": tail(0.01), "worst5_share_of_losses": tail(0.05),
           "mae_median": float(np.median(mae)), "mae_p95": float(np.quantile(mae, 0.95)), "mae_max": float(mae.max()),
           "mae_over_10pct": float((mae > 0.10).mean()),
           "worst": [{"coin": r.coin, "day": str(r.day)[:10], "pnl": float(r.pnl), "mae": float(r.mae)}
                     for r in worst.itertuples()]}
    if has_hit:
        h = hit == 1
        out.update({"hit_share": float(h.mean()), "hits": {"n": int(h.sum()), "pts": float(p[h].sum() * 100),
                                                           "avg": float(p[h].mean()) if h.any() else float("nan")},
                    "misses": {"n": int((~h).sum()), "pts": float(p[~h].sum() * 100),
                               "avg": float(p[~h].mean()) if (~h).any() else float("nan"),
                               "closed_in_profit": float((p[~h] > 0).mean()) if (~h).any() else float("nan")}})
    return out


def _day_rows(tr, g, day, market_day) -> dict:
    sub = tr[tr["day"].values == day]
    coins = sub.groupby("coin")["pnl"].sum().sort_values(key=np.abs, ascending=False).head(3)
    return {"day": str(day)[:10], "n": int(len(sub)), "avg": float(g.loc[day, "mean"]), "total_pts":
            float(g.loc[day, "sum"] * 100), "market": float(market_day.get(day, np.nan)),
            "coins": [(c, float(v * 100)) for c, v in coins.items()]}


def days(tr: pd.DataFrame, market_day: pd.Series, k: int = 3) -> dict:
    """Section [2]: per-day concentration (days ranked by the average per trade)."""
    g = tr.groupby("day")["pnl"].agg(["mean", "sum", "count"])
    order = g.sort_values("mean")
    worst, best = list(order.index[:k]), list(order.index[-k:][::-1])
    total = float(g["sum"].sum() * 100)
    neg, pos = float(-g["sum"][g["sum"] < 0].sum()), float(g["sum"][g["sum"] > 0].sum())
    w_sum, b_sum = float(g.loc[worst, "sum"].sum()), float(g.loc[best, "sum"].sum())
    return {"n_days": int(len(g)), "positive_days": float((g["mean"] > 0).mean()),
            "median_day": float(g["mean"].median()), "std_day": float(g["mean"].std(ddof=1)) if len(g) > 1 else 0.0,
            "worst": [_day_rows(tr, g, d, market_day) for d in worst],
            "best": [_day_rows(tr, g, d, market_day) for d in best], "total_pts": total,
            "without_worst": total - w_sum * 100, "without_best": total - b_sum * 100,
            "without_both": total - (w_sum + b_sum) * 100,
            "worst_share_of_negative": float(-min(w_sum, 0.0) / neg) if neg > 0 else float("nan"),
            "best_share_of_positive": float(max(b_sum, 0.0) / pos) if pos > 0 else float("nan")}


def market_link(tr: pd.DataFrame, market_day: pd.Series, long_all_day: pd.Series) -> dict:
    """Section [3]: daily mean of the strategy vs the same-side market day: correlation, OLS slope / intercept, and the
    strategy's average on days the market went up (long-all > 0) vs down."""
    s = tr.groupby("day")["pnl"].mean()
    m = market_day.reindex(s.index).to_numpy(dtype=np.float64)
    up = long_all_day.reindex(s.index).to_numpy(dtype=np.float64) > 0
    sv = s.to_numpy(dtype=np.float64)
    ok = np.isfinite(m)
    slope, intercept = (np.polyfit(m[ok], sv[ok], 1) if ok.sum() > 2 else (np.nan, np.nan))
    return {"corr": metrics.pearson(sv, m), "slope": float(slope), "intercept": float(intercept),
            "up_days_share": float(up.mean()), "strategy_on_up_days": float(sv[up].mean()) if up.any() else np.nan,
            "strategy_on_down_days": float(sv[~up].mean()) if (~up).any() else np.nan}


def calibration(tr: pd.DataFrame) -> dict:
    """Section [4]: is the predicted move calibrated on the selected trades?"""
    pm, rm = tr["pred_move"].values, tr["real_move"].values
    return {"pred_mean": float(pm.mean()), "real_mean": float(rm.mean()), "reached_full": float((rm >= pm).mean()),
            "reached_half": float((rm >= 0.5 * pm).mean()), "corr": metrics.pearson(pm, rm)}


def diagnose(tr: pd.DataFrame, market_day: pd.Series, long_all_day: pd.Series) -> dict:
    """All four sections for one trade frame (``market_day``: same-side all-rows day; ``long_all_day``: market)."""
    return {"payoff": payoff(tr), "days": days(tr, market_day), "market": market_link(tr, market_day, long_all_day),
            "calibration": calibration(tr)}


def _pct(v: float, digits: int = 2) -> str:
    return f"{v * 100:+.{digits}f}%"


def render(title: str, d: dict) -> str:
    p, dy, mk, c = d["payoff"], d["days"], d["market"], d["calibration"]
    lines = [f"{title}   n={p['n']:,}   avg net {_pct(p['avg'], 3)}   total {p['total_pts']:+.0f}%-pts",
             f"[1] win% = {p['win'] * 100:.1f}%" + (f"  (target reached in {p['hit_share'] * 100:.1f}%)"
                                                     if "hit_share" in p else "")
             + f"   avg win = {_pct(p['avg_win'])}   avg loss = {_pct(p['avg_loss'])}   payoff = {p['payoff']:.2f}",
             f"    break-even win rate for this payoff = {p['breakeven_win'] * 100:.1f}%  -> actual "
             f"{p['win'] * 100:.1f}%  => {'ABOVE' if p['win'] > p['breakeven_win'] else 'BELOW'} break-even"]
    if "hits" in p:
        h, m = p["hits"], p["misses"]
        lines += [f"    trades that HIT the target: {h['n']:,}  contribute {h['pts']:+.0f}%-pts (avg {_pct(h['avg'])})",
                  f"    trades that MISSED it     : {m['n']:,}  contribute {m['pts']:+.0f}%-pts (avg {_pct(m['avg'])})"
                  f"  | of which closed in profit: {m['closed_in_profit'] * 100:.0f}%"]
    q = p["quantiles"]
    lines += ["    per-trade quantiles %: " + " ".join(f"{k}={v * 100:+.2f}" for k, v in q.items()),
              f"    worst 1% of trades = {p['worst1_share_of_losses'] * 100:.0f}% of ALL losses ; worst 5% = "
              f"{p['worst5_share_of_losses'] * 100:.0f}%",
              f"    adverse excursion: median {p['mae_median'] * 100:.1f}%  p95 {p['mae_p95'] * 100:.1f}%  max "
              f"{p['mae_max'] * 100:.0f}% ; trades with excursion >10%: {p['mae_over_10pct'] * 100:.1f}%",
              "    worst trades: " + " | ".join(f"{w['coin']} {w['day']} {w['pnl'] * 100:+.0f}% (exc "
                                                f"{w['mae'] * 100:.0f}%)" for w in p["worst"]),
              f"[2] DAYS ({dy['n_days']} days)   positive days {dy['positive_days'] * 100:.0f}%   median day "
              f"{_pct(dy['median_day'])}/trade   std of daily mean {dy['std_day'] * 100:.2f}%"]
    for name in ("worst", "best"):
        lines.append(f"    {name.upper()} 3 days (by avg per trade):")
        for r in dy[name]:
            coins = ", ".join(f"{cn} {v:+.0f}" for cn, v in r["coins"])
            lines.append(f"      {r['day']}: {r['n']:>5} trades, avg {_pct(r['avg'])}/trade, total "
                         f"{r['total_pts']:+.0f}%-pts | same-side market that day {_pct(r['market'])} | coins: {coins}")
    lines += [f"    total {dy['total_pts']:+.0f}%-pts | without worst 3 days: {dy['without_worst']:+.0f} | without "
              f"best 3 days: {dy['without_best']:+.0f} | without both: {dy['without_both']:+.0f}",
              f"    worst 3 days = {dy['worst_share_of_negative'] * 100:.0f}% of all negative-day PnL ; best 3 days = "
              f"{dy['best_share_of_positive'] * 100:.0f}% of all positive-day PnL",
              f"[3] MARKET: corr(strategy day, same-side market day) = {mk['corr']:+.2f}   slope = {mk['slope']:+.2f}"
              f"   intercept = {_pct(mk['intercept'], 3)}/trade/day",
              f"    market UP days {mk['up_days_share'] * 100:.0f}%: strategy avg {_pct(mk['strategy_on_up_days'])} ;"
              f" DOWN days: strategy avg {_pct(mk['strategy_on_down_days'])}",
              f"[4] PREDICTED MOVE: predicted mean {c['pred_mean'] * 100:.2f}% vs realised mean "
              f"{c['real_mean'] * 100:.2f}% ; reached the full prediction in {c['reached_full'] * 100:.0f}% ; half of"
              f" it in {c['reached_half'] * 100:.0f}% ; corr(pred, realised) = {c['corr']:+.2f}"]
    return "\n".join(lines)
