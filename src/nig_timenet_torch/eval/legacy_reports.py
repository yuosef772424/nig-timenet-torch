"""PURPOSE: The TF evaluation reports that had no counterpart here, over a battery table: tearsheet with the equity /
    drawdown of the trades and of the daily portfolio, integrity (lag scan, naive baselines, bias), confidence
    calibration, output-conditioned hit rates, per-coin best / worst and the worst / best single samples.
TAGS: legacy reports, tearsheet, equity curve, portfolio, drawdown, sortino, lag scan, calibration, brier, best worst,
    output conditioned, tf parity
PITFALLS: pure functions of a battery table (``eval/battery_adapter.BATTERY_COLUMNS``: mu / y_ret in return units,
    p_dir, end_time naive UTC); the trade rule is the battery's (direction from p_dir on the top ``top_share`` |mu|
    rows, cost per side, so a round trip pays 2 x cost); per-trade numbers are unannualised; the equity of the trades
    risks ``risk_per_trade`` of the capital per trade in time order (TF ``simulate_equity_curve``), the daily
    portfolio puts the whole capital equally on the day's trades. Exploratory (rule C5).

Sources (crypto-signal-prediction @ 1899a2d): ``evaluation/tearsheet.py`` (compute_trading_performance_metrics,
simulate_equity_curve), ``evaluation/integrity.py`` (naive baseline, lag scan, directional bias, worst / best
combinations), ``evaluation/trust_calibration.py`` (ECE / Brier), ``evaluation/output_conditioned.py`` (by move,
confidence and batch), ``evaluation/plots.py`` (the figures: ``eval/legacy_plots.py``).
"""

import math

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.battery_lib import round_trip
from nig_timenet_torch.stats import metrics

__all__ = ["trades", "tearsheet", "equity_curve", "daily_portfolio", "lag_scan", "integrity", "calibration",
           "conditioned", "coin_ranking", "extreme_samples", "move_over_sd", "TOP_SHARE", "RISK_PER_TRADE"]

TOP_SHARE = 0.10        # share of rows traded: the strongest |mu| (the battery's top-|mu| rule)
RISK_PER_TRADE = 0.02   # capital share risked per trade in the trade equity curve (TF tearsheet used 0.02 - 0.1)


def trades(table: pd.DataFrame, cost: float, top_share: float = TOP_SHARE) -> pd.DataFrame:
    """The traded rows in time order: the top ``top_share`` |mu| rows, side from p_dir, ``net`` = side x y_ret minus
    the round trip (fractions, not %)."""
    if not 0 < top_share <= 1:
        raise ValueError(f"top_share must be in (0, 1], got {top_share}")
    k = max(1, int(len(table) * top_share))
    idx = np.argsort(-np.abs(table["mu"].to_numpy()), kind="stable")[:k]
    out = table.iloc[idx].copy()
    out["side"] = np.where(out["p_dir"].to_numpy() > 0.5, 1.0, -1.0)
    out["gross"] = out["side"] * out["y_ret"]
    out["net"] = out["gross"] - round_trip(cost)
    return out.sort_values("end_time", kind="stable").reset_index(drop=True)


def equity_curve(returns: np.ndarray, risk: float = RISK_PER_TRADE, start: float = 1.0) -> dict:
    """Capital after each trade when ``risk`` of it is put on every trade (a loss below -100 % of the stake is
    clipped there); drawdown in fractions of the running peak."""
    r = np.clip(np.asarray(returns, dtype=float), -1.0, None)
    equity = start * np.concatenate([[1.0], np.cumprod(1.0 + risk * r)])
    peak = np.maximum.accumulate(equity)
    return {"equity": equity, "drawdown": equity / peak - 1.0}


def tearsheet(returns: np.ndarray, risk: float = RISK_PER_TRADE) -> dict:
    """TF tearsheet numbers of a series of trade returns: n, win rate, mean, Sharpe and Sortino (per trade), Omega
    at 0 (for equal-weight trades it equals the profit factor; both kept, as in TF), profit factor, SQN
    (sqrt(min(n, 100)) x mean / sd), payoff, the largest win / loss streaks, and the final capital / max drawdown of
    ``equity_curve`` at ``risk``."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return {"n": int(len(r))}
    sd, down = r.std(ddof=1), r[r < 0]
    dsd = math.sqrt(float(np.mean(np.square(np.minimum(r, 0.0)))))
    gain, loss = r[r > 0].sum(), -down.sum()
    eq = equity_curve(r, risk)
    signs = np.sign(r)
    streaks = {s: 0 for s in (1.0, -1.0)}
    run, last = 0, 0.0
    for s in signs:
        run = run + 1 if s == last else 1
        last = s
        if s in streaks:
            streaks[s] = max(streaks[s], run)
    pf = float(gain / loss) if loss > 0 else math.inf
    return {"n": int(len(r)), "win_rate": float((r > 0).mean()), "mean": float(r.mean()),
            "median": float(np.median(r)), "sharpe": float(r.mean() / sd) if sd > 0 else 0.0,
            "sortino": float(r.mean() / dsd) if dsd > 0 else math.inf, "omega": pf, "profit_factor": pf,
            "sqn": float(math.sqrt(min(len(r), 100)) * r.mean() / sd) if sd > 0 else 0.0,
            "payoff": float(r[r > 0].mean() / -down.mean()) if len(down) and (r > 0).any() else math.nan,
            "max_win_streak": int(streaks[1.0]), "max_loss_streak": int(streaks[-1.0]),
            "final_equity": float(eq["equity"][-1]), "max_drawdown": float(eq["drawdown"].min()), "risk": risk}


def daily_portfolio(trade_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per UTC day: trades, the equal-weight mean net return of the day's trades, the compounding equity of a
    portfolio that splits the whole capital over each day's trades, and its drawdown."""
    day = pd.to_datetime(trade_rows["end_time"]).dt.floor("D")
    g = trade_rows.groupby(day)["net"]
    out = pd.DataFrame({"trades": g.size(), "ret": g.mean()})
    out["equity"] = np.cumprod(1.0 + np.clip(out["ret"].to_numpy(), -1.0, None))
    out["drawdown"] = out["equity"] / np.maximum.accumulate(out["equity"]) - 1.0
    out.index.name = "day"
    return out


def lag_scan(table: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    """TF lag scan per lag k in [-max_lag, max_lag]: the mean over coins of the Pearson correlation of mu at bar t with
    the realised return of bar t + k (same coin, rows in time order). k = 0 is the skill; a peak at k < 0 means the
    model repeats a move that has already happened (late copying)."""
    rows = []
    groups = [g.sort_values("end_time") for _, g in table.groupby("coin") if len(g) > 2 * max_lag + 3]
    for k in range(-max_lag, max_lag + 1):
        vals = []
        for g in groups:
            mu, y = g["mu"].to_numpy(), g["y_ret"].to_numpy()
            a, b = (mu[: len(mu) - k], y[k:]) if k >= 0 else (mu[-k:], y[: len(y) + k])
            vals.append(metrics.pearson(a, b))
        v = np.asarray(vals, dtype=float)
        rows.append({"lag": k, "mean_r": float(np.nanmean(v)) if np.isfinite(v).any() else math.nan,
                     "coins": int(np.isfinite(v).sum())})
    return pd.DataFrame(rows)


def integrity(table: pd.DataFrame, max_lag: int = 3) -> dict:
    """TF integrity diagnostics: error of mu against the naive zero forecast and the persistence forecast (last bar's
    return), direction hit of mu against the momentum sign, the share predicted up against realised up, and the
    lag scan with the lag of its peak."""
    y, mu = table["y_ret"].to_numpy(), table["mu"].to_numpy()
    out = {"rows": int(len(table)), "mae_model": metrics.mae(y, mu), "mae_zero": metrics.mae(y, np.zeros_like(y)),
           "share_pred_up": float((mu > 0).mean()), "share_true_up": float((y > 0).mean()),
           "hit_model": float((np.sign(mu) == np.sign(y))[y != 0].mean()) if (y != 0).any() else math.nan,
           "bias_mean_mu_minus_y": float(np.mean(mu - y))}
    if "last_ret" in table:
        last = table["last_ret"].to_numpy()
        out["mae_persistence"] = metrics.mae(y, last)
        ok = (y != 0) & np.isfinite(last)
        out["hit_momentum"] = float((np.sign(last) == np.sign(y))[ok].mean()) if ok.any() else math.nan
    scan = lag_scan(table, max_lag)
    out["lag_scan"] = scan.to_dict("records")
    best = scan.loc[scan["mean_r"].abs().idxmax()] if scan["mean_r"].notna().any() else None
    out["lag_of_peak"] = None if best is None else int(best["lag"])
    return out


def calibration(table: pd.DataFrame, n_bins: int = 10) -> dict:
    """TF trust report on the direction probability p_dir against up = y_ret > 0: ECE and Brier with the base-rate
    baseline, and the reliability bins (count, mean label, mean p)."""
    rows = table["y_ret"].to_numpy() != 0
    up, p = (table["y_ret"].to_numpy()[rows] > 0).astype(float), table["p_dir"].to_numpy()[rows]
    counts, mean_y, mean_p = metrics.ece_bins(up, p, n_bins)
    base = np.full_like(p, up.mean() if len(up) else 0.5)
    return {"rows": int(rows.sum()), "ece": metrics.ece(up, p, n_bins), "brier": metrics.brier(up, p),
            "brier_base_rate": metrics.brier(up, base), "auc": metrics.auc(up, p),
            "bins": [{"count": int(c), "mean_up": float(a), "mean_p": float(b)}
                     for c, a, b in zip(counts, mean_y, mean_p, strict=True)]}


def _hit_rows(frame: pd.DataFrame, key) -> pd.DataFrame:
    rows = []
    for name, g in frame.groupby(key, observed=True):
        hit = int((np.sign(g["mu"]) == np.sign(g["y_ret"])).sum())
        lo, hi = metrics.wilson_interval(hit, len(g))
        rows.append({"group": str(name), "rows": int(len(g)), "hit": hit / len(g), "ci_lo": lo, "ci_hi": hi,
                     "mean_mu": float(g["mu"].mean()), "mean_y": float(g["y_ret"].mean()),
                     "mean_signed": float((np.sign(g["mu"]) * g["y_ret"]).mean())})
    return pd.DataFrame(rows)


def conditioned(table: pd.DataFrame, n_buckets: int = 5, n_batches: int = 10) -> dict:
    """TF output-conditioned analysis: the direction hit rate (Wilson CI) and the mean signed return by bucket of the
    predicted move |mu|, by side, by bucket of the predicted sd (confidence) and by consecutive time batch."""
    t = table[table["y_ret"] != 0].copy()
    t["move"] = pd.qcut(np.abs(t["mu"]), n_buckets, labels=False, duplicates="drop")
    t["side"] = np.where(t["mu"] > 0, "up", "down")
    t["conf"] = pd.qcut(t["sigma"], n_buckets, labels=False, duplicates="drop")
    order = np.argsort(t["end_time"].to_numpy(), kind="stable")
    t["batch"] = np.empty(len(t), dtype=int)
    t.iloc[order, t.columns.get_loc("batch")] = np.arange(len(t)) * n_batches // max(len(t), 1)
    return {"by_move": _hit_rows(t, "move"), "by_side": _hit_rows(t, "side"), "by_sd": _hit_rows(t, "conf"),
            "by_batch": _hit_rows(t, "batch")}


def coin_ranking(table: pd.DataFrame, min_rows: int = 30) -> pd.DataFrame:
    """Per coin (>= ``min_rows`` rows): rows, Pearson r and rank IC of mu with y_ret, hit rate and MAE, sorted from
    the best (highest r) to the worst coin (TF integrity: the best / worst combinations that get charts)."""
    rows = []
    for coin, g in table.groupby("coin"):
        if len(g) < min_rows:
            continue
        y, mu = g["y_ret"].to_numpy(), g["mu"].to_numpy()
        rows.append({"coin": coin, "rows": int(len(g)), "r": metrics.pearson(y, mu), "rank_ic": metrics.spearman(y, mu),
                     "hit": float((np.sign(mu) == np.sign(y))[y != 0].mean()) if (y != 0).any() else math.nan,
                     "mae": metrics.mae(y, mu)})
    out = pd.DataFrame(rows, columns=["coin", "rows", "r", "rank_ic", "hit", "mae"])
    return out.sort_values("r", ascending=False, na_position="last").reset_index(drop=True)


def extreme_samples(trade_rows: pd.DataFrame, n: int = 10) -> dict:
    """The ``n`` worst and ``n`` best TRADES by net return (coin, exact end_time, side, mu, sigma, realised), the rows
    to chart on raw candles (``eval/sample_charts.py``: ``samples_csv``)."""
    cols = [c for c in ("coin", "end_time", "side", "mu", "sigma", "p_dir", "y_ret", "net") if c in trade_rows]
    srt = trade_rows.sort_values("net", kind="stable")
    return {"worst": srt.head(n)[cols].reset_index(drop=True), "best": srt.tail(n)[cols][::-1].reset_index(drop=True)}


def move_over_sd(table: pd.DataFrame, cost: float, ks: tuple = (0.25, 0.5, 1.0)) -> pd.DataFrame:
    """TF trade selection (``select_and_rank_trades``): trade only the rows whose predicted move exceeds ``k`` times
    the predicted sd (|mu| > k sigma), side from p_dir, net of the round trip; one tearsheet row per k plus the
    share of rows kept (the battery's section 6 keeps LOW sigma instead)."""
    rows = []
    for k in ks:
        keep = table[np.abs(table["mu"]) > k * table["sigma"]]
        net = np.where(keep["p_dir"] > 0.5, 1.0, -1.0) * keep["y_ret"].to_numpy() - round_trip(cost)
        ts = tearsheet(net)
        rows.append({"k": k, "kept_share": len(keep) / max(len(table), 1), "days": int(
            pd.to_datetime(keep["end_time"]).dt.floor("D").nunique()) if len(keep) else 0,
            **{c: ts.get(c, math.nan) for c in ("n", "win_rate", "mean", "sharpe", "profit_factor")}})
    return pd.DataFrame(rows)
