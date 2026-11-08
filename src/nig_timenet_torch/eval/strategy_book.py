"""PURPOSE: Every use of the model outputs as a trade rule over a battery table (direction, top-|mu| grid, limit at mu,
    limit at the predicted extreme, take-profit), each with trivial controls, on VAL and TEST with a day-block CI.
TAGS: t-profit, strategy book, trade rules, trivial controls, volatility baseline, day block bootstrap, val and test
PITFALLS: cost is PER SIDE (2x per round trip, battery_lib.SIDES); every threshold (aleatoric, volatility, top-k)
    comes from VAL and is applied unchanged to TEST; high / low enter only as price levels, the side of a
    direction trade comes from the close head (rule D); TP rules use the 2-bar realised extreme, no path order.

A trade frame has one row per trade: ``pnl`` (after cost), ``hit`` (target reached / order filled; NaN when the rule
has no target), ``side`` (+1 long, -1 short), ``coin``, ``day``, ``mae`` (adverse excursion >= 0 from the entry),
``pred_move`` / ``real_move`` (the predicted and realised favourable move the rule relies on). The battery table
(``eval.battery_adapter``) must carry the extra columns ``ale_high``, ``ale_low`` (aleatoric std, return units),
``vol`` (a volatility baseline in return units) and ``last_ret`` (last bar's close return) for the controls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.battery_lib import SIDES
from nig_timenet_torch.stats.metrics import block_bootstrap_ci

__all__ = ["direction_trades", "limit_at_mu_trades", "limit_extreme_trades", "tp_trades", "summarize", "book",
           "FILTER_QUANTILES", "TOP_PCTS", "TP_SHARE"]

FILTER_QUANTILES = (None, 0.5, 0.8, 0.9)
TOP_PCTS = (0.05, 0.10, 0.20)
TP_SHARE = 0.5


def _frame(df, idx, pnl, side, hit, mae, pred_move, real_move) -> pd.DataFrame:
    return pd.DataFrame({"pnl": pnl, "hit": hit, "side": side, "coin": df["coin"].values[idx],
                         "day": df["end_time"].values[idx].astype("datetime64[D]"), "mae": mae,
                         "pred_move": pred_move, "real_move": real_move})


def _adverse_market(df, idx, side) -> np.ndarray:
    """Adverse excursion of a market entry at the last close: the 2-bar max drop (long) or max rise (short)."""
    return np.where(side > 0, np.maximum(0.0, -df["fut_low"].values[idx]), np.maximum(0.0, df["fut_high"].values[idx]))


def direction_trades(df, idx, side, cost: float) -> pd.DataFrame:
    """Market entry at the last close, exit at the close of the horizon, on rows ``idx`` with ``side`` (+1 / -1)."""
    idx, side = np.asarray(idx), np.broadcast_to(np.asarray(side, dtype=np.float64), (len(idx),))
    y = df["y_ret"].values[idx]
    return _frame(df, idx, side * y - SIDES * cost, side, np.full(len(idx), np.nan), _adverse_market(df, idx, side),
                  np.abs(df["mu"].values[idx]), side * y)


def limit_at_mu_trades(df, idx, side, cost: float) -> pd.DataFrame:
    """The battery's 'REALISTIC EXECUTION - limit at mu' rule (eval/battery.py::limit_execution), exactly: market
    entry at the last close; 'hit' when the CLOSE return reaches mu (y >= mu long, y <= mu short); a hit is paid
    |mu|, a miss the close return; cost per side x2."""
    idx = np.asarray(idx)
    side = np.broadcast_to(np.asarray(side, dtype=np.float64), (len(idx),))
    mu, y = df["mu"].values[idx], df["y_ret"].values[idx]
    hit = np.where(side > 0, y >= mu, y <= mu)
    pnl = np.where(hit, np.abs(mu), side * y) - SIDES * cost
    return _frame(df, idx, pnl, side, hit.astype(float), _adverse_market(df, idx, side), np.abs(mu), side * y)


def limit_extreme_trades(df, mask, side: float, cost: float, pred_high=None, pred_low=None) -> pd.DataFrame:
    """Limit buy at pred_low (< 0) or limit short at pred_high (> 0), filled when the realised 2-bar extreme reaches
    it, exit at the close (battery_profit.limit_highlow): only filled orders are trades (``hit`` = 1)."""
    ph = df["pred_high"].values if pred_high is None else pred_high
    pl = df["pred_low"].values if pred_low is None else pred_low
    pred = pl if side > 0 else ph
    fut = df["fut_low"].values if side > 0 else df["fut_high"].values
    fill = np.asarray(mask) & ((pred < 0) if side > 0 else (pred > 0)) & ((fut <= pred) if side > 0 else (fut >= pred))
    idx = np.where(fill)[0]
    entry, y = 1.0 + pred[idx], df["y_ret"].values[idx]
    pnl = side * ((1.0 + y) - entry) / entry - SIDES * cost
    mae = np.maximum(0.0, side * (entry - (1.0 + fut[idx])) / entry)
    return _frame(df, idx, pnl, np.full(len(idx), side), np.ones(len(idx)), mae, np.abs(pred[idx]),
                  np.abs(fut[idx]))


def tp_trades(df, mask, side: float, cost: float, share: float = TP_SHARE, pred_high=None, pred_low=None,
              need_close_down: bool = True) -> pd.DataFrame:
    """Market entry at the last close with a take-profit at ``share`` of the predicted extreme; exit at the TP when
    the 2-bar extreme reaches it, else at the close. Long: rows with pred_high > 0 (the 'LONG mirror'). Short: rows
    with pred_low < 0 and (when ``need_close_down``) mu < 0 (the 'SHORT C' rule). No stop, so no path order."""
    ph = df["pred_high"].values if pred_high is None else pred_high
    pl = df["pred_low"].values if pred_low is None else pred_low
    if side > 0:
        rows = np.asarray(mask) & (ph > 0)
        pred, fut = ph, df["fut_high"].values
    else:
        rows = np.asarray(mask) & (pl < 0) & ((df["mu"].values < 0) if need_close_down else True)
        pred, fut = pl, df["fut_low"].values
    idx = np.where(rows)[0]
    tp, y = share * pred[idx], df["y_ret"].values[idx]
    hit = (fut[idx] >= tp) if side > 0 else (fut[idx] <= tp)
    pnl = np.where(hit, side * tp, side * y) - SIDES * cost
    sides = np.full(len(idx), side)
    return _frame(df, idx, pnl, sides, hit.astype(float), _adverse_market(df, idx, sides), np.abs(pred[idx]),
                  side * fut[idx])


def summarize(tr: pd.DataFrame, level: float = 0.95, seed: int = 0) -> dict:
    """n, win rate, mean, day-block bootstrap CI at ``level``, per-trade Sharpe, total (%-points) of a trade frame."""
    p = tr["pnl"].values
    if len(p) < 2:
        return {"n": int(len(p))}
    mean, lo, hi = block_bootstrap_ci(p, tr["day"].values, level=level, seed=seed)
    sd = p.std(ddof=1)
    return {"n": int(len(p)), "win": float((p > 0).mean()), "avg": mean, "ci_lo": lo, "ci_hi": hi,
            "sharpe": float(mean / sd) if sd > 0 else 0.0, "total_pts": float(p.sum() * 100),
            "hit": float(np.nanmean(tr["hit"].values)) if np.isfinite(tr["hit"].values).any() else None}


def _top(df, field_vals, pct, mask) -> np.ndarray:
    base = np.where(mask)[0]
    k = max(1, int(len(base) * pct))
    return base[np.argsort(-np.abs(field_vals[base]), kind="stable")[:k]]


def _direction_uses(d, cost, rng):
    """(use, control, trades) of the close-direction rules on one split."""
    scored = d["dir_true"].values != 0
    side = np.where(d["p_dir"].values > 0.5, 1.0, -1.0)
    all_idx = np.where(scored)[0]
    yield "close direction, all rows", "model", direction_trades(d, all_idx, side[all_idx], cost)
    yield "close direction, all rows", "always long", direction_trades(d, all_idx, 1.0, cost)
    yield "close direction, all rows", "always short", direction_trades(d, all_idx, -1.0, cost)
    yield "close direction, all rows", "shuffled direction", direction_trades(
        d, all_idx, rng.permutation(side[all_idx]), cost)
    yield "close direction, all rows", "last-bar sign", direction_trades(
        d, all_idx, np.where(d["last_ret"].values[all_idx] > 0, 1.0, -1.0), cost)
    for pct in TOP_PCTS:
        for name, sg in (("LONG", 1.0), ("SHORT", -1.0)):
            m = scored & (side == sg)
            use = f"{name} top{int(pct * 100)}% |mu| (market exit)"
            idx = _top(d, d["mu"].values, pct, m)
            yield use, "model", direction_trades(d, idx, sg, cost)
            yield use, "volatility selection", direction_trades(d, _top(d, d["vol"].values, pct, m), sg, cost)
            yield use, "random selection", direction_trades(d, rng.choice(np.where(m)[0], len(idx), replace=False), sg,
                                                            cost)
            yield f"{name} top{int(pct * 100)}% (limit at mu)", "model", limit_at_mu_trades(d, idx, sg, cost)


def _level_uses(d, cost, thr, vol_scale, rng):
    """(use, control, trades) of the rules that use the high / low heads as price levels on one split."""
    n = len(d)
    ph, pl = d["pred_high"].values, d["pred_low"].values
    perm = rng.permutation(n)
    shuf = {"pred_high": ph[perm], "pred_low": pl[perm]}
    volp = {"pred_high": vol_scale["high"] * d["vol"].values, "pred_low": -vol_scale["low"] * d["vol"].values}
    every = np.ones(n, dtype=bool)
    for name, side in (("LONG limit at pred_low", 1.0), ("SHORT limit at pred_high", -1.0)):
        yield name, "model", limit_extreme_trades(d, every, side, cost)
        yield name, "shuffled levels", limit_extreme_trades(d, every, side, cost, **shuf)
        yield name, "volatility levels", limit_extreme_trades(d, every, side, cost, **volp)
    for q in FILTER_QUANTILES:
        tag = "none" if q is None else f"ale>=Q{int(q * 100)}"
        for name, side, ale in (("LONG TP=0.5*pred_high", 1.0, "ale_high"),
                                ("SHORT C TP=0.5*pred_low", -1.0, "ale_low")):
            sel = every if q is None else d[ale].values >= thr[(ale, q)]
            vsel = every if q is None else d["vol"].values >= thr[("vol", q)]
            use = f"{name} [{tag}]"
            model = tp_trades(d, sel, side, cost)
            yield use, "model", model
            yield use, "shuffled levels", tp_trades(d, sel, side, cost, **shuf)
            yield use, "volatility levels", tp_trades(d, sel, side, cost, **volp)
            yield use, "volatility selection", tp_trades(d, vsel, side, cost)
            if len(model):
                rnd = np.zeros(n, dtype=bool)
                rnd[rng.choice(n, min(n, int(sel.sum())), replace=False)] = True
                yield use, "random selection", tp_trades(d, rnd, side, cost)
                rows = np.where(sel & ((ph > 0) if side > 0 else (pl < 0) & (d["mu"].values < 0)))[0]
                yield use, "market, same rows, no TP", direction_trades(d, rows, side, cost)


def book(val: pd.DataFrame, test: pd.DataFrame, cost: float = 0.0008, seed: int = 0) -> list[dict]:
    """Every (use, control) on VAL and TEST: one dict per (use, control, split) with ``summarize`` stats and the
    trade frame (key ``trades``). Thresholds (aleatoric / volatility quantiles) and the volatility-level scale (median
    pred / median vol) come from VAL."""
    thr = {(c, q): float(np.quantile(val[c].values, q)) for c in ("ale_high", "ale_low", "vol")
           for q in FILTER_QUANTILES if q is not None}
    vol_scale = {"high": float(np.median(val["pred_high"].values) / np.median(val["vol"].values)),
                 "low": float(-np.median(val["pred_low"].values) / np.median(val["vol"].values))}
    out = []
    for split, d in (("VAL", val), ("TEST", test)):
        rng = np.random.default_rng(seed)
        for gen in (_direction_uses(d, cost, rng), _level_uses(d, cost, thr, vol_scale, rng)):
            for use, control, tr in gen:
                out.append({"use": use, "control": control, "split": split, **summarize(tr), "trades": tr})
    return out
