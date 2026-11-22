"""Battery profit sections (T-profit): close-direction trades, limit orders at predicted high / low, sigma sizing

PURPOSE: The T-profit sections of the battery: per-side-cost profit with trivial baselines and a block-bootstrap CI
    that is widened for the number of variants tried; limit orders at the predicted high / low; sigma as position size.
TAGS: t-profit, profit after cost, limit order, position size, block bootstrap, trivial baselines, battery
PITFALLS: cost is charged PER SIDE (2x per round trip), the same as every older battery section (battery_lib
    round_trip). Direction comes from the close head only; high / low are price levels (fills), never a direction. A CI
    that excludes 0 is exploratory evidence until the owner asks for strict confirmation (rule C5). Blocks are
    calendar days; a table without end_time falls back to FALLBACK_BLOCK_ROWS-row chunks (the note says so).

Every section is a pure function over the battery table of ``eval.battery_adapter`` (limit sections also need its
pred_high / pred_low / fut_high / fut_low columns) and returns the battery's ``{"title", "rows", "note", ...}``
shape plus a ``"stats"`` dict of the numbers. ``n_variants`` is the number of model / rule variants tried so far
(multiple testing): the CI level is ``battery_lib.bonferroni_level(n_variants)``.
"""

from __future__ import annotations

import numpy as np

from nig_timenet_torch.eval.battery_lib import (
    MIN_TRADES,
    SIDES,
    bonferroni_level,
    limit_at_extreme,
    net_pnl,
    scored_rows,
    section,
    trade_side,
    trade_stats,
)
from nig_timenet_torch.eval.day_blocks import calendar_day, day_blocks
from nig_timenet_torch.stats.metrics import block_bootstrap_ci

__all__ = ["profit_sections", "profit_close", "limit_highlow", "sigma_sizing", "DEFAULT_SEED", "FALLBACK_BLOCK_ROWS"]

DEFAULT_SEED = 0
#: rows per bootstrap block when the table has no end_time (no days to group by)
FALLBACK_BLOCK_ROWS = 50


def _level(n_variants: int) -> float:
    if n_variants < 1:
        raise ValueError("n_variants must be >= 1")
    return bonferroni_level(n_variants)


def _blocks(df) -> np.ndarray:
    """One block per calendar day of end_time (rows of one day move together); without timestamps, chunks of
    ``FALLBACK_BLOCK_ROWS`` rows."""
    if "end_time" in df:
        return day_blocks(calendar_day(df["end_time"]), 1)
    return np.arange(len(df)) // FALLBACK_BLOCK_ROWS


def _line(label: str, pnl: np.ndarray, blocks: np.ndarray, level: float, seed: int) -> tuple[str, dict]:
    s = trade_stats(pnl)
    if s is None:
        return f"  {label:<34}(n<{MIN_TRADES})", {}
    mean, lo, hi = block_bootstrap_ci(pnl, blocks, level=level, seed=seed)
    s.update({"ci_lo": lo, "ci_hi": hi, "level": level})
    return (f"  {label:<34}n={s['n']:>8,}  avg={s['avg'] * 100:+.4f}%  sharpe={s['sharpe']:+.3f}  "
            f"CI{level * 100:.1f}%=[{lo * 100:+.4f}%, {hi * 100:+.4f}%]", s)


def profit_close(test, cost: float, n_variants: int = 1, seed: int = DEFAULT_SEED) -> dict:
    """Trade the close direction (p_dir > 0.5) on every non-flat row, market exit at the close, cost per side.
    Trivial baselines on the same rows: always up, always down, a shuffled direction, and the last candle's sign
    (when ``last_ret`` exists). Only the model line beating the best of them, with a CI above 0, is profit."""
    keep = scored_rows(test)
    d = test[keep]
    y, blocks = d["y_ret"].values, _blocks(d)
    level, rng = _level(n_variants), np.random.default_rng(seed)
    side = trade_side(d)
    rows, stats = [], {}
    cands = [("model (close p_dir)", side), ("always up", np.ones(len(d))), ("always down", -np.ones(len(d))),
             ("shuffled direction", rng.permutation(side))]
    if "last_ret" in d:
        cands.append(("last candle sign (momentum)", np.where(d["last_ret"].values > 0, 1.0, -1.0)))
    for name, sg in cands:
        line, s = _line(name, net_pnl(sg * y, cost), blocks, level, seed)
        rows.append(line)
        stats[name] = s
    ci = "day-block bootstrap" if "end_time" in d else f"{FALLBACK_BLOCK_ROWS}-row blocks (no end_time)"
    return {**section("T-PROFIT - close direction trades after cost (TEST)", rows,
                       f"cost {cost:.4%} per side x{SIDES}; CI = {ci} at {level:.1%} "
                       f"(n_variants={n_variants}). Profit = model beats every baseline row AND its CI excludes 0."),
            "stats": stats}


def limit_highlow(test, cost: float, n_variants: int = 1, seed: int = DEFAULT_SEED) -> dict:
    """Limit orders at the predicted low (long) and predicted high (short), exit at the close. A price event, not a
    direction claim. Baselines per side: the SAME orders with shuffled predictions, a constant offset (the model's
    median), and a plain market entry at the last close on the same rows. A model that only knows the typical range
    scores like the constant-offset row."""
    need = {"pred_high", "pred_low", "fut_high", "fut_low"}
    if not need <= set(test.columns):
        return section("T-PROFIT - LIMIT ORDERS - SKIPPED", ["  (needs the high / low heads in the table)"],
                        "Train/evaluate with the high and low heads to enable this section.")
    level, rng = _level(n_variants), np.random.default_rng(seed)
    y, blocks_all = test["y_ret"].values, _blocks(test)
    rows, stats = [], {}
    for name, side, pcol, fcol in (("LONG at pred_low", 1.0, "pred_low", "fut_low"),
                                   ("SHORT at pred_high", -1.0, "pred_high", "fut_high")):
        pred, fut = test[pcol].values, test[fcol].values
        valid_model = (pred < 0) if side > 0 else (pred > 0)
        const = np.full(len(pred), float(np.median(pred[valid_model])) if valid_model.any() else 0.0)
        variants = [("model", pred), ("shuffled predictions", rng.permutation(pred)), ("constant offset", const)]
        for label, p in variants:
            pnl, fill, valid = limit_at_extreme(side, p, fut, y, cost)
            line, s = _line(f"{name} - {label}", pnl, blocks_all[fill], level, seed)
            share = float(fill.sum() / max(int(valid.sum()), 1))
            rows.append(line + f"  fill={share * 100:.1f}%")
            stats[f"{name} - {label}"] = {**s, "fill": share}
        market = net_pnl(side * y[valid_model], cost)
        line, s = _line(f"{name} - market at last close", market, blocks_all[valid_model], level, seed)
        rows.append(line)
        stats[f"{name} - market at last close"] = s
    return {**section("T-PROFIT - LIMIT ORDERS at the predicted high / low (TEST), exit at the close", rows,
                       "Unfilled orders are no trade (pnl of filled orders only, fill% beside it). The model must beat "
                       "its shuffled and constant-offset rows: those know nothing about THIS bar's range."),
            "stats": stats}


def sigma_sizing(test, cost: float, n_variants: int = 1, seed: int = DEFAULT_SEED) -> dict:
    """Uncertainty as position size: trade the close direction with weight 1 / sigma (normalised to mean 1, so the
    average exposure equals the flat book). Baselines: equal weights and weights from SHUFFLED sigma. Cost scales
    with the size. A sigma that knows nothing scores like the shuffled row."""
    keep = scored_rows(test) & (test["sigma"].values > 0)
    d = test[keep]
    y, blocks = d["y_ret"].values, _blocks(d)
    level, rng = _level(n_variants), np.random.default_rng(seed)
    side = trade_side(d)
    inv = 1.0 / d["sigma"].values
    rows, stats = [], {}
    for name, w in (("sized by 1/sigma", inv / inv.mean()), ("equal weight", np.ones(len(d))),
                    ("sized by shuffled 1/sigma", rng.permutation(inv) / inv.mean())):
        line, s = _line(name, w * net_pnl(side * y, cost), blocks, level, seed)
        rows.append(line)
        stats[name] = s
    return {**section("T-PROFIT - SIGMA AS POSITION SIZE (TEST)", rows,
                       "Mean weight 1 in every row (same average exposure). If the 1/sigma row does not beat equal "
                       "and shuffled weights, sigma is not a size signal."), "stats": stats}


def profit_sections(test, cost: float, n_variants: int = 1) -> dict:
    """The three T-profit sections in battery order."""
    return {"profit_close": profit_close(test, cost, n_variants), "profit_limit": limit_highlow(test, cost, n_variants),
            "profit_sigma": sigma_sizing(test, cost, n_variants)}
