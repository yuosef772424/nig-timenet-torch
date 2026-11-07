"""Evaluation battery: many-angle judgment of a prediction table

PURPOSE: Ten many-angle sections plus classics and a ranked verdict over a tidy table (step, mu, sigma, p_dir, y_ret,
    dir_true, last_ret, coin, end_time); pure functions, no model.
TAGS: battery, evaluation, trade grid, verdict, uncertainty filter, weekly stability, flip test
PITFALLS: columns are the battery table, not the PredictionTable (build it with eval.battery_adapter); cost 0.0008 is a
    pinned default; Sharpe is per trade, not annualised.
SOURCE: ported from feat/ar-stream 7c9736b, f052cf6 (owner's work)

The evaluation battery: a many-angle judgment of one run, the way a trader would interrogate it before trusting
money to it. Every section is a pure function over the tidy table, so the notebook stays a thin runner (rule E4/G6).
Sections: sanity (IC + bias), confidence calibration, anti-cowardice, trading grids (market exit and limit-at-mu),
an uncertainty filter, shock-regime splits, weekly stability, per-coin, the flip test, the classics ported from the
old project (battery_classics) and a ranked verdict.
"""

from __future__ import annotations

import numpy as np

from nig_timenet_torch.eval.battery_classics import (
    market_neutral,
    persistence_baseline,
    selective_evaluation,
    table_integrity,
)
from nig_timenet_torch.eval.battery_lib import _HEADER, _dir_pnl, _fmt, _scored, _section, _stats, _top_abs
from nig_timenet_torch.eval.battery_profit import profit_sections
from nig_timenet_torch.stats.metrics import spearman

__all__ = ["battery", "full_battery", "show", "table_integrity", "persistence_baseline",
           "selective_evaluation", "market_neutral"]


def show(section: dict, echo=print) -> None:
    """Print one battery section the way the notebook does: the title, its rows, then the reading line.
    Lets the battery notebook run every test as its own re-runnable cell with the same output shape."""
    echo("=" * 100)
    echo(section["title"])
    for row in section["rows"]:
        echo(row)
    if section.get("note"):
        echo("-> " + section["note"])


def sanity(val, test) -> dict:
    """Section 1: is the ranking sane (IC of mu vs the true return) and is the model biased?"""
    rows = []
    for name, df in (("VAL", val), ("TEST", test)):
        keep = _scored(df)
        mu, y = df["mu"].values, df["y_ret"].values
        ic = spearman(mu[keep], y[keep])
        acc = float(((mu[keep] > 0) == (y[keep] > 0)).mean())
        rows.append(f"  {name:<5} IC(mu, y_ret)={ic:+.4f}  dir_acc={acc:.4f}  "
                    f"P(mu>0)={float((mu > 0).mean()):.3f}  P(y>0)={float((y > 0).mean()):.3f}  "
                    f"n={int(keep.sum()):,}")
    return _section("SANITY - IC, direction accuracy, up-bias", rows,
                    "IC > 0 means the model's magnitude ranks moves correctly; a big P(mu>0) - P(y>0) gap is bias.")


def confidence_buckets(test, n_buckets: int = 5) -> dict:
    """Section 2: does accuracy rise with the model's own confidence (|p_dir - 0.5|)?"""
    df = test[_scored(test)]
    edges = np.quantile((df["p_dir"] - 0.5).abs(), np.linspace(0, 1, n_buckets + 1))
    rows = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        m = ((df["p_dir"] - 0.5).abs() >= lo) & ((df["p_dir"] - 0.5).abs() <= hi)
        sub = df[m]
        if len(sub) < 30:
            continue
        acc = float(((sub["p_dir"] > 0.5) == (sub["y_ret"] > 0)).mean())  # the direction it would TRADE
        pnl = _dir_pnl(sub, np.arange(len(sub)), 0.0)
        rows.append(f"  confidence [{lo:.3f}, {hi:.3f}]  n={len(sub):>7,}  dir_acc={acc:.4f}"
                    f"  avg_pnl/trade={pnl.mean() * 100:+.4f}%")
    return _section("CONFIDENCE CALIBRATION (TEST)", rows,
                    "A reliable model's accuracy rises monotonically with its own confidence; flat means the "
                    "probability carries no information.")


def cowardice(test) -> dict:
    """Section 3: anti-cowardice - does the model predict real magnitudes or hide near zero?"""
    keep = _scored(test)
    mu, y = np.abs(test["mu"].values[keep]), np.abs(test["y_ret"].values[keep])
    ratio = float(mu.mean() / max(y.mean(), 1e-12))
    qs = (0.25, 0.50, 0.75, 0.90)
    qline = "  ".join(f"q{int(q * 100)}: |mu|={np.quantile(mu, q):+.4f} vs |y|={np.quantile(y, q):+.4f}" for q in qs)
    hidden = float((mu < 0.25 * np.median(y)).mean())
    rows = [f"  mean |mu| / mean |y_ret| = {ratio:.3f}   (0 = coward, ~1 = honest magnitude)",
            f"  {qline}",
            f"  share of predictions under a quarter of the median true move: {hidden * 100:.1f}%"]
    return _section("ANTI-COWARDICE (TEST)", rows,
                    "With the czar loss this should sit well above the v1 numbers; a tiny ratio means the model "
                    "hedges by predicting ~0 and the trading grid below is meaningless.")


def trade_grid(test, cost: float, pcts: tuple = (0.05, 0.10, 0.20)) -> dict:
    """Section 4: trade the predicted direction, market exit at the true close, top-% by |mu|."""
    rows = [_HEADER]
    for pct in pcts:
        for side, sign in (("LONG", 1.0), ("SHORT", -1.0)):
            mask = (np.where(test["p_dir"].values > 0.5, 1.0, -1.0) == sign) & _scored(test)
            idx = _top_abs(test, "mu", pct, mask)
            rows.append(_fmt(f"{side} top{int(pct * 100)}% |mu| (market exit)",
                             _stats(sign * test["y_ret"].values[idx] - cost)))
    rows.append(_fmt("ALL rows, trade the direction (market exit)",
                     _stats(np.where(test["p_dir"].values > 0.5, 1, -1) * test["y_ret"].values
                            - cost * np.ones(len(test)))))
    return _section("TRADING GRID (TEST) - market exit", rows,
                    f"cost={cost:.4%} per trade. Compare with the persistence baseline before believing any row.")


def limit_execution(test, cost: float, pcts: tuple = (0.05, 0.10, 0.20)) -> dict:
    """Section 5: realistic execution - a limit order AT the predicted mu; if the market never reaches
    it, fall back to the market exit (the true close). This is the closest to a tradeable backtest."""
    rows = [f"{'strategy':<44}{'n':>8}{'fill%':>8}{'avg%':>10}{'Sharpe':>9}{'PF':>7}"]
    mu, y = test["mu"].values, test["y_ret"].values
    for pct in pcts:
        for side, sign in (("LONG", 1.0), ("SHORT", -1.0)):
            mask = (np.where(test["p_dir"].values > 0.5, 1.0, -1.0) == sign) & _scored(test)
            idx = _top_abs(test, "mu", pct, mask)
            hit = (y[idx] >= mu[idx]) if sign > 0 else (y[idx] <= mu[idx])
            pnl = np.where(hit, np.abs(mu[idx]) - cost, sign * y[idx] - cost)
            s = _stats(pnl)
            if s:
                rows.append(f"{f'{side} top{int(pct * 100)}% (limit at mu)':<44}{s['n']:>8,}"
                           f"{float(hit.mean()) * 100:>7.1f}%{s['avg'] * 100:>+9.4f}%"
                           f"{s['sharpe']:>+9.3f}{s['pf']:>7.2f}")
    return _section("REALISTIC EXECUTION (TEST) - limit at mu, fallback close", rows,
                    "A high fill% with a positive avg means the model's magnitude is honest; fills with negative "
                    "avg mean the limit sits inside noise.")


def uncertainty_filter(val, test, cost: float, pct: float = 0.10, quantiles: tuple = (0.3, 0.5, 0.7)) -> dict:
    """Section 6: keep only LOW-sigma predictions (the model's own confidence in its magnitude):
    thresholds are the VAL quantiles, applied to TEST - no leakage."""
    rows = [_HEADER]
    base = val.iloc[_top_abs(val, "mu", pct, _scored(val))]
    for q in quantiles:
        thr = float(np.quantile(base["sigma"].values, q))
        idx = _top_abs(test, "mu", pct, _scored(test) & (test["sigma"].values <= thr))
        rows.append(_fmt(f"top{int(pct * 100)}% |mu| with sigma<=Q{int(q * 100)}(val)",
                         _stats(_dir_pnl(test, idx, cost))))
    rows.append(_fmt("top{int}% |mu| unfiltered".replace("{int}", str(int(pct * 100))),
                     _stats(_dir_pnl(test, _top_abs(test, "mu", pct, _scored(test)), cost))))
    return _section("UNCERTAINTY FILTER (TEST) - low sigma only", rows,
                    "If accuracy/PnL improves as sigma tightens, the model KNOWS when it does not know.")


def shock_split(test, cost: float, n_groups: int = 5, pct: float = 0.10) -> dict:
    """Section 7: calm vs wild - accuracy and PnL by quintile of the TRUE move size. The owner's core
    question: does the model stay sane exactly where the market is unpredictable?"""
    keep = _scored(test)
    strength = np.zeros(len(test), dtype=bool)
    strength[keep] = True
    rows = []
    size = np.abs(test["y_ret"].values)
    edges = np.quantile(size[keep], np.linspace(0, 1, n_groups + 1))
    for i in range(n_groups):
        m = strength & (size >= edges[i]) & (size <= edges[i + 1])
        if m.sum() < 50:
            continue
        sub = test[m]
        acc = float(((sub["mu"] > 0) == (sub["y_ret"] > 0)).mean())
        pnl = _stats(_dir_pnl(test, np.where(m)[0], cost))
        rows.append(f"  move q{int((i + 1) * 100 / n_groups):<3} |y| in [{edges[i]:.4f}, {edges[i + 1]:.4f}]"
                    f"  n={len(sub):>7,}  dir_acc={acc:.4f}  "
                    f"pnl avg={pnl['avg'] * 100:+.4f}%  sharpe={pnl['sharpe']:+.3f}" if pnl else
                    f"  move q{int((i + 1) * 100 / n_groups):<3} (n<50)")
    return _section("SHOCK REGIMES (TEST) - calm to wildest quintile", rows,
                    "Accuracy is EXPECTED to decay toward the wildest quintile (those moves are news/liquidity, "
                    "not pattern); what matters is that the PnL does not explode there.")


def weekly_stability(test, cost: float, pct: float = 0.10, min_n: int = 50) -> dict:
    """Section 8: per-week accuracy and PnL of the top-pct trades - one lucky month is not an edge."""
    if "end_time" not in test:
        return _section("WEEKLY STABILITY - SKIPPED", ["  (windows carry no end timestamps)"],
                        "Rebuild the windows with the frame index to enable this section.")
    scored = _scored(test)
    idx = set(_top_abs(test, "mu", pct, scored).tolist())
    weeks = test["end_time"].values.astype("datetime64[D]")
    rows, pos = [], 0
    for w in np.unique(weeks):
        wm = [i for i in np.where(scored)[0] if weeks[i] == w and i in idx]
        if len(wm) < min_n:
            continue
        pnl = _dir_pnl(test, np.array(wm), cost)
        acc = float(((test["mu"].values[wm] > 0) == (test["y_ret"].values[wm] > 0)).mean())
        pos += int(pnl.mean() > 0)
        sd = pnl.std(ddof=1)
        rows.append(f"  week {w}  n={len(wm):>6,}  dir_acc={acc:.4f}  avg={pnl.mean() * 100:+.4f}%"
                    f"  sharpe={pnl.mean() / sd if sd > 0 else 0.0:+.3f}")
    total = len(rows)
    return _section("WEEKLY STABILITY (TEST)", rows or ["  (no week reached the min trade count)"],
                    (f"{pos} of {total} weeks positive - an edge that lives in one week only is noise."
                     if total else "no week had enough trades"))


def per_coin(test, cost: float, pct: float = 0.10, min_n: int = 30) -> dict:
    """Section 9: per-coin accuracy and PnL of the top-pct trades, best to worst."""
    if "coin" not in test:
        return _section("PER-COIN - SKIPPED", ["  (no coin column: single-coin run)"])
    rows = []
    for coin in np.unique(test["coin"].values):
        m = (test["coin"].values == coin) & _scored(test)
        idx = _top_abs(test, "mu", pct, m)
        if len(idx) < min_n:
            continue
        pnl = _stats(_dir_pnl(test, idx, cost))
        acc = float(((test["mu"].values[idx] > 0) == (test["y_ret"].values[idx] > 0)).mean())
        rows.append((coin, len(idx), acc, pnl))
    rows.sort(key=lambda r: -(r[3]["sharpe"] if r[3] else -9))
    lines = [f"  {'coin':<12}{'n':>7}{'dir_acc':>9}{'avg%':>10}{'Sharpe':>9}{'PF':>7}"]
    for coin, n, acc, s in rows:
        if s:
            lines.append(f"  {coin:<12}{n:>7,}{acc:>9.4f}{s['avg'] * 100:>+9.4f}%"
                        f"{s['sharpe']:>+9.3f}{s['pf']:>7.2f}")
    return _section("PER-COIN (TEST)", lines,
                    f"{len(rows)} coins with >= {min_n} scored top-{int(pct * 100)}% trades")


def flip_test(test, cost: float, pct: float = 0.10) -> dict:
    """Section 10: is the direction inverted? Compare trading the signal against trading AGAINST it."""
    idx = _top_abs(test, "mu", pct, _scored(test))
    pnl = _dir_pnl(test, idx, cost)
    flipped = -pnl - 2 * cost * np.ones(len(pnl))
    rows = [f"  trade the direction : Sharpe={pnl.mean() / max(pnl.std(ddof=1), 1e-12):+.3f}  "
            f"avg={pnl.mean() * 100:+.4f}%",
            f"  trade AGAINST it   : Sharpe={flipped.mean() / max(flipped.std(ddof=1), 1e-12):+.3f}  "
            f"avg={flipped.mean() * 100:+.4f}%"]
    return _section("FLIP TEST (TEST)", rows,
                    "If 'against' wins, the model found the pattern but inverted its sign.")


def battery(val, test, cost: float = 0.0008) -> dict:
    """All fourteen sections over the VAL/TEST prediction tables (same order as the notebook prints them)."""
    return {
        "integrity": table_integrity(val, test),
        "sanity": sanity(val, test),
        "confidence": confidence_buckets(test),
        "cowardice": cowardice(test),
        "trades": trade_grid(test, cost),
        "limit": limit_execution(test, cost),
        "uncertainty": uncertainty_filter(val, test, cost),
        "shocks": shock_split(test, cost),
        "weekly": weekly_stability(test, cost),
        "per_coin": per_coin(test, cost),
        "flip": flip_test(test, cost),
        "baseline": persistence_baseline(test, cost),
        "selective": selective_evaluation(val, test, cost),
        "neutral": market_neutral(val, test, cost),
    }


def verdict(test, cost: float, sections: dict) -> dict:
    """The ranked bottom line: the best realistic strategy by TEST Sharpe, with the direction edge,
    the shock sanity and the weekly consistency beside it - so the verdict is never one lucky number."""
    keep = _scored(test)
    side = np.where(test["p_dir"].values > 0.5, 1.0, -1.0)
    best = {"name": "none", "sharpe": float("nan")}
    for pct in (0.05, 0.10, 0.20):
        for name, sign in (("LONG", 1.0), ("SHORT", -1.0)):
            idx = _top_abs(test, "mu", pct, keep & (side == sign))
            s = _stats(sign * test["y_ret"].values[idx] - cost)
            if s and (best["sharpe"] != best["sharpe"] or s["sharpe"] > best["sharpe"]):
                best = {"name": f"{name} top{int(pct * 100)}%", "sharpe": s["sharpe"], "stats": s}
    if "stats" not in best:  # no strategy reached 50 trades (a smoke run or a tiny split)
        out = _section("VERDICT", ["  BEST: none (no strategy reached 50 trades)", "  الحكم: غير قابل للتداول"],
                       "tradeable=False marginal=False")
        out.update(best=best, tradeable=False)
        return out
    b = best["stats"]
    label = ("قابل للتداول نظرياً (Sharpe>0.10)" if best["sharpe"] > 0.10
             else "هامشي - يحتاج تحقق" if best["sharpe"] > 0.05 else "غير قابل للتداول")
    rows = [f"  BEST: {best['name']}  Sharpe={best['sharpe']:+.3f}  PF={b['pf']:.2f}  "
            f"n={b['n']:,}  avg={b['avg'] * 100:+.4f}%",
            f"  الحكم: {label}",
            f"  weekly consistency: {sections['weekly']['note']}",
            "  (the verdict weighs the BEST strategy only - read the shock and weekly sections before "
            "trusting it: an edge that dies in the wildest quintile or lives in one week is not an edge)"]
    out = _section("VERDICT", rows,
                  f"tradeable={best['sharpe'] > 0.10} marginal={0.05 < best['sharpe'] <= 0.10}")
    out["best"] = best
    out["tradeable"] = bool(best["sharpe"] > 0.10)
    return out


def full_battery(val, test, cost: float = 0.0008, n_variants: int = 1) -> dict:
    """The sections, the T-profit sections (battery_profit; ``n_variants`` = variants tried so far, for the CI
    level) AND the verdict in one call - what the notebook prints, in order."""
    sections = battery(val, test, cost)
    return {**sections, **profit_sections(test, cost, n_variants), "verdict": verdict(test, cost, sections)}
