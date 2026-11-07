"""Battery classics: the old TF project checks over the tidy table

PURPOSE: Four classic checks of the old TF project over the tidy table: integrity, persistence baseline, selective
    evaluation (VAL thresholds, TEST judgment), market-neutral portfolio.
TAGS: battery classics, integrity, persistence baseline, selective evaluation, market neutral
PITFALLS: market_neutral needs >= 4 coins plus end_time and persistence needs last_ret (both skip otherwise).
SOURCE: ported from feat/ar-stream 7c9736b, f052cf6 (owner's work)

Ported from the crypto-signal-prediction evaluation package; each re-derived as a pure function over the table,
never over raw features.
"""

from __future__ import annotations

import numpy as np

from nig_timenet_torch.eval.battery_lib import _HEADER, _dir_pnl, _fmt, _scored, _section, _stats, _top_abs
from nig_timenet_torch.stats.metrics import spearman


def table_integrity(val, test) -> dict:
    """Section 11: judge the TABLE before the model - missing values, a dir_true that disagrees with
    y_ret's sign, a p_dir outside [0, 1], or VAL windows that overlap TEST in time all poison every
    other section. The old project's run_full_verification opened with exactly this data check."""
    rows = []
    for name, df in (("VAL", val), ("TEST", test)):
        cols = [c for c in ("mu", "sigma", "p_dir", "y_ret") if c in df]
        bad = int(df[cols].isna().any(axis=1).sum())
        mism = int((np.sign(df["y_ret"]) != df["dir_true"]).sum())
        rows.append(f"  {name:<5} n={len(df):,}  NaN rows={bad}  dir_true mismatches={mism}"
                    f"  p_dir in [{df['p_dir'].min():.3f}, {df['p_dir'].max():.3f}]")
    if "end_time" in val and "end_time" in test:
        late = int((val["end_time"].values >= test["end_time"].values.min()).sum())
        rows.append(f"  VAL windows ending at/after the first TEST end_time: {late:,}  "
                    f"(VAL max {val['end_time'].max()} / TEST min {test['end_time'].min()})")
    return _section("TABLE INTEGRITY - the data first", rows,
                    "A nonzero NaN/mismatch count or VAL/TEST overlap means leakage or a broken dump: "
                    "fix the run before reading on.")


def persistence_baseline(test, cost: float, pct: float = 0.10) -> dict:
    """Section 12: what does the LAST candle alone say? The old project asked whether the model adds
    anything over candle shape; the floor here is momentum - trade sign(last candle's return) on the
    same rows the model trades. If the model cannot beat this line, the 32-candle history adds nothing."""
    keep = _scored(test)
    y, mu, last = test["y_ret"].values, test["mu"].values, test["last_ret"].values
    rows = [f"  {'predictor':<28}{'dir_acc':>9}{'IC':>9}"]
    for name, pred in (("model (mu)", mu), ("last candle (momentum)", last)):
        rows.append(f"  {name:<28}{float(((pred[keep] > 0) == (y[keep] > 0)).mean()):>9.4f}"
                    f"{spearman(pred[keep], y[keep]):>+9.4f}")
    rows.append("")
    rows.append(_HEADER)
    idx = _top_abs(test, "mu", pct, keep)
    for name, sign in (("model direction", np.where(mu[idx] > 0, 1.0, -1.0)),
                       ("momentum direction", np.where(last[idx] > 0, 1.0, -1.0))):
        rows.append(_fmt(f"{name}, top{int(pct * 100)}% |mu| rows", _stats(sign * y[idx] - cost)))
    return _section("PERSISTENCE BASELINE (TEST) - the last candle alone", rows,
                    "Momentum is the cheapest possible edge; a model that only ties it learned the "
                    "previous candle, not the market.")


def selective_evaluation(val, test, cost: float, quantiles: tuple = (0.25, 0.50, 0.75)) -> dict:
    """Section 13: the old project's selective evaluation - every confidence threshold is chosen on
    VAL only and judged on TEST: does accuracy actually rise with confidence, or does it only look
    that way on the split the threshold came from?"""
    rows = []
    for q in quantiles:
        thr = float(np.quantile((val["p_dir"] - 0.5).abs(), q))
        for name, df in (("VAL", val), ("TEST", test)):
            m = ((df["p_dir"] - 0.5).abs() >= thr) & _scored(df)
            if int(m.sum()) < 50:
                rows.append(f"  confidence >= Q{int(q * 100)}(val)  {name:<5}(n<50)")
                continue
            acc = float(((df["p_dir"].values[m] > 0.5) == (df["y_ret"].values[m] > 0)).mean())
            pnl = _stats(_dir_pnl(df, np.where(m)[0], cost))
            rows.append(f"  confidence >= Q{int(q * 100)}(val)  {name:<5} n={int(m.sum()):>7,}"
                        f"  dir_acc={acc:.4f}  sharpe={pnl['sharpe']:+.3f}  avg={pnl['avg'] * 100:+.4f}%")
    return _section("SELECTIVE EVALUATION - VAL thresholds, TEST judgment", rows,
                    "Consistent only if the confident slice beats the overall accuracy on TEST too; "
                    "a VAL-only rise is the split flattering itself.")


def market_neutral(val, test, cost: float, quantiles: tuple = (0.2, 0.3, 0.4)) -> dict:
    """Section 14: the old project's market-neutral portfolio - rank the coins by mu each period,
    long the top q against shorting the bottom q, so the market's whole move cancels and only the
    model's relative ranking can pay. q is chosen on VAL only; TEST rows are shown for transparency."""
    if "coin" not in test or "end_time" not in test:
        return _section("MARKET-NEUTRAL PORTFOLIO - SKIPPED", ["  (needs coin and end_time columns)"],
                        "Single-coin runs cannot go neutral; run the panel notebook for this section.")
    if test["coin"].nunique() < 4:
        return _section("MARKET-NEUTRAL PORTFOLIO - SKIPPED",
                        [f"  (only {test['coin'].nunique()} coins: a cross-section needs >= 4)"],
                        "Run the panel notebook over more coins to enable this section.")

    def _period_pnl(df, q):
        rets = []
        for _, g in df.groupby("end_time"):
            if len(g) < 4:
                continue
            k = max(1, int(len(g) * q))
            ranked = g.sort_values("mu")["y_ret"].values
            rets.append(ranked[-k:].mean() - ranked[:k].mean())
        return np.asarray(rets) - 2 * cost

    best = max(quantiles, key=lambda q: float(_period_pnl(val, q).mean()))
    rows = [_HEADER]
    for q in quantiles:
        mark = " (VAL choice)" if q == best else ""
        rows.append(_fmt(f"long top{int(q * 100)}% / short bottom{int(q * 100)}%{mark}",
                         _stats(_period_pnl(test, q))))
    return _section("MARKET-NEUTRAL PORTFOLIO (TEST)", rows,
                    "Each period pays the full round trip on BOTH legs (2x cost) - the harshest cost "
                    "assumption; a positive VAL-chosen q on TEST is real relative-ranking skill.")


