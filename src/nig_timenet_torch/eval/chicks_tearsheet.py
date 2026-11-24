"""PURPOSE: The TF tearsheet of the chicks report on percent returns: Sharpe, Sortino, Omega, profit factor, SQN,
    drawdown, equity curve, and the significance report (t test, bootstrap, permutation, baselines).
TAGS: chicks, tearsheet, sharpe, sortino, omega, profit factor, sqn, equity curve, drawdown, significance, permutation
    test, bootstrap, momentum baseline, reversal baseline, market neutral, tf parity
PITFALLS: returns are in PERCENT (not fractions) and clipped at -100 % per trade; ``risk_per_trade`` is the share of
    the capital put on each trade (0.1 per trade, 1.0 for a portfolio of equal-weight periods); the TF SQN is capped
    at 100 trades and the Sharpe is per trade unless ``periods_per_year`` is given (then sqrt(periods) annualised);
    ``eval.legacy_reports.tearsheet`` is a DIFFERENT tearsheet (fractions, the battery's trades): this one is the
    chicks report's, over the flat frame's close rows, and the two are not interchangeable; the significance report
    shuffles each period's signs ``n_perm`` times with a seeded generator (``seed``), so it is reproducible, and
    ``market_neutral`` subtracts each timestamp's mean return before trading.
SOURCE: crypto-signal-prediction ``evaluation/tearsheet.py``; the Student-t tail is ``stats.tdist.t_sf`` (no scipy).
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.chicks_text import Say
from nig_timenet_torch.stats.tdist import t_sf

__all__ = ["compute_trading_performance_metrics", "simulate_equity_curve", "print_performance_verdict",
           "signal_significance_report", "RISK_PER_TRADE", "N_PERM", "N_BOOT", "MIN_PERIOD_SECONDS_NS"]

RISK_PER_TRADE = 0.1             # the capital share put on each trade of a per-trade equity curve
N_PERM = 1000                    # permutations of the significance test
N_BOOT = 2000                    # bootstrap draws of the Sharpe interval
MIN_PERIOD_SECONDS_NS = 60e9     # a median time step below a minute is not a real time (sequence numbers)


def simulate_equity_curve(returns_pct, initial_capital: float = 10000.0,
                          risk_per_trade: float = RISK_PER_TRADE) -> dict:
    """{'equity', 'drawdown_pct', 'running_max'}: the capital after each trade when ``risk_per_trade`` of it is put on
    every trade (returns in percent, clipped at -100)."""
    r = np.maximum(np.asarray(returns_pct, dtype=np.float64), -100.0)
    equity = np.empty(len(r) + 1)
    equity[0] = initial_capital
    for i, ret in enumerate(r):
        equity[i + 1] = equity[i] + equity[i] * risk_per_trade * (ret / 100.0)
    running_max = np.maximum.accumulate(equity)
    return {"equity": equity, "drawdown_pct": (equity - running_max) / running_max * 100, "running_max": running_max}


def compute_trading_performance_metrics(returns_pct, risk_free_pct: float = 0.0, risk_per_trade: float = RISK_PER_TRADE,
                                        periods_per_year: float | None = None) -> dict:
    """Sharpe, Sortino, max drawdown, profit factor, win rate, Omega, SQN and the total return of a series of returns
    in percent; {} for an empty series. Sharpe and Sortino are per trade unless ``periods_per_year`` annualises them."""
    r = np.asarray(returns_pct, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return {}
    r = np.maximum(r, -100.0)
    excess = r - risk_free_pct
    ann = np.sqrt(periods_per_year) if periods_per_year else 1.0
    sharpe = float(np.mean(excess) / (np.std(excess) + 1e-9)) * ann
    downside = excess[excess < 0]
    sortino = float(np.mean(excess) / (np.std(downside) + 1e-9)) * ann if len(downside) > 0 else float("inf")
    eq_curve = simulate_equity_curve(r, initial_capital=100.0, risk_per_trade=risk_per_trade)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    gains_omega, losses_omega = np.sum(np.maximum(r, 0)), np.sum(np.maximum(-r, 0))
    return {"n_trades": len(r), "sharpe_basis": "annualized" if periods_per_year else "per_trade",
            "sharpe_ratio": sharpe, "sortino_ratio": sortino, "max_drawdown_pct": float(eq_curve["drawdown_pct"].min()),
            "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
            "win_rate_pct": float(np.mean(r > 0)) * 100,
            "omega_ratio": float(gains_omega / losses_omega) if losses_omega > 0 else float("inf"),
            "sqn": float(np.mean(r) / (np.std(r) + 1e-9)) * np.sqrt(min(len(r), 100)),
            "total_return_pct": float(eq_curve["equity"][-1] - 100.0), "avg_return_per_trade_pct": float(np.mean(r)),
            "risk_per_trade": risk_per_trade}


def print_performance_verdict(m: dict, say: Say | None = None) -> None:
    """The quick qualitative verdict printed under the tearsheet numbers."""
    say = say or Say()
    if not m:
        say("tear_no_data")
        return
    if m.get("sharpe_basis") == "annualized":
        s = m["sharpe_ratio"]
        say("tear_sharpe_ann", v=s, j=say.text("tear_good" if s > 1 else "tear_weak" if s > 0 else "tear_neg"))
    else:
        say("tear_sharpe_trade", v=m["sharpe_ratio"])
    say("tear_sortino", v=m["sortino_ratio"])
    d = m["max_drawdown_pct"]
    say("tear_dd", v=d, j=say.text("tear_dd_ok" if d > -20 else "tear_dd_high"))
    pf = m["profit_factor"]
    say("tear_pf", v=pf, j=say.text("tear_pf_strong" if pf > 2 else "tear_pf_ok" if pf > 1 else "tear_pf_bad"))
    say("tear_omega", v=m["omega_ratio"])
    q = m["sqn"]
    say("tear_sqn", v=q, j=say.text("tear_sqn_top" if q > 3 else "tear_sqn_ok" if q > 2 else "tear_sqn_weak"))
    say("tear_total", v=m["total_return_pct"], n=m["n_trades"],
        u=say.text("tear_unit_period" if m.get("sharpe_basis") == "annualized" else "tear_unit_trade"))


def _period_stats(p, periods_per_year, n_boot: int = N_BOOT, rng=None) -> dict:
    p = np.asarray(p, dtype=np.float64)
    p = p[np.isfinite(p)]
    T = len(p)
    if T < 3:
        return {"n_periods": T}
    sd, m = float(np.std(p, ddof=1)), float(np.mean(p))
    t = m / (sd + 1e-12) * np.sqrt(T)
    ann = np.sqrt(periods_per_year) if periods_per_year else 1.0
    rng = rng or np.random.default_rng(0)
    bs = p[rng.integers(0, T, size=(n_boot, T))]
    boot_sharpe = bs.mean(1) / (bs.std(1, ddof=1) + 1e-12) * ann
    half = T // 2

    def sh(q):
        return float(np.mean(q) / (np.std(q, ddof=1) + 1e-12) * ann) if len(q) > 2 else float("nan")

    return {"n_periods": T, "mean_pct": m, "sharpe": m / (sd + 1e-12) * ann, "t_stat": float(t),
            "p_value_one_sided": float(t_sf(float(t), T - 1)),
            "sharpe_ci95": (float(np.percentile(boot_sharpe, 2.5)), float(np.percentile(boot_sharpe, 97.5))),
            "first_half_sharpe": sh(p[:half]), "second_half_sharpe": sh(p[half:]),
            "first_half_sum_pct": float(np.sum(p[:half])), "second_half_sum_pct": float(np.sum(p[half:]))}


def _print_significance(out: dict, market_neutral: bool, say: Say) -> None:
    print("\n" + "=" * 100)
    say("sig_title", mn=say.text("sig_mn") if market_neutral else "")
    print("=" * 100)
    say("sig_table_head")
    for name, key in (("model", "sig_model"), ("momentum", "sig_momentum"), ("reversal", "sig_reversal")):
        s = out[name]
        if "sharpe" not in s:
            continue
        lo, hi = s["sharpe_ci95"]
        print(f"{say.text(key):<12} {s['sharpe']:>7.2f} {f'[{lo:+.2f}, {hi:+.2f}]':>17} {s['t_stat']:>6.2f} "
              f"{s['p_value_one_sided']:>7.3f} {s['first_half_sum_pct']:>+9.2f} {s['second_half_sum_pct']:>+9.2f}")
    pm = out["permutation"]
    say("sig_perm", n=pm["n_perm"], o=pm["observed_mean_pct"], nl=pm["null_mean_pct"], p95=pm["null_p95_pct"],
        p=pm["p_value"])
    say("sig_agree", a=out["agreement_with_reversal_pct"])
    m = out["model"]
    if "sharpe" in m:
        beats = all(m["sharpe"] > out[b].get("sharpe", -np.inf) for b in ("momentum", "reversal"))
        say("sig_skill", v=say.text("sig_yes" if pm["p_value"] < 0.05 else "sig_no_skill"), p=pm["p_value"])
        periods = int(np.ceil(4 * m["n_periods"] / m["t_stat"] ** 2)) if m["t_stat"] > 0 else "∞"
        say("sig_profit", v=say.text("sig_sig" if m["t_stat"] >= 2 else "sig_not_sig"), t=m["t_stat"], k=periods)
        say("sig_rules", v=say.text("sig_beats" if beats else "sig_not_beats"))
        if m["first_half_sum_pct"] > 0 >= m["second_half_sum_pct"]:
            say("sig_regime")


def signal_significance_report(close_df: pd.DataFrame, trade_cost_pct: float = 0.08, market_neutral: bool = True,
                               periods_per_year: float | None = None, n_perm: int = N_PERM, seed: int = 0,
                               verbose: bool = True, say: Say | None = None) -> dict:
    """Is the profit a signal or luck? ``close_df``: the close rows of the flat frame (asset, entry, true,
    predicted_change, timestamp). Returns the stats of the model, of the momentum and of the reversal rule, the
    permutation test and the agreement with the reversal rule; {} when there are fewer than 3 timestamps."""
    say = say or Say()
    df = close_df[["asset", "entry", "true", "predicted_change", "timestamp"]].dropna()
    df = df.sort_values(["timestamp", "asset"], kind="mergesort").reset_index(drop=True)
    if df["timestamp"].nunique() < 3:
        return {}
    ret = ((df["true"] - df["entry"]) / (df["entry"].abs() + 1e-7) * 100).values
    _, g = np.unique(df["timestamp"].values, return_inverse=True)
    if market_neutral:
        ret = ret - (np.bincount(g, ret) / np.bincount(g))[g]
    prev = df.groupby("asset")["entry"].shift(1).values
    past = (df["entry"].values - prev) / (np.abs(prev) + 1e-7)
    past_sign = np.where(np.isfinite(past), np.sign(past), 0.0)
    model_sign = np.sign(df["predicted_change"].values)

    def period_returns(sign):
        r = sign * ret - trade_cost_pct * (sign != 0)
        n = np.bincount(g, (sign != 0).astype(float))
        return np.where(n > 0, np.bincount(g, r) / np.maximum(n, 1), np.nan)

    rng = np.random.default_rng(seed)
    obs = period_returns(model_sign)
    out = {"model": _period_stats(obs, periods_per_year, rng=rng),
           "momentum": _period_stats(period_returns(past_sign), periods_per_year, rng=rng),
           "reversal": _period_stats(period_returns(-past_sign), periods_per_year, rng=rng)}
    obs_mean = float(np.nanmean(obs))
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = np.lexsort((rng.random(len(g)), g))
        null[k] = np.nanmean(period_returns(model_sign[perm]))
    out["permutation"] = {"n_perm": n_perm, "observed_mean_pct": obs_mean, "null_mean_pct": float(null.mean()),
                          "null_p95_pct": float(np.percentile(null, 95)),
                          "p_value": float((1 + np.sum(null >= obs_mean)) / (1 + n_perm))}
    valid = (past_sign != 0) & (model_sign != 0)
    out["agreement_with_reversal_pct"] = (float(np.mean(model_sign[valid] == -past_sign[valid]) * 100) if valid.any()
                                          else float("nan"))
    if verbose:
        _print_significance(out, market_neutral, say)
    return out
