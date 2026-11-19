"""Battery scorecard: selections x side judged by days, uncertainty buckets, the long / short payoff asymmetry

PURPOSE: Battery sections judging a use of the outputs by its DAYS (mean day with block CI, winning days, worst day,
    CVaR, drawdown) per side and selection (top |mu|, confidence, low sigma); uncertainty buckets; the short exp payoff.
TAGS: scorecard, day distribution, cvar, drawdown, loss concentration, confidence, uncertainty, top k, long short,
    short squeeze, convexity, log return, battery
PITFALLS: y_ret and mu are SIMPLE returns (the adapter decodes the log label with exp); a short pays -y_ret, so its
    loss grows like exp(log move) while its gain is bounded (asymmetry section). Thresholds come from VAL (never TEST);
    the CI level is Bonferroni over the section's rows x n_variants. A day is the calendar day of end_time; a day's
    P&L is the equal-weight mean of the trades of that day.

Owner request 2026-11-19: "make the evaluation take into account the confidence, the distribution of the losses over
the days, the size of the uncertainty and the top". The older battery judges a selection by its per-trade Sharpe
(``verdict``); a trade-level number hides that one to three days decide the sign of a book (baseline-001) and that a
short loses more than it wins per trade although it wins more often (short-side-001). Every section here is a pure
function over the battery table (``eval.battery_adapter``) and returns the battery's ``{"title", "rows", "note"}``
shape plus ``"stats"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.battery_lib import _scored, _section, round_trip
from nig_timenet_torch.stats.metrics import block_bootstrap_ci

__all__ = ["scorecard_sections", "scorecard", "uncertainty_buckets", "payoff_asymmetry", "scorecard_verdict",
           "day_profile", "selections", "TOP_SHARES", "TAIL_SHARE", "DAY_BLOCK", "MIN_TRADES", "N_BUCKETS"]

#: the "top" shares of the selections (of the VAL rows, by |mu|)
TOP_SHARES = (0.05, 0.10, 0.20)
#: the tail of the days / trades read as "the worst" (CVaR, loss concentration)
TAIL_SHARE = 0.05
#: days per bootstrap block (neighbouring days move together)
DAY_BLOCK = 5
#: a selection x side row with fewer trades is not scored
MIN_TRADES = 50
#: buckets of the uncertainty and confidence sections (VAL quantile edges)
N_BUCKETS = 5

_SIDES = (("LONG", 1.0), ("SHORT", -1.0), ("BOTH", 0.0))


def _days(df) -> np.ndarray:
    if "end_time" not in df:
        raise ValueError("the scorecard needs end_time (a day is the unit of judgment)")
    return pd.to_datetime(df["end_time"]).to_numpy().astype("datetime64[D]")


def _side(df) -> np.ndarray:
    return np.where(df["p_dir"].to_numpy() > 0.5, 1.0, -1.0)


def _conf(df) -> np.ndarray:
    """Direction confidence of the close class head: distance of p_dir from a coin flip."""
    return np.abs(df["p_dir"].to_numpy() - 0.5)


def _snr(df) -> np.ndarray:
    """Predicted move over its own uncertainty."""
    return np.abs(df["mu"].to_numpy()) / np.maximum(df["sigma"].to_numpy(), 1e-12)


def day_profile(pnl: np.ndarray, days: np.ndarray, level: float, seed: int = 0) -> dict:
    """The day distribution of a book: every trade's ``pnl`` (net, simple return) is averaged within its day (an
    equal-weight book), then: days, share of winning days, mean day with a ``DAY_BLOCK``-day block CI, median day,
    worst day, CVaR (mean of the worst ``TAIL_SHARE`` of days), the share of all losing-day P&L held by those worst
    days, the max drawdown of the summed day P&L, the mean day without the best ``TAIL_SHARE`` of days, and the mean
    day of the BUSY days (at least the median number of trades): an equal-weight day mean gives a day with one trade
    the weight of a day with a hundred, so a side that loses on its crowded days can still show a positive mean day."""
    pnl = np.asarray(pnl, dtype=np.float64)
    keys, inv = np.unique(days, return_inverse=True)
    counts = np.bincount(inv)
    day = np.bincount(inv, weights=pnl) / counts
    n = day.shape[0]
    k = max(1, int(np.ceil(n * TAIL_SHARE)))
    order = np.sort(day)
    mean, lo, hi = block_bootstrap_ci(day, np.arange(n) // DAY_BLOCK, level=level, seed=seed)
    losing = day[day < 0].sum()
    curve = np.cumsum(day)
    return {"days": int(n), "win_days": float((day > 0).mean()), "mean_day": float(mean), "ci_lo": lo, "ci_hi": hi,
            "median_day": float(np.median(day)), "worst_day": float(order[0]), "cvar": float(order[:k].mean()),
            "tail_loss_share": float(order[:k][order[:k] < 0].sum() / losing) if losing < 0 else 0.0,
            "max_dd": float((np.maximum.accumulate(np.maximum(curve, 0.0)) - curve).max()),
            "mean_wo_best": float(order[:-k].mean()) if n > k else float("nan"),
            "mean_busy": float(day[counts >= np.median(counts)].mean())}


def selections(val, test) -> dict[str, np.ndarray]:
    """Named row masks over TEST, every threshold a VAL quantile (fitted before TEST is read): all rows, top shares
    by |mu|, the most confident fifth, the lowest-sigma fifth, the best fifth by |mu| / sigma, and two combinations
    (top fifth by |mu| or confidence AND sigma at most its VAL median)."""
    def q(arr, share):
        return float(np.quantile(arr, share))

    keep = _scored(test)
    amu_v, amu_t = np.abs(val["mu"].to_numpy()), np.abs(test["mu"].to_numpy())
    sig_v, sig_t = val["sigma"].to_numpy(), test["sigma"].to_numpy()
    low_sig = sig_t <= q(sig_v, 0.5)
    out = {"all rows": keep}
    for share in TOP_SHARES:
        out[f"top {int(share * 100)}% |mu|"] = keep & (amu_t >= q(amu_v, 1.0 - share))
    out["top 20% confidence"] = keep & (_conf(test) >= q(_conf(val), 0.8))
    out["low 20% sigma"] = keep & (sig_t <= q(sig_v, 0.2))
    out["top 20% |mu|/sigma"] = keep & (_snr(test) >= q(_snr(val), 0.8))
    out["top 20% |mu| + sigma<=median"] = out["top 20% |mu|"] & low_sig
    out["top 20% conf + sigma<=median"] = out["top 20% confidence"] & low_sig
    return out


def _level(n_rows: int, n_variants: int) -> float:
    if n_variants < 1:
        raise ValueError("n_variants must be >= 1")
    return 1.0 - 0.05 / (max(n_rows, 1) * n_variants)


def scorecard(val, test, cost: float, n_variants: int = 1) -> dict:
    """Every selection x side (LONG / SHORT / BOTH), net of the round trip, judged by its trades AND its days."""
    side, y, days, rt = _side(test), test["y_ret"].to_numpy(), _days(test), round_trip(cost)
    masks = selections(val, test)
    level = _level(len(masks) * len(_SIDES), n_variants)
    head = (f"  {'selection':<30}{'side':<6}{'n':>8}{'days':>6}{'win%':>7}{'avg%':>9}{'win d%':>8}"
            f"{'mean d% [CI]':>28}{'worst d%':>10}{'CVaR%':>8}{'tail%':>7}{'maxDD%':>8}{'w/o best%':>10}{'busy d%':>9}")
    rows, stats = [head], {}
    for name, mask in masks.items():
        for label, sg in _SIDES:
            idx = np.where(mask & (side == sg))[0] if sg else np.where(mask)[0]
            if len(idx) < MIN_TRADES:
                rows.append(f"  {name:<30}{label:<6}(n<{MIN_TRADES})")
                continue
            pnl = side[idx] * y[idx] - rt
            p = day_profile(pnl, days[idx], level)
            p.update(n=int(len(idx)), win=float((pnl > 0).mean()), avg=float(pnl.mean()))
            stats[f"{name} | {label}"] = p
            rows.append(f"  {name:<30}{label:<6}{p['n']:>8,}{p['days']:>6}{p['win'] * 100:>6.1f}%"
                        f"{p['avg'] * 100:>+8.3f}%{p['win_days'] * 100:>7.1f}%"
                        f"{p['mean_day'] * 100:>+9.3f} [{p['ci_lo'] * 100:+.3f}, {p['ci_hi'] * 100:+.3f}]"
                        f"{p['worst_day'] * 100:>+10.2f}{p['cvar'] * 100:>+8.2f}{p['tail_loss_share'] * 100:>6.0f}%"
                        f"{p['max_dd'] * 100:>8.1f}{p['mean_wo_best'] * 100:>+10.3f}{p['mean_busy'] * 100:>+9.3f}")
    out = _section(
        "SCORECARD (TEST) - selection x side, judged by DAYS (thresholds from VAL)", rows,
        f"net of {rt:.4%} per trade; a day = equal-weight mean of that day's trades; CI {level * 100:.2f}% from "
        f"{DAY_BLOCK}-day blocks (Bonferroni over {len(masks) * len(_SIDES)} rows x n_variants={n_variants}); CVaR = "
        f"mean of the worst {TAIL_SHARE:.0%} days; tail% = their share of all losing-day P&L; maxDD on the summed "
        f"day P&L; w/o best = mean day without the best {TAIL_SHARE:.0%} days; busy = mean day of the days with at "
        "least the median number of trades (where most trades are). A use is real only if its mean day CI "
        "is above 0, it keeps winning days > 50 %, and its losses do not live in a few days (the verdict asks for a "
        "positive mean day without the best days).")
    out["stats"] = stats
    return out


def _bucket_edges(val_values: np.ndarray) -> np.ndarray:
    return np.unique(np.quantile(val_values, np.linspace(0.0, 1.0, N_BUCKETS + 1)[1:-1]))


def uncertainty_buckets(val, test, cost: float) -> dict:
    """TEST rows cut by VAL quantiles of the predicted sigma and of the direction confidence: per bucket the hit
    rate, the mean |realised move|, the net per trade of the LONG and the SHORT rows, and the short's exp() cost."""
    side, y, rt = _side(test), test["y_ret"].to_numpy(), round_trip(cost)
    keep = _scored(test)
    rows, stats = [], {}
    for name, v_val, v_test in (("sigma", val["sigma"].to_numpy(), test["sigma"].to_numpy()),
                                ("confidence", _conf(val), _conf(test))):
        edges = _bucket_edges(v_val)
        b = np.searchsorted(edges, v_test, side="right")
        rows.append(f"  by {name} (VAL quintile edges {', '.join(f'{e:.4f}' for e in edges)})")
        rows.append(f"    {'bucket':<8}{'n':>8}{'hit':>7}{'mean|y|%':>10}{'LONG avg%':>11}{'SHORT avg%':>12}"
                    f"{'SHORT exp cost%':>17}")
        for i in range(len(edges) + 1):
            m = keep & (b == i)
            if m.sum() < MIN_TRADES:
                continue
            lm, sm = m & (side > 0), m & (side < 0)
            hit = float((side[m] * y[m] > 0).mean())
            long_avg = float((y[lm] - rt).mean()) if lm.any() else float("nan")
            short_avg = float((-y[sm] - rt).mean()) if sm.any() else float("nan")
            exp_cost = float((-y[sm] + np.log1p(y[sm])).mean()) if sm.any() else float("nan")
            stats[f"{name} q{i + 1}"] = {"n": int(m.sum()), "hit": hit, "long_avg": long_avg, "short_avg": short_avg,
                                         "short_exp_cost": exp_cost}
            rows.append(f"    q{i + 1:<7}{int(m.sum()):>8,}{hit:>7.3f}{np.abs(y[m]).mean() * 100:>10.3f}"
                        f"{long_avg * 100:>+11.3f}{short_avg * 100:>+12.3f}{exp_cost * 100:>+17.3f}")
    out = _section("UNCERTAINTY AND CONFIDENCE BUCKETS (TEST, edges from VAL)", rows,
                   "A model that knows when it does not know has its hit rate and net rising toward low sigma / high "
                   "confidence. 'SHORT exp cost' = mean of log(1 + y) - y on the short rows: what the short pays "
                   "because a price rise costs more than an equal log fall gains (it grows with sigma).")
    out["stats"] = stats
    return out


def _asym_line(label: str, sg: float, y: np.ndarray, sigma: np.ndarray, mu: np.ndarray, rt: float) -> tuple:
    gross = sg * y
    net = gross - rt
    log_part = sg * np.log1p(y)
    s_log = np.log1p(sigma / (1.0 + mu))
    win, loss = net[net > 0], net[net < 0]
    worst = np.sort(net)[: max(1, len(net) // 100)]
    s = {"n": int(len(net)), "win": float((net > 0).mean()),
         "avg_win": float(win.mean()) if len(win) else 0.0, "avg_loss": float(loss.mean()) if len(loss) else 0.0,
         "net": float(net.mean()), "log_part": float(log_part.mean()), "exp_part": float((gross - log_part).mean()),
         "pred_exp_part": float((sg * 0.5 * s_log**2).mean()), "sd": float(y.std()),
         "worst1_share": float(worst.sum() / loss.sum()) if loss.sum() < 0 else 0.0}
    s["loss_win"] = -s["avg_loss"] / s["avg_win"] if s["avg_win"] > 0 else float("nan")
    pct = {k: s[k] * 100 for k in ("win", "avg_win", "avg_loss", "log_part", "exp_part", "pred_exp_part", "net", "sd",
                                   "worst1_share")}
    return (f"  {label:<32}{s['n']:>7,}{pct['win']:>6.1f}%{pct['avg_win']:>+8.3f}{pct['avg_loss']:>+8.3f}"
            f"{s['loss_win']:>6.2f}{pct['log_part']:>+9.3f}{pct['exp_part']:>+9.3f}{pct['pred_exp_part']:>+10.3f}"
            f"{pct['net']:>+9.3f}{pct['sd']:>7.2f}{pct['worst1_share']:>7.0f}%", s)


def payoff_asymmetry(val, test, cost: float) -> dict:
    """Why a side can win more often and still lose: per side and top share, the trade P&L split into the part a
    symmetric LOG forecast is paid (side x log(1 + y)) and the exp() part (side x (y - log(1 + y)): >= 0 for a long,
    <= 0 for a short), next to the exp() part the model's own sigma predicts (side x s^2 / 2, s the log-space sd).
    The last rows are the mean-corrected rule: trade only when the predicted LOG move beats the exp() part plus cost."""
    side, y, rt = _side(test), test["y_ret"].to_numpy(), round_trip(cost)
    mu, sigma = test["mu"].to_numpy(), test["sigma"].to_numpy()
    keep = _scored(test)
    head = (f"  {'selection':<32}{'n':>7}{'win%':>7}{'avgW%':>8}{'avgL%':>8}{'L/W':>6}{'log%':>9}{'exp%':>9}"
            f"{'pred exp%':>10}{'net%':>9}{'sd%':>7}{'worst1%':>8}")
    rows, stats = [head], {}
    amu_v = np.abs(val["mu"].to_numpy())
    for share in (*TOP_SHARES, 1.0):
        top = keep & (np.abs(mu) >= float(np.quantile(amu_v, 1.0 - share))) if share < 1.0 else keep
        for label, sg in _SIDES[:2]:
            idx = np.where(top & (side == sg))[0]
            name = f"{label} {'all rows' if share == 1.0 else f'top {int(share * 100)}% |mu|'}"
            if len(idx) < MIN_TRADES:
                rows.append(f"  {name:<32}(n<{MIN_TRADES})")
                continue
            line, stats[name] = _asym_line(name, sg, y[idx], sigma[idx], mu[idx], rt)
            rows.append(line)
    m_log = np.log1p(mu)
    drag = 0.5 * np.log1p(sigma / (1.0 + mu)) ** 2
    for label, sg, rule in (("LONG  log mu + s^2/2 > cost", 1.0, m_log + drag > rt),
                            ("SHORT -log mu - s^2/2 > cost", -1.0, -m_log - drag > rt)):
        idx = np.where(keep & rule)[0]
        if len(idx) < MIN_TRADES:
            rows.append(f"  {label:<32}(n<{MIN_TRADES})")
            continue
        line, stats[label] = _asym_line(label, sg, y[idx], sigma[idx], mu[idx], rt)
        rows.append(line)
    out = _section(
        "LONG vs SHORT PAYOFF - the exp() of a log forecast (TEST)", rows,
        "log% + exp% = gross; net = gross - round trip. The model is trained on log(future / close), a loss symmetric "
        "in up and down, but a trade is paid in price: a long earns e^l - 1 >= l, a short earns 1 - e^l <= -l. So a "
        "short that is RIGHT in log units (log% > 0) can still lose: its exp% is about -s^2/2 per trade and grows with "
        "the volatility of the rows it picks. L/W = mean loss over mean win; worst1% = share of all losses held by the "
        "worst 1% of trades. The two mean-corrected rows use mu and sigma only (a variant: count it).")
    out["stats"] = stats
    return out


def scorecard_verdict(card: dict) -> dict:
    """The best scorecard row by the LOWER bound of its mean-day CI (not by a per-trade Sharpe), with its day
    shape beside it. A row passes only with the CI above 0, winning days above 50 %, a mean day that stays above 0
    without its best ``TAIL_SHARE`` of days (an edge that lives in a few days is not an edge) and a positive mean on
    its busy days (an edge that lives on thin days is not tradeable at size)."""
    stats = card.get("stats", {})
    if not stats:
        out = _section("SCORECARD VERDICT", ["  no selection x side reached the minimum trades"], "pass=False")
        out.update(best=None, passed=False)
        return out
    name, best = max(stats.items(), key=lambda kv: (kv[1]["ci_lo"] if kv[1]["ci_lo"] == kv[1]["ci_lo"] else -1e9))
    passed = bool(best["ci_lo"] > 0 and best["win_days"] > 0.5 and best["mean_wo_best"] > 0
                  and best["mean_busy"] > 0)
    n_pos = sum(1 for s in stats.values() if s["mean_day"] > 0)
    rows = [f"  BEST (by mean-day CI low): {name}  mean day {best['mean_day'] * 100:+.3f}% "
            f"[{best['ci_lo'] * 100:+.3f}, {best['ci_hi'] * 100:+.3f}]  winning days {best['win_days'] * 100:.1f}%  "
            f"worst day {best['worst_day'] * 100:+.2f}%  without the best days {best['mean_wo_best'] * 100:+.3f}%  "
            f"busy days {best['mean_busy'] * 100:+.3f}%",
            f"  rows with a positive mean day: {n_pos} of {len(stats)} (a share near half is what noise gives)",
            f"  pass: {passed}"]
    out = _section("SCORECARD VERDICT", rows, f"pass={passed} (judged by days, Bonferroni CI)")
    out.update(best=name, passed=passed)
    return out


def scorecard_sections(val, test, cost: float, n_variants: int = 1) -> dict:
    """The four sections in the order the battery prints them (skipped when TEST has no end_time)."""
    if "end_time" not in test:
        return {}
    card = scorecard(val, test, cost, n_variants)
    return {"scorecard": card, "buckets": uncertainty_buckets(val, test, cost),
            "asymmetry": payoff_asymmetry(val, test, cost), "scorecard_verdict": scorecard_verdict(card)}
