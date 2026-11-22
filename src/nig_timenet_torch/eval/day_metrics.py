"""PURPOSE: The owner's day metrics of a multi-coin model: the share of winning DAYS of the equal-weight sign
    portfolio and of the market-neutral top / bottom book, gross and net, with baselines, CIs and paired tests.
TAGS: winning days, day portfolio, absolute direction, market neutral, long short, top k, relative rank ic,
    baselines, always long, market sign, momentum, reversal, paired block bootstrap, t-profit
PITFALLS: a frame is ONE stage of ONE run with the columns of ``FRAME_COLUMNS``; a day is the UTC calendar day of the
    prediction time and needs ``min_rows`` rows; rows whose score is NaN or 0 do not trade; metric 2 needs a score
    that DIFFERS across the coins of a timestamp (a constant score gives no day). Baselines whose sign is unknown
    (last-bar asymmetry) are oriented on the VAL frame, never on the frame they are scored on. Cost is charged per
    side: a row is a round trip (2 sides); the long and the short leg of metric 2 are half a unit each (2 sides per
    unit of gross exposure). Day P&L of overlapping labels (stride < horizon) is the mean over the tranches.

Trivial models (rule C1): a score that is independent of the future scores 50 % winning days on a market with a
symmetric return (CI around 0.5) and exactly minus the cost after cost; ``always_long`` scores the market's own share
of up days, which is why it is a baseline every score must beat.
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.battery_lib import DEFAULT_COST_PER_SIDE, net_pnl
from nig_timenet_torch.eval.day_blocks import DAY_BLOCK_DAYS, calendar_day, day_blocks
from nig_timenet_torch.stats import metrics

__all__ = ["FRAME_COLUMNS", "absolute_days", "neutral_days", "relative_rank_ic", "day_summary", "paired_days",
           "baseline_scores", "orient", "day_report", "accept", "render", "ABSOLUTE_BASELINES", "NEUTRAL_BASELINES"]

FRAME_COLUMNS = ("time", "coin", "mu", "sigma", "y_ret", "ret_1d", "asym")
ABSOLUTE_BASELINES = ("always_long", "yesterday_market_sign", "last_bar_asymmetry")
NEUTRAL_BASELINES = ("momentum_1d", "reversal_1d", "last_bar_asymmetry")
MIN_ROWS = 3            # rows a day needs to count


def _days(frame: pd.DataFrame) -> np.ndarray:
    return calendar_day(frame["time"])


def absolute_days(frame: pd.DataFrame, score, cost: float = 0.0, min_rows: int = MIN_ROWS) -> pd.Series:
    """Metric 1: per day, the mean over the day's rows of ``sign(score_i) * y_ret_i - 2 * cost`` (rows with a NaN or
    zero score do not trade). Index = day."""
    s = np.sign(np.nan_to_num(np.asarray(score, dtype=np.float64)))
    pnl = pd.Series(np.where(s != 0, net_pnl(s * frame["y_ret"].to_numpy(np.float64), cost), np.nan))
    g = pnl.groupby(_days(frame))
    return g.mean()[g.count() >= min_rows]


def _timestamp_spread(frame: pd.DataFrame, score, top_share: float) -> pd.Series:
    """Per timestamp: ``stats.metrics.long_short_spreads`` of ``score`` against the realised return."""
    keys, vals = metrics.long_short_spreads(frame["y_ret"].to_numpy(np.float64), score,
                                            frame["time"].to_numpy().astype("datetime64[ns]"), top_share)
    return pd.Series(vals, index=pd.DatetimeIndex(keys), dtype=np.float64)


def neutral_days(frame: pd.DataFrame, score, top_share: float = 0.2, cost: float = 0.0,
                 min_rows: int = 1) -> pd.Series:
    """Metric 2: per day, the mean over the day's timestamps of the top-k / bottom-k spread of ``score`` (which is
    ranked inside the timestamp, so ``score`` and ``score - timestamp mean`` give the same book), minus one round trip
    (2 x cost) per unit of gross exposure."""
    # half-unit convention: the spread (mean of the top k minus mean of the bottom k) is already per unit of gross
    # exposure (the long and the short leg are half a unit each), and a unit of exposure pays one round trip
    spread = net_pnl(_timestamp_spread(frame, score, top_share), cost)
    if not len(spread):
        return pd.Series(dtype=np.float64)
    g = spread.groupby(calendar_day(spread.index))
    return g.mean()[g.count() >= min_rows]


def relative_rank_ic(frame: pd.DataFrame, score) -> float:
    """Mean over timestamps of the Spearman correlation of ``score`` with the realised move RELATIVE to the market
    (y_i - mean_j y_j). TRIVIAL: a score independent of the future gives 0; a constant score gives NaN."""
    t = frame["time"].to_numpy().astype(np.int64)
    y = frame["y_ret"].to_numpy(np.float64)
    rel = y - pd.Series(y).groupby(t).transform("mean").to_numpy()
    return metrics.rank_ic(rel, np.asarray(score, dtype=np.float64), t)


def day_summary(days: pd.Series, level: float = 0.95, block_days: int = DAY_BLOCK_DAYS, seed: int = 0) -> dict:
    """{n_days, win_share, win_lo, win_hi (block bootstrap over ``block_days``-day blocks), wilson_lo, wilson_hi
    (days as independent), mean, mean_lo, mean_hi, sharpe_day}."""
    v = days.to_numpy(np.float64)
    n = len(v)
    if n == 0:
        return {"n_days": 0}
    blocks = day_blocks(days.index, block_days)
    win, wl, wh = metrics.block_bootstrap_ci((v > 0).astype(np.float64), blocks, level=level, seed=seed)
    mean, ml, mh = metrics.block_bootstrap_ci(v, blocks, level=level, seed=seed)
    wil = metrics.wilson_interval(int((v > 0).sum()), n)
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    return {"n_days": n, "win_share": win, "win_lo": wl, "win_hi": wh, "wilson_lo": wil[0], "wilson_hi": wil[1],
            "mean": mean, "mean_lo": ml, "mean_hi": mh, "sharpe_day": mean / sd if sd and sd > 0 else float("nan")}


def paired_days(model: pd.Series, base: pd.Series, level: float = 0.95, block_days: int = DAY_BLOCK_DAYS,
                seed: int = 0) -> dict:
    """Model minus baseline on the days BOTH have: the difference of the mean day P&L and of the winning-day share,
    each with a paired block-bootstrap CI; ``wins`` = the lower end of BOTH CIs is above 0."""
    both = pd.concat([model.rename("m"), base.rename("b")], axis=1).dropna()
    if len(both) < 2:
        return {"n_days": int(len(both)), "wins": False}
    blocks = day_blocks(both.index, block_days)
    d = (both["m"] - both["b"]).to_numpy()
    w = (both["m"] > 0).to_numpy(np.float64) - (both["b"] > 0).to_numpy(np.float64)
    dm, dl, dh = metrics.block_bootstrap_ci(d, blocks, level=level, seed=seed)
    wm, wl, wh = metrics.block_bootstrap_ci(w, blocks, level=level, seed=seed)
    return {"n_days": int(len(both)), "d_mean": dm, "d_mean_lo": dl, "d_mean_hi": dh, "d_win": wm, "d_win_lo": wl,
            "d_win_hi": wh, "wins": bool(dl > 0 and wl > 0)}


def orient(val: pd.DataFrame, score) -> float:
    """+1 / -1: the sign that gives ``score`` a non-negative rank IC with the realised return on the VAL frame."""
    ic = metrics.rank_ic(val["y_ret"].to_numpy(np.float64), np.asarray(score, dtype=np.float64),
                         val["time"].to_numpy().astype(np.int64))
    return -1.0 if np.isfinite(ic) and ic < 0 else 1.0


def baseline_scores(frame: pd.DataFrame, val: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    """name -> score per row. ``always_long`` +1; ``yesterday_market_sign`` the sign of the mean over the coins of
    the timestamp of their trailing 24 h return (``ret_1d``); ``momentum_1d`` / ``reversal_1d`` +- the coin's own
    trailing 24 h return (ranked inside the timestamp: the previous day's relative-move ranking); ``last_bar_asymmetry``
    (DS1) log(H / C) + log(L / C) of the last bar, oriented on ``val`` (as it is when ``val`` is None)."""
    r1 = frame["ret_1d"].to_numpy(np.float64)
    market = pd.Series(r1).groupby(frame["time"].to_numpy().astype(np.int64)).transform("mean").to_numpy()
    asym = frame["asym"].to_numpy(np.float64)
    sign = orient(val, val["asym"].to_numpy(np.float64)) if val is not None else 1.0
    return {"always_long": np.ones(len(frame)), "yesterday_market_sign": np.sign(market), "momentum_1d": r1,
            "reversal_1d": -r1, "last_bar_asymmetry": sign * asym}


def _section(frame, scores: dict, model: str, days_fn, cost: float, level: float, seed: int) -> dict:
    gross = {k: days_fn(frame, s, cost=0.0) for k, s in scores.items()}
    net = {k: days_fn(frame, s, cost=cost) for k, s in scores.items()}
    out = {k: {"gross": day_summary(gross[k], level, seed=seed), "net": day_summary(net[k], level, seed=seed)}
           for k in scores}
    for k in scores:
        if k != model:
            out[k]["model_minus_this"] = paired_days(gross[model], gross[k], level, seed=seed)
    out[model]["beats_every_baseline"] = bool(all(out[k]["model_minus_this"]["wins"] for k in scores if k != model))
    return out


def day_report(frame: pd.DataFrame, val: pd.DataFrame | None = None, *, cost: float = DEFAULT_COST_PER_SIDE,
               top_share: float = 0.2, level: float = 0.95, seed: int = 0) -> dict:
    """Both metrics of one stage for the model (``mu``) and every baseline: per score the gross and net day summary
    and, for a baseline, the paired model-minus-baseline comparison on the gross days. Keys: ``absolute``,
    ``neutral`` ({score name: {...}}), ``absolute_relative_sign`` (metric 1 with the sign of ``mu`` minus its
    timestamp mean; information), ``relative_rank_ic`` ({score name: value}), ``settings``."""
    missing = [c for c in FRAME_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"day_report: the frame lacks {missing}")
    base = baseline_scores(frame, val)
    mu = frame["mu"].to_numpy(np.float64)
    absolute = {"model": mu, **{k: base[k] for k in ABSOLUTE_BASELINES}}
    neutral = {"model": mu, **{k: base[k] for k in NEUTRAL_BASELINES}}
    rel = mu - pd.Series(mu).groupby(frame["time"].to_numpy().astype(np.int64)).transform("mean").to_numpy()
    return {"absolute": _section(frame, absolute, "model", absolute_days, cost, level, seed),
            # information only (not the pre-registered model score): the sign of the mean RELATIVE to its timestamp,
            # i.e. the same equal-weight book made market-neutral by construction
            "absolute_relative_sign": {"gross": day_summary(absolute_days(frame, rel), level, seed=seed),
                                       "net": day_summary(absolute_days(frame, rel, cost=cost), level, seed=seed)},
            "neutral": _section(frame, neutral, "model",
                                lambda f, s, cost: neutral_days(f, s, top_share=top_share, cost=cost), cost, level,
                                seed),
            "relative_rank_ic": {k: relative_rank_ic(frame, s) for k, s in neutral.items()},
            "settings": {"cost_per_side": cost, "top_share": top_share, "level": level, "block_days": DAY_BLOCK_DAYS,
                         "rows": int(len(frame)), "coins": int(frame["coin"].nunique()),
                         "timestamps": int(frame["time"].nunique())}}


def accept(reports: list[dict]) -> dict:
    """The owner's accept rule over the seeds of one arm (rule C4): on >= 2 seeds, a significant paired win over
    EVERY baseline, per metric. ``reports`` = one ``day_report`` per seed."""
    out = {}
    for metric in ("absolute", "neutral"):
        wins = [bool(r[metric]["model"]["beats_every_baseline"]) for r in reports]
        out[metric] = {"seeds": len(wins), "seeds_beating_every_baseline": int(sum(wins)),
                       "accepted": len(wins) >= 2 and all(wins)}
    return out


def _row(name: str, r: dict) -> str:
    g, n = r["gross"], r["net"]
    if not g.get("n_days"):
        return f"| {name} | 0 | | | | |"
    p = r.get("model_minus_this")
    vs = "" if p is None else (f"{p['d_win'] * 100:+.1f} pp [{p['d_win_lo'] * 100:+.1f}, {p['d_win_hi'] * 100:+.1f}]"
                               f"{' WIN' if p['wins'] else ''}" if p.get("n_days", 0) >= 2 else "n/a")
    return (f"| {name} | {g['n_days']} | {g['win_share'] * 100:.1f} % [{g['win_lo'] * 100:.1f}, "
            f"{g['win_hi'] * 100:.1f}] | {g['mean'] * 100:+.4f} % | {n['mean'] * 100:+.4f} % "
            f"[{n['mean_lo'] * 100:+.4f}, {n['mean_hi'] * 100:+.4f}] | {vs} |")


def render(report: dict, title: str = "") -> str:
    """Markdown tables of one ``day_report``."""
    s = report["settings"]
    lines = [f"### {title}".rstrip(), "",
             f"{s['rows']} rows, {s['coins']} coins, {s['timestamps']} timestamps; cost {s['cost_per_side']} per side; "
             f"CI {s['level']:.0%} from {s['block_days']}-day blocks.", ""]
    for metric, head in (("absolute", "Metric 1, absolute direction (equal-weight sign portfolio)"),
                         ("neutral", f"Metric 2, market-neutral top / bottom {s['top_share']:.0%}")):
        lines += [f"**{head}**", "",
                  "| score | days | winning days, gross [CI] | mean day, gross | mean day, net [CI] | "
                  "model - this, winning days [CI] |", "|---|---|---|---|---|---|"]
        lines += [_row(k, v) for k, v in report[metric].items()]
        if metric == "absolute":
            lines.append(_row("model, sign of (mean - timestamp mean) (information)", report["absolute_relative_sign"]))
        lines.append("")
    ic = ", ".join(f"{k} {v:+.4f}" for k, v in report["relative_rank_ic"].items())
    lines.append(f"Cross-sectional rank IC of the relative move: {ic}.")
    return "\n".join(lines) + "\n"
