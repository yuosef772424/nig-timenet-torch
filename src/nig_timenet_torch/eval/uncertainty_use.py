"""PURPOSE: Is the predicted uncertainty USEFUL on a panel: does the sd rank the absolute error across coins and
    across time, is it sharp and calibrated, is the direction biased, does selecting by confidence win more days.
TAGS: uncertainty use, sd ranks error, across coins, across time, t-sharp, calibration, direction bias, confidence
    selection, winning days, panel diagnostics
PITFALLS: works on the day-metric frame (``eval.day_metrics.FRAME_COLUMNS``; one stage of one run); a confidence
    threshold is fitted on the VAL frame and applied to the scored frame, never fitted on it; "across time" is a
    correlation inside each coin over its timestamps (it needs >= ``MIN_GROUP`` rows per coin), "across coins" inside
    each timestamp. Every number is descriptive: no trade rule is chosen here.

Trivial models (rule C1): a constant sd gives NaN rank correlations (no ranking), sharpness 0 or 1 and whatever
coverage the constant gives; a model whose mean is 0 everywhere has no direction (share_pred_up 0).
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.day_metrics import absolute_days, day_summary, neutral_days
from nig_timenet_torch.stats import metrics

__all__ = ["sd_ranks_error", "sharp_and_calibrated", "direction_bias", "confidence_selection", "uncertainty_report",
           "render"]

MIN_GROUP = 20
Z80 = 1.2815515655446004


def sd_ranks_error(frame: pd.DataFrame, mu: str = "mu", sigma: str = "sigma", y: str = "y_ret") -> dict:
    """Spearman of the predictive sd with |y - mu|: ``across_coins`` = mean over timestamps (inside a timestamp),
    ``across_time`` = mean over coins (inside a coin), ``pooled`` = all rows. A useful sd is positive on all three."""
    err = (frame[y] - frame[mu]).abs().to_numpy(np.float64)
    sd = frame[sigma].to_numpy(np.float64)
    t = frame["time"].to_numpy().astype(np.int64)
    coin = pd.factorize(frame["coin"])[0]
    return {"across_coins": metrics.rank_ic(err, sd, t), "across_time": metrics.rank_ic(err, sd, coin),
            "pooled": metrics.spearman(err, sd)}


def sharp_and_calibrated(frame: pd.DataFrame, mu: str = "mu", sigma: str = "sigma", y: str = "y_ret") -> dict:
    """T-sharp (share of rows whose sd is below the predicted move |mu|), the coverage of mu +- 1.28 sd (0.80 when
    Gaussian and calibrated) and the sd of the standardised error (1.0 when calibrated)."""
    m, s, yy = (frame[c].to_numpy(np.float64) for c in (mu, sigma, y))
    z = (yy - m) / np.where(s > 0, s, np.nan)
    return {"t_sharp": metrics.width_below_share(s, np.abs(m)),
            "coverage_80": metrics.interval_coverage(yy, m - Z80 * s, m + Z80 * s),
            "z_sd": float(np.nanstd(z)), "mean_sd": float(np.mean(s)), "mean_abs_mu": float(np.mean(np.abs(m))),
            "mean_abs_error": float(np.mean(np.abs(yy - m)))}


def direction_bias(frame: pd.DataFrame) -> dict:
    """Mean predicted sign against the realised one: over all rows, per coin (correlation over coins of the share of
    predicted-up rows with the share of realised-up rows) and per day (the same over days), plus the share of
    timestamps in which every coin gets the SAME predicted sign (the model is then a market-timing rule)."""
    up_p, up_y = (frame["mu"] > 0).to_numpy(np.float64), (frame["y_ret"] > 0).to_numpy(np.float64)
    df = pd.DataFrame({"p": up_p, "y": up_y, "coin": frame["coin"].to_numpy(),
                       "day": frame["time"].to_numpy().astype("datetime64[D]"), "t": frame["time"].to_numpy()})
    per_coin, per_day = df.groupby("coin")[["p", "y"]].mean(), df.groupby("day")[["p", "y"]].mean()
    per_t = df.groupby("t")["p"].mean()
    return {"share_pred_up": float(up_p.mean()), "share_real_up": float(up_y.mean()),
            "per_coin_corr": metrics.pearson(per_coin["p"].to_numpy(), per_coin["y"].to_numpy()),
            "per_coin_pred_up_min": float(per_coin["p"].min()), "per_coin_pred_up_max": float(per_coin["p"].max()),
            "per_day_corr": metrics.pearson(per_day["p"].to_numpy(), per_day["y"].to_numpy()),
            "one_sign_timestamps": float(((per_t == 0) | (per_t == 1)).mean()),
            "hit_rate": float((np.sign(frame["mu"]) == np.sign(frame["y_ret"])).mean())}


def confidence_selection(frame: pd.DataFrame, val: pd.DataFrame, keep: float = 0.2, cost: float = 0.0008,
                         top_share: float = 0.2) -> dict:
    """Winning days of metric 1 and 2 on (a) all rows, (b) the rows whose confidence |mu| / sd is at least the VAL
    quantile ``1 - keep``, (c) the rows whose sd is at most the VAL quantile ``keep``; (d) the rows whose |mu| is at
    least the VAL quantile ``1 - keep`` (a rule that needs no uncertainty: what (b) must beat to be worth having);
    (e) all rows, the book of metric 2 ranked by mu / sd instead of mu."""
    def conf(f):
        return (f["mu"].abs() / f["sigma"].where(f["sigma"] > 0)).to_numpy(np.float64)
    rules = {"all_rows": np.ones(len(frame), bool),
             "top_confidence": conf(frame) >= np.nanquantile(conf(val), 1.0 - keep),
             "low_sd": frame["sigma"].to_numpy() <= np.nanquantile(val["sigma"].to_numpy(), keep),
             "top_abs_mu": frame["mu"].abs().to_numpy() >= np.nanquantile(val["mu"].abs().to_numpy(), 1.0 - keep)}
    out = {}
    scores = {name: np.where(mask, frame["mu"].to_numpy(np.float64), np.nan) for name, mask in rules.items()}
    # every row trades, but the book of metric 2 is ranked by mu / sd (a small move of a calm coin counts as much as
    # a large move of a wild one): the use of the sd that keeps the whole cross-section
    scores["ranked_by_mu_over_sd"] = conf(frame) * np.sign(frame["mu"].to_numpy(np.float64))
    for name, score in scores.items():
        out[name] = {"rows_share": float(np.isfinite(score).mean()),
                     "absolute_gross": day_summary(absolute_days(frame, score)),
                     "absolute_net": day_summary(absolute_days(frame, score, cost=cost)),
                     "neutral_gross": day_summary(neutral_days(frame, score, top_share=top_share))}
    return out


def uncertainty_report(frame: pd.DataFrame, val: pd.DataFrame, *, cost: float = 0.0008) -> dict:
    """Everything above for the close head, plus ``sd_ranks_error`` and ``sharp_and_calibrated`` of the high / low
    heads when the frame carries them (``pred_<t>``, ``sigma_<t>``, ``fut_<t>``)."""
    heads = {"close": ("mu", "sigma", "y_ret")}
    heads.update({t: (f"pred_{t}", f"sigma_{t}", f"fut_{t}") for t in ("high", "low") if f"pred_{t}" in frame.columns})
    return {"sd_ranks_error": {t: sd_ranks_error(frame, *cols) for t, cols in heads.items()},
            "sharp_and_calibrated": {t: sharp_and_calibrated(frame, *cols) for t, cols in heads.items()},
            "direction_bias": direction_bias(frame), "confidence_selection": confidence_selection(frame, val,
                                                                                                    cost=cost)}


def render(report: dict) -> str:
    """Markdown of one ``uncertainty_report``."""
    lines = ["| head | sd vs abs error: across coins | across time | pooled | T-sharp | coverage 80 % | z sd |",
             "|---|---|---|---|---|---|---|"]
    for t, r in report["sd_ranks_error"].items():
        c = report["sharp_and_calibrated"][t]
        lines.append(f"| {t} | {r['across_coins']:+.3f} | {r['across_time']:+.3f} | {r['pooled']:+.3f} | "
                     f"{c['t_sharp']:.3f} | {c['coverage_80']:.3f} | {c['z_sd']:.2f} |")
    b = report["direction_bias"]
    lines += ["", f"Direction: predicted up {b['share_pred_up']:.3f} of rows (realised {b['share_real_up']:.3f}), hit "
                  f"rate {b['hit_rate']:.3f}; per-coin share predicted up {b['per_coin_pred_up_min']:.2f} .. "
                  f"{b['per_coin_pred_up_max']:.2f} (correlation with the realised share {b['per_coin_corr']:+.2f}); "
                  f"per-day correlation {b['per_day_corr']:+.2f}; timestamps with one sign for every coin "
                  f"{b['one_sign_timestamps']:.2f}.", "",
              "| rows kept | share | metric 1 winning days, gross | metric 1 mean day, net | metric 2 winning days |",
              "|---|---|---|---|---|"]
    for name, r in report["confidence_selection"].items():
        a, n, m = r["absolute_gross"], r["absolute_net"], r["neutral_gross"]
        if not a.get("n_days"):
            lines.append(f"| {name} | {r['rows_share']:.2f} | no days | | |")
            continue
        m2 = f"{m['win_share'] * 100:.1f} % ({m['n_days']} d)" if m.get("n_days") else "no days"
        lines.append(f"| {name} | {r['rows_share']:.2f} | {a['win_share'] * 100:.1f} % ({a['n_days']} d) | "
                     f"{n['mean'] * 100:+.4f} % | {m2} |")
    return "\n".join(lines) + "\n"
