"""Verification block c: selective direction accuracy and R:R trades over the signals frame of a run

PURPOSE: Block c of the integrated verification (old TF section 7-b): does the accuracy rise with confidence (slice
    picked on VAL, judged on TEST) and do R:R trades with a model-set stop pay, each against a random-direction control.
TAGS: verification, selective direction, confidence slice, rr trades, stop distance, wilson, day t-stat
PITFALLS: every threshold is chosen on VAL and judged on TEST; the frame is the battery table (+ ``conf``, ``up``,
    ``day``, ``fut_high`` / ``fut_low`` as returns of the last close) built by ``eval.verification.signals_frame``;
    a daily candle cannot say which of target and stop was touched first, so touching both counts as a loss;
    the trade cost is ``round_trip(cost_per_side)`` in return units (the battery's one cost), in R it is cost / d; the
    nig stop distance is the NIG predictive std (the old TF code used the Student-t scale, ~sqrt(alpha / (alpha - 1))
    smaller); the day t-stat resamples nothing: it is mean / se of the per-day mean R (days are the independent unit).
SOURCE: ported from crypto-signal-prediction workflow/selective_eval.py (selective_direction_report,
    rr_trading_report, simulate_rr_trades; thresholds and verdict logic unchanged, columns renamed to the battery's).
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.battery_lib import round_trip
from nig_timenet_torch.eval.verification_text import t
from nig_timenet_torch.stats import metrics

__all__ = ["direction_scores", "selective_direction_report", "rr_trading_report", "simulate_rr_trades",
           "STOP_DISTANCE", "PASS", "FAIL"]

PASS, FAIL = "✅", "❌"       # verdict symbols of the summary table (✅ / ❌), the old table's own
N_BINS = 10
QUANTILES = (0.5, 0.7, 0.8, 0.9, 0.95)
MIN_DAYS_T = 3                        # a day t-stat needs more than 2 days


def direction_scores(df: pd.DataFrame) -> dict:
    """{name: (side +-1 per row, confidence score)} of every candidate confidence source the frame can give; the higher
    the score the more confident; every one predicts the direction of the close after the horizon."""
    out = {}
    if df.attrs.get("p_dir_source") == "close class head":
        p = df["p_dir"].to_numpy()
        out["class_margin"] = (np.where(p >= 0.5, 1, -1), np.abs(p - 0.5))
    mu = df["mu"].to_numpy()
    side = np.where(mu >= 0, 1, -1)
    out["nig_edge"] = (side, np.abs(mu) / (df["sigma"].to_numpy() + 1e-12))     # |expected move| / uncertainty
    if "conf" in df:
        out["conf_head"] = (side, df["conf"].to_numpy())
    if "class_margin" in out:
        out["agree_margin"] = (side, np.where(out["class_margin"][0] == side, out["class_margin"][1], -np.inf))
    return out


def _slice_trend(score: np.ndarray, correct: np.ndarray) -> float:
    """Spearman between the confidence decile and its accuracy (+1 = accuracy rises with confidence everywhere)."""
    ok = np.isfinite(score)
    s, c = score[ok], correct[ok].astype(float)
    if len(s) < N_BINS:
        return float("nan")
    edges = np.unique(np.quantile(s, np.linspace(0, 1, N_BINS + 1)))
    b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    accs = [(i, c[b == i].mean()) for i in range(len(edges) - 1) if (b == i).any()]
    if len(accs) <= 2:
        return float("nan")
    return metrics.spearman(np.array([i for i, _ in accs], dtype=float), np.array([a for _, a in accs]))


def _top_slice_threshold(score: np.ndarray, correct: np.ndarray, target_acc: float, min_n: int):
    """The largest-coverage threshold on VAL whose top slice reaches ``target_acc`` (>= ``min_n`` rows), else None."""
    ok = np.isfinite(score)
    s, c = score[ok], correct[ok].astype(float)
    order = np.argsort(-s, kind="stable")
    k = np.arange(1, len(s) + 1)
    hit = np.where((np.cumsum(c[order]) / k >= target_acc) & (k >= min_n))[0]
    return None if len(hit) == 0 else float(s[order][hit[-1]])


def _slice_stats(df: pd.DataFrame, mask: np.ndarray, correct: np.ndarray) -> dict:
    n, k = int(mask.sum()), int(correct[mask].sum())
    lo, hi = metrics.wilson_interval(k, n)
    return {"n": n, "n_days": int(df.loc[mask, "day"].nunique()), "coverage": n / max(len(df), 1),
            "accuracy": k / n if n else float("nan"), "lo": lo, "hi": hi}


def selective_direction_report(val: pd.DataFrame, test: pd.DataFrame, target_acc: float = 0.65,
                               min_n: int = 200, lang: str = "en") -> pd.DataFrame:
    """Per confidence source: is the accuracy monotone in confidence on VAL and on TEST, then the VAL threshold that
    reaches ``target_acc`` applied to TEST; passes when TEST holds >= ``min_n`` rows at >= ``target_acc`` with a Wilson
    lower bound above 0.5. The Wilson interval assumes independent rows (the coins of one day are not): read
    ``test_dates``."""
    v_scores, t_scores = direction_scores(val), direction_scores(test)
    rows = []
    for name, (v_side, v_s) in v_scores.items():
        t_side, t_s = t_scores[name]
        v_ok = (v_side == np.where(val["up"] == 1, 1, -1)).astype(int)
        t_ok = (t_side == np.where(test["up"] == 1, 1, -1)).astype(int)
        thr = _top_slice_threshold(v_s, v_ok, target_acc, min_n)
        row = {"score": name, "acc_all_test": t_ok.mean(), "monotonic_val": _slice_trend(v_s, v_ok),
               "monotonic_test": _slice_trend(t_s, t_ok), "threshold": thr}
        if thr is None:
            row["verdict"] = f"{FAIL} " + t(lang, "sel_none", acc=target_acc)
        else:
            vs, ts = _slice_stats(val, v_s >= thr, v_ok), _slice_stats(test, t_s >= thr, t_ok)
            row.update({"val_acc": vs["accuracy"], "val_n": vs["n"], "test_acc": ts["accuracy"], "test_n": ts["n"],
                        "test_dates": ts["n_days"], "test_coverage": ts["coverage"], "test_wilson_lo": ts["lo"],
                        "test_wilson_hi": ts["hi"]})
            held = ts["n"] >= min_n and ts["accuracy"] >= target_acc and ts["lo"] > 0.5
            row["verdict"] = f"{PASS} " + t(lang, "sel_pass") if held else f"{FAIL} " + t(lang, "sel_fail")
        rows.append(row)
    return pd.DataFrame(rows)


def _nig_distance(df: pd.DataFrame, side: np.ndarray, stop_k: float) -> np.ndarray:
    return stop_k * df["sigma"].to_numpy()


def _level_distance(df: pd.DataFrame, side: np.ndarray, stop_k: float) -> np.ndarray:
    """Distance to the PREDICTED low (buy) / high (sell), a return of the entry."""
    return np.where(side > 0, -df["pred_low"].to_numpy(), df["pred_high"].to_numpy())


#: stop mode -> distance d of the stop (a return of the entry); the target sits at ``rr`` x d. An unknown name raises.
STOP_DISTANCE = {"nig": _nig_distance, "model_levels": _level_distance}


def simulate_rr_trades(df: pd.DataFrame, side: np.ndarray, rr: float, stop_mode: str, cost_per_side: float,
                       stop_k: float = 1.0) -> pd.DataFrame:
    """One trade per row: entry at the last close, stop at distance d, target at ``rr`` x d, judged on the realised
    high / low / close of the horizon; no touch = exit at the close. ``r`` is in units of the risk d, after the round
    trip of ``cost_per_side``; ``exit`` is tp / sl / both (counted a loss) / close / invalid (no usable d)."""
    if stop_mode not in STOP_DISTANCE:
        raise ValueError(f"stop_mode {stop_mode!r} is not one of {sorted(STOP_DISTANCE)}")
    d = STOP_DISTANCE[stop_mode](df, side, stop_k)
    valid = np.isfinite(d) & (d > 1e-5)
    d = np.where(valid, d, np.nan)
    hi, lo, ret = df["fut_high"].to_numpy(), df["fut_low"].to_numpy(), df["y_ret"].to_numpy()
    hit_tp = np.where(side > 0, hi >= rr * d, lo <= -rr * d)
    hit_sl = np.where(side > 0, lo <= -d, hi >= d)
    r = np.where(hit_sl, -1.0, np.where(hit_tp, rr, side * ret / d)) - round_trip(cost_per_side) / d
    kind = np.where(hit_sl & hit_tp, "both", np.where(hit_sl, "sl", np.where(hit_tp, "tp", "close")))
    return pd.DataFrame({"r": np.where(valid, r, np.nan), "exit": np.where(valid, kind, "invalid"),
                         "day": df["day"].to_numpy()})


def _trade_summary(tr: pd.DataFrame) -> dict:
    t = tr[tr["exit"] != "invalid"]
    if len(t) == 0:
        return {"n": 0}
    daily = t.groupby("day")["r"].mean()
    ok = len(daily) >= MIN_DAYS_T and daily.std() > 0
    t_days = float(daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))) if ok else 0.0
    return {"n": len(t), "n_days": int(t["day"].nunique()), "win_rate": float((t["exit"] == "tp").mean()),
            "both_hit": float((t["exit"] == "both").mean()), "expectancy_R": float(t["r"].mean()), "t_days": t_days}


def rr_trading_report(val: pd.DataFrame, test: pd.DataFrame, *, rr: float = 2.0, stop_mode: str = "nig",
                      cost_per_side: float, min_trades: int = 200, seed: int = 0, lang: str = "en") -> pd.DataFrame:
    """Per confidence source: the quantile threshold with the best mean R on VAL (>= ``min_trades`` trades), applied to
    TEST, beside the same rows with a random direction. Passes when VAL and TEST both earn, the TEST day t-stat is
    >= 2 and TEST beats the random direction (a random walk wins ~1/3 at rr = 2: a 40 % win rate alone is nothing)."""
    rng = np.random.default_rng(seed)
    v_scores, t_scores = direction_scores(val), direction_scores(test)
    rows = []
    for name, (v_side, v_s) in v_scores.items():
        t_side, t_s = t_scores[name]
        v_tr = simulate_rr_trades(val, v_side, rr, stop_mode, cost_per_side)
        best = None
        for q in QUANTILES if np.isfinite(v_s).any() else ():
            thr = float(np.quantile(v_s[np.isfinite(v_s)], q))
            st = _trade_summary(v_tr[v_s >= thr])
            if st["n"] >= min_trades and (best is None or st["expectancy_R"] > best[1]["expectancy_R"]):
                best = (thr, st)
        if best is None:
            rows.append({"score": name, "verdict": f"{FAIL} " + t(lang, "tr_few", n=min_trades)})
            continue
        thr, vst = best
        sel = t_s >= thr
        tst = _trade_summary(simulate_rr_trades(test[sel], t_side[sel], rr, stop_mode, cost_per_side))
        base = _trade_summary(simulate_rr_trades(test[sel], rng.choice([-1, 1], size=int(sel.sum())), rr, stop_mode,
                                                 cost_per_side))
        passed = (tst.get("n", 0) >= min_trades and vst["expectancy_R"] > 0 and tst.get("expectancy_R", -1) > 0
                  and tst.get("t_days", 0.0) >= 2.0 and tst.get("expectancy_R", -1) > base.get("expectancy_R", np.inf))
        rows.append({"score": name, "threshold": thr, "val_n": vst["n"], "val_win": vst["win_rate"],
                     "val_exp_R": vst["expectancy_R"], "test_n": tst.get("n", 0), "test_dates": tst.get("n_days", 0),
                     "test_win": tst.get("win_rate"), "test_both_hit": tst.get("both_hit"),
                     "test_exp_R": tst.get("expectancy_R"), "test_t_days": tst.get("t_days"),
                     "random_dir_win": base.get("win_rate"), "random_dir_exp_R": base.get("expectancy_R"),
                     "verdict": f"{PASS} " + t(lang, "tr_pass") if passed else f"{FAIL} " + t(lang, "tr_fail")})
    return pd.DataFrame(rows)
