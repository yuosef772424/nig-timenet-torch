"""PURPOSE: The TF trust and calibration report over the flat frame (ECE, Brier, trust score, reliability curve) and the
    post-hoc confidence calibration fitted on VAL only, applied to TEST.
TAGS: chicks, trust report, calibration, reliability curve, ece, brier, trust score, isotonic, platt, confidence head,
    class margin, post hoc calibration, tf parity
PITFALLS: calibrators are fitted on the validation frame ONLY (``Role.CALIBRATION``, rule B4) and never see test; one
    calibrator per head, the signal it reads is chosen on val: the confidence head when its Spearman correlation with
    being right is >= 0.03 and p < 0.05, else ``|p_up - 0.5|`` of the class head (a DIFFERENT target: the class
    head's own direction hit, so its ECE is not comparable with the confidence row), else the constant val accuracy;
    calibration repairs bias (ECE), it does not create discrimination; the binning is by quantiles (``pd.qcut``,
    ten bins), which is NOT the equal-width bin of ``stats.metrics.ece`` (so the ECE here differs from the battery's);
    ``calibrated_df`` keeps the raw ``confidence`` untouched and adds ``confidence_cal``, ``calib_raw_p``,
    ``calib_correct`` and ``calib_signal``.
SOURCE: crypto-signal-prediction ``evaluation/trust_calibration.py`` (generate_trust_report, diagnose_confidence,
    fit_confidence_calibrators, apply_confidence_calibration, calibration_report); the learned map is the registered
    ``eval.calibrator`` component, its state is plain data.
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.roles import Role
from nig_timenet_torch.eval.calibrator import CALIBRATORS
from nig_timenet_torch.eval.chicks_outputs import DEFAULT_OUTPUT_DIR, save_or_print
from nig_timenet_torch.eval.chicks_text import Say
from nig_timenet_torch.stats import metrics
from nig_timenet_torch.stats.tdist import t_sf

__all__ = ["generate_trust_report", "diagnose_confidence", "fit_confidence_calibrators",
           "apply_confidence_calibration", "calibration_report", "print_calibration_report", "CALIB_MIN_RHO",
           "CALIB_MIN_STD", "CALIB_MIN_N"]

CALIB_MIN_RHO = 0.03     # the least positive Spearman rho (with p < 0.05) for an informative confidence head
CALIB_MIN_STD = 1e-3     # a confidence with a smaller standard deviation is "constant"
CALIB_MIN_N = 50         # the fewest val rows of a head to fit a map (below: the constant val accuracy)


def _quantile_bins(values: np.ndarray, n_bins: int):
    """The TF binning: ``n_bins`` quantile bins (fewer when the values repeat), equal-width ones if that fails."""
    try:
        return pd.qcut(values, q=min(n_bins, len(np.unique(values))), duplicates="drop")
    except Exception:  # noqa: BLE001 - the TF fallback
        return pd.cut(values, bins=n_bins)


def _binned_gap(p, y, n_bins: int = 10) -> float:
    """Sum over quantile bins of weight x |mean p - mean y| (the TF ECE of ``generate_trust_report``)."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return float("nan")
    g = pd.DataFrame({"p": p, "y": y, "b": _quantile_bins(p, n_bins)}).groupby("b", observed=True)
    return float(sum(len(x) / len(p) * abs(x["p"].mean() - x["y"].mean()) for _, x in g))


def _uncertainty_error_link(sub: pd.DataFrame):
    common = sub.dropna(subset=["uncertainty", "abs_error"])
    if len(common) <= 2 or not common["uncertainty"].std() > 1e-9:
        return None
    u, e = common["uncertainty"], common["abs_error"]
    if "entry" in common.columns:
        scale = np.abs(common["entry"]).replace(0, np.nan)      # relative to the entry price, by ranks
        u, e = u / scale, e / scale
    return float(pd.Series(u).corr(pd.Series(e), method="spearman"))


def _group_metrics(sub: pd.DataFrame, n_bins: int) -> dict:
    sub = sub.dropna(subset=["confidence"])
    if len(sub) == 0:
        return {}
    correct, conf = sub["correct"].astype(float).to_numpy(), sub["confidence"].astype(float).to_numpy()
    bin_df = pd.DataFrame({"conf": conf, "correct": correct, "bin": _quantile_bins(conf, n_bins)})
    ece = mce = 0.0
    rows = []
    for name, g in bin_df.groupby("bin", observed=True):
        if len(g) == 0:
            continue
        gap = abs(g["conf"].mean() - g["correct"].mean())
        ece += len(g) / len(bin_df) * gap
        mce = max(mce, gap)
        rows.append({"bin": str(name), "avg_confidence": g["conf"].mean(), "avg_accuracy": g["correct"].mean(),
                     "gap": gap, "count": len(g)})
    corr = metrics.pearson(conf, correct) if np.std(conf) > 1e-9 and np.std(correct) > 1e-9 else 0.0
    lo, hi = metrics.wilson_interval(int(correct.sum()), len(correct))
    has_unc = "uncertainty" in sub and sub["uncertainty"].notna().any()
    return {"n_samples": len(sub), "ece": ece, "mce": mce, "brier_score": float(np.mean((conf - correct) ** 2)),
            "confidence_accuracy_corr": corr,
            "uncertainty_error_corr": (_uncertainty_error_link(sub) if "abs_error" in sub.columns
                                       and "uncertainty" in sub.columns else None),
            "sharpness_avg_uncertainty": float(sub["uncertainty"].mean()) if has_unc else None,
            "mean_confidence": float(conf.mean()), "mean_accuracy": float(correct.mean()),
            "accuracy_ci95_lo": lo, "accuracy_ci95_hi": hi,
            "trust_score_0_100": 100.0 * max(0.0, 1 - ece) * (0.5 + 0.5 * (corr + 1) / 2),
            "_reliability_rows": rows}


def _trust_verdict(trust_score, ece, corr, say: Say) -> None:
    say("trust_verdict_head")
    if trust_score is None:
        return
    key = "trust_good" if trust_score >= 75 else "trust_mid" if trust_score >= 50 else "trust_bad"
    say(key, s=trust_score)
    if ece is not None:
        say("ece_good" if ece < 0.05 else "ece_mid" if ece < 0.15 else "ece_bad", e=ece)


def generate_trust_report(df: pd.DataFrame, group_by: str | None = "target", n_bins: int = 10, verbose: bool = True,
                          calibrators: dict | None = None, out_dir: str = DEFAULT_OUTPUT_DIR,
                          say: Say | None = None) -> dict:
    """{'overall', 'by_group' (DataFrame), 'reliability_curve' (DataFrame), 'calibration' (ECE / Brier before and
    after, or None), 'calibrated_df'} of a flat frame; ``calibrators`` come from ``fit_confidence_calibrators``."""
    say = say or Say()
    overall = _group_metrics(df, n_bins)
    by_group, reliability = [], []
    if group_by and group_by in df.columns:
        for gval, sub in df.groupby(group_by):
            m = _group_metrics(sub, n_bins)
            if not m:
                continue
            for rr in m.pop("_reliability_rows"):
                reliability.append({**rr, group_by: gval})
            by_group.append({group_by: gval, **m})
    by_group_df, reliability_df = pd.DataFrame(by_group), pd.DataFrame(reliability)
    overall.pop("_reliability_rows", None)
    calibration_df = calibrated_df = None
    if calibrators:
        calibrated_df = apply_confidence_calibration(df, calibrators)
        calibration_df = calibration_report(calibrated_df, n_bins=n_bins, verbose=False)
    if verbose:
        print("=" * 100)
        say("trust_title")
        print("=" * 100)
        say("trust_overall")
        for k, v in overall.items():
            print(f"   • {k}: {v:.4f}" if isinstance(v, float) else f"   • {k}: {v}")
        _trust_verdict(overall.get("trust_score_0_100"), overall.get("ece"), overall.get("confidence_accuracy_corr"),
                       say)
        if not by_group_df.empty:
            say("trust_by_group", g=group_by)
            cols = [group_by, "n_samples", "trust_score_0_100", "ece", "mce", "confidence_accuracy_corr",
                    "mean_confidence", "mean_accuracy", "accuracy_ci95_lo", "accuracy_ci95_hi"]
            save_or_print(by_group_df[[c for c in cols if c in by_group_df.columns]].round(4), "trust_report_by_group",
                          out_dir=out_dir, say=say)
        if calibration_df is not None:
            print_calibration_report(calibration_df, say)
    return {"overall": overall, "by_group": by_group_df, "reliability_curve": reliability_df,
            "calibration": calibration_df, "calibrated_df": calibrated_df}


def diagnose_confidence(conf, correct) -> dict:
    """Does the confidence head carry information? verdict: 'constant' | 'no_signal' | 'informative'."""
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    std = float(np.std(conf))
    rho = p = float("nan")
    if std >= CALIB_MIN_STD and np.std(correct) > 0 and len(conf) > 10:
        rho = metrics.spearman(conf, correct)
        n = len(conf)
        t = rho * np.sqrt((n - 2) / max(1.0 - rho * rho, 1e-300))
        p = float(2.0 * t_sf(abs(float(t)), n - 2)) if np.isfinite(rho) else float("nan")
    if std < CALIB_MIN_STD:
        verdict = "constant"
    elif np.isfinite(rho) and rho >= CALIB_MIN_RHO and p < 0.05:
        verdict = "informative"
    else:
        verdict = "no_signal"
    return {"n": int(len(conf)), "conf_std": std, "conf_unique": int(len(np.unique(np.round(conf, 6)))),
            "spearman_rho": rho, "spearman_p": p, "verdict": verdict}


def _class_signal(sub: pd.DataFrame):
    """(max(p_up, 1 - p_up), |p_up - 0.5|, the class head's direction hit) or None without p_up / the price change."""
    if "p_up" not in sub.columns or "price_change" not in sub.columns:
        return None
    p = sub["p_up"].to_numpy(dtype=np.float64)
    chg = sub["price_change"].to_numpy(dtype=np.float64)
    if not (np.isfinite(p) & np.isfinite(chg)).all():
        return None
    return np.maximum(p, 1 - p), np.abs(p - 0.5), ((p >= 0.5) == (chg > 0)).astype(np.float64)


def _fit_state(method: str, scores, labels) -> dict:
    return CALIBRATORS.entry(method).obj().fit(scores, labels, role=Role.CALIBRATION)


def fit_confidence_calibrators(val_df: pd.DataFrame, method: str = "isotonic", group_by: str = "target",
                               min_n: int = CALIB_MIN_N, verbose: bool = True, say: Say | None = None) -> dict:
    """One calibrator per head of the VAL frame (never test): {head: {'method', 'signal', 'label', 'state' (the
    calibrator's plain state, None for the constant), 'n_val', 'base_rate', 'diag'}}. An unknown ``method`` raises."""
    say = say or Say()
    CALIBRATORS.entry(method)
    out = {}
    for g, sub in val_df.dropna(subset=["confidence", "correct"]).groupby(group_by):
        correct = sub["correct"].to_numpy(dtype=np.float64)
        diag = diagnose_confidence(sub["confidence"].to_numpy(), correct)
        cls = _class_signal(sub)
        entry = {"method": method, "n_val": int(len(sub)), "base_rate": float(correct.mean()), "diag": diag}
        if len(sub) >= min_n and diag["verdict"] == "informative":
            entry.update(signal="confidence", label="correct", state=_fit_state(method, sub["confidence"], correct))
        elif len(sub) >= min_n and cls is not None:
            _, margin, lab = cls
            entry.update(signal="class_margin", label="class_correct", base_rate=float(lab.mean()),
                         state=_fit_state(method, 0.5 + margin, lab))
        else:
            entry.update(signal="base_rate", label="correct", state=None)
        out[g] = entry
        if verbose:
            d = diag
            warn = "" if d["verdict"] == "informative" else say.text(
                "calib_warn", why=say.text("calib_const" if d["verdict"] == "constant" else "calib_nosig"),
                std=d["conf_std"], uniq=d["conf_unique"], rho=d["spearman_rho"],
                to=say.text("calib_to_margin" if entry["signal"] == "class_margin" else "calib_to_base"))
            print(say.text("calib_fit", g=g, method=method, n=d["n"], signal=entry["signal"]) + warn)
    return out


def _predict(cal: dict, scores) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if cal["state"] is None:
        return np.full(np.shape(scores), cal["base_rate"], dtype=np.float64)
    return CALIBRATORS.entry(cal["method"]).obj().apply(scores, cal["state"])


def apply_confidence_calibration(df: pd.DataFrame, calibrators: dict, group_by: str = "target") -> pd.DataFrame:
    """A copy of ``df`` with ``confidence_cal`` (the calibrated probability), ``calib_raw_p`` (the raw one, in the same
    target's space), ``calib_correct`` (that target) and ``calib_signal``; a head without a calibrator stays NaN."""
    out = df.copy()
    for c in ("confidence_cal", "calib_raw_p", "calib_correct"):
        out[c] = np.nan
    out["calib_signal"] = None
    for g, cal in calibrators.items():
        m = (out[group_by] == g).to_numpy()
        if not m.any():
            continue
        sub = out[m]
        if cal["signal"] == "class_margin":
            cls = _class_signal(sub)
            if cls is None:
                continue
            raw_p, margin, lab = cls
            cal_p = _predict(cal, 0.5 + margin)
        else:
            raw_p = sub["confidence"].to_numpy(dtype=np.float64)
            lab = sub["correct"].to_numpy(dtype=np.float64)
            cal_p = _predict(cal, raw_p)
        out.loc[m, "calib_raw_p"] = raw_p
        out.loc[m, "confidence_cal"] = np.asarray(cal_p, dtype=np.float64)
        out.loc[m, "calib_correct"] = lab
        out.loc[m, "calib_signal"] = cal["signal"]
    return out


def calibration_report(cal_df: pd.DataFrame, n_bins: int = 10, group_by: str = "target", verbose: bool = True,
                       say: Say | None = None) -> pd.DataFrame:
    """ECE and Brier before and after the calibration, per head, on ``cal_df`` (the output of
    ``apply_confidence_calibration``, i.e. test)."""
    rows = []
    for g, sub in cal_df.dropna(subset=["confidence_cal", "calib_correct"]).groupby(group_by):
        p0, p1, y = (sub[c].to_numpy(dtype=np.float64) for c in ("calib_raw_p", "confidence_cal", "calib_correct"))
        rows.append({group_by: g, "n": int(len(sub)), "signal": sub["calib_signal"].iloc[0],
                     "ece_before": _binned_gap(p0, y, n_bins), "ece_after": _binned_gap(p1, y, n_bins),
                     "brier_before": float(np.mean((p0 - y) ** 2)), "brier_after": float(np.mean((p1 - y) ** 2)),
                     "mean_p_before": float(p0.mean()), "mean_p_after": float(p1.mean()),
                     "mean_accuracy": float(y.mean())})
    res = pd.DataFrame(rows)
    if verbose:
        print_calibration_report(res, say)
    return res


def print_calibration_report(res: pd.DataFrame, say: Say | None = None) -> None:
    say = say or Say()
    if res is None or res.empty:
        return
    say("calib_head")
    print(res.round(4).to_string(index=False))
    for _, r in res.iterrows():
        if r["signal"] == "class_margin":
            say("calib_margin_note", t=r["target"])
        elif r["signal"] == "base_rate":
            say("calib_base_note", t=r["target"])
    say("calib_footer")
