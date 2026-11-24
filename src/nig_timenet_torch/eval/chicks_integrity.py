"""PURPOSE: The TF model-integrity diagnostics per coin x head: error against the naive forecast, the lag of best
    correlation (late copying), a momentum baseline, directional bias and the worst / best figures.
TAGS: chicks, integrity diagnostics, naive baseline, lag scan, late copying, momentum baseline, directional bias,
    mae improvement, tf parity
PITFALLS: works on the PRICE-LEVEL arrays of each coin x head (``true_real``, ``pred_real``, ``entry_price``), where
    the naive forecast is the entry and so the predicted line always hugs the actual one: judge by ``mae_improvement_%``
    and ``best_lag``, not by the picture; a combination with fewer than 5 samples is left out; the lag range is
    ``min(max_lag, n // 3)``; a combination is "healthy" when its best lag is 0, its MAE beats the naive by more than
    10 % and its direction hit beats ``max(55, momentum + 3)``; the figures are drawn only for the worst
    ``plot_worst_n`` and the best ``plot_best_n`` by MAE improvement (the full table is always saved when long) and
    only when ``make_plots`` is true.
SOURCE: crypto-signal-prediction ``evaluation/integrity.py`` (run_integrity_diagnostics and its helpers).
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.chicks_outputs import DEFAULT_OUTPUT_DIR, save_or_print
from nig_timenet_torch.eval.chicks_plots import plot_actual_vs_predicted, plot_lag_correlation
from nig_timenet_torch.eval.chicks_text import Say
from nig_timenet_torch.stats import metrics

__all__ = ["run_integrity_diagnostics", "compute_point_metrics", "cross_correlation_lag_scan",
           "momentum_direction_accuracy", "directional_bias", "MAX_LAG", "HEALTHY_MAE_IMPROVEMENT"]

MAX_LAG = 5
HEALTHY_MAE_IMPROVEMENT = 10.0     # percent the MAE must beat the naive forecast by to count as healthy
MIN_SAMPLES = 5


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE and correlation of a forecast (correlation 0.0 when either side is constant)."""
    corr = metrics.pearson(y_true, y_pred) if np.std(y_true) > 1e-9 and np.std(y_pred) > 1e-9 else 0.0
    return {"mae": metrics.mae(y_true, y_pred), "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))), "corr": corr}


def cross_correlation_lag_scan(y_actual: np.ndarray, y_pred: np.ndarray, max_lag: int = MAX_LAG):
    """(lags, correlations, best_lag): at which shift are actual and predicted most correlated (0 healthy)."""
    lags = list(range(-max_lag, max_lag + 1))
    corrs = []
    for lag in lags:
        a, p = ((y_actual[:lag], y_pred[-lag:]) if lag < 0 else (y_actual[lag:], y_pred[:-lag]) if lag > 0
                else (y_actual, y_pred))
        corrs.append(metrics.pearson(a, p) if len(a) > 1 and np.std(a) > 1e-9 and np.std(p) > 1e-9 else np.nan)
    valid = [c if not np.isnan(c) else -np.inf for c in corrs]
    return lags, corrs, lags[int(np.argmax(valid))]


def momentum_direction_accuracy(y_actual: np.ndarray, entry_price: np.ndarray) -> float:
    """Direction hit of the momentum rule "the next move repeats the last actual move" (a fair directional baseline)."""
    actual_dir = np.sign(y_actual - entry_price)
    if len(actual_dir) < 2:
        return float("nan")
    pred_dir, now = actual_dir[:-1], actual_dir[1:]
    mask = now != 0
    return float(np.mean(pred_dir[mask] == now[mask])) if mask.sum() else float("nan")


def directional_bias(y_actual: np.ndarray, y_pred: np.ndarray, say: Say | None = None) -> dict:
    """Are the errors biased one way (always optimistic or pessimistic)?"""
    say = say or Say()
    errors = y_pred - y_actual
    bias_mean = float(np.mean(errors))
    ref = np.sign(bias_mean) if bias_mean != 0 else 0
    return {"bias_mean": bias_mean, "bias_pct": float(np.mean(errors / (np.abs(y_actual) + 1e-9)) * 100),
            "same_sign_pct": float(np.mean(np.sign(errors) == ref)) * 100 if ref != 0 else float("nan"),
            "direction_label": say.text("bias_up" if bias_mean > 0 else "bias_down" if bias_mean < 0
                                        else "bias_flat")}


def _one(asset: str, target: str, d: dict, max_lag: int, say: Say) -> dict | None:
    y_actual = np.asarray(d["true_real"], dtype=np.float64)
    y_pred = np.asarray(d["pred_real"], dtype=np.float64)
    entry = np.asarray(d["entry_price"], dtype=np.float64)
    n = len(y_actual)
    if n < MIN_SAMPLES:
        return None
    model_m, naive_m = compute_point_metrics(y_actual, y_pred), compute_point_metrics(y_actual, entry)
    move = np.sign(y_actual - entry)
    dir_model = (float(np.mean(np.sign(y_pred - entry)[move != 0] == move[move != 0])) if np.any(move != 0)
                 else float("nan"))
    dir_momentum = momentum_direction_accuracy(y_actual, entry)
    bias = directional_bias(y_actual, y_pred, say)
    lags, lag_corrs, best_lag = cross_correlation_lag_scan(y_actual, y_pred, max(1, min(max_lag, n // 3)))
    return {"asset": asset, "target": target, "n": n, "model_mae": model_m["mae"], "naive_mae": naive_m["mae"],
            "mae_improvement_%": (naive_m["mae"] - model_m["mae"]) / (naive_m["mae"] + 1e-12) * 100,
            "model_corr": model_m["corr"], "naive_corr": naive_m["corr"],
            "dir_acc_model_%": dir_model * 100 if not np.isnan(dir_model) else np.nan,
            "dir_acc_momentum_%": dir_momentum * 100 if not np.isnan(dir_momentum) else np.nan,
            "best_lag": best_lag, "bias_pct": bias["bias_pct"], "bias_same_sign_%": bias["same_sign_pct"],
            "bias_label": bias["direction_label"], "_y_actual": y_actual, "_y_pred": y_pred, "_y_naive": entry,
            "_lags": lags, "_lag_corrs": lag_corrs}


def _draw_extremes(df: pd.DataFrame, worst_n: int, best_n: int, out_dir: str, say: Say) -> None:
    ranked = df.sort_values("mae_improvement_%")
    keep = list(dict.fromkeys(list(ranked.head(worst_n).index) + list(ranked.tail(best_n).index)))
    to_plot = ranked.loc[keep]
    if len(to_plot) > 0:
        say("integ_plots", w=worst_n, b=best_n)
    for _, r in to_plot.iterrows():
        label = f"{r['asset']} — {r['target']}"
        plot_actual_vs_predicted(r["_y_actual"], r["_y_pred"], r["_y_naive"],
                                 title=f"{label} (MAE improve: {r['mae_improvement_%']:.1f}%)", save_dir=out_dir,
                                 filename=f"integrity_{r['asset']}_{r['target']}_actual_vs_pred")
        plot_lag_correlation(r["_lags"], r["_lag_corrs"], r["best_lag"], title=f"Lag Scan — {label}", save_dir=out_dir,
                             filename=f"integrity_{r['asset']}_{r['target']}_lag_scan")


def run_integrity_diagnostics(per_asset_results: pd.DataFrame, targets, max_lag: int = MAX_LAG,
                              plot_worst_n: int = 2, plot_best_n: int = 1, out_dir: str = DEFAULT_OUTPUT_DIR,
                              verbose: bool = True, make_plots: bool = True, say: Say | None = None) -> pd.DataFrame:
    """The summary table of the integrity checks, one row per coin x head (empty when no head has enough samples)."""
    say = say or Say()
    rows = []
    for _, row in per_asset_results.iterrows():
        for name in targets:
            if name in row and isinstance(row[name], dict) and "true_real" in row[name]:
                item = _one(row["asset"], name, row[name], max_lag, say)
                if item is not None:
                    rows.append(item)
    if not rows:
        if verbose:
            say("integ_none")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    healthy = df[(df["best_lag"] == 0) & (df["mae_improvement_%"] > HEALTHY_MAE_IMPROVEMENT)
                 & (df["dir_acc_model_%"] > np.maximum(55, df["dir_acc_momentum_%"] + 3))]
    display_df = df.drop(columns=["_y_actual", "_y_pred", "_y_naive", "_lags", "_lag_corrs"])
    if verbose:
        print("=" * 100)
        say("integ_title")
        print("=" * 100)
        save_or_print(display_df.round(3), "integrity_diagnostics_summary", out_dir=out_dir, say=say)
        say("integ_healthy", h=len(healthy), n=len(df))
        if len(healthy) < len(df):
            flagged = display_df[~display_df.index.isin(healthy.index)]
            say("integ_flagged")
            save_or_print(flagged[["asset", "target", "mae_improvement_%", "dir_acc_model_%", "dir_acc_momentum_%",
                                   "best_lag"]].round(2), "integrity_flagged", out_dir=out_dir, say=say)
        if make_plots:
            _draw_extremes(df, plot_worst_n, plot_best_n, out_dir, say)
    return display_df
