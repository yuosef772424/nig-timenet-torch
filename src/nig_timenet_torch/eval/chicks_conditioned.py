"""PURPOSE: The TF output-conditioned reports over the flat frame: direction hit rate by predicted move, by confidence
    bucket and by consecutive batch (a z-score per batch finds a regression).
TAGS: chicks, output conditioned, movement analysis, confidence bucket, batch performance, regression detection,
    z score, tf parity
PITFALLS: the column names and the direction words of the result tables are Arabic, exactly as the TF files had them
    (``عدد`` count, ``نسبة_نجاح`` win rate %, ``متوسط_الثقة`` mean confidence, ``متوسط_الخطأ`` mean error, ``حالة``
    status): other code (the batch figure) reads them by name, ``lang`` only changes the printed sentences; batches
    are consecutive blocks of the frame in its row order (coin by coin), so a batch can span a coin boundary; the
    movement bins are in percent (-inf, -2, -0.5, 0, 0.5, 2, inf).
SOURCE: crypto-signal-prediction ``evaluation/output_conditioned.py``.
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.chicks_outputs import DEFAULT_OUTPUT_DIR, save_or_print
from nig_timenet_torch.eval.chicks_text import Say

__all__ = ["analyze_by_predicted_movement", "analyze_by_confidence_bucket", "analyze_by_batch", "MOVE_BINS",
           "BATCH_Z_THRESHOLD"]

MOVE_BINS = (-np.inf, -2, -0.5, 0, 0.5, 2, np.inf)    # predicted move in percent of the entry
BATCH_Z_THRESHOLD = 1.5                               # |z| of a batch's win rate that marks it as unusual

_COUNT, _WIN, _CONF, _ERR, _STATE = "عدد", "نسبة_نجاح", "متوسط_الثقة", "متوسط_الخطأ", "حالة"


def analyze_by_predicted_movement(df: pd.DataFrame, magnitude_col: str = "predicted_change_pct", bins=None,
                                  verbose: bool = True, out_dir: str = DEFAULT_OUTPUT_DIR,
                                  say: Say | None = None) -> pd.DataFrame:
    """Direction hit rate by the direction and size of the predicted move, per head."""
    say = say or Say()
    if magnitude_col not in df.columns:
        raise ValueError(say.text("move_no_col", c=magnitude_col))
    bins = list(MOVE_BINS if bins is None else bins)
    labels = [f"({bins[i]:.1f}%, {bins[i + 1]:.1f}%]" for i in range(len(bins) - 1)]
    work = df.copy()
    work["direction"] = np.where(work[magnitude_col] > 0, "صعود", np.where(work[magnitude_col] < 0, "هبوط", "ثابت"))
    work["movement_bucket"] = pd.cut(work[magnitude_col], bins=bins, labels=labels)
    grouped = work.groupby(["target", "direction", "movement_bucket"], observed=True).agg(
        عدد=("correct", "size"), نسبة_نجاح=("correct", "mean"), متوسط_الثقة=("confidence", "mean")).reset_index()
    grouped[_WIN] = (grouped[_WIN] * 100).round(2)
    grouped[_CONF] = grouped[_CONF].round(3)
    grouped = grouped[grouped[_COUNT] > 0].sort_values(["target", "direction"])
    if verbose:
        print("=" * 100)
        say("move_title")
        print("=" * 100)
        save_or_print(grouped, "movement_analysis", out_dir=out_dir, say=say)
        best, worst = grouped.loc[grouped[_WIN].idxmax()], grouped.loc[grouped[_WIN].idxmin()]
        for key, r in (("move_best", best), ("move_worst", worst)):
            say(key, t=r["target"], d=r["direction"], b=r["movement_bucket"], w=r[_WIN], n=int(r[_COUNT]))
    return grouped


def analyze_by_confidence_bucket(df: pd.DataFrame, n_buckets: int = 5, verbose: bool = True,
                                 out_dir: str = DEFAULT_OUTPUT_DIR, say: Say | None = None) -> pd.DataFrame:
    """Direction hit rate by confidence bucket (equal counts), per head."""
    say = say or Say()
    work = df.dropna(subset=["confidence"]).copy()
    work["conf_bucket"] = pd.qcut(work["confidence"], q=n_buckets, duplicates="drop")
    grouped = work.groupby(["target", "conf_bucket"], observed=True).agg(
        عدد=("correct", "size"), نسبة_نجاح=("correct", "mean"), متوسط_الثقة=("confidence", "mean")).reset_index()
    grouped[_WIN] = (grouped[_WIN] * 100).round(2)
    if verbose:
        print("\n" + "=" * 100)
        say("conf_title")
        print("=" * 100)
        save_or_print(grouped, "confidence_bucket_analysis", out_dir=out_dir, say=say)
    return grouped


def analyze_by_batch(df: pd.DataFrame, batch_size: int = 100, target: str | None = None,
                     z_threshold: float = BATCH_Z_THRESHOLD, verbose: bool = True, out_dir: str = DEFAULT_OUTPUT_DIR,
                     say: Say | None = None) -> pd.DataFrame:
    """Win rate and error per consecutive block of ``batch_size`` samples, the z-score of each block's win rate
    against all blocks, and the blocks with ``|z| >= z_threshold`` marked as a regression or an improvement."""
    say = say or Say()
    work = (df if target is None else df[df["target"] == target]).reset_index(drop=True)
    work["batch_id"] = work.index // batch_size
    named = {"عدد": ("correct", "size"), "نسبة_نجاح": ("correct", "mean"), "متوسط_الثقة": ("confidence", "mean"),
             **({"متوسط_الخطأ": ("abs_error", "mean")} if "abs_error" in work.columns else {})}
    stats = work.groupby("batch_id").agg(**named).reset_index()
    mean_wr, std_wr = stats[_WIN].mean(), stats[_WIN].std()
    stats["z_score"] = (stats[_WIN] - mean_wr) / (std_wr + 1e-9)
    stats[_STATE] = np.where(stats["z_score"] <= -z_threshold, say.text("batch_regression"),
                             np.where(stats["z_score"] >= z_threshold, say.text("batch_improved"),
                                      say.text("batch_normal")))
    stats[_WIN] = (stats[_WIN] * 100).round(2)
    if verbose:
        print("\n" + "=" * 100)
        say("batch_title", b=batch_size, t=f", target={target}" if target else "")
        print("=" * 100)
        save_or_print(stats.round(4), f"batch_performance{'_' + target if target else ''}", out_dir=out_dir, say=say)
        anomalies = stats[stats[_STATE] != say.text("batch_normal")]
        best = stats.loc[stats[_WIN].idxmax()]
        if not anomalies.empty:
            say("batch_anoms", b=int(best["batch_id"]), w=best[_WIN])
            for _, row in anomalies.iterrows():
                say("batch_anom_line", b=int(row["batch_id"]), w=row[_WIN], s=row[_STATE], d=best[_WIN] - row[_WIN])
        else:
            say("batch_stable")
    return stats
