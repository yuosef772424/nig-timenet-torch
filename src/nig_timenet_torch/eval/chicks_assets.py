"""PURPOSE: The first block of the TF chicks report: judge a finished stage coin by coin, print each coin's latest
    sample and performance, the decoding verification, the coverage and the per-coin and pooled tables.
TAGS: chicks, test all assets, per asset results, coverage, verification summary, wilson interval, aggregated
    metrics, asset summary, tf parity
PITFALLS: the model is NOT called here: the stage's PredictionTable already holds its outputs (``StageData``), the
    TF code pooled the windows and called the model once, which is the table's job now; a coin with fewer than
    ``min_samples_per_asset`` rows is skipped only when ``eval_all_rows`` is False; a coin whose evaluation raises is
    reported and skipped (the TF behaviour), so read the printed errors; the tables that grow big are saved to
    ``out_dir`` (``per_asset_summary.csv``, ``per_asset_ci.csv`` when over 40 rows); coins keep the order they first
    appear in the table; the row number in the latest table counts inside the coin.
SOURCE: crypto-signal-prediction ``evaluation/all_assets.py`` (test_all_assets_v4, summarize_verification,
    print_verification_summary, _overall_metrics_table, per_asset_ci_table, calculate_aggregated_metrics_v4) and
    ``evaluation/unified.py`` (_print_v4_report).
"""

import traceback

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.chicks_decode import (
    DIRECTION_TOLERANCE,
    StageData,
    decode_stage,
    evaluate_decoded,
    nan_max,
    nan_mean,
    slice_decoded,
    verify_decoding,
)
from nig_timenet_torch.eval.chicks_flat import latest_table_frame, print_latest_frame
from nig_timenet_torch.eval.chicks_outputs import fmt_price, save_or_print
from nig_timenet_torch.eval.chicks_text import Say
from nig_timenet_torch.stats import metrics
from nig_timenet_torch.stats.nig import NIG_UNC_MAX

__all__ = ["test_all_assets", "summarize_verification", "print_verification_summary", "overall_table",
           "per_asset_ci_table", "aggregate_results", "MIN_SAMPLES_PER_ASSET"]

MIN_SAMPLES_PER_ASSET = 20     # the TF floor of rows per coin when ``eval_all_rows`` is False


def _col(g: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(g[name], errors="coerce") if name in g else pd.Series(np.nan, index=g.index)


def summarize_verification(verification_df: pd.DataFrame, say: Say | None = None) -> pd.DataFrame:
    """One row per head: the verification reports of every coin folded together."""
    say = say or Say()
    if verification_df is None or verification_df.empty:
        return pd.DataFrame()
    rows = []
    for tgt, g in verification_df.groupby("target", sort=False):
        n = _col(g, "n_samples") if "n_samples" in g else pd.Series(1.0, index=g.index)

        def wmean(name, g=g, n=n):
            v = _col(g, name)
            m = v.notna() & n.notna() & (n > 0)
            return float(np.average(v[m], weights=n[m])) if m.any() else float("nan")

        n_failed = int((g["status"] == "FAILED").sum()) if "status" in g else 0
        kind = g["kind"].iloc[0] if "kind" in g else ""
        rows.append({"target": tgt, "kind": kind, "assets": len(g),
                     "status": say.text("status_ok") if n_failed == 0 else say.text("status_failed", n=n_failed),
                     "roundtrip_max_diff": nan_max(_col(g, "roundtrip_max_diff")),
                     "true_consistency_max_diff": nan_max(_col(g, "true_value_consistency_max_diff")),
                     "space_ratio_mean": nan_mean(_col(g, "space_consistency_ratio_mean")),
                     "mae_scaled": wmean("mae_scaled"), "mae_real": wmean("mae_real")})
    return pd.DataFrame(rows)


def print_verification_summary(verification_df: pd.DataFrame, say: Say | None = None) -> pd.DataFrame:
    """Print the decoding verification once for all coins (and return it)."""
    say = say or Say()
    summary = summarize_verification(verification_df, say)
    if summary.empty:
        return summary
    print(f"\n{'=' * 100}")
    say("ver_head")
    print("=" * 100)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    n_failed = int((verification_df["status"] == "FAILED").sum()) if "status" in verification_df else 0
    if n_failed == 0:
        say("ver_all_ok")
    else:
        say("ver_failed", n=n_failed)
    say("ver_note")
    return summary


def overall_table(all_results: list, targets) -> pd.DataFrame:
    """Pooled metrics per head over all coins, weighted by samples (MAE, MAPE, win rate and its Wilson interval)."""
    rows = []
    for name in targets:
        errs, pcts, dirs = [], [], []
        for res in all_results:
            r = res.get(name)
            if not isinstance(r, dict) or "error_abs" not in r:
                continue
            v = np.asarray(r.get("has_truth", np.ones(len(r["error_abs"]), dtype=bool)), dtype=bool)
            errs.append(np.asarray(r["error_abs"])[v])
            pcts.append(np.asarray(r["error_pct"])[v])
            dirs.append(np.asarray(r["direction_correct"])[v])
        if errs:
            e, p, d = np.concatenate(errs), np.concatenate(pcts), np.concatenate(dirs)
            lo, hi = metrics.wilson_interval(int(d.sum()), len(d))
            rows.append({"target": name, "n": int(len(e)), "MAE": nan_mean(e), "MAPE_%": nan_mean(p),
                         "Win_%": float(d.mean() * 100) if len(d) else float("nan"),
                         "Win_CI95_lo_%": lo * 100, "Win_CI95_hi_%": hi * 100})
    return pd.DataFrame(rows)


def per_asset_ci_table(all_results: list, targets) -> pd.DataFrame:
    """One row per coin x head: n, the direction accuracy % and its Wilson 95 % interval (small n: wide)."""
    rows = []
    for res in all_results:
        for name in targets:
            r = res.get(name)
            if not isinstance(r, dict) or "direction_correct" not in r:
                continue
            v = np.asarray(r.get("has_truth", np.ones(len(r["direction_correct"]), dtype=bool)), dtype=bool)
            d = np.asarray(r["direction_correct"])[v]
            lo, hi = metrics.wilson_interval(int(d.sum()), len(d))
            rows.append({"asset": res.get("asset"), "target": name, "n": int(len(d)),
                         "acc_%": float(d.mean() * 100) if len(d) else float("nan"),
                         "ci95_lo_%": lo * 100, "ci95_hi_%": hi * 100})
    return pd.DataFrame(rows)


def aggregate_results(results_df: pd.DataFrame, targets) -> dict:
    """Totals over coins: number of coins and samples, and per head the sample-weighted and the plain mean MAE."""
    out = {"total_assets": len(results_df), "total_samples": int(results_df["n_samples"].sum()),
           "avg_samples_per_asset": float(results_df["n_samples"].mean())}
    weights = (results_df["n_samples"] / results_df["n_samples"].sum()).to_numpy()
    for name in targets:
        maes, w_used = [], []
        for pos, (_, row) in enumerate(results_df.iterrows()):
            if "error_abs" in row[name]:
                maes.append(nan_mean(row[name]["error_abs"]))
                w_used.append(weights[pos])
        if maes:
            m, w = np.asarray(maes, dtype=np.float64), np.asarray(w_used, dtype=np.float64)
            ok = np.isfinite(m)
            if ok.any():
                out[f"{name}_weighted_mae"] = float(np.average(m[ok], weights=w[ok]))
                out[f"{name}_mean_mae"] = float(m[ok].mean())
    return out


def _print_asset_report(res: dict, targets, n_samples: int, n_display: int, say: Say) -> None:
    """The compact per-coin block: the latest sample of every head, then one performance line per head."""
    names = [t for t in targets if isinstance(res.get(t), dict)]
    k = int(min(n_display, n_samples)) if n_display else 0
    if k > 0:
        table = latest_table_frame(res, names, k, None)
        print(say.text("latest_head", k=k))
        print_latest_frame(table, legend=False, say=say)
    lines = [say.text("perf_line", name=name, mae=fmt_price(nan_mean(res[name]["error_abs"])),
                      mape=nan_mean(res[name]["error_pct"]), win=res[name].get("win_rate", float("nan")),
                      corr=res[name].get("correlation", 0.0), n=int(res[name].get("n_valid", n_samples)))
             for name in names if "error_abs" in res[name]]
    if lines:
        print(say.text("perf_head"))
        print("\n".join(lines))


def _evaluate_assets(stage: StageData, decoded: dict, kept: list, n_total: int, verbose: bool, n_display: int,
                     range_frac: float, run_verification: bool, say: Say) -> tuple:
    """(results, verification rows, summary rows) of the kept coins, in table order; a failing coin is reported."""
    results, ver_rows, summary_rows = [], [], []
    for idx, asset, rows in kept:
        n_samples = len(rows)
        try:
            res = evaluate_decoded(slice_decoded(decoded, rows), range_frac)
            if run_verification:
                ver_rows += [{"asset": asset, "n_samples": n_samples, **rep} for rep in verify_decoding(res).values()]
            if verbose:
                print(f"\n{'─' * 100}")
                print(say.text("asset_head", i=idx, n=n_total, asset=asset, rows=n_samples))
                _print_asset_report(res, stage.targets, n_samples, n_display, say)
            row = {"asset": asset, "n": n_samples}
            for t in stage.targets:
                if "error_abs" in res[t]:
                    row[f"{t}_MAE"] = nan_mean(res[t]["error_abs"])
                    row[f"{t}_win%"] = res[t].get("win_rate", float("nan"))
            summary_rows.append(row)
            res["asset"], res["n_samples"] = asset, n_samples
            results.append(res)
        except Exception as e:  # noqa: BLE001 - the TF report keeps going past one coin that fails
            print(say.text("asset_error", asset=asset, e=e))
            if verbose:
                traceback.print_exc()
    return results, ver_rows, summary_rows


def test_all_assets(stage: StageData, *, out_dir: str, say: Say | None = None, verbose: bool = True,
                    n_display: int = 1, range_frac: float = DIRECTION_TOLERANCE,
                    min_samples_per_asset: int = MIN_SAMPLES_PER_ASSET, run_verification: bool = True,
                    eval_all_rows: bool = True, unc_max: float = NIG_UNC_MAX, class_head: bool = False) -> dict:
    """The TF ``test_all_assets_v4`` on a finished stage: ``per_asset_results`` (one row per coin, one dict per head),
    ``aggregated_results``, ``verification_summary``, ``asset_summary``, ``asset_ci``, ``coverage``, ``targets``."""
    say = say or Say()
    asset_col = stage.table["asset"].astype(str).to_numpy()
    order = list(pd.unique(asset_col))
    rows_of = {a: np.flatnonzero(asset_col == a) for a in order}
    min_eff = 1 if eval_all_rows else int(min_samples_per_asset)
    skipped = [(a, len(r)) for a, r in rows_of.items() if len(r) < min_eff]
    kept = [(i, a, rows_of[a]) for i, a in enumerate(order, 1) if len(rows_of[a]) >= min_eff]
    if verbose:
        print(f"\n{'=' * 100}")
        print(say.text("tested_head", n=len(order), k=n_display))
        print("=" * 100)
    decoded = decode_stage(stage, unc_max, class_head) if kept else {}
    if verbose and kept:
        print(say.text("targets_line", names=list(stage.targets)))
    results, ver_rows, summary_rows = _evaluate_assets(stage, decoded, kept, len(order), verbose, n_display, range_frac,
                                                       run_verification, say)
    if not results:
        print(say.text("none_tested"))
        return {"per_asset_results": pd.DataFrame(), "aggregated_results": {}, "verification_summary": pd.DataFrame(),
                "asset_summary": pd.DataFrame(), "asset_ci": pd.DataFrame(),
                "coverage": {"test_rows": len(stage.table), "evaluated_rows": 0, "skipped_assets": skipped},
                "targets": tuple(stage.targets)}
    results_df = pd.DataFrame(results)
    per_asset_rows = results_df["n_samples"].to_numpy()
    coverage = {"test_rows": len(stage.table), "evaluated_rows": int(per_asset_rows.sum()), "test_assets": len(order),
                "evaluated_assets": int(len(results_df)), "skipped_assets": skipped,
                "eval_all_rows": bool(eval_all_rows), "rows_per_asset_min": int(per_asset_rows.min()),
                "rows_per_asset_median": float(np.median(per_asset_rows)),
                "rows_per_asset_max": int(per_asset_rows.max())}
    asset_ci, verification_df = per_asset_ci_table(results, stage.targets), (
        pd.DataFrame(ver_rows) if ver_rows else pd.DataFrame())
    asset_summary = pd.DataFrame(summary_rows)
    if verbose:
        _print_totals(results, stage, verification_df, coverage, asset_summary, asset_ci, run_verification, out_dir,
                      min_samples_per_asset, say)
    return {"per_asset_results": results_df, "aggregated_results": aggregate_results(results_df, stage.targets),
            "verification_summary": verification_df, "asset_summary": asset_summary, "asset_ci": asset_ci,
            "coverage": coverage, "targets": tuple(stage.targets)}


def _print_totals(results, stage, verification_df, cov, asset_summary, asset_ci, run_verification, out_dir,
                  min_samples_per_asset, say) -> None:
    """Everything printed after the coins: verification, coverage, per-coin tables, the total per head."""
    if run_verification and not verification_df.empty:
        print_verification_summary(verification_df, say)
    print(f"\n{'=' * 100}")
    tail = "" if cov["eval_all_rows"] else say.text("cov_not_all", n=min_samples_per_asset)
    print(say.text("cov_line", ev=cov["evaluated_rows"], tr=cov["test_rows"], ea=cov["evaluated_assets"],
                   ta=cov["test_assets"], lo=cov["rows_per_asset_min"], med=cov["rows_per_asset_median"],
                   hi=cov["rows_per_asset_max"]) + tail)
    if cov["skipped_assets"]:
        skipped = cov["skipped_assets"]
        print(say.text("cov_skipped", n=len(skipped), items=skipped[:8], more=" …" if len(skipped) > 8 else ""))
    if cov["rows_per_asset_median"] < 50:
        print(say.text("cov_small_median"))
    if len(asset_summary) > 1:
        print(f"\n{'=' * 100}")
        print(say.text("asset_summary_head"))
        print("=" * 100)
        save_or_print(asset_summary.round(4), "per_asset_summary", out_dir=out_dir, say=say)
    if not asset_ci.empty:
        print(f"\n{'=' * 100}")
        print(say.text("asset_ci_head"))
        print("=" * 100)
        save_or_print(asset_ci.round(2), "per_asset_ci", out_dir=out_dir, say=say)
    overall = overall_table(results, stage.targets)
    if not overall.empty:
        print(f"\n{'=' * 100}")
        print(say.text("overall_head"))
        print("=" * 100)
        print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
