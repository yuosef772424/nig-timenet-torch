"""PURPOSE: The TF run_full_analysis for a finished test stage: per-coin judgement, flat frame, trust and calibration,
    conditioned reports, trades, patterns, integrity, tearsheet and summary; saves every table and figure.
TAGS: chicks, run full analysis, full report, trust report, trade selection, tearsheet, equity curve, integrity,
    pattern discovery, executive summary, pnl market neutral, tf parity
PITFALLS: the TF function called the model; here the stage is a PredictionTable already (``StageData``), the model is
    not needed; the calibrators are fitted on ``val_stage`` ONLY (``Role.CALIBRATION``) and never on test, and
    ``calibrators`` given by the caller replace that fit; the tearsheet reads the close head only and prices each
    trade ``direction x true change - cost`` (``trade_cost_pct`` is a ROUND-TRIP percent, 0.08 = 0.04 % a side);
    with timestamps the trades of one timestamp are one equal-weight portfolio period, without them the curve is
    sequential and says it is not executable; ``pnl_market_neutral`` subtracts the timestamp mean return (needed with
    relative targets); a report that cannot run (too few samples) is skipped with a printed warning, never raised.
SOURCE: crypto-signal-prediction ``evaluation/full_analysis.py`` (run_full_analysis).
"""

import os

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.chicks_assets import test_all_assets
from nig_timenet_torch.eval.chicks_conditioned import (
    analyze_by_batch,
    analyze_by_confidence_bucket,
    analyze_by_predicted_movement,
)
from nig_timenet_torch.eval.chicks_decode import DIRECTION_TOLERANCE, StageData
from nig_timenet_torch.eval.chicks_flat import build_flat_frame
from nig_timenet_torch.eval.chicks_integrity import run_integrity_diagnostics
from nig_timenet_torch.eval.chicks_outputs import ensure_output_dir, save_or_print
from nig_timenet_torch.eval.chicks_patterns import detect_success_failure_patterns
from nig_timenet_torch.eval.chicks_plots import (
    plot_batch_performance,
    plot_calibration_curve,
    plot_confidence_vs_error,
    plot_equity_curve,
    plot_error_distribution,
)
from nig_timenet_torch.eval.chicks_tearsheet import (
    MIN_PERIOD_SECONDS_NS,
    compute_trading_performance_metrics,
    print_performance_verdict,
    signal_significance_report,
    simulate_equity_curve,
)
from nig_timenet_torch.eval.chicks_text import Say
from nig_timenet_torch.eval.chicks_trades import (
    MIN_EDGE_RATIO,
    TOP_N,
    print_trade_selection,
    select_and_rank_trades,
    trade_accuracy_pct,
)
from nig_timenet_torch.eval.chicks_trust import fit_confidence_calibrators, generate_trust_report
from nig_timenet_torch.stats import metrics
from nig_timenet_torch.stats.nig import NIG_UNC_MAX

__all__ = ["run_full_analysis", "BATCH_ANALYSIS_SIZE", "TRADE_COST_PCT", "CLOSE_TARGET"]

BATCH_ANALYSIS_SIZE = 100
TRADE_COST_PCT = 0.08            # round-trip cost in percent (0.04 % a side)
CLOSE_TARGET = "close"
_YEAR_NS = 365.25 * 86400e9


def _val_calibrators(val_stage, targets, method: str, kw: dict, say: Say, verbose: bool) -> tuple:
    """(calibrators, val flat frame) fitted on the validation stage only (None, None when it has no results)."""
    res = test_all_assets(val_stage, verbose=False, run_verification=False, **kw)
    if res["per_asset_results"].empty:
        if verbose:
            say("full_val_empty")
        return None, None
    val_flat = build_flat_frame(res["per_asset_results"], targets)
    return fit_confidence_calibrators(val_flat, method=method, verbose=verbose, say=say), val_flat


def _trade_block(continuous_df, out_dir, min_edge, sort_by, top_n, verbose, say) -> tuple:
    candidates = select_and_rank_trades(continuous_df, min_edge_ratio=min_edge, sort_by=sort_by, top_n=None, say=say)
    selection = candidates.head(int(top_n)).copy() if top_n is not None else candidates.copy()
    if not selection.empty:
        save_or_print(selection, "trade_selection", out_dir=out_dir, verbose=False)
    if verbose:
        print_trade_selection(selection, original=continuous_df, title=say.text("full_trade_title", k=min_edge,
                                                                                  s=sort_by),
                              max_rows=top_n, say=say)
        ok_col = next((c for c in ("direction_correct", "correct", "ok")
                       if c in continuous_df.columns and c in candidates.columns), None)
        if ok_col and not candidates.empty:
            acc, n = trade_accuracy_pct(candidates[ok_col])
            base_acc, base_n = trade_accuracy_pct(continuous_df[ok_col])
            if base_n and n:
                say("full_trade_rate", a=acc, n=n, b=base_acc, bn=base_n)
    return candidates, selection


def _patterns_block(flat_df, out_dir, make_plots, verbose, say):
    try:
        return detect_success_failure_patterns(flat_df, verbose=verbose, out_dir=out_dir, make_plots=make_plots,
                                               say=say)
    except ValueError as e:
        if verbose:
            say("full_pat_skip", e=e)
        return None


def _period_series(close_df: pd.DataFrame, cost: float, neutral_wanted: bool) -> dict:
    """The returns the tearsheet judges: per timestamp (equal-weight portfolio) when every row has a time, else per
    trade; plus the labels of the figure and the printed basis."""
    true_change = (close_df["true"] - close_df["entry"]) / (np.abs(close_df["entry"]) + 1e-7) * 100
    direction = np.sign(close_df["predicted_change"])
    ts = close_df["timestamp"] if "timestamp" in close_df else pd.Series(np.nan, index=close_df.index)
    neutral = bool(neutral_wanted) and bool(ts.notna().all())
    if neutral:
        true_change = true_change - true_change.groupby(ts.values).transform("mean")
    returns = direction * true_change - cost
    timed = bool(ts.notna().all() and ts.nunique() > 2)
    if not timed:
        return {"timed": False, "neutral": neutral, "returns": returns.values, "risk": 0.1, "ppy": None, "n_periods": 0}
    period = returns.groupby(ts.values).mean().sort_index()
    step = np.diff(period.index.values.astype(np.float64))
    step_ns = float(np.median(step)) if len(step) else 0.0
    return {"timed": True, "neutral": neutral, "returns": period.values, "risk": 1.0, "n_periods": len(period),
            "ppy": (_YEAR_NS / step_ns) if step_ns >= MIN_PERIOD_SECONDS_NS else None}


def _trading_block(continuous_df, out_dir, cost, neutral_wanted, make_plots, verbose, say):
    if continuous_df.empty or CLOSE_TARGET not in continuous_df["target"].unique():
        return None
    close_df = continuous_df[continuous_df["target"] == CLOSE_TARGET].copy()
    s = _period_series(close_df, cost, neutral_wanted)
    tm = compute_trading_performance_metrics(s["returns"], periods_per_year=s["ppy"], risk_per_trade=s["risk"])
    if s["timed"]:
        title = f"Equity Curve — Close (equal-weight portfolio per period, cost {cost}%)"
        basis = say.text("full_basis_portfolio", p=s["n_periods"], t=len(close_df), c=cost)
        if s["neutral"]:
            basis += say.text("full_basis_neutral")
            title += " — market-neutral"
    else:
        title, basis = "Equity Curve — Close (per trade, NO timestamps: sequential, not executable)", say.text(
            "full_basis_untimed")
    if verbose and tm:
        print("\n" + "=" * 100)
        say("full_tear_title")
        print(f"   ({basis})")
        print("=" * 100)
        print_performance_verdict(tm, say)
    if tm and s["timed"]:
        try:
            tm["significance"] = signal_significance_report(close_df, trade_cost_pct=cost, market_neutral=s["neutral"],
                                                            periods_per_year=s["ppy"], verbose=verbose, say=say)
        except Exception as e:  # noqa: BLE001 - the TF report goes on without it
            print(say.text("full_sig_fail", e=e))
    if make_plots and tm:
        eq = simulate_equity_curve(s["returns"], risk_per_trade=s["risk"])
        plot_equity_curve(eq["equity"], eq["drawdown_pct"], title=title, save_dir=out_dir)
    return tm


def _summary_block(res: dict, flat_df, trust, trade_selection, integrity_df, tm, out_dir, say: Say) -> None:
    print("\n" + "=" * 100)
    say("full_done")
    print("=" * 100)
    overall_trust = trust["overall"].get("trust_score_0_100")
    lo, hi = metrics.wilson_interval(int(flat_df["correct"].sum()), len(flat_df))
    say("full_n", n=len(flat_df))
    say("full_acc", a=flat_df["correct"].mean() * 100, lo=lo * 100, hi=hi * 100)
    cov = res.get("coverage") or {}
    if cov.get("test_rows"):
        say("full_cov", ev=cov["evaluated_rows"], tr=cov["test_rows"], ea=cov.get("evaluated_assets", 0),
            ta=cov.get("test_assets", 0))
    if "unc_clipped" in flat_df.columns and flat_df["unc_clipped"].any():
        say("full_clipped", m=NIG_UNC_MAX, n=int(flat_df["unc_clipped"].sum()), p=flat_df["unc_clipped"].mean() * 100)
    cal = trust.get("calibration")
    if cal is not None and not cal.empty:
        say("full_ece", s=" | ".join(f"{r['target']}: {r['ece_before']:.3f}←{r['ece_after']:.3f} ({r['signal']})"
                                     for _, r in cal.iterrows()))
    if trade_selection is not None:
        say("full_trades", n=len(trade_selection))
    if overall_trust is not None:
        say("full_trust", s=overall_trust)
    if not res["verification_summary"].empty:
        failed = (res["verification_summary"]["status"] == "FAILED").sum()
        say("full_decode_ok") if failed == 0 else say("full_decode_bad", n=failed)
    if integrity_df is not None and not integrity_df.empty:
        say("full_beats_naive", p=(integrity_df["mae_improvement_%"] > 10).mean() * 100)
    if tm:
        say("full_sharpe", b=say.text("full_ann" if tm.get("sharpe_basis") == "annualized" else "full_per_trade"),
            s=tm["sharpe_ratio"], d=tm["max_drawdown_pct"])
    say("full_saved", p=os.path.abspath(out_dir))


def run_full_analysis(stage: StageData, *, out_dir: str, val_stage: StageData | None = None,
                      calibrators: dict | None = None, calibration_method: str = "isotonic",
                      batch_analysis_size: int = BATCH_ANALYSIS_SIZE, make_plots: bool = True,
                      run_integrity_check: bool = True, verbose: bool = True, n_display: int = 1,
                      run_trade_selection: bool = True, trade_min_edge_ratio: float = MIN_EDGE_RATIO,
                      trade_sort_by: str = "both", trade_top_n: int | None = TOP_N,
                      trade_cost_pct: float = TRADE_COST_PCT, pnl_market_neutral: bool = False,
                      eval_all_rows: bool = True, range_frac: float = DIRECTION_TOLERANCE,
                      unc_max: float = NIG_UNC_MAX, class_head: bool = False, say: Say | None = None) -> dict:
    """Every report of the chicks analysis of ``stage`` (the test split), files in ``out_dir``. ``val_stage`` (the
    validation split) fits the confidence calibrators (``calibration_method``: "isotonic" | "platt"), or give
    ``calibrators`` ready. Returns the TF dict: the per-asset results, ``flat_df``, ``trust_report``, ``calibrators``,
    ``movement_analysis``, ``confidence_analysis``, ``trade_candidates``, ``trade_selection``, ``batch_analysis``,
    ``pattern_discovery``, ``integrity_diagnostics`` and ``trading_metrics``."""
    say = say or Say()
    ensure_output_dir(out_dir)
    kw = {"out_dir": out_dir, "say": say, "range_frac": range_frac, "unc_max": unc_max, "class_head": class_head,
          "eval_all_rows": True}
    res = test_all_assets(stage, verbose=verbose, n_display=n_display, **{**kw, "eval_all_rows": eval_all_rows})
    per_asset = res["per_asset_results"]
    if per_asset.empty:
        say("full_no_results")
        return res
    flat_df = build_flat_frame(per_asset, stage.targets)
    save_or_print(flat_df, "flat_predictions_all", out_dir=out_dir, verbose=False)
    if calibrators is None and val_stage is not None:
        calibrators, _ = _val_calibrators(val_stage, stage.targets, calibration_method, kw, say, verbose)
    trust = generate_trust_report(flat_df, group_by="target", verbose=verbose, calibrators=calibrators, out_dir=out_dir,
                                  say=say)
    if make_plots and not trust["reliability_curve"].empty:
        for tgt in flat_df["target"].unique():
            plot_calibration_curve(trust["reliability_curve"], group_val=tgt, save_dir=out_dir)
        plot_confidence_vs_error(flat_df, save_dir=out_dir)
    continuous_df = flat_df[flat_df["kind"] == "continuous"]
    movement = batch = None
    if not continuous_df.empty:
        movement = analyze_by_predicted_movement(continuous_df, verbose=verbose, out_dir=out_dir, say=say)
        batch = analyze_by_batch(continuous_df, batch_size=batch_analysis_size, verbose=verbose, out_dir=out_dir,
                                 say=say)
        if make_plots:
            plot_error_distribution(continuous_df, save_dir=out_dir)
            if batch is not None and not batch.empty:
                plot_batch_performance(batch, save_dir=out_dir)
    confidence_analysis = analyze_by_confidence_bucket(flat_df, verbose=verbose, out_dir=out_dir, say=say)
    candidates = selection = None
    if run_trade_selection and not continuous_df.empty:
        candidates, selection = _trade_block(continuous_df, out_dir, trade_min_edge_ratio, trade_sort_by,
                                             trade_top_n, verbose, say)
    patterns = _patterns_block(flat_df, out_dir, make_plots, verbose, say)
    integrity_df = (run_integrity_diagnostics(per_asset, stage.targets, out_dir=out_dir, verbose=verbose,
                                              make_plots=make_plots, say=say) if run_integrity_check else None)
    tm = _trading_block(continuous_df, out_dir, trade_cost_pct, pnl_market_neutral, make_plots, verbose, say)
    if verbose:
        _summary_block(res, flat_df, trust, selection, integrity_df, tm, out_dir, say)
    return {**res, "flat_df": flat_df, "trust_report": trust, "calibrators": calibrators,
            "movement_analysis": movement, "confidence_analysis": confidence_analysis, "trade_candidates": candidates,
            "trade_selection": selection, "batch_analysis": batch, "pattern_discovery": patterns,
            "integrity_diagnostics": integrity_df, "trading_metrics": tm}
