# chicks-port-001: the TF project's chicks report in this project (W1; owner 2026-11-23; a parity port, no model result)

Failure reading: not applicable (a parity port of reports, no model arm was run or judged; C7 reads model arms)

What the owner asked (`docs/handoff/2026-11-23-reports-charts-pipeline.md`, item 2): the function `run_chicks` of the TF
project (`workflow/run.py`, which calls `evaluation/full_analysis.py::run_full_analysis`) gives, for a finished model on
the test split, a full report; this project must give the SAME reports, the SAME figures and the SAME printed text from
a finished torch run. Source: `crypto-signal-prediction` (read-only checkout, `evaluation/*.py` as named below).
New code: `src/nig_timenet_torch/eval/chicks_*.py` (12 modules), `workflow/chicks_run.py`, `eval/calibrator.py` (Platt),
`stats/tdist.py`, `stats/nig.py::nig_uncertainty_bounded`. Proof: `tests/parity/test_chicks_parity.py` against golden
files made by running the OLD code (`tests/parity/chicks_golden/README.md`), `tests/eval/test_chicks_*.py`,
`tests/workflow/test_chicks_run.py`.

## What was there before the port, per old report

(a) already here with the same numbers, (b) here but different (other table, other units, other formulas, other file
names), (c) missing. Every row is now covered by the new modules and by the parity test (last column).

| old report / figure / printed block (`run_full_analysis`) | before: file:function here | before | now (new module : function) |
|---|---|---|---|
| per coin: last sample table and one performance line per head (`unified._print_v4_report`) | `eval/latest_table.py:latest_table` (return units, LONG / SHORT, other columns) | b | `chicks_assets:test_all_assets`, `chicks_flat:latest_table_frame` |
| decoding verification once for all coins (`verify.verify_decoding`, `all_assets.print_verification_summary`) | `eval/verification.py` is the integrated verification, a different report; `workflow/run_common.py:check_decode` is a guard | b | `chicks_decode:verify_decoding`, `chicks_assets:print_verification_summary` |
| coverage line, per-coin summary, per-coin x head accuracy with Wilson CI, pooled total per head | `eval/legacy_reports.py:coin_ranking` (r, rank IC, hit) only | b | `chicks_assets:overall_table`, `per_asset_ci_table`, `aggregate_results` |
| `flat_predictions_all.csv` (the one frame every report reads) | none | c | `chicks_flat:build_flat_frame` |
| trust report: ECE, MCE, Brier, confidence-accuracy correlation, uncertainty-error correlation, trust score, verdict | `eval/legacy_reports.py:calibration` (equal-width ECE / Brier of `p_dir`) | b | `chicks_trust:generate_trust_report` |
| reliability curve `calibration_curve_<target>.png` | `eval/legacy_plots.py` (one figure per stage, other file name) | b | `chicks_plots:plot_calibration_curve` |
| post-hoc calibration (isotonic / Platt on VAL, applied to TEST, ECE before and after) | `eval/calibrator.py:Calibrator` isotonic only, not wired to a report | b | `calibrator:Calibrator`, `PlattCalibrator`; `chicks_trust:fit_confidence_calibrators`, `calibration_report` |
| `confidence_vs_error_all.png` | none | c | `chicks_plots:plot_confidence_vs_error` |
| `error_distribution_all.png` | `eval/legacy_plots.py` (3-panel error figure, other titles) | b | `chicks_plots:plot_error_distribution` |
| success rate by predicted move / by confidence bucket / by consecutive batch (Arabic tables, z-score, regression finder) | `eval/legacy_reports.py:conditioned` (move quintile, side, sd, batch with Wilson CI) | b | `chicks_conditioned:*` |
| `batch_performance.png` | `eval/legacy_plots.py` (hit by batch with error bars) | b | `chicks_plots:plot_batch_performance` |
| trade selection (min edge ratio, sort by confidence / move / both / edge, top N, `trade_selection.csv`, hit rate before / after) | `eval/trade_selection.py:select_trades` (latest table, return units); `legacy_reports.move_over_sd` | b | `chicks_trades:select_and_rank_trades`, `print_trade_selection` |
| pattern discovery per head (tree rules, importances, K-Means, isolation forest, 2 figures per head) | none | c | `chicks_patterns:detect_success_failure_patterns` |
| integrity per coin x head (naive baseline, lag scan, momentum, bias; csv; worst / best figures) | `eval/legacy_reports.py:integrity`, `lag_scan` (return units, mean over coins), `legacy_plots.price_panels` | b | `chicks_integrity:run_integrity_diagnostics` |
| tearsheet numbers and verdict (Sharpe, Sortino, Omega, profit factor, SQN, drawdown, total return) | `eval/legacy_reports.py:tearsheet` (fractions, other Sortino / Omega) | b | `chicks_tearsheet:compute_trading_performance_metrics`, `print_performance_verdict` |
| equity and drawdown figure | `eval/legacy_plots.py` (4-panel trades + daily) | b | `chicks_plots:plot_equity_curve`, `chicks_tearsheet:simulate_equity_curve` |
| significance report (t test, bootstrap Sharpe CI, permutation test, momentum and reversal rules, halves) | none (`eval/battery*` has other block-bootstrap and placebo tests) | c | `chicks_tearsheet:signal_significance_report` |
| market-neutral PnL option, equal-weight portfolio per timestamp, annualisation | none | c | `chicks_full:_period_series` (`pnl_market_neutral`) |
| executive summary printed block | none | c | `chicks_full:_summary_block` |
| output manager (`save_or_print`, `save_figure`, the file names) | none | c | `chicks_outputs` |
| the Arabic text of all of this | only `verification_text.py` | c | `chicks_text` (`ar` = the TF text, `en` added) |

Not part of `run_full_analysis`, so not ported here: `evaluation/legacy_uncertainty.py`, `legacy_assets.py`,
`io_patterns.py`, `live.py` (other sections of the TF notebook; `eval/latest_table.py` already covers the live table).

## How the parity is shown

`tests/parity/make_chicks_golden.py` runs the OLD `run_full_analysis` unchanged (TensorFlow replaced by an empty stub, the
model call replaced by a fake that replays the table's columns) on a small known world, two cases: daily bars with the
defaults, and 6 h bars with Platt, market neutral, the confidence rule, top 5. The test then runs the new report on the
same arrays and compares: the printed text line by line (about 500 lines per case), every saved file (number by number),
the list of PNG names and, per figure, titles, axis labels, legend and the data of lines, bars and scatters
(`figsig.py`), and the returned tables (trust, calibration, movement, batches, trades, patterns, integrity, tearsheet,
significance). 10 parity tests, all equal at relative 1e-6 (Platt tables at 1e-4).

## What is NOT equal to the old report, and why

| difference | cause | effect |
|---|---|---|
| price slope (the TF `iqr`) is a central difference of the codec's decode | the torch codecs have no slope method | relative error ~1e-11 (linear) / ~1e-9 (log codec); the uncertainty in price units differs at that level |
| Platt map | scikit-learn's lbfgs stops at its own tolerance, this project's Newton fit is to 1e-12 | calibrated probability differs by up to ~3e-5; ECE after calibration at the 5th digit |
| `p_up` is NaN by default (`class_head=False`) | the TF 'return' mode never read the class head (`class_key` was set for entry_range only); with it NaN, the pattern features and the calibration fallback are exactly the TF ones | `ChicksSettings.class_head=True` reads the class head: one more pattern feature and the class-margin fallback |
| pattern figures follow `make_plots` | the TF drew them whenever `verbose` was true | `make_plots=False` also stops them |
| every table / figure goes to `out_dir` | the TF wrote the pattern, movement, confidence, batch and trust tables to a fixed `analysis_outputs/` whatever `out_dir` was | none with the default `out_dir` |
| `model_mae` etc. drop NaN rows (`stats.metrics.mae`) | rule C2 (one MAE) | only a row with no true price (a live row) differs; the TF propagated NaN |
| `confidence_calibrators.json` is written | rule B4 (the learned state is stored with the run) | one extra file; `save_calibrators=False` removes it |
| the decoding verification's round trip compares the decode with its first-order expansion | the TF compared it with its own `scaled * iqr + median` | identical for the linear (return) codec; for the log codec it reports 0, not a check |
| `lang="en"` | new | all sentences English; the Arabic column names of the tables stay (the figures read them), the figures are English in both languages |
| no model call, no `batch_size`, `timestamp_col`, `timestamp_key` | the stage is a finished PredictionTable | none |

## Not verified

- A real-size run (hundreds of coins, 300 000 test rows): only the 6-coin fixture artifact (`tests/workflow/test_chicks_run.py`,
  slow) and the synthetic worlds were run; the per-coin loop and the random forest on 340 000 rows were not timed here.
- The `logret_close` codec and `eval_all_rows=False`, `trade_top_n=None`, `n_display > 1`: no old golden (code reads the
  same path; the unit tests cover the filter and the codec slope).
- Figures were compared by data and titles, not by pixels; nobody looked at them on a screen.
- Target modes that do not decode to a price forecast (magnitude, entry_range, '+relative') are skipped with a printed line,
  as `battery_step` does; the TF `entry_range` specs are not ported.
