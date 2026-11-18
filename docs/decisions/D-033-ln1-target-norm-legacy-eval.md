# D-033 — Legacy TF evaluation ported, target normalisation, panel logret_std arm, failure-chart audit (owner 2026-11-17)

Date: 2026-11-17
Owner decision: owner message 2026-11-17 (Arabic): port the tests of the old project that have no counterpart here (the chart of the best and worst predictions, the portfolio chart and the rest); build the target-normalisation component; let the panel model's normalisation support logret_std with relative_rank_loss; audit the earlier model reports, using Haiku to read the charts BEFORE normalisation of the samples the model failed on badly.
Why: the TF evaluation (crypto-signal-prediction `evaluation/`: tearsheet, integrity, trust_calibration, output_conditioned, trade_selection, plots) had no port: no equity / portfolio chart, no best / worst charts, no lag scan, no Brier. The port standardised the inputs (InputScaler) and scaled the NIG heads by one number, but never centred / scaled each head's labels as TF `logret_std` did (N-01). The worst rows of the reports were never looked at on raw prices.

## What changes behaviour
Every line is a NEW default of a NEW option or a NEW function; no existing default changes and the default run is unchanged:
- `target_norm` = `'none'` builds the same model as before (test `test_none_builds_the_same_model_and_fit_is_a_no_op`); `'logret_std'` is opt-in per model override and is used only by the new panel arm `relative_rank_loss_logret_std`, which is NOT in the panel job's default arm list (`PANEL_001_ARMS`, the same five arms as before).
- `eval/legacy_reports`, `eval/legacy_plots`, `workflow/legacy_report`: off the run path (a CLI over saved battery tables). `top_share` 0.1 and `cost` 0.0008 are the battery's own top-|mu| share and cost per side; `risk` 0.02 is the TF tearsheet's capital share per trade; `max_lag` 3, 5 buckets, 10 batches, 10 bins, 10 samples, 30 rows per coin and k in (0.25, 0.5, 1.0) are report sizes (TF used 5 lags, 10 batches, 10 bins).
- `eval/sample_charts` and `data.fetch.vision_klines`: a chart tool; `http` None = the real archive, `context_days` 45 = the 64-bar 4h window plus a month for the volume median; the `Sample` fields default to "unknown".
- The backtest (owner messages 2026-11-17: "a separate historical test for any model and any strategy ... one input, optional features that can be absent or switched off"; then "the main project is torch, but the structure must be separate so it can move to any project; it connects to the main system at one point, through a bridge"; "it must run on a finer frame than the training frames"; "it must work with the type of the model output: absolute when it decodes without loss, otherwise per the type the target represents") lives in the separate package `src/nig_backtest/`, outside `nig_timenet_torch`, so its defaults are not in this project's pin file. None of them touches the default run. They follow the battery's conventions where one exists (fee 0.0008 per side, risk 0.02 per trade, 2 bars = the artifact horizon, ATR 14, `next_open` entry = no lookahead, stop first when a bar touches both levels = the conservative reading, output kind `score` = direction and ranking only, so no exit assumes a price the output does not give). Everywhere else they are "off / all": no target, no stop, no cap on open trades, every coin, stage and present feature. Third owner message 2026-11-17 ("the VAL -> TEST point is right, with a switch to turn it off; train, val and test results apart, val and test may be merged, train never; prices loaded separately into one compressed store; candles matched against the last candle and target kept in the training data; costs none / fixed / by the size of the move; several overlap modes; ranked outputs on real prices with direction rules written as code; the report over every chosen batch, the charts per batch"): the protocol is ON by default (`Protocol.enabled` True: a TEST number is shown only for the variant chosen on VAL, the safe reading; off = exploration, marked in the table); `merge_val_test` False (periods apart unless asked); `costs.mode` "fixed" (the old fee behaviour); `book.per_coin` 1 and `book.admit` "first" (the earlier one-trade-per-coin behaviour); `max_missing_share` None (missing coins are reported with a warning, not a stop, until the owner sets a share); `side` "score" (the output kind's own direction); charts per batch of 50 coins (a size, not a behaviour).

- `nig_timenet_torch.data.fetch.vision_klines(http)`: `<new>` -> `None`
- `nig_timenet_torch.eval.legacy_reports.calibration(n_bins)`: `<new>` -> `10`
- `nig_timenet_torch.eval.legacy_reports.coin_ranking(min_rows)`: `<new>` -> `30`
- `nig_timenet_torch.eval.legacy_reports.conditioned(n_batches)`: `<new>` -> `10`
- `nig_timenet_torch.eval.legacy_reports.conditioned(n_buckets)`: `<new>` -> `5`
- `nig_timenet_torch.eval.legacy_reports.equity_curve(risk)`: `<new>` -> `0.02`
- `nig_timenet_torch.eval.legacy_reports.equity_curve(start)`: `<new>` -> `1.0`
- `nig_timenet_torch.eval.legacy_reports.extreme_samples(n)`: `<new>` -> `10`
- `nig_timenet_torch.eval.legacy_reports.integrity(max_lag)`: `<new>` -> `3`
- `nig_timenet_torch.eval.legacy_reports.lag_scan(max_lag)`: `<new>` -> `3`
- `nig_timenet_torch.eval.legacy_reports.move_over_sd(ks)`: `<new>` -> `(0.25, 0.5, 1.0)`
- `nig_timenet_torch.eval.legacy_reports.tearsheet(risk)`: `<new>` -> `0.02`
- `nig_timenet_torch.eval.legacy_plots.pred_vs_true(title)`: `<new>` -> `''`
- `nig_timenet_torch.eval.legacy_plots.top_coins(n)`: `<new>` -> `3`
- `nig_timenet_torch.eval.legacy_reports.trades(top_share)`: `<new>` -> `0.1`
- `nig_timenet_torch.eval.sample_charts.Sample(note)`: `<new>` -> `''`
- `nig_timenet_torch.eval.sample_charts.Sample(pnl)`: `<new>` -> `nan`
- `nig_timenet_torch.eval.sample_charts.Sample(side)`: `<new>` -> `''`
- `nig_timenet_torch.eval.sample_charts.Sample(source)`: `<new>` -> `''`
- `nig_timenet_torch.eval.sample_charts.Sample(time)`: `<new>` -> `False`
- `nig_timenet_torch.eval.sample_charts.load_candles(context_days)`: `<new>` -> `45`
- `nig_timenet_torch.eval.sample_charts.load_candles(http)`: `<new>` -> `None`
- `nig_timenet_torch.eval.sample_charts.plot_sample(extra)`: `<new>` -> `None`
- `nig_timenet_torch.eval.sample_charts.run(http)`: `<new>` -> `None`
- `nig_timenet_torch.model.builder.NigTimeNetV2(target_norm)`: `<new>` -> `'none'`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(target_norm)`: `<new>` -> `'none'`
- `nig_timenet_torch.workflow.legacy_report.legacy_from_folder(out_dir)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.legacy_report.legacy_report(cost)`: `<new>` -> `0.0008`
- `nig_timenet_torch.workflow.legacy_report.legacy_report(out_dir)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.legacy_report.legacy_report(plots)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.legacy_report.legacy_report(top_share)`: `<new>` -> `0.1`
- `nig_timenet_torch.workflow.legacy_report.stage_report(n_samples)`: `<new>` -> `10`
- `nig_timenet_torch.workflow.legacy_report.stage_report(top_share)`: `<new>` -> `0.1`
