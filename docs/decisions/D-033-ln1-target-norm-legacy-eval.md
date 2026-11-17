# D-033 — Legacy TF evaluation ported, target normalisation, panel logret_std arm, failure-chart audit (owner 2026-11-17)

Date: 2026-11-17
Owner decision: owner message 2026-11-17 (Arabic): port the tests of the old project that have no counterpart here (the chart of the best and worst predictions, the portfolio chart and the rest); build the target-normalisation component; let the panel model's normalisation support logret_std with relative_rank_loss; audit the earlier model reports, using Haiku to read the charts BEFORE normalisation of the samples the model failed on badly.
Why: the TF evaluation (crypto-signal-prediction `evaluation/`: tearsheet, integrity, trust_calibration, output_conditioned, trade_selection, plots) had no port: no equity / portfolio chart, no best / worst charts, no lag scan, no Brier. The port standardised the inputs (InputScaler) and scaled the NIG heads by one number, but never centred / scaled each head's labels as TF `logret_std` did (N-01). The worst rows of the reports were never looked at on raw prices.

## What changes behaviour
Every line is a NEW default of a NEW option or a NEW function; no existing default changes and the default run is unchanged:
- `target_norm` = `'none'` builds the same model as before (test `test_none_builds_the_same_model_and_fit_is_a_no_op`); `'logret_std'` is opt-in per model override and is used only by the new panel arm `relative_rank_loss_logret_std`, which is NOT in the panel job's default arm list (`PANEL_001_ARMS`, the same five arms as before).
- `eval/legacy_reports`, `eval/legacy_plots`, `workflow/legacy_report`: off the run path (a CLI over saved battery tables). `top_share` 0.1 and `cost` 0.0008 are the battery's own top-|mu| share and cost per side; `risk` 0.02 is the TF tearsheet's capital share per trade; `max_lag` 3, 5 buckets, 10 batches, 10 bins, 10 samples, 30 rows per coin and k in (0.25, 0.5, 1.0) are report sizes (TF used 5 lags, 10 batches, 10 bins).
- `eval/sample_charts` and `data.fetch.vision_klines`: a chart tool; `http` None = the real archive, `context_days` 45 = the 64-bar 4h window plus a month for the volume median; the `Sample` fields default to "unknown".

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
