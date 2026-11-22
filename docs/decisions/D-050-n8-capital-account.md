# D-050 — Evaluation in a capital account (audit E1-E3, E5)

Date: 2026-11-22
Owner decision: owner message 2026-11-22 ("start the next step n8, work on the latest branch") on ROADMAP N8 (owner 2026-11-21, from `docs/results/independent-review-001.md`): "each a switch whose default reproduces today (rule B1), with a test: an `account` section (fixed slot per coin, cash for unused slots, fee on turnover, compounding; CAGR / Sharpe / max drawdown / per year) beside always-long, BTC buy-and-hold and a random-coin timing placebo at the same timestamps and counts; a symmetric trim (best AND worst 5 % of days) and the median day, printed for the benchmark too"
Why: the independent review (2026-11-21) found the measurement, not the signal, behind part of the "no profit after cost" readings: E1 the scorecard verdict drops only the best 5 % of days and puts no benchmark through that test (any long crypto book fails it, BTC buy-and-hold included); E2 the "mean day" is the equal-weight mean of a day's trades, not an account return (the same book ruins an account at one share per timestamp and earns +14 % a year at fixed slots); E3 every 8 h row pays a full round trip even when the coin is kept; E5 TEST drops the `dir_true == 0` rows, chosen by the outcome (2.5 % of rows).

## What changes behaviour
- New section `account` (`eval.account.account_section`, after `scorecard_verdict` in `battery.SECTION_ORDER`, also in
  `battery_profit.json`): the long book (TEST rows with mu >= the fold's VAL 90th percentile of mu, long whatever
  p_dir says: the audit's book) in a capital account: one sub-book per phase of the row step inside the label
  horizon (the horizon read from `last_close`, or `account_books`), a fixed slot of 1 / slots per pick (slots = the
  coins of TEST), cash otherwise, the cost per side on the weight TURNOVER, compounded; CAGR, Sharpe (annualised),
  max drawdown, exposure, each year. Beside it: the same book paying a round trip every period (the battery's charge,
  E3), with all its capital on the picks (audit a3), at 0.02 / 0.05 / 0.10 / 0.15 % per side (a5), always long on the
  same slots, BTCUSDT buy-and-hold, the equal-weight basket held with the book's breadth as exposure (a6), and 20
  draws of a placebo that buys random coins at the same timestamps in the same numbers (mean, 95th percentile, share
  of draws the book beats). Every TEST row with a finite y_ret and mu is used (flat rows kept). A table without
  end_time / coin / a readable horizon prints the section as SKIPPED. Every existing section and number is unchanged.
- Scorecard (`eval.battery_scorecard`): `day_profile` adds `mean_sym_trim` (the mean day without the best AND worst
  5 % of days); the scorecard table prints it and the median day as two new columns; `benchmark_days` puts every row
  long and BTCUSDT long through the same day test, printed under the table and kept in `card["benchmarks"]`, never in
  `card["stats"]` (so the verdict's best row is unchanged); `scorecard_verdict(trim)` = `TRIM_KEYS[trim]`: `best`
  (default, the rule until now: `mean_wo_best > 0`) or `symmetric` (`mean_sym_trim > 0`); the verdict prints the
  trim and the benchmarks' trimmed mean day.
- `full_battery(drop_flat)` (default True = today): False marks the tables with `battery_lib.keep_flat`, and
  `battery_lib.scored_rows` then keeps the flat rows in every section (E5).
- `BatterySettings` gains `verdict_trim` ("best"), `drop_flat` (True) and `account_books` (0 = read the horizon),
  passed by `battery_step`; `run_tables` / `battery_from_folder` take the same three.
- `workflow.battery_run.account_from_folders` (and `python -m nig_timenet_torch.workflow.battery_run <fold folders>`)
  runs the account over the TEST of several fold folders, each with its own VAL threshold: the audit's a3-a6 view of
  a rolling run.

- `nig_timenet_torch.eval.account.account_curve(fee)`: `<new>` -> `'turnover'`
- `nig_timenet_torch.eval.account.account_frame(quantile)`: `<new>` -> `0.9`
- `nig_timenet_torch.eval.account.account_section(benchmark)`: `<new>` -> `'BTCUSDT'`
- `nig_timenet_torch.eval.account.account_section(cost)`: `<new>` -> `0.0008`
- `nig_timenet_torch.eval.account.account_section(fee_grid)`: `<new>` -> `(0.0002, 0.0005, 0.001, 0.0015)`
- `nig_timenet_torch.eval.account.account_section(n_books)`: `<new>` -> `None`
- `nig_timenet_torch.eval.account.account_section(n_placebo)`: `<new>` -> `20`
- `nig_timenet_torch.eval.account.account_section(quantile)`: `<new>` -> `0.9`
- `nig_timenet_torch.eval.account.account_section(seed)`: `<new>` -> `0`
- `nig_timenet_torch.eval.account.account_section(slots)`: `<new>` -> `None`
- `nig_timenet_torch.eval.account.account_stats(exposure)`: `<new>` -> `nan`
- `nig_timenet_torch.eval.account.book_weights(sizing)`: `<new>` -> `'slots'`
- `nig_timenet_torch.eval.account.horizon_steps(step)`: `<new>` -> `None`
- `nig_timenet_torch.eval.battery.full_battery(account_books)`: `<new>` -> `None`
- `nig_timenet_torch.eval.battery.full_battery(drop_flat)`: `<new>` -> `True`
- `nig_timenet_torch.eval.battery.full_battery(trim)`: `<new>` -> `'best'`
- `nig_timenet_torch.eval.battery_scorecard.scorecard_sections(trim)`: `<new>` -> `'best'`
- `nig_timenet_torch.eval.battery_scorecard.scorecard_verdict(trim)`: `<new>` -> `'best'`
- `nig_timenet_torch.workflow.battery_run.account_from_folders(account_books)`: `<new>` -> `0`
- `nig_timenet_torch.workflow.battery_run.account_from_folders(cost)`: `<new>` -> `0.0008`
- `nig_timenet_torch.workflow.battery_run.battery_from_folder(account_books)`: `<new>` -> `0`
- `nig_timenet_torch.workflow.battery_run.battery_from_folder(drop_flat)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.battery_run.battery_from_folder(trim)`: `<new>` -> `'best'`
- `nig_timenet_torch.workflow.battery_run.run_tables(account_books)`: `<new>` -> `0`
- `nig_timenet_torch.workflow.battery_run.run_tables(drop_flat)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.battery_run.run_tables(trim)`: `<new>` -> `'best'`
- `nig_timenet_torch.workflow.settings.BatterySettings(account_books)`: `<new>` -> `0`
- `nig_timenet_torch.workflow.settings.BatterySettings(drop_flat)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.settings.BatterySettings(verdict_trim)`: `<new>` -> `'best'`
- `nig_timenet_torch.workflow.settings.EvalSettings(battery)`: `BatterySettings(enabled=True, cost_per_side=0.0008, n_variants=1, day_metrics=False)` -> `BatterySettings(enabled=True, cost_per_side=0.0008, n_variants=1, day_metrics=False, verdict_trim='best', drop_flat=True, account_books=0)`
- `nig_timenet_torch.workflow.settings.RunSettings(evaluation)`: `EvalSettings(chunk=20000, health_max_rows=100000, battery=BatterySettings(enabled=True, cost_per_side=0.0008, n_variants=1, day_metrics=False))` -> `EvalSettings(chunk=20000, health_max_rows=100000, battery=BatterySettings(enabled=True, cost_per_side=0.0008, n_variants=1, day_metrics=False, verdict_trim='best', drop_flat=True, account_books=0))`
