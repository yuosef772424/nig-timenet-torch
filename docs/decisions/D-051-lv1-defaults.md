# D-051 — Latest-samples table, trade selection, chart and live prediction (owner 2026-11-23)

Date: 2026-11-23
Owner decision: owner message 2026-11-23 (translated): "yes, build the list - start with the first (the latest-samples table) then the second (trade selection); it should show the confidence, the uncertainty, the expected and the true move, the error, the entry point and the expected and the true target; I also want the way to draw it; and I want the project ready for live trading: it fetches the data from Binance, prepares it, loads the model from a folder path and predicts, so that the model predicts more than one candle as a historical test and the last one is live"
Why: the TF project printed the last N samples per coin (`evaluation/latest_table.py`, `trade_selection.py`, `live.py`) and this project had no counterpart, so a trained model could not be looked at on its newest bars or used on live candles; every line below is a parameter default of a function that did not exist before (nothing existing changes its default).

## What changes behaviour
- New modules only: `eval/latest_table.py` (the table), `eval/trade_selection.py` (edge over uncertainty, ranking), `eval/latest_plots.py` (the chart), `workflow/latest_report.py` (CSV / markdown / PNG, a CLI over a run folder), `workflow/live_model.py` (`live_model.json` beside `best.pt`: a model loads from a folder path), `workflow/live_run.py` (fetch -> prepare -> load -> predict -> report), `data/fetch.recent_klines` (closed candles of the live API), `data/sources.frames_loader`, `data/build.build_live` and `prepare_asset(last_n=)` (the windows of the newest bars with the dataset code). `prepare_asset(last_n=None)` is the old behaviour byte for byte (`tests/data` green).
- Not a default but a behaviour: `eval.battery_adapter.to_battery_table` keeps a row whose label is NaN with `dir_true` 0 (before, the NaN was cast to an integer: a garbage value and a warning); rows with a finite label are unchanged.

- `nig_timenet_torch.data.build.build_live(lock)`: `<new>` -> `None`
- `nig_timenet_torch.data.fetch.recent_klines(http)`: `<new>` -> `None`
- `nig_timenet_torch.data.fetch.recent_klines(pause)`: `<new>` -> `1.0`
- `nig_timenet_torch.data.fetch.recent_klines(tries)`: `<new>` -> `3`
- `nig_timenet_torch.data.fetch.recent_klines(url)`: `<new>` -> `'https://fapi.binance.com/fapi/v1/klines'`
- `nig_timenet_torch.data.windows.prepare_asset(last_n)`: `<new>` -> `None`
- `nig_timenet_torch.eval.latest_plots.latest_chart(bar)`: `<new>` -> `None`
- `nig_timenet_torch.eval.latest_plots.latest_chart(candles)`: `<new>` -> `None`
- `nig_timenet_torch.eval.latest_plots.latest_chart(context_bars)`: `<new>` -> `24`
- `nig_timenet_torch.eval.latest_plots.latest_chart(horizon)`: `<new>` -> `None`
- `nig_timenet_torch.eval.latest_plots.latest_charts(candles)`: `<new>` -> `None`
- `nig_timenet_torch.eval.latest_table.format_latest_table(digits)`: `<new>` -> `2`
- `nig_timenet_torch.eval.latest_table.latest_table(coins)`: `<new>` -> `None`
- `nig_timenet_torch.eval.latest_table.latest_table(n)`: `<new>` -> `5`
- `nig_timenet_torch.eval.trade_selection.select_trades(cost)`: `<new>` -> `0.0008`
- `nig_timenet_torch.eval.trade_selection.select_trades(k)`: `<new>` -> `1.0`
- `nig_timenet_torch.eval.trade_selection.select_trades(min_after_cost)`: `<new>` -> `None`
- `nig_timenet_torch.eval.trade_selection.select_trades(min_confidence)`: `<new>` -> `0.0`
- `nig_timenet_torch.eval.trade_selection.select_trades(rank_by)`: `<new>` -> `'combined'`
- `nig_timenet_torch.eval.trade_selection.select_trades(require_agreement)`: `<new>` -> `False`
- `nig_timenet_torch.eval.trade_selection.select_trades(top)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.latest_report.latest_from_folder(out_dir)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.latest_report.latest_from_folder(stage)`: `<new>` -> `'val'`
- `nig_timenet_torch.workflow.latest_report.latest_report(candles)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.latest_report.latest_report(charts)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.latest_report.latest_report(coins)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.latest_report.latest_report(cost)`: `<new>` -> `0.0008`
- `nig_timenet_torch.workflow.latest_report.latest_report(k)`: `<new>` -> `1.0`
- `nig_timenet_torch.workflow.latest_report.latest_report(n)`: `<new>` -> `5`
- `nig_timenet_torch.workflow.latest_report.latest_report(out_dir)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.latest_report.latest_report(rank_by)`: `<new>` -> `'combined'`
- `nig_timenet_torch.workflow.latest_report.latest_report(show_latest)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.latest_report.latest_report(top)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.live_model.export_from_folder(architecture)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.live_model.export_from_folder(cross_section)`: `<new>` -> `False`
- `nig_timenet_torch.workflow.live_model.export_from_folder(input_timeframes)`: `<new>` -> `()`
- `nig_timenet_torch.workflow.live_model.export_from_folder(overrides)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.live_model.export_live_model(folder)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.live_model.load_live_model(device)`: `<new>` -> `'cpu'`
- `nig_timenet_torch.workflow.live_run.drop_stale(max_bars)`: `<new>` -> `2.0`
- `nig_timenet_torch.workflow.live_run.fetch_universe(http)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.live_run.fetch_universe(url)`: `<new>` -> `'https://fapi.binance.com/fapi/v1/klines'`
- `nig_timenet_torch.workflow.live_run.fetch_universe(workers)`: `<new>` -> `8`
- `nig_timenet_torch.workflow.live_run.live_run(bars)`: `<new>` -> `1000`
- `nig_timenet_torch.workflow.live_run.live_run(charts)`: `<new>` -> `True`
- `nig_timenet_torch.workflow.live_run.live_run(coins)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.live_run.live_run(cost)`: `<new>` -> `0.0008`
- `nig_timenet_torch.workflow.live_run.live_run(data_settings)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.live_run.live_run(device)`: `<new>` -> `'cpu'`
- `nig_timenet_torch.workflow.live_run.live_run(http)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.live_run.live_run(k)`: `<new>` -> `1.0`
- `nig_timenet_torch.workflow.live_run.live_run(klines_url)`: `<new>` -> `'https://fapi.binance.com/fapi/v1/klines'`
- `nig_timenet_torch.workflow.live_run.live_run(last)`: `<new>` -> `6`
- `nig_timenet_torch.workflow.live_run.live_run(out_dir)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.live_run.live_run(rank_by)`: `<new>` -> `'combined'`
- `nig_timenet_torch.workflow.live_run.live_run(top)`: `<new>` -> `None`
- `nig_timenet_torch.workflow.live_run.live_table(device)`: `<new>` -> `'cpu'`
