# D-034 — Panel walk-forwards: std target normalisation, LS selection, two timeframes 4h + 1h (owner 2026-11-18)

Date: 2026-11-18
Owner decision: owner messages 2026-11-18 (Arabic): "test the panel model with std normalisation of the batch data, use the rolling training"; "run another process based on relative_rank_loss_ls for the best version"; "also prepare a process with two frames, the 4 hour frame and the 1 hour frame, then prepare the process and give me the run command so that if the limit ends I run it on Colab after the running trainings end".
Why: the run steps refused an artifact with more than one timeframe (`workflow/run_common.input_key`, ROADMAP N5) and no artifact held a 1h window. The 1h window is added to the existing artifact row by row (same rows, labels, coins, splits), so a run with and without it differs only in that input. The channels and the alignment were checked against the Binance archive on BTCUSDT 2021-05-20 (artifact = USD-M futures, `last_candles` timestamp = the 4h bar OPEN time, X = log(high / low / close over the previous close); this also settles B-020 on raw klines for that coin).

## What changes behaviour
Every line is a NEW default of a NEW option or function; the default run is unchanged (owner 2026-11-19 added: a CI for the kept-row rules of the panel report, which adds keys and columns and no default; and the ordered means on prices with a target normalisation):
- `add_market_channel` (owner 2026-11-19, the USDT supply channel): `days` `7` = the 7-day growth the model-free look measured (docs/results/dominance-look-001.md, addendum 2); `overwrite` `False` = never replace an artifact silently. An off-path tool: it writes a new derived artifact and changes no run default.
- `order_in_label_units` `False` (NigTimeNetV2, build_nig_timenet_v2) = the order of the means applied where it was (in the heads' space, normalised when `target_norm` is set); True needs the ordered means and a fitted `target_norm` (raises otherwise) and orders the means on prices after the inverse map: the measured cause of the high / low ranking loss of the std arm (B-079, docs/results/panel-rolling-std-001.md); used only by the new panel arm `relative_rank_loss_logret_std_label_order`, not in any default arm list.
- `input_timeframes` `()` (LoaderSettings, `job_run_settings`, `flat_inputs`, `input_keys`) and `[]` (RollingConfig) = the artifact's one frame, exactly as before (a two-frame artifact is still refused unless frames are named); the `RunSettings(data)` line is the same LoaderSettings with the new field at that default. Named frames are put side by side on the channel axis (windows of one length), the simplest reading of two frames; separate encoder branches (the model already builds them) are a later arm.
- `add_timeframe`: `tf` `'1h'` and `window` `32` = the owner's frame and the source's window length; `cache` None = no candle cache (the Colab chain passes a Drive folder); `fetch` None = the Binance archive (`data.fetch.vision_klines`); `workers` 16 = parallel downloads (sizes, not behaviour); `overwrite` False = never replace an artifact silently.

- `nig_timenet_torch.data_io.loading.LoaderSettings(input_timeframes)`: `<new>` -> `()`
- `nig_timenet_torch.data_io.lower_timeframe.add_timeframe(cache)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.lower_timeframe.add_timeframe(fetch)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.lower_timeframe.add_timeframe(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.data_io.lower_timeframe.add_timeframe(tf)`: `<new>` -> `'1h'`
- `nig_timenet_torch.data_io.lower_timeframe.add_timeframe(window)`: `<new>` -> `32`
- `nig_timenet_torch.data_io.lower_timeframe.add_timeframe(workers)`: `<new>` -> `16`
- `nig_timenet_torch.data_io.market_channel.add_market_channel(days)`: `<new>` -> `7`
- `nig_timenet_torch.data_io.market_channel.add_market_channel(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.jobs.rolling.RollingConfig(input_timeframes)`: `<new>` -> `[]`
- `nig_timenet_torch.model.builder.NigTimeNetV2(order_in_label_units)`: `<new>` -> `False`
- `nig_timenet_torch.model.builder.build_nig_timenet_v2(order_in_label_units)`: `<new>` -> `False`
- `nig_timenet_torch.workflow.flat_run.flat_inputs(timeframes)`: `<new>` -> `()`
- `nig_timenet_torch.workflow.run_common.input_keys(timeframes)`: `<new>` -> `()`
- `nig_timenet_torch.workflow.settings.RunSettings(data)`: `LoaderSettings(directory='.', name='preprocessing_output', format='auto', mmap=True, local_dir='auto', allow_unchecked=False, verify_hashes=False, expect_base_tf=None)` -> `LoaderSettings(directory='.', name='preprocessing_output', format='auto', mmap=True, local_dir='auto', allow_unchecked=False, verify_hashes=False, expect_base_tf=None, input_timeframes=())`
- `nig_timenet_torch.workflow.study_settings.job_run_settings(input_timeframes)`: `<new>` -> `()`
