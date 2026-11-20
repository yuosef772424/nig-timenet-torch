# D-038 — Indicator channels (RSI, stochastic, volume) + architecture / size look + rank-loss walk-forward at lr 1e-5, long-only reading (owner 2026-11-19)

Date: 2026-11-19
Owner decision: owner message 2026-11-19: "add indicators like RSI and the stochastic", follow-up "you can load the data again and add volume if it is useful"
Why: the artifact `c572b1b36960` holds only three channels (log high / low / close over the previous close); the indicators need the coin's own candles with warm-up history, so a derived-artifact tool (same pattern as `market_channel` / `lower_timeframe`) adds them without changing rows, labels or splits. No existing default changes; every line below is a parameter of the new tool.

## What changes behaviour
- `cache` None = no candle cache (tests); Colab passes a Drive folder so a cut build resumes. `fetch` None = the public Binance archive (`data.fetch.vision_klines`); tests inject candles. `overwrite` False = an existing destination raises. `workers` 16 = the thread count `lower_timeframe` uses (the build of 2026-11-19 took 509 s for 535 coins). `x_close` None = no check against the artifact's own close channel (the artifact build always passes it: rows whose candles disagree get zeros and are counted).

- `btc_relative` (owner message 2026-11-19: inputs minus BTC per bar, RSI / stochastic on the coin / BTC price,
  BTC as a feature): `cache`, `fetch`, `overwrite`, `workers` mean what they mean for `add_indicator_channels`;
  `coin_channels(is_btc)` False = a normal coin (True only for BTC's own rows: zero relative channels).
  `relative_close_labels(fetch, overwrite)` (BR1, the next hypothesis of `docs/results/btc-relative-001.md`): the
  same meaning as above.
  `market_residual` (owner 2026-11-19, method 3): `add_market_residual(candles)` None = candles from the archive
  through the cache (`cache`, `fetch` as above); a callable (tests, the local vision_slice raw files) replaces it.
- `RollingSettings.fold_ids` (AI2, owner message 2026-11-20: "can all folds run in parallel, since each fold trains
  an independent model", then "try it with the next run"): None = every fold, exactly as before. A tuple runs only
  those fold indices, so several processes share one study and one `out`; the process that finds every fold x seed
  on disk writes `rolling_summary.*`, the pooled days and the manifest (the others print "rolling part OK"). Needs
  `retrain_every_k == 1`. Side effect for every run: `rolling_summary.json` `settings` gains the key
  `fold_ids: null`.

- `nig_timenet_torch.data_io.btc_relative.add_btc_relative(cache)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.btc_relative.add_btc_relative(fetch)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.btc_relative.add_btc_relative(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.data_io.btc_relative.add_btc_relative(workers)`: `<new>` -> `16`
- `nig_timenet_torch.data_io.btc_relative.coin_channels(is_btc)`: `<new>` -> `False`
- `nig_timenet_torch.data_io.btc_relative.relative_close_labels(fetch)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.btc_relative.relative_close_labels(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.data_io.indicator_channels.add_indicator_channels(cache)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.indicator_channels.add_indicator_channels(fetch)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.indicator_channels.add_indicator_channels(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.data_io.indicator_channels.add_indicator_channels(workers)`: `<new>` -> `16`
- `nig_timenet_torch.data_io.indicator_channels.indicator_windows(x_close)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.market_residual.add_market_residual(cache)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.market_residual.add_market_residual(candles)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.market_residual.add_market_residual(fetch)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.market_residual.add_market_residual(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.workflow.settings.RollingSettings(fold_ids)`: `<new>` -> `None`
