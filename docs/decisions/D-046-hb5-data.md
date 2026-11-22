# D-046 — Hardening 2026-11b, package HB5: data structure (owner 2026-11-21)

Date: 2026-11-22
Owner decision: owner message 2026-11-21: "apply the improvement suggestions, fix the unclear places and launch a hardening campaign for the project; the goal is better structure and more clarity" (plan: `docs/hardening/PLAN-2026-11b.md`, package HB5); owner message 2026-11-22: "continue the work, preferably with Sonnet agents for each task"
Why: the audit `docs/hardening/audit-2026-11b/D_data_data_io.md` (rows D-01..D-10, D-17, D-19, D-23, D-32, E-19, re-verified by the agent against the code) found the five derived-artifact tools each re-implementing one pipeline, the indicator math three times, tools importing each other's private names, two candle caches, the npy_dir writer far from its reader, a `data` <-> `data_io` import cycle held by 17 lazy imports, two column maps of the target systems, two `OHLCV_AGG`, three spellings of the label time, data tools importing a torch module for a label key, three coin-name normalisers and the archive download twice.

## What changes behaviour
- Artifacts: none. A harness wrote every derived artifact, environment, native build and vision slice before and
  after on the test fixtures: 564 files byte-identical (manifests without id / created, derivation records without
  seconds); the tools keep their `producer` strings (`nig_timenet_torch.data_io.<tool>`) for that reason.
- Module paths: the derived tools and the vision slice live in `data/` (`lower_timeframe`, `indicator_channels`,
  `btc_relative`, `market_channel`, `market_residual`, `vision_slice`; the `nig-*` console scripts point there);
  `data_io/artifact_writer` (was `data/`), `data_io/lock` (was `data/`), `core/timeframe` (was `data/`); new
  `data/derive.py`, `data/channel_math.py`, `core/coins.py`, `core/label_time.py`. `data_io` no longer imports `data`.
  A notebook or script that imports an old path must use the new one.
- Candle cache: one file per coin and frame (`<symbol>_<tf>.parquet`, with the fetched range in its attrs); a cache file
  narrower than the columns asked or not covering the range is refetched (it was used silently).
- An unsupported `reg_target_mode` raises ValueError naming the supported modes (was KeyError).
- A reference coin given without the USDT suffix is recognised by the builder (one normaliser, `core.coins.norm_coin`).
- The removed keys below are module moves: the same defaults are pinned at the new paths.

- `nig_timenet_torch.data.artifact_writer.git_head(repo)`: `None` -> `<removed>`
- `nig_timenet_torch.data.artifact_writer.write_artifact(lock)`: `None` -> `<removed>`
- `nig_timenet_torch.data.artifact_writer.write_artifact(overwrite)`: `False` -> `<removed>`
- `nig_timenet_torch.data.artifact_writer.write_artifact(producer)`: `''` -> `<removed>`
- `nig_timenet_torch.data.artifact_writer.write_artifact(source_commit)`: `''` -> `<removed>`
- `nig_timenet_torch.data.btc_relative.add_btc_relative(cache)`: `<new>` -> `None`
- `nig_timenet_torch.data.btc_relative.add_btc_relative(fetch)`: `<new>` -> `None`
- `nig_timenet_torch.data.btc_relative.add_btc_relative(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.data.btc_relative.add_btc_relative(workers)`: `<new>` -> `16`
- `nig_timenet_torch.data.btc_relative.coin_channels(is_btc)`: `<new>` -> `False`
- `nig_timenet_torch.data.btc_relative.relative_close_labels(fetch)`: `<new>` -> `None`
- `nig_timenet_torch.data.btc_relative.relative_close_labels(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.data.channel_math.stoch_slow(high)`: `<new>` -> `None`
- `nig_timenet_torch.data.channel_math.stoch_slow(low)`: `<new>` -> `None`
- `nig_timenet_torch.data.channel_math.to_grid(columns)`: `<new>` -> `None`
- `nig_timenet_torch.data.channel_math.to_grid(first)`: `<new>` -> `None`
- `nig_timenet_torch.data.channel_math.to_grid(size)`: `<new>` -> `None`
- `nig_timenet_torch.data.derive.coin_candles(cols)`: `<new>` -> `('high', 'low', 'close', 'quote_volume')`
- `nig_timenet_torch.data.derive.make_parser(cache)`: `<new>` -> `True`
- `nig_timenet_torch.data.derive.make_parser(workers)`: `<new>` -> `True`
- `nig_timenet_torch.data.derive.per_coin_map(progress_every)`: `<new>` -> `25`
- `nig_timenet_torch.data.derive.write_derived(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.data.indicator_channels.add_indicator_channels(cache)`: `<new>` -> `None`
- `nig_timenet_torch.data.indicator_channels.add_indicator_channels(fetch)`: `<new>` -> `None`
- `nig_timenet_torch.data.indicator_channels.add_indicator_channels(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.data.indicator_channels.add_indicator_channels(workers)`: `<new>` -> `16`
- `nig_timenet_torch.data.indicator_channels.indicator_windows(x_close)`: `<new>` -> `None`
- `nig_timenet_torch.data.lower_timeframe.add_timeframe(cache)`: `<new>` -> `None`
- `nig_timenet_torch.data.lower_timeframe.add_timeframe(fetch)`: `<new>` -> `None`
- `nig_timenet_torch.data.lower_timeframe.add_timeframe(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.data.lower_timeframe.add_timeframe(tf)`: `<new>` -> `'1h'`
- `nig_timenet_torch.data.lower_timeframe.add_timeframe(window)`: `<new>` -> `32`
- `nig_timenet_torch.data.lower_timeframe.add_timeframe(workers)`: `<new>` -> `16`
- `nig_timenet_torch.data.market_channel.add_market_channel(days)`: `<new>` -> `7`
- `nig_timenet_torch.data.market_channel.add_market_channel(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.data.market_residual.add_market_residual(beta_bars)`: `<new>` -> `180`
- `nig_timenet_torch.data.market_residual.add_market_residual(cache)`: `<new>` -> `None`
- `nig_timenet_torch.data.market_residual.add_market_residual(candles)`: `<new>` -> `None`
- `nig_timenet_torch.data.market_residual.add_market_residual(fetch)`: `<new>` -> `None`
- `nig_timenet_torch.data.market_residual.add_market_residual(label)`: `<new>` -> `'residual'`
- `nig_timenet_torch.data.market_residual.add_market_residual(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.data.market_residual.rolling_beta(bars)`: `<new>` -> `180`
- `nig_timenet_torch.data.vision_slice.DEFAULT_COINS`: `<new>` -> `('BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT', 'TRXUSDT', 'ATOMUSDT', 'NEARUSDT', 'UNIUSDT', 'ETCUSDT', 'FILUSDT', 'XLMUSDT', 'AAVEUSDT', 'ALGOUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'EOSUSDT', 'XTZUSDT', 'THETAUSDT', 'VETUSDT', 'ICPUSDT', 'GRTUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'INJUSDT', 'RUNEUSDT', 'DYDXUSDT', 'CRVUSDT', 'SNXUSDT', 'COMPUSDT', 'ZECUSDT', 'DASHUSDT', 'ENJUSDT', 'CHZUSDT', 'KAVAUSDT', 'GALAUSDT', 'IMXUSDT', 'LDOUSDT', 'FETUSDT', 'EGLDUSDT', 'HBARUSDT', 'QNTUSDT', 'NEOUSDT', 'IOTAUSDT', 'ZILUSDT', 'BATUSDT', 'ONEUSDT', 'CELOUSDT', 'ANKRUSDT', 'SUSHIUSDT')`
- `nig_timenet_torch.data.vision_slice.build_logret_dataset(horizon)`: `<new>` -> `2`
- `nig_timenet_torch.data.vision_slice.build_logret_dataset(interval)`: `<new>` -> `'4h'`
- `nig_timenet_torch.data.vision_slice.build_logret_dataset(stride)`: `<new>` -> `1`
- `nig_timenet_torch.data.vision_slice.build_logret_dataset(time_seal)`: `<new>` -> `None`
- `nig_timenet_torch.data.vision_slice.build_logret_dataset(window)`: `<new>` -> `32`
- `nig_timenet_torch.data.vision_slice.fetch_coin(get)`: `<new>` -> `None`
- `nig_timenet_torch.data.vision_slice.fetch_month(get)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.artifact_writer.git_head(repo)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.artifact_writer.write_artifact(lock)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.artifact_writer.write_artifact(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.data_io.artifact_writer.write_artifact(producer)`: `<new>` -> `''`
- `nig_timenet_torch.data_io.artifact_writer.write_artifact(source_commit)`: `<new>` -> `''`
- `nig_timenet_torch.data_io.btc_relative.add_btc_relative(cache)`: `None` -> `<removed>`
- `nig_timenet_torch.data_io.btc_relative.add_btc_relative(fetch)`: `None` -> `<removed>`
- `nig_timenet_torch.data_io.btc_relative.add_btc_relative(overwrite)`: `False` -> `<removed>`
- `nig_timenet_torch.data_io.btc_relative.add_btc_relative(workers)`: `16` -> `<removed>`
- `nig_timenet_torch.data_io.btc_relative.coin_channels(is_btc)`: `False` -> `<removed>`
- `nig_timenet_torch.data_io.btc_relative.relative_close_labels(fetch)`: `None` -> `<removed>`
- `nig_timenet_torch.data_io.btc_relative.relative_close_labels(overwrite)`: `False` -> `<removed>`
- `nig_timenet_torch.data_io.indicator_channels.add_indicator_channels(cache)`: `None` -> `<removed>`
- `nig_timenet_torch.data_io.indicator_channels.add_indicator_channels(fetch)`: `None` -> `<removed>`
- `nig_timenet_torch.data_io.indicator_channels.add_indicator_channels(overwrite)`: `False` -> `<removed>`
- `nig_timenet_torch.data_io.indicator_channels.add_indicator_channels(workers)`: `16` -> `<removed>`
- `nig_timenet_torch.data_io.indicator_channels.indicator_windows(x_close)`: `None` -> `<removed>`
- `nig_timenet_torch.data_io.lower_timeframe.add_timeframe(cache)`: `None` -> `<removed>`
- `nig_timenet_torch.data_io.lower_timeframe.add_timeframe(fetch)`: `None` -> `<removed>`
- `nig_timenet_torch.data_io.lower_timeframe.add_timeframe(overwrite)`: `False` -> `<removed>`
- `nig_timenet_torch.data_io.lower_timeframe.add_timeframe(tf)`: `'1h'` -> `<removed>`
- `nig_timenet_torch.data_io.lower_timeframe.add_timeframe(window)`: `32` -> `<removed>`
- `nig_timenet_torch.data_io.lower_timeframe.add_timeframe(workers)`: `16` -> `<removed>`
- `nig_timenet_torch.data_io.market_channel.add_market_channel(days)`: `7` -> `<removed>`
- `nig_timenet_torch.data_io.market_channel.add_market_channel(overwrite)`: `False` -> `<removed>`
- `nig_timenet_torch.data_io.market_residual.add_market_residual(beta_bars)`: `180` -> `<removed>`
- `nig_timenet_torch.data_io.market_residual.add_market_residual(cache)`: `None` -> `<removed>`
- `nig_timenet_torch.data_io.market_residual.add_market_residual(candles)`: `None` -> `<removed>`
- `nig_timenet_torch.data_io.market_residual.add_market_residual(fetch)`: `None` -> `<removed>`
- `nig_timenet_torch.data_io.market_residual.add_market_residual(label)`: `'residual'` -> `<removed>`
- `nig_timenet_torch.data_io.market_residual.add_market_residual(overwrite)`: `False` -> `<removed>`
- `nig_timenet_torch.data_io.market_residual.rolling_beta(bars)`: `180` -> `<removed>`
- `nig_timenet_torch.data_io.npy_dir.save(export)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.vision_slice.DEFAULT_COINS`: `('BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT', 'TRXUSDT', 'ATOMUSDT', 'NEARUSDT', 'UNIUSDT', 'ETCUSDT', 'FILUSDT', 'XLMUSDT', 'AAVEUSDT', 'ALGOUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'EOSUSDT', 'XTZUSDT', 'THETAUSDT', 'VETUSDT', 'ICPUSDT', 'GRTUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'INJUSDT', 'RUNEUSDT', 'DYDXUSDT', 'CRVUSDT', 'SNXUSDT', 'COMPUSDT', 'ZECUSDT', 'DASHUSDT', 'ENJUSDT', 'CHZUSDT', 'KAVAUSDT', 'GALAUSDT', 'IMXUSDT', 'LDOUSDT', 'FETUSDT', 'EGLDUSDT', 'HBARUSDT', 'QNTUSDT', 'NEOUSDT', 'IOTAUSDT', 'ZILUSDT', 'BATUSDT', 'ONEUSDT', 'CELOUSDT', 'ANKRUSDT', 'SUSHIUSDT')` -> `<removed>`
- `nig_timenet_torch.data_io.vision_slice.build_logret_dataset(horizon)`: `2` -> `<removed>`
- `nig_timenet_torch.data_io.vision_slice.build_logret_dataset(interval)`: `'4h'` -> `<removed>`
- `nig_timenet_torch.data_io.vision_slice.build_logret_dataset(stride)`: `1` -> `<removed>`
- `nig_timenet_torch.data_io.vision_slice.build_logret_dataset(time_seal)`: `None` -> `<removed>`
- `nig_timenet_torch.data_io.vision_slice.build_logret_dataset(window)`: `32` -> `<removed>`
- `nig_timenet_torch.data_io.vision_slice.fetch_coin(get)`: `None` -> `<removed>`
- `nig_timenet_torch.data_io.vision_slice.fetch_month(get)`: `None` -> `<removed>`
