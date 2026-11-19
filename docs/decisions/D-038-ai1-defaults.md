# D-038 — Indicator channels (RSI, stochastic, volume) + architecture / size look + rank-loss walk-forward at lr 1e-5, long-only reading (owner 2026-11-19)

Date: 2026-11-19
Owner decision: owner message 2026-11-19: "add indicators like RSI and the stochastic", follow-up "you can load the data again and add volume if it is useful"
Why: the artifact `c572b1b36960` holds only three channels (log high / low / close over the previous close); the indicators need the coin's own candles with warm-up history, so a derived-artifact tool (same pattern as `market_channel` / `lower_timeframe`) adds them without changing rows, labels or splits. No existing default changes; every line below is a parameter of the new tool.

## What changes behaviour
- `cache` None = no candle cache (tests); Colab passes a Drive folder so a cut build resumes. `fetch` None = the public Binance archive (`data.fetch.vision_klines`); tests inject candles. `overwrite` False = an existing destination raises. `workers` 16 = the thread count `lower_timeframe` uses (the build of 2026-11-19 took 509 s for 535 coins). `x_close` None = no check against the artifact's own close channel (the artifact build always passes it: rows whose candles disagree get zeros and are counted).

- `nig_timenet_torch.data_io.indicator_channels.add_indicator_channels(cache)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.indicator_channels.add_indicator_channels(fetch)`: `<new>` -> `None`
- `nig_timenet_torch.data_io.indicator_channels.add_indicator_channels(overwrite)`: `<new>` -> `False`
- `nig_timenet_torch.data_io.indicator_channels.add_indicator_channels(workers)`: `<new>` -> `16`
- `nig_timenet_torch.data_io.indicator_channels.indicator_windows(x_close)`: `<new>` -> `None`
