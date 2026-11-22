"""Native data pipeline and derived-artifact tools (above data_io): raw candles -> features -> windows -> targets -> a
contract artifact (``nig-data``), the Binance downloaders (fetch, vision_slice) and the five derived-artifact tools
(lower_timeframe, indicator_channels, btc_relative, market_channel, market_residual; shared: derive, channel_math).
Settings live in `settings.DataSettings`; run it with `build.build_dataset`."""
