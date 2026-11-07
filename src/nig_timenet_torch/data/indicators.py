"""Indicators and feature list

PURPOSE: Compute pandas_ta_classic indicators by name plus the custom families, and derive the feature list from
    what really ran.
TAGS: indicators, pandas_ta_classic, feature list, synthetic frame, exclude from features
PITFALLS: needs the [data] extra; the feature list is never hand-written, it is what add_indicators yields on a
    synthetic frame.

Technical indicators (pandas_ta_classic by name) + the custom families, and the feature list derived from them.

The feature list is never written by hand: it is whatever `add_indicators` produces on a synthetic frame with the
current settings, minus ``exclude_from_features`` (so it always matches the indicators that really ran).
"""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import OHLCV_COLUMNS
from nig_timenet_torch.data.custom_features import add_custom_features
from nig_timenet_torch.data.settings import FeatureSettings
from nig_timenet_torch.data.ta_backend import ta_module

__all__ = ["add_indicators", "synthetic_ohlcv", "base_feature_columns", "OHLCV"]

OHLCV = OHLCV_COLUMNS

#: indicators whose parameters are tuples (one call per tuple) -> the keyword names of the call
_TUPLE_PARAMS = {"stoch": ("fast_k", "slow_k"), "stochrsi": ("rsi_length", "stoch_length"),
                 "macd": ("fast", "slow", "signal"), "ppo": ("fast", "slow", "signal")}


def _call_indicator(df: pd.DataFrame, name: str, values) -> None:
    accessor = df.ta
    if not hasattr(accessor, name):
        raise ValueError(f"indicator {name!r} does not exist in pandas_ta_classic (FeatureSettings.indicator_settings)")
    method = getattr(accessor, name)
    if name in _TUPLE_PARAMS:
        for params in values:
            method(append=True, **dict(zip(_TUPLE_PARAMS[name], params, strict=False)))
    elif name == "donchian":
        for length in values:
            method(lower_length=length, upper_length=length, append=True)
    else:
        for length in values:
            method(length=length, append=True)


def add_indicators(data, features: FeatureSettings, tf_order: tuple[str, ...]) -> pd.DataFrame:
    """OHLCV frame -> frame with indicators, custom features and static indicators; rows with NaN/inf are dropped."""
    ta_module()
    df = pd.DataFrame(data).copy()
    for name, values in features.indicator_settings.items():
        _call_indicator(df, name, values)
    if features.custom.enabled:
        df = add_custom_features(df, features.custom, tf_order)
    for name in features.static_indicators:
        if name == "cross_value":
            df.ta.cross_value(value=50, append=True)
        elif hasattr(df.ta, name):
            getattr(df.ta, name)(append=True, **features.static_indicator_params.get(name, {}))
        else:
            raise ValueError(f"static indicator {name!r} does not exist in pandas_ta_classic")
    df = df.dropna()
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def synthetic_ohlcv(n: int = 1500, seed: int = 0, start: str = "2023-01-01", freq: str = "1h") -> pd.DataFrame:
    """A geometric random walk with plausible candles and volume; used to derive the feature list and in tests."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.004, n)) * close
    out = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                        "low": np.minimum(open_, close) - spread, "close": close,
                        "volume": rng.lognormal(10.0, 0.5, n)},
                       index=pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp"))
    return out


def base_feature_columns(features: FeatureSettings, tf_order: tuple[str, ...]) -> list[str]:
    """Feature columns of the per-coin stage (before the cross-asset columns), in the order the indicators ran."""
    probe = add_indicators(synthetic_ohlcv(1500), features, tf_order)
    return [c for c in probe.columns if c not in set(features.exclude_from_features)]
