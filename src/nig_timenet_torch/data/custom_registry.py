"""The registry of custom feature families and the `Bars` view they all read.

A family is one function ``fn(bars, settings)`` that adds its columns to ``bars.df``; adding a family is one new
function plus one ``@CUSTOM_FEATURES.decorator(...)`` line, nothing else changes. Families run in ascending `order`
(the order of the reference pipeline), which is also the column order of the feature matrix.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nig_timenet_torch.core.registry import Contract, Registry

EPS = 1e-12

CUSTOM_FEATURES = Registry("custom_feature", Contract("custom_feature", required_meta=("prefixes", "order")))


@dataclass
class Bars:
    """OHLCV frame (a copy, columns are added in place) + the float64 series every family needs."""
    df: pd.DataFrame
    tf_order: tuple[str, ...]

    def __post_init__(self):
        df = self.df
        self.close = df["close"].astype("float64")
        self.high = df["high"].astype("float64")
        self.low = df["low"].astype("float64")
        self.open_ = df["open"].astype("float64") if "open" in df else self.close.shift(1).bfill()
        self.log_close = np.log(self.close.clip(lower=EPS))
        self.ret_1 = self.log_close.diff()

    def ta(self) -> Any:
        """The pandas_ta_classic accessor of the frame (`df.ta`)."""
        from nig_timenet_torch.data.ta_backend import ta_module
        ta_module()
        return self.df.ta


def hour_features_meaningful(tf_order: tuple[str, ...]) -> bool:
    """Hour-of-day only carries information when some frame is shorter than a day (otherwise it is constant)."""
    from nig_timenet_torch.data.timeframe import normalize_tf
    try:
        smallest = min(pd.Timedelta(normalize_tf(tf)) for tf in tf_order)
    except (ValueError, TypeError):
        return True
    return smallest < pd.Timedelta(days=1)
