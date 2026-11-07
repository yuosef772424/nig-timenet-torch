"""Run custom feature families

PURPOSE: Run the registered custom feature families over a frame in registry order.
TAGS: custom features, families, registry order, feature matrix
PITFALLS: families run in ascending order, which is also the column order of the feature matrix; reordering changes
    the artifact.

Run the registered custom feature families over a frame (see custom_registry).
"""

import pandas as pd

from nig_timenet_torch.data import custom_advanced, custom_basic  # noqa: F401  (import registers the families)
from nig_timenet_torch.data.custom_registry import CUSTOM_FEATURES, Bars
from nig_timenet_torch.data.settings import CustomFeatureSettings

__all__ = ["add_custom_features", "family_names"]


def family_names() -> tuple[str, ...]:
    """Registered family names in the order they run."""
    return tuple(e.name for e in sorted(CUSTOM_FEATURES.entries(), key=lambda e: e.meta["order"]))


def add_custom_features(data: pd.DataFrame, settings: CustomFeatureSettings, tf_order: tuple[str, ...]) -> pd.DataFrame:
    """Copy of `data` (needs high/low/close, optionally open/volume) with every enabled family's columns added."""
    bars = Bars(pd.DataFrame(data).copy(), tuple(tf_order))
    for entry in sorted(CUSTOM_FEATURES.entries(), key=lambda e: e.meta["order"]):
        entry.obj(bars, settings)
    return bars.df
