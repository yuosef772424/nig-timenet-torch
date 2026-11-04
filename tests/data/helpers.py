"""Shared builders for the data-pipeline tests."""

from nig_timenet_torch.data.settings import DataSettings

COINS = tuple(f"C{i}USDT" for i in range(4))


def tiny_settings(two_tf: bool = False, **extra) -> DataSettings:
    """The synthetic 4-coin / 40-day setup the TF parity fixtures were built with."""
    patch = {"source.kind": "synthetic", "source.coins": COINS, "source.synthetic_days": 40,
             "windows.tf_order": ("1h", "4h") if two_tf else ("1h",),
             "windows.window_sizes": {"1h": 16, "4h": 8} if two_tf else {"1h": 16},
             "windows.higher_tf_mode": "closed" if two_tf else "legacy",
             "features.market.reference_symbol": "C0USDT", **extra}
    return DataSettings().updated(patch)


def longer_settings(days: int = 100, **extra) -> DataSettings:
    """Same synthetic coins over more days, for rolling studies (several folds)."""
    return tiny_settings(**{"source.synthetic_days": days, **extra})
