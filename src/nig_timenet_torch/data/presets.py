"""Named patches for `DataSettings.updated(...)`, each a dict of dotted keys. A preset is one entry here; the
defaults of DataSettings already are the production hourly preset, the others are small changes on top."""

from nig_timenet_torch.data.settings import DataSettings

__all__ = ["PRESETS", "preset", "TINY"]

TINY = {"windows.window_sizes": {"1h": 16}, "source.kind": "synthetic", "source.synthetic_days": 70,
        "source.coins": tuple(f"C{i}USDT" for i in range(6)),
        "features.market.reference_symbol": "C0USDT"}

PRESETS = {
    "hourly_w32_s8": {},
    "hourly_4h_closed": {"windows.tf_order": ("1h", "4h"), "windows.window_sizes": {"1h": 32, "4h": 32},
                         "windows.higher_tf_mode": "closed", "runtime.x_storage_dtype": "float16"},
    "dated_splits": {"output.split_dates": {"train_end": "2025-06-24", "val_end": "2025-11-21"},
                     "output.holdout_start": "2026-07-15"},
    "tiny": TINY,
}


def preset(*names: str, base: DataSettings | None = None) -> DataSettings:
    """`base` (default: DataSettings()) with the named presets applied in order."""
    settings = base or DataSettings()
    for n in names:
        if n not in PRESETS:
            raise KeyError(f"unknown preset {n!r}; known: {sorted(PRESETS)}")
        settings = settings.updated(PRESETS[n])
    return settings
