"""Time-frame names: pandas offset aliases are case sensitive and deprecated for hours, so names are normalised once."""

import re

import pandas as pd

__all__ = ["normalize_tf", "tf_timedelta", "smallest_tf"]

_UNITS = {"m": "min", "min": "min", "h": "h", "d": "D", "w": "W"}
_PATTERN = re.compile(r"^\s*(\d*)\s*([A-Za-z]+)\s*$")


def normalize_tf(tf: str) -> str:
    """'1H' / '1h' / '60m' style names -> a pandas alias ('1h', '4h', '1D', '15min', '1W'); anything else raises."""
    match = _PATTERN.match(str(tf))
    if not match:
        raise ValueError(f"unknown time frame {tf!r}; examples: '15m', '1h', '4h', '1d', '1w'")
    count, unit = match.group(1) or "1", match.group(2)
    unit = unit.lower()
    if unit not in _UNITS:
        raise ValueError(f"unknown time frame unit in {tf!r}; use one of m, h, d, w")
    return f"{int(count)}{_UNITS[unit]}"


def tf_timedelta(tf: str) -> pd.Timedelta:
    return pd.Timedelta(normalize_tf(tf))


def smallest_tf(tfs) -> str:
    """The shortest of the given frames (the base frame whenever `base_tf` is not given)."""
    return min(tfs, key=tf_timedelta)
