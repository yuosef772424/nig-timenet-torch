"""Label time

PURPOSE: The two times a window's label has, named once: the OPEN of the label's last bar (the time seal asks
    about it) and the CLOSE of that bar (when the label is known: purging asks about it).
TAGS: label time, label bar open, label bar close, time seal, purge, leakage, rule d1, rule d2
PITFALLS: for bar-aligned data ``open < seal  <=>  close <= seal`` (close = open + one bar), so the seal check on
    the open and the purge check on the close agree; use the open for the seal (writer, loader) and the close for
    purging (split plan, the vision slice's seal) and never mix them inside one comparison. ``ts`` is the window's
    last bar OPEN time (``last_candles`` timestamp), ``horizon`` the dataset's forecast_horizon in bars.
"""

__all__ = ["label_bar_open", "label_bar_close"]


def label_bar_open(ts, horizon: int, bar_ns: int):
    """OPEN time (ns) of the label's last bar: ``ts + horizon * bar`` (scalar or numpy array)."""
    return ts + int(horizon) * int(bar_ns)


def label_bar_close(ts, horizon: int, bar_ns: int):
    """CLOSE time (ns) of the label's last bar = when the label is known: ``ts + (horizon + 1) * bar``."""
    return ts + (int(horizon) + 1) * int(bar_ns)
