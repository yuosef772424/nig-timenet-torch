"""Day blocks: the calendar day of a timestamp and the block ids of a day-block bootstrap

PURPOSE: One definition of the calendar day of a timestamp and of the block id of a day (the unit a block bootstrap
    resamples), shared by the T-profit sections, the scorecard, the day metrics, the strategy book, deep side, gates.
TAGS: day block, calendar day, block bootstrap, bootstrap block, cluster by day, battery
PITFALLS: a day is the UTC calendar day of the timestamp (tz-aware input is read as UTC); the block SIZE is each
    section's own choice and is kept: 1-day blocks in T-profit, strategy book, deep side and gates (rows of one day
    move together), DAY_BLOCK_DAYS = 5 consecutive calendar days in the day metrics (labels overlap across midnight,
    volatility clusters); the scorecard cuts blocks of DAY_BLOCK_DAYS consecutive OBSERVED days (an index over its
    unique days, not a calendar offset: a gap in the days does not widen its blocks), so it uses only the constant.

A block id is only a label: the bootstrap (``stats.metrics.block_bootstrap_ci``) resamples the rows that share an id,
so any monotone relabelling of the same partition gives the same numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["calendar_day", "day_blocks", "DAY_BLOCK_DAYS"]

#: consecutive days resampled together by the multi-day sections (the scorecard, the day metrics)
DAY_BLOCK_DAYS = 5


def calendar_day(times) -> np.ndarray:
    """``datetime64[D]`` of every timestamp (an array, Series or index of datetimes; integers are read as ns)."""
    return pd.to_datetime(times).values.astype("datetime64[D]")


def day_blocks(days, block_days: int) -> np.ndarray:
    """Block id (int) of every day: consecutive CALENDAR days, ``block_days`` per block, counted from the first day
    present. ``block_days`` = 1 puts the rows of one day in one block."""
    d = np.asarray(days).astype("datetime64[D]").astype(np.int64)
    return (d - (d.min() if len(d) else 0)) // block_days
