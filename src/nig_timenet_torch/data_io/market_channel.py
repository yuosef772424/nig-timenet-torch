"""PURPOSE: Add a market-level input channel from an outside DAILY series (e.g. the USDT supply) to a contract artifact:
    for every bar of every window, the log growth of the series over k days as known the day BEFORE that bar.
TAGS: market channel, outside data, usdt supply, stablecoin, daily series, log growth, no lookahead, derived artifact
PITFALLS: the value at a bar is the growth up to the previous UTC day (a daily point stamped day D is never used on day
    D), so nothing of the bar's own day leaks; bars before the series has k + 1 days get 0 and are counted
    (``missing_share``); the channel is the same for every coin of a timestamp (the relative inputs give it 0 in their
    two relative blocks); rows, labels, coins and splits stay the source's. Owner task 2026-11-19 (USDT supply).

    python -m nig_timenet_torch.data_io.market_channel SRC DST --series FILE.csv --name usdt_supply_g7 --days 7
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io import npy_dir
from nig_timenet_torch.data_io.artifact import read_manifest

__all__ = ["daily_growth", "bar_values", "add_market_channel", "DERIVATION_JSON", "main"]

DERIVATION_JSON = "derivation_market_channel.json"


def daily_growth(series: pd.Series, days: int) -> pd.Series:
    """log(v_d / v_{d - days}) on a daily UTC grid (gaps forward-filled; the index is the day)."""
    s = series.sort_index()
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC").floor("D") if s.index.tz is not None else \
        pd.DatetimeIndex(s.index).tz_localize("UTC").floor("D")
    s = s[~s.index.duplicated(keep="last")].asfreq("D").ffill()
    return np.log(s.astype(float)).diff(days)


def bar_values(growth: pd.Series, last_open_ns: np.ndarray, base_tf: str, window: int) -> tuple[np.ndarray, np.ndarray]:
    """(values (n, window) float32, valid (n,) bool): for each window ending at the bar that OPENS at ``last_open_ns``,
    the growth known the day before each of its ``window`` bars (bar j opens at last - (window - 1 - j) * base_tf)."""
    step = pd.Timedelta(base_tf).value
    offs = (np.arange(window) - (window - 1)) * step
    opens = np.asarray(last_open_ns, dtype=np.int64)[:, None] + offs[None, :]          # (n, window) bar open times
    day = 86400 * 10**9
    prev_day = (opens // day - 1) * day                                                  # the previous UTC day, ns
    g = growth.dropna()
    keys = g.index.as_unit("ns").asi8
    pos = np.searchsorted(keys, prev_day)
    hit = (pos < len(keys)) & (keys[np.minimum(pos, len(keys) - 1)] == prev_day)
    vals = np.where(hit, g.to_numpy(np.float64)[np.minimum(pos, len(keys) - 1)], 0.0).astype(np.float32)
    return vals, hit.all(axis=1)


def add_market_channel(src, dst, series: pd.Series, *, name: str, days: int = 7, overwrite: bool = False) -> dict:
    """Write ``dst``: the contract artifact ``src`` with one more channel ``name`` in every ``X_<tf>`` window (the base
    frame's), its rows, labels, coins and splits unchanged. Returns the derivation record (also in ``dst``)."""
    from nig_timenet_torch.data.artifact_writer import write_artifact

    src, dst = Path(src), Path(dst)
    man = read_manifest(src)
    ds = npy_dir.load(src, mmap=False)
    if name in ds["feature_order"]:
        raise ValueError(f"{src} already has a channel {name!r}")
    if len(ds["timeframes"]) != 1:
        raise ValueError(f"{src} holds {list(ds['timeframes'])}: add the channel to a one-frame artifact")
    key = f"X_{ds['base_tf']}"
    x = np.asarray(ds[key])
    ts = np.asarray(ds["last_candles"])[:, TS_COL].astype(np.int64)
    vals, ok = bar_values(daily_growth(series, days), ts, ds["base_tf"], x.shape[1])
    ds = {**ds, key: np.concatenate([x, vals[:, :, None].astype(x.dtype)], axis=-1),
          "feature_order": [*ds["feature_order"], name]}
    manifest = write_artifact(ds, dst, source_commit=man.source_commit, overwrite=overwrite,
                              producer=f"nig_timenet_torch.data_io.market_channel (+{name} on {man.data_fingerprint})")
    record = {"source_artifact": str(src), "source_fingerprint": man.data_fingerprint,
              "fingerprint": manifest["export"]["data_sha256"], "added_channel": name, "growth_days": days,
              "rule": "log growth over `growth_days` days, value of the previous UTC day for every bar",
              "series_first": str(series.index.min()), "series_last": str(series.index.max()),
              "rows": int(len(ts)), "missing_rows": int((~ok).sum()), "missing_share": float((~ok).mean())}
    (dst / DERIVATION_JSON).write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nig_timenet_torch.data_io.market_channel", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--series", required=True, help="CSV: first column the date, second column the value")
    ap.add_argument("--name", required=True)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    raw = pd.read_csv(a.series, index_col=0)
    series = pd.Series(raw.iloc[:, 0].to_numpy(), index=pd.to_datetime(raw.index, utc=True))
    r = add_market_channel(a.src, a.dst, series, name=a.name, days=a.days, overwrite=a.overwrite)
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
