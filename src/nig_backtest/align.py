"""PURPOSE: Do the candles match the data the model was trained on? Compares a price source with the per-sample last
    candle and target kept BEFORE encoding (last high / low / close, future close / max high / min low) at each shift.
TAGS: alignment check, archive vs artifact, last candle, future target, time shift, bar open time, data integrity
PITFALLS: the reference ``time`` is the OPEN time of the sample's last bar (the artifact's timestamp); finer candles
    are resampled to the reference ``interval`` (left-closed) before the comparison; a mismatch that disappears at
    another shift means the two sources disagree on bar time (B-020), not on prices. Relative tolerance ``rtol``.
"""

import numpy as np
import pandas as pd

from nig_backtest.prices import PriceSource

__all__ = ["REFERENCE_COLUMNS", "check_alignment"]

REFERENCE_COLUMNS = ("coin", "time", "last_high", "last_low", "last_close", "future_close", "future_high_max",
                     "future_low_min")


def _resample(bars: pd.DataFrame, interval: pd.Timedelta) -> pd.DataFrame:
    return bars.resample(interval, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def _coin_rows(ref: pd.DataFrame, bars: pd.DataFrame, h: int, shift: int) -> pd.DataFrame:
    t = bars.index.to_numpy()
    i = np.searchsorted(t, ref["time"].to_numpy(), side="left") + shift
    ok = (i >= 0) & (i + h < len(t))
    i = np.clip(i, 0, len(t) - 1)
    hi, lo, cl = (bars[k].to_numpy() for k in ("high", "low", "close"))
    got = {"last_high": hi[i], "last_low": lo[i], "last_close": cl[i]}
    if h >= 1:
        fut = np.clip(i[:, None] + np.arange(1, h + 1)[None, :], 0, len(t) - 1)
        got.update(future_close=cl[fut[:, -1]], future_high_max=hi[fut].max(axis=1), future_low_min=lo[fut].min(axis=1))
    return pd.DataFrame({k: np.where(ok, v, np.nan) for k, v in got.items()}, index=ref.index)


def check_alignment(prices: PriceSource, reference: pd.DataFrame, horizon: str, interval: str,
                    shifts=(-1, 0, 1), rtol: float = 1e-3) -> dict:
    """For each shift (in reference bars): the share of samples whose six reference values match the candles within
    ``rtol``, per column; the best shift; the coins without candles; the worst mismatches at the best shift."""
    missing = [c for c in REFERENCE_COLUMNS if c not in reference]
    if missing:
        raise ValueError(f"alignment reference without {missing}")
    step = pd.Timedelta(interval)
    h = int(pd.Timedelta(horizon) / step)
    per_shift, no_prices, frames = {}, [], {s: [] for s in shifts}
    for coin, ref in reference.groupby("coin"):
        try:
            bars = prices.bars(str(coin))
        except KeyError:
            no_prices.append(str(coin))
            continue
        if bars.index.to_series().diff().min() < step:
            bars = _resample(bars, step)
        for s in shifts:
            frames[s].append(pd.concat([ref, _coin_rows(ref, bars, h, s).add_prefix("got_")], axis=1))
    cols = REFERENCE_COLUMNS[2:] if h >= 1 else REFERENCE_COLUMNS[2:5]
    for s in shifts:
        if not frames[s]:
            continue
        f = pd.concat(frames[s])
        ok = {c: np.isclose(f[f"got_{c}"], f[c], rtol=rtol, atol=0.0) for c in cols}
        per_shift[s] = {"rows": int(len(f)), **{c: float(v.mean()) for c, v in ok.items()},
                        "all": float(np.logical_and.reduce(list(ok.values())).mean())}
        frames[s] = f.assign(ok=np.logical_and.reduce(list(ok.values())))
    best = max(per_shift, key=lambda s: per_shift[s]["all"]) if per_shift else None
    worst = pd.DataFrame()
    if best is not None:
        f = frames[best]
        f = f.assign(rel_err=np.abs(f["got_last_close"] / f["last_close"] - 1.0))
        worst = f[~f["ok"]].nlargest(10, "rel_err")
    note = "" if h >= 1 else "future values not checked: the horizon is shorter than one candle of the interval"
    return {"per_shift": per_shift, "best_shift": best, "coins_without_prices": no_prices, "worst": worst,
            "note": note}
