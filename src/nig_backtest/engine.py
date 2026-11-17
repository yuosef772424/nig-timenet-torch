"""PURPOSE: Turn selected signal rows into trades on RAW candles: entry after the signal time, exit at the first of
    take-profit / stop / N bars (vectorised per coin), costs, excursions; coins loaded in batches, one at a time.
TAGS: backtest engine, trade simulation, take profit, stop loss, time exit, gap fill, excursion, per coin, batches,
    no lookahead
PITFALLS: entry is the open of the first bar that opens at or after ``time`` (``next_open``) or the close of the bar
    that closes at ``time`` (``signal_close``, which must exist exactly); ATR is known at entry (the bar before, or
    the last ``atr_timeframe`` bar closed by then); a bar touching target and stop counts per ``exit.both_hit``; a gap
    through a level fills at the bar's open (better for a target, worse for a stop); a path cut by the end of the
    data exits at its last close (``data_end``). The candles may be FINER than the model's frame (``exit.hold`` as a
    duration): finer candles order target and stop inside the model's bar. Cross-section rules (top_share,
    per_time_top_k) see every coin: selection runs on the whole signal table, only the candles load per coin.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nig_backtest._base import OHLCV as OHLCV_COLUMNS
from nig_backtest.contract import SignalSet
from nig_backtest.prices import PriceSource, atr, bar_interval
from nig_backtest.rules import Strategy, check_strategy, level_fraction, select_entries

__all__ = ["Backtest", "simulate_coin", "run_backtest", "hold_bars", "TRADE_COLUMNS", "EXIT_REASONS"]

EXIT_REASONS = ("take_profit", "stop_loss", "time", "data_end")
TRADE_COLUMNS = ("coin", "time", "side", "score", "entry_time", "entry_price", "exit_time", "exit_price",
                 "exit_reason", "bars_held", "gross", "net", "mfe", "mae", "atr_entry", "target_frac", "stop_frac",
                 "win", "direction_right")


@dataclass
class Backtest:
    """The trades of one strategy over one signal set, with what each entry rule kept and why rows were skipped."""
    strategy: Strategy
    trades: pd.DataFrame
    rule_trace: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)


def _paths(arr: np.ndarray, start: np.ndarray, n: int) -> np.ndarray:
    idx = start[:, None] + np.arange(n)[None, :]
    inside = idx < len(arr)
    return np.where(inside, arr[np.clip(idx, 0, len(arr) - 1)], np.nan)


def _first(mask: np.ndarray) -> np.ndarray:
    """First True per row, or the row length when none."""
    return np.where(mask.any(axis=1), mask.argmax(axis=1), mask.shape[1])


def _entry_points(rows: pd.DataFrame, bars: pd.DataFrame, how: str):
    t = bars.index.to_numpy()
    times = rows["time"].to_numpy()
    if how == "next_open":
        i0 = np.searchsorted(t, times, side="left")
        ok = i0 < len(t)
        return i0, ok, bars["open"].to_numpy()[np.minimum(i0, len(t) - 1)]
    dt = np.timedelta64(bar_interval(bars).value, "ns")
    j = np.searchsorted(t, times - dt, side="left")
    jc = np.minimum(j, len(t) - 1)
    ok = (j < len(t)) & (t[jc] == times - dt) & (j + 1 < len(t))
    return j + 1, ok, bars["close"].to_numpy()[jc]


def _hits(side, entry, tgt, stp, H, L):
    """First bar of target and stop per trade (row length = never)."""
    tp_px, sl_px = entry * (1 + side * tgt), entry * (1 - side * stp)
    long = side[:, None] > 0
    with np.errstate(invalid="ignore"):
        hit_tp = np.where(long, H >= tp_px[:, None], L <= tp_px[:, None]) & np.isfinite(tp_px)[:, None]
        hit_sl = np.where(long, L <= sl_px[:, None], H >= sl_px[:, None]) & np.isfinite(sl_px)[:, None]
    return _first(hit_tp), _first(hit_sl), tp_px, sl_px


def hold_bars(s: Strategy, bars: pd.DataFrame, horizon: str | None = None) -> int:
    """How many PRICE bars a trade is held: ``exit.max_bars``, or ``exit.hold`` (a duration, or "horizon") divided by
    the candles' interval, which must divide it (a 4h model on 1h candles: "8h" = 8 bars)."""
    hold = s.exit.hold
    if hold is None:
        return s.exit.max_bars
    dur = pd.Timedelta(horizon if hold == "horizon" else hold)
    step = bar_interval(bars)
    n, rest = divmod(dur, step)
    if rest != pd.Timedelta(0) or n < 1:
        raise ValueError(f"exit.hold {dur} is not a whole number of the candles' {step} bars")
    return int(n)


def _atr_at_entry(bars: pd.DataFrame, i0: np.ndarray, x) -> np.ndarray:
    """ATR known when the trade opens: of the price bar before the entry bar, or (``atr_timeframe``) of the last bar of
    that frame that closed at or before the entry bar's open (candles resampled, left-closed)."""
    if x.atr_timeframe is None:
        return atr(bars, x.atr_bars)[np.maximum(i0 - 1, 0)]
    tf = pd.Timedelta(x.atr_timeframe)
    coarse = bars.resample(tf, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    a = atr(coarse, x.atr_bars)
    closes = (coarse.index + tf).to_numpy()
    j = np.searchsorted(closes, bars.index.to_numpy()[i0], side="right") - 1
    return np.where(j >= 0, a[np.maximum(j, 0)], np.nan)


def simulate_coin(rows: pd.DataFrame, bars: pd.DataFrame, s: Strategy, horizon: str | None = None
                  ) -> tuple[pd.DataFrame, dict]:
    """Trades of one coin (``rows`` = its selected signals with ``side``) on its candles; returns (trades, skipped).
    The candles may be finer than the model's frame: the entry is the first bar opening at / after ``time``."""
    x, n = s.exit, hold_bars(s, bars, horizon)
    rows = rows.sort_values("time", kind="stable").reset_index(drop=True)
    i0, ok, entry = _entry_points(rows, bars, s.entry.entry_price)
    skipped = {"no_bars": int((~ok).sum())}
    rows, i0, entry = rows[ok].reset_index(drop=True), i0[ok], entry[ok]
    if rows.empty:
        return pd.DataFrame(columns=list(TRADE_COLUMNS)), skipped
    atr_e = _atr_at_entry(bars, i0, x)
    Op, H, L, C = (_paths(bars[k].to_numpy(), i0, n) for k in OHLCV_COLUMNS[:4])
    side = rows["side"].to_numpy()
    tgt = level_fraction(rows, x.take_profit, x.take_profit_unit, atr_e, entry, "target")
    stp = level_fraction(rows, x.stop_loss, x.stop_loss_unit, atr_e, entry, "stop")
    k_tp, k_sl, tp_px, sl_px = _hits(side, entry, tgt, stp, H, L)
    if x.both_hit == "stop_first":
        k_tp = np.where(k_tp == k_sl, n, k_tp)
    else:
        k_sl = np.where(k_tp == k_sl, n, k_sl)
    n_in = np.isfinite(C).sum(axis=1)
    k = np.minimum(np.minimum(k_tp, k_sl), n_in - 1)
    r = np.arange(len(rows))
    reason = np.where(k == k_tp, "take_profit", np.where(k == k_sl, "stop_loss",
                                                         np.where(n_in == n, "time", "data_end")))
    o_k = Op[r, k]
    with np.errstate(invalid="ignore"):
        fill_tp = np.where(side > 0, np.fmax(tp_px, o_k), np.fmin(tp_px, o_k))   # a gap past the level fills at open
        fill_sl = np.where(side > 0, np.fmin(sl_px, o_k), np.fmax(sl_px, o_k))
    exit_px = np.where(reason == "take_profit", fill_tp, np.where(reason == "stop_loss", fill_sl, C[r, k]))
    held = np.arange(n)[None, :] <= k[:, None]
    hi, lo = np.nanmax(np.where(held, H, np.nan), axis=1), np.nanmin(np.where(held, L, np.nan), axis=1)
    gross = side * (exit_px / entry - 1.0)
    t = bars.index.to_numpy()
    dt = np.timedelta64(bar_interval(bars).value, "ns")
    out = rows.copy()
    out = out.assign(entry_time=t[i0], entry_price=entry, exit_time=t[np.minimum(i0 + k, len(t) - 1)] + dt,
                     exit_price=exit_px, exit_reason=reason, bars_held=k + 1, gross=gross,
                     net=gross - 2 * (s.costs.fee_per_side + s.costs.slippage_per_side),
                     mfe=np.where(side > 0, hi / entry - 1, 1 - lo / entry),
                     mae=np.where(side > 0, lo / entry - 1, 1 - hi / entry), atr_entry=atr_e, target_frac=tgt,
                     stop_frac=stp, _exit_bar=i0 + k, _entry_bar=i0)
    if s.entry.one_per_coin:
        out, skipped["overlap"] = _one_per_coin(out)
    out["held"] = out["exit_time"] - out["entry_time"]
    out["win"] = out["net"] > 0
    out["direction_right"] = out["gross"] > 0
    if "y_true" in out:      # did the LABEL the model was trained on go the traded way (whatever the exit did)?
        out["label_right"] = np.sign(out["y_true"].to_numpy()) == out["side"].to_numpy()
    return out.drop(columns=["_exit_bar", "_entry_bar"]), skipped


def _one_per_coin(trades: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    keep, last_exit = [], -1
    for i, (a, b) in enumerate(zip(trades["_entry_bar"].to_numpy(), trades["_exit_bar"].to_numpy(), strict=True)):
        if a > last_exit:
            keep.append(i)
            last_exit = b
    return trades.iloc[keep].reset_index(drop=True), len(trades) - len(keep)


def run_backtest(signals: SignalSet, prices: PriceSource, strategy: Strategy, *, coins=None,
                 batch_coins: int = 50, progress=None) -> Backtest:
    """Select the entries on the whole signal table (cross-section rules see every coin), then simulate coin by coin
    in batches of ``batch_coins`` (the candles of one coin are loaded once and may leave the cache after)."""
    check_strategy(strategy, signals.enabled, signals.absolute, signals.horizon)
    rows = signals.read(coins)
    picked = select_entries(rows, strategy)
    trace = picked.attrs.get("rule_trace", [])
    parts, skipped = [], {"no_bars": 0, "overlap": 0, "no_prices": 0}
    names = sorted(picked["coin"].unique())
    for b in range(0, len(names), max(1, batch_coins)):
        for coin in names[b:b + batch_coins]:
            sub = picked[picked["coin"] == coin]
            try:
                bars = prices.bars(coin)
            except KeyError:
                skipped["no_prices"] += len(sub)
                continue
            trades, sk = simulate_coin(sub, bars, strategy, signals.horizon)
            for key, v in sk.items():
                skipped[key] += v
            parts.append(trades)
        if progress is not None:
            progress(min(b + batch_coins, len(names)), len(names))
    trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(TRADE_COLUMNS))
    trades = trades.sort_values(["entry_time", "coin"], kind="stable").reset_index(drop=True)
    return Backtest(strategy, trades, trace + [("traded", len(trades))], skipped)
