"""PURPOSE: Charts and numeric facts of single samples on RAW candles (before any normalisation): the input window of 4h
    bars, the event day in 1h bars and the volume, so a person (or a cheaper model) can see why a prediction failed.
TAGS: sample chart, raw candles, before normalisation, failure sample, best worst samples, candlestick, volume, facts
PITFALLS: the candles come from the public Binance futures archive (``data.fetch.vision_klines``), the source of the
    TF artifact, but a coin delisted or renamed since can be missing (``facts`` then says ``bars_4h`` 0 and the chart
    is skipped). When the report gives only a DATE, the event is that UTC day: ``facts`` measures the largest 2-bar
    (8 h, the artifact horizon) move inside it, which is an upper bound of the sample's own move. matplotlib is
    imported inside the plot function only (an optional dependency, extra ``plots``).

Why (owner, 2026-11-17): the worst rows of the earlier reports (``docs/results/baseline-001.md``: shorts that lost
the 172 % label clip) were listed as numbers only; nobody had looked at what the market did before and during them.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Sample", "facts", "plot_sample", "WINDOW_BARS", "HORIZON_BARS", "load_candles", "read_samples", "run",
           "main"]

WINDOW_BARS = 64     # 4h bars of context drawn before the event day (about the TF input window)
HORIZON_BARS = 2     # the artifact's label horizon (2 bars of 4h)


@dataclass(frozen=True)
class Sample:
    """One row to chart: ``coin`` (e.g. ``AIOTUSDT``), ``date`` (UTC day or exact bar time), ``side`` (``long`` /
    ``short`` / ``""``), ``pnl`` (the reported result in %, NaN if unknown), ``source`` (report and line) and a
    free ``note``; ``time`` is True when ``date`` is an exact bar time instead of a day."""
    coin: str
    date: str
    side: str = ""
    pnl: float = math.nan
    source: str = ""
    note: str = ""
    time: bool = False

    @property
    def tag(self) -> str:
        return f"{self.coin}_{pd.Timestamp(self.date).strftime('%Y%m%d%H%M' if self.time else '%Y%m%d')}"


def load_candles(sample: Sample, http=None, context_days: int = 45) -> dict[str, pd.DataFrame]:
    """{"4h": frame, "1h": frame} around the sample from the public archive (``context_days`` before, 2 days after)."""
    from nig_timenet_torch.data.fetch import vision_klines
    day = pd.Timestamp(sample.date, tz="UTC").floor("D")
    lo, hi = day - pd.Timedelta(days=context_days), day + pd.Timedelta(days=2)
    return {"4h": vision_klines(sample.coin, "4h", lo, hi, http),
            "1h": vision_klines(sample.coin, "1h", day - pd.Timedelta(days=2), hi, http)}


def _event(sample: Sample) -> tuple[pd.Timestamp, pd.Timestamp]:
    t = pd.Timestamp(sample.date, tz="UTC")
    if sample.time:
        return t, t + pd.Timedelta(hours=4 * HORIZON_BARS)
    return t.floor("D"), t.floor("D") + pd.Timedelta(days=1)


def facts(sample: Sample, c4: pd.DataFrame) -> dict:
    """Numbers about the sample from the RAW 4h candles: the largest 2-bar up / down move from a bar close inside the
    event, the day's return, the volume against the 30 days before, the moves and the volatility before the event,
    flat (stale) bars and the bars available before it (a proxy of how long the coin has traded)."""
    out = {"coin": sample.coin, "date": sample.date, "side": sample.side, "pnl": sample.pnl, "bars_4h": int(len(c4))}
    if len(c4) < HORIZON_BARS + 2:
        return out
    start, end = _event(sample)
    close, high, low = c4["close"].to_numpy(), c4["high"].to_numpy(), c4["low"].to_numpy()
    t = c4.index
    inside = np.flatnonzero((t >= start) & (t < end))
    before = c4[t < start]
    up, down = [], []
    for i in inside:
        j = slice(i + 1, min(i + 1 + HORIZON_BARS, len(c4)))
        if j.start < j.stop:
            up.append(high[j].max() / close[i] - 1.0)
            down.append(low[j].min() / close[i] - 1.0)
    logret = np.diff(np.log(before["close"].to_numpy())) if len(before) > 1 else np.array([])
    w = logret[-WINDOW_BARS:]
    day_vol = c4["quote_volume"][(t >= start) & (t < end)].sum()
    daily = before["quote_volume"].resample("1D").sum()
    out.update({
        "max_up_2bar_pct": 100 * float(max(up)) if up else math.nan,
        "max_down_2bar_pct": 100 * float(min(down)) if down else math.nan,
        "event_return_pct": 100 * float(close[inside[-1]] / before["close"].iloc[-1] - 1.0)
        if len(inside) and len(before) else math.nan,
        "volume_vs_30d_median": float(day_vol / daily.tail(30).median()) if len(daily) and daily.tail(30).median() > 0
        else math.nan,
        "ret_7d_before_pct": 100 * float(np.expm1(logret[-42:].sum())) if len(logret) else math.nan,
        "ret_30d_before_pct": 100 * float(np.expm1(logret[-180:].sum())) if len(logret) else math.nan,
        "window_vol_4h_pct": 100 * float(w.std()) if len(w) > 1 else math.nan,
        "window_max_abs_bar_pct": 100 * float(np.expm1(np.abs(w).max())) if len(w) else math.nan,
        "window_flat_bars": int((before["high"] == before["low"]).tail(WINDOW_BARS).sum()),
        "bars_before_event": int(len(before)),
        "fetched_from": str(t[0]),
        "event_close_vs_window_high": float(close[inside[-1]] / before["high"].tail(WINDOW_BARS).max())
        if len(inside) and len(before) else math.nan,
    })
    return out


def _candles(ax, frame: pd.DataFrame, width: float) -> None:
    x = frame.index.to_pydatetime()
    o, h, lo, c = (frame[k].to_numpy() for k in ("open", "high", "low", "close"))
    colors = np.where(c >= o, "#2a9d8f", "#e76f51")
    ax.vlines(x, lo, h, colors=colors, linewidth=0.8)
    ax.bar(x, np.abs(c - o) + 1e-12 * np.maximum(c, 1e-12), bottom=np.minimum(o, c), width=width, color=colors,
           edgecolor=colors, linewidth=0.5)


def plot_sample(sample: Sample, candles: dict[str, pd.DataFrame], out_dir, extra: dict | None = None) -> Path | None:
    """Three panels on RAW prices: the 4h window before the event and the event (event shaded, the 2-bar horizon
    after each bar), the quote volume, and the event day +- 1 day in 1h bars; a text box with ``facts``. Returns the
    PNG path, or None when the archive has no candles of the coin there."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c4, c1 = candles["4h"], candles["1h"]
    if len(c4) < 4:
        return None
    start, end = _event(sample)
    lo = start - pd.Timedelta(hours=4 * WINDOW_BARS)
    w4 = c4[(c4.index >= lo) & (c4.index < end + pd.Timedelta(hours=4 * HORIZON_BARS))]
    w1 = c1[(c1.index >= start - pd.Timedelta(days=1)) & (c1.index < end + pd.Timedelta(days=1))]
    f = {**facts(sample, c4), **(extra or {})}
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [3, 1, 3]})
    _candles(axes[0], w4, width=4 / 24 * 0.7)
    axes[1].bar(w4.index.to_pydatetime(), w4["quote_volume"].to_numpy(), width=4 / 24 * 0.7, color="#6c757d")
    if len(w1):
        _candles(axes[2], w1, width=1 / 24 * 0.7)
    axes[0].set_yscale("log")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.axvspan(start.to_pydatetime(), end.to_pydatetime(), color="#ffd166", alpha=0.25)
        ax.grid(alpha=0.25)
    pnl = "" if math.isnan(sample.pnl) else f"  reported {sample.pnl:+.0f}%"
    axes[0].set_title(f"{sample.coin}  {sample.date}  side={sample.side or '-'}{pnl}   (4h bars, RAW prices, log "
                      f"scale, {WINDOW_BARS} bars before the event; shaded = event)", fontsize=10)
    axes[1].set_title("quote volume (USDT), 4h", fontsize=9)
    axes[2].set_title("event day +- 1 day, 1h bars (RAW prices, log scale)", fontsize=9)
    keys = ("max_up_2bar_pct", "max_down_2bar_pct", "event_return_pct", "volume_vs_30d_median", "ret_7d_before_pct",
            "ret_30d_before_pct", "window_vol_4h_pct", "window_max_abs_bar_pct", "window_flat_bars",
            "bars_before_event", "fetched_from")
    text = "\n".join(f"{k}: {f[k]:.2f}" if isinstance(f.get(k), float) else f"{k}: {f.get(k)}" for k in keys if k in f)
    axes[0].text(0.01, 0.98, text, transform=axes[0].transAxes, va="top", fontsize=7.5, family="monospace",
                 bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#adb5bd"})
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.tight_layout()
    out = Path(out_dir) / f"{sample.tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def read_samples(path) -> list[Sample]:
    """Samples from a CSV with the columns of ``Sample`` (``coin`` and ``date`` required)."""
    frame = pd.read_csv(path, dtype=str).fillna("")
    out = []
    for r in frame.to_dict("records"):
        out.append(Sample(r["coin"], r["date"], r.get("side", ""), float(r["pnl"]) if r.get("pnl") else math.nan,
                          r.get("source", ""), r.get("note", ""), str(r.get("time", "")).lower() in ("1", "true")))
    return out


def run(samples: list[Sample], out_dir, http=None) -> pd.DataFrame:
    """Chart every sample into ``out_dir`` and write ``facts.csv`` there; a coin without archive candles gets a row
    with ``chart`` empty."""
    rows = []
    for s in samples:
        try:
            candles = load_candles(s, http)
        except Exception as exc:  # noqa: BLE001 - one missing coin must not stop the batch; the reason is recorded
            rows.append({"coin": s.coin, "date": s.date, "chart": "", "error": repr(exc)[:200]})
            continue
        path = plot_sample(s, candles, out_dir)
        rows.append({**facts(s, candles["4h"]), "source": s.source, "note": s.note,
                     "chart": "" if path is None else path.name})
    table = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(out_dir) / "facts.csv", index=False)
    return table


def main(argv=None) -> None:
    """``python -m nig_timenet_torch.eval.sample_charts SAMPLES.csv OUT_DIR``"""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("samples")
    ap.add_argument("out")
    a = ap.parse_args(argv)
    print(run(read_samples(a.samples), a.out).to_string())


if __name__ == "__main__":
    main()
