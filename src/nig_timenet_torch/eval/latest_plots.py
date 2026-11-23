"""PURPOSE: The chart of a coin's latest samples in original prices: the candles (when given), each row's entry, its
    expected target with the uncertainty band, the true target, and below it the expected vs true move with the error.
TAGS: latest chart, latest samples plot, entry target chart, uncertainty band, expected vs true move, live row,
    candlestick, matplotlib, price units
PITFALLS: matplotlib is an optional dependency (extra ``plots``) imported inside the plot function only, Agg backend; a
    row is drawn from its entry (the prediction time, the close of its last bar) to entry + ``horizon``; the horizon is
    inferred from the spacing of the rows unless given, so pass it for a single row; a LIVE row (no truth) is drawn
    dashed with no true marker; candles are indexed by bar OPEN time (UTC; naive or aware), ``bar`` is the bar
    length (a ``pd.Timedelta``); one PNG per coin, the file name holds the coin and the last time.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.eval.latest_table import LATEST_COLUMNS
from nig_timenet_torch.eval.sample_charts import draw_candles

__all__ = ["latest_chart", "latest_charts"]

GREEN, RED, GREY, INK = "#2a9d8f", "#e76f51", "#6c757d", "#264653"


def _horizon(times: pd.Series, horizon) -> pd.Timedelta:
    if horizon is not None:
        return pd.Timedelta(horizon)
    gaps = pd.to_datetime(times).sort_values().diff().dropna()
    if gaps.empty:
        raise ValueError("latest_chart: one row has no spacing to infer the horizon from; pass horizon=")
    return pd.Timedelta(gaps.median())


def _row_artists(ax, r: pd.Series, h: pd.Timedelta) -> None:
    t0, t1 = pd.Timestamp(r["time"]), pd.Timestamp(r["time"]) + h
    colour = GREEN if r["signal"] == "LONG" else RED
    live = r["status"] == "live"
    ax.plot([t0, t1], [r["entry"], r["target"]], color=colour, lw=1.6, ls="--" if live else "-", marker="o",
            markersize=3, zorder=3)
    ax.vlines(t1, r["target_lo"], r["target_hi"], color=colour, lw=4, alpha=0.35, zorder=2)
    if not live:
        ax.scatter([t1], [r["true_target"]], marker="D", s=26, color=INK, zorder=4)
        ax.annotate("" if pd.isna(r["ok"]) else ("ok" if r["ok"] else "wrong"), (t1, r["true_target"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points", color=INK)
    else:
        ax.annotate("LIVE", (t1, r["target"]), fontsize=7, xytext=(4, 4), textcoords="offset points", color=colour)


def _move_panel(ax, d: pd.DataFrame) -> None:
    x = np.arange(len(d))
    w = 0.38
    ax.bar(x - w / 2, d["move_pct"], w, color=[GREEN if s == "LONG" else RED for s in d["signal"]], alpha=0.85,
           yerr=d["uncertainty_pct"], error_kw={"ecolor": GREY, "lw": 1, "capsize": 2}, label="expected move +-sd")
    ax.bar(x + w / 2, d["true_move_pct"].fillna(0.0), w, color=INK, alpha=0.8, label="true move")
    for i, (_, r) in enumerate(d.iterrows()):
        txt = "live" if r["status"] == "live" else f"err {r['error_pct']:+.2f}"
        ax.annotate(txt, (i, 0), fontsize=7, ha="center", xytext=(0, -12), textcoords="offset points", color=GREY)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([pd.Timestamp(t).strftime("%m-%d %H:%M") for t in d["time"]], rotation=20, fontsize=8)
    ax.set_ylabel("move, % of entry")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)


def latest_chart(latest: pd.DataFrame, coin: str, out_dir, candles: pd.DataFrame | None = None, *, horizon=None,
                 bar=None, context_bars: int = 24, title: str = "") -> Path:
    """One PNG for ``coin``: price panel (candles of ``candles`` when given, every row's entry -> expected target with
    the +-sd band and the true target) and the move panel (expected +-sd vs true, with the error). ``latest`` is an
    ``eval.latest_table.latest_table`` frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    missing = [c for c in LATEST_COLUMNS if c not in latest.columns]
    if missing:
        raise ValueError(f"latest_chart: the table lacks {missing}; pass the output of latest_table")
    d = latest[latest["coin"] == coin].sort_values("time")
    if d.empty:
        raise ValueError(f"latest_chart: no rows for {coin!r}; the table holds {sorted(set(latest['coin']))}")
    h = _horizon(d["time"], horizon)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [3, 2]})
    view = None
    if candles is not None and len(candles):
        if candles.index.tz is not None:                      # the table's times are naive UTC
            candles = candles.set_axis(candles.index.tz_convert("UTC").tz_localize(None))
        b = pd.Timedelta(bar) if bar is not None else pd.Timedelta(candles.index.to_series().diff().median())
        lo = pd.Timestamp(d["time"].min()) - b * context_bars
        view = candles[(candles.index >= lo) & (candles.index <= pd.Timestamp(d["time"].max()) + h)]
        if len(view):
            draw_candles(top, view, width=b / pd.Timedelta(days=1) * 0.7)
    for _, r in d.iterrows():
        _row_artists(top, r, h)
    first = pd.Timestamp(d["time"].min())
    left = min(first, view.index.min()) if view is not None and len(view) else first
    top.set_xlim(left - h / 2, pd.Timestamp(d["time"].max()) + h + h / 2)   # the live row's target must be inside
    top.set_title(f"{title} {coin}: entry -> expected target (band = +-sd) and true target (diamond); dashed = live"
                  .strip(), fontsize=10)
    top.set_ylabel("price")
    top.grid(alpha=0.25)
    top.tick_params(axis="x", labelrotation=20, labelsize=8)
    _move_panel(bottom, d)
    fig.tight_layout()
    out = Path(out_dir) / f"latest_{coin}_{pd.Timestamp(d['time'].max()).strftime('%Y%m%d_%H%M')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def latest_charts(latest: pd.DataFrame, out_dir, candles: dict | None = None, **kw) -> list[Path]:
    """``latest_chart`` for every coin of ``latest`` (candles: {coin: frame}), in coin order."""
    return [latest_chart(latest, c, out_dir, (candles or {}).get(c), **kw) for c in sorted(set(latest["coin"]))]
