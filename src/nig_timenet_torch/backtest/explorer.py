"""PURPOSE: Look at the trades: filter them (coins, winners / losers, right / wrong direction, exit reason, side, stage,
    time, size of the result), group them, and draw any trade on its raw candles with entry, exit, target and stop.
TAGS: trade explorer, trade filter, winners, losers, accurate trades, trade chart, gallery, candles, backtest
PITFALLS: filters only narrow (each returns a new TradeBook); a chart reads the candles through the same PriceSource
    as the run, so it shows exactly the prices the trade was simulated on; ``gallery`` draws at most ``n`` charts.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.backtest.portfolio import by_group
from nig_timenet_torch.backtest.prices import PriceSource, bar_interval

__all__ = ["TradeBook"]


class TradeBook:
    """A filterable view of a trade table (``Backtest.trades``)."""

    def __init__(self, trades: pd.DataFrame):
        self.trades = trades.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.trades)

    def where(self, *, coins=None, outcome: str | None = None, direction_right: bool | None = None,
              label_right: bool | None = None, exit_reason=None, side: str | None = None, stage=None, start=None,
              end=None, min_net=None, max_net=None, query: str | None = None) -> "TradeBook":
        """Trades that meet every given condition: ``outcome`` "win" / "loss" (net > 0 / <= 0), ``direction_right``
        (gross > 0), ``label_right`` (the model's label went the traded way; needs y_true), ``exit_reason`` (one or
        several of take_profit / stop_loss / time / data_end), ``side`` "long" / "short", ``stage``, entry time in
        [start, end), net in [min_net, max_net], and a pandas ``query``."""
        t = self.trades
        m = np.ones(len(t), dtype=bool)
        if coins is not None:
            m &= t["coin"].isin([coins] if isinstance(coins, str) else list(coins)).to_numpy()
        if outcome is not None:
            if outcome not in ("win", "loss"):
                raise ValueError(f"outcome must be 'win' or 'loss', not {outcome!r}")
            m &= t["win"].to_numpy() == (outcome == "win")
        if direction_right is not None:
            m &= t["direction_right"].to_numpy() == bool(direction_right)
        if label_right is not None:
            if "label_right" not in t:
                raise ValueError("label_right needs the y_true feature in the signals")
            m &= t["label_right"].to_numpy() == bool(label_right)
        if exit_reason is not None:
            m &= t["exit_reason"].isin([exit_reason] if isinstance(exit_reason, str) else list(exit_reason)).to_numpy()
        if side is not None:
            m &= (t["side"].to_numpy() > 0) == (side == "long")
        if stage is not None:
            m &= t["stage"].isin([stage] if isinstance(stage, str) else list(stage)).to_numpy()
        if start is not None:
            m &= (t["entry_time"] >= pd.Timestamp(start)).to_numpy()
        if end is not None:
            m &= (t["entry_time"] < pd.Timestamp(end)).to_numpy()
        if min_net is not None:
            m &= t["net"].to_numpy() >= min_net
        if max_net is not None:
            m &= t["net"].to_numpy() <= max_net
        out = t[m]
        return TradeBook(out.query(query) if query else out)

    def worst(self, n: int = 10) -> pd.DataFrame:
        return self.trades.nsmallest(n, "net")

    def best(self, n: int = 10) -> pd.DataFrame:
        return self.trades.nlargest(n, "net")

    def by(self, key: str = "coin") -> pd.DataFrame:
        """The group table (``portfolio.by_group``) by a column, or by "month" of the entry."""
        if key == "month":
            return by_group(self.trades, self.trades["entry_time"].dt.to_period("M").astype(str))
        return by_group(self.trades, key)

    def plot(self, i: int, prices: PriceSource, out_dir, context_bars: int = 48) -> Path:
        """Chart of trade ``i`` (row position) on its raw candles: ``context_bars`` before the entry, the holding
        bars, the entry / exit marks and the target / stop lines."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from nig_timenet_torch.eval.sample_charts import draw_candles

        tr = self.trades.iloc[i]
        bars = prices.bars(tr["coin"])
        dt = bar_interval(bars)
        w = bars[(bars.index >= tr["entry_time"] - context_bars * dt) & (bars.index < tr["exit_time"] + 4 * dt)]
        fig, ax = plt.subplots(figsize=(12, 5))
        draw_candles(ax, w, width=dt / pd.Timedelta(days=1) * 0.7)
        ax.axvspan(tr["entry_time"], tr["exit_time"], color="#ffd166", alpha=0.25)
        ax.scatter([tr["entry_time"]], [tr["entry_price"]], marker="^" if tr["side"] > 0 else "v", s=80,
                   color="#264653", zorder=5, label="entry")
        ax.scatter([tr["exit_time"]], [tr["exit_price"]], marker="x", s=80, color="#e76f51", zorder=5, label="exit")
        for frac, sign, color, label in ((tr["target_frac"], 1, "#2a9d8f", "target"),
                                         (tr["stop_frac"], -1, "#e63946", "stop")):
            if np.isfinite(frac):
                ax.axhline(tr["entry_price"] * (1 + sign * tr["side"] * frac), color=color, ls="--", lw=1, label=label)
        conf = f"  conf {tr['confidence']:.3g}" if "confidence" in tr and pd.notna(tr.get("confidence")) else ""
        ax.set_title(f"{tr['coin']}  {'LONG' if tr['side'] > 0 else 'SHORT'}  {tr['entry_time']:%Y-%m-%d %H:%M}  "
                     f"score {tr['score']:+.4g}{conf}  exit {tr['exit_reason']} after {int(tr['bars_held'])} bars  "
                     f"net {100 * tr['net']:+.2f}%  (raw prices)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = Path(out_dir) / f"{i:05d}_{tr['coin']}_{tr['entry_time']:%Y%m%d%H%M}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=85)
        plt.close(fig)
        return path

    def gallery(self, prices: PriceSource, out_dir, n: int = 20, order: str = "worst") -> list[Path]:
        """Charts of the ``n`` worst / best / first trades and an ``index.md`` listing them."""
        pick = {"worst": lambda: self.trades.nsmallest(n, "net"), "best": lambda: self.trades.nlargest(n, "net"),
                "first": lambda: self.trades.head(n)}[order]()
        paths = [self.plot(int(self.trades.index.get_loc(i)), prices, out_dir) for i in pick.index]
        lines = [f"# {order} {len(paths)} trades", "", "| chart | coin | entry | side | exit | net % |",
                 "|---|---|---|---|---|---|"]
        for p, (_, tr) in zip(paths, pick.iterrows(), strict=True):
            lines.append(f"| [{p.name}]({p.name}) | {tr['coin']} | {tr['entry_time']:%Y-%m-%d %H:%M} | "
                         f"{'long' if tr['side'] > 0 else 'short'} | {tr['exit_reason']} | {100 * tr['net']:+.2f} |")
        (Path(out_dir) / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return paths
