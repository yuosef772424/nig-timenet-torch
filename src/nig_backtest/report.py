"""PURPOSE: Write a backtest to a folder: the trades (parquet), the summary (markdown + JSON), the rule trace, the
    equity / drawdown chart, the tables by coin / month / exit, and galleries of the worst and best trades.
TAGS: backtest report, trades parquet, summary markdown, equity chart, rule trace, gallery, report writer
PITFALLS: charts need matplotlib (extra ``plots``) and, for the galleries, the PriceSource of the run; ``gallery``
    0 writes no trade charts. The summary covers every chosen coin, per period; the charts are drawn per batch of coins
    (``batch_NNN/``) so a large run stays drawable. Exploratory: one strategy on one signal set is a look.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nig_backtest.engine import Backtest
from nig_backtest.explorer import TradeBook
from nig_backtest.portfolio import by_period, equity
from nig_backtest.rules import strategy_to_dict

__all__ = ["write_report", "md_table", "plot_curves"]


def md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """A markdown table (no tabulate dependency); a '|' inside a cell is escaped."""
    if frame is None or frame.empty:
        return "(none)"
    def f(v):
        return f"{v:.{digits}g}" if isinstance(v, (float, np.floating)) else str(v).replace("|", "\\|")
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rows = ["| " + " | ".join(f(v) for v in r) + " |" for r in frame.itertuples(index=False)]
    return "\n".join([head, "|" + "---|" * len(frame.columns), *rows])


def plot_curves(curves: dict, path, title: str = "") -> Path:
    """Equity (top) and drawdown (bottom) of one or several trade curves ({label: portfolio.equity frame})."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for label, c in curves.items():
        if len(c):
            ax[0].plot(pd.to_datetime(c["exit_time"]), c["equity"], lw=1.2, label=label)
            ax[1].plot(pd.to_datetime(c["exit_time"]), 100 * c["drawdown"], lw=1.0)
    ax[0].axhline(1.0, color="#6c757d", ls="--", lw=0.8)
    ax[0].set_title(title or "equity (trades booked at exit)", fontsize=10)
    ax[0].legend(fontsize=7)
    ax[1].set_ylabel("drawdown %")
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return Path(path)


def _period_lines(name: str, summ: dict, trades: pd.DataFrame) -> list[str]:
    pt = summ["per_trade"]
    lines = [f"## {name}", "", f"Trades {summ['trades']}, final equity {summ['final_equity']:.4f}, max drawdown "
             f"{100 * summ['max_drawdown']:.2f} %, days {summ['days']}, positive days {summ['positive_days']:.1%}, "
             f"mean cost {100 * summ['mean_cost']:.3f} %", "",
             "Per trade: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in pt.items()),
             "", "By exit:", "", md_table(summ["by_exit"]), "", "By side:", "", md_table(summ["by_side"]), ""]
    book = TradeBook(trades)
    if len(book):
        lines += ["By month:", "", md_table(book.by("month")), "", "Worst coins (sum of net):", "",
                  md_table(book.by("coin").nsmallest(10, "sum_net")), "", "Best coins:", "",
                  md_table(book.by("coin").nlargest(10, "sum_net")), ""]
    return lines


def _batch_charts(bt: Backtest, out: Path, prices, gallery: int, batch_coins: int) -> list[str]:
    """Per batch of ``batch_coins`` coins: the equity of each period and the worst / best trades, in batch_NNN/."""
    names = sorted(bt.trades["coin"].unique())
    index = []
    for b in range(0, len(names), max(1, batch_coins)):
        part = names[b:b + batch_coins]
        folder = out / f"batch_{b // max(1, batch_coins):03d}"
        folder.mkdir(parents=True, exist_ok=True)
        trades = bt.trades[bt.trades["coin"].isin(part)]
        curves = {p: equity(trades[trades["period"] == p], bt.strategy.book.risk_per_trade)
                  for p in pd.unique(trades["period"])}
        plot_curves(curves, folder / "equity.png", f"{bt.strategy.name}: coins {part[0]} .. {part[-1]} "
                    f"({len(part)}), equity per period")
        if prices is not None and gallery:
            book = TradeBook(trades)
            book.gallery(prices, folder / "worst", n=gallery, order="worst")
            book.gallery(prices, folder / "best", n=gallery, order="best")
        index.append(f"| [{folder.name}]({folder.name}/) | {part[0]} .. {part[-1]} | {len(part)} | {len(trades)} |")
    return ["## Charts per batch", "", "| batch | coins | n coins | trades |", "|---|---|---|---|", *index, ""]


def write_report(bt: Backtest, out_dir, prices=None, gallery: int = 12, plots: bool = True,
                 merge_val_test: bool = False, batch_coins: int = 50) -> dict:
    """Everything about one backtest into ``out_dir``: the summary of EVERY chosen coin per period (train alone; val
    and test also merged when ``merge_val_test``), the trades, and the charts drawn per batch of ``batch_coins``
    coins. Returns {period: summary}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = bt.strategy
    periods = by_period(bt.trades, s.book.risk_per_trade, merge_val_test)
    bt.trades.to_parquet(out / "trades.parquet", index=False)
    sk = {k: v for k, v in bt.skipped.items() if k != "coins_without_prices"}
    lines = [f"# Backtest `{s.name}`", "", "Strategy:", "", "```json",
             json.dumps(strategy_to_dict(s), indent=1, default=str), "```", "",
             "Rule trace (rows kept after each decision): " + " -> ".join(f"{k} {v}" for k, v in bt.rule_trace),
             "", f"Skipped: {sk}", ""]
    if bt.skipped.get("no_prices"):
        lines += [f"**WARNING: {bt.skipped['no_prices']} selected signals ({bt.skipped['missing_share']:.1%}) have no "
                  f"candles (coins: {', '.join(bt.skipped['coins_without_prices'][:20])}). Delisted coins bias the "
                  f"result upward; set max_missing_share to stop such a run.**", ""]
    lines += ["Periods are reported apart: train is in-sample and never merged with val / test.", ""]
    for name, summ in periods.items():
        mask = bt.trades["period"].isin(["val", "test"]) if name == "val+test" else bt.trades["period"] == name
        lines += _period_lines(name, summ, bt.trades[mask])
    if plots and len(bt.trades):
        lines += _batch_charts(bt, out, prices, gallery, batch_coins)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    js = {p: {k: v for k, v in summ.items() if not isinstance(v, pd.DataFrame)} for p, summ in periods.items()}
    (out / "summary.json").write_text(json.dumps({"strategy": strategy_to_dict(s), "rule_trace": bt.rule_trace,
                                                  "skipped": bt.skipped, "periods": js}, indent=1, default=str),
                                      encoding="utf-8")
    return periods
