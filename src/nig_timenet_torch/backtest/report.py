"""PURPOSE: Write a backtest to a folder: the trades (parquet), the summary (markdown + JSON), the rule trace, the
    equity / drawdown chart, the tables by coin / month / exit, and galleries of the worst and best trades.
TAGS: backtest report, trades parquet, summary markdown, equity chart, rule trace, gallery, report writer
PITFALLS: charts need matplotlib (extra ``plots``) and, for the galleries, the PriceSource of the run; ``gallery``
    0 writes no trade charts. Exploratory (rule C5): one strategy on one signal set is a look.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nig_timenet_torch.backtest.engine import Backtest
from nig_timenet_torch.backtest.explorer import TradeBook
from nig_timenet_torch.backtest.portfolio import summary
from nig_timenet_torch.backtest.rules import strategy_to_dict

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


def write_report(bt: Backtest, out_dir, prices=None, gallery: int = 12, plots: bool = True) -> dict:
    """Everything about one backtest into ``out_dir``; returns the summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = bt.strategy
    summ = summary(bt.trades, s.book.risk_per_trade, s.book.max_open)
    bt.trades.to_parquet(out / "trades.parquet", index=False)
    book = TradeBook(bt.trades)
    lines = [f"# Backtest `{s.name}` (exploratory, rule C5)", "", "Strategy:", "", "```json",
             json.dumps(strategy_to_dict(s), indent=1, default=str), "```", "",
             "Rule trace (rows kept after each decision): " + " -> ".join(f"{k} {v}" for k, v in bt.rule_trace),
             f"Skipped: {bt.skipped}", "",
             f"Trades {summ['trades']} (skipped by max_open {summ['skipped_max_open']}), final equity "
             f"{summ['final_equity']:.4f}, max drawdown {100 * summ['max_drawdown']:.2f} %, days {summ['days']}, "
             f"positive days {summ['positive_days']:.1%}", "",
             "Per trade: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                                       for k, v in summ["per_trade"].items()), ""]
    for key in ("by_exit", "by_side", "by_stage"):
        if key in summ:
            lines += [f"## {key.replace('_', ' ')}", "", md_table(summ[key]), ""]
    if len(book):
        lines += ["## by month", "", md_table(book.by("month")), "", "## worst coins (sum of net)", "",
                  md_table(book.by("coin").nsmallest(10, "sum_net")), "", "## best coins", "",
                  md_table(book.by("coin").nlargest(10, "sum_net")), ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    js = {k: v for k, v in summ.items() if not isinstance(v, pd.DataFrame)}
    (out / "summary.json").write_text(json.dumps({"strategy": strategy_to_dict(s), "rule_trace": bt.rule_trace,
                                                  "skipped": bt.skipped, **js}, indent=1, default=str),
                                      encoding="utf-8")
    if plots and len(bt.trades):
        plot_curves({s.name: summ["curve"]}, out / "equity.png", f"{s.name}: equity and drawdown")
        if prices is not None and gallery:
            book.gallery(prices, out / "worst", n=gallery, order="worst")
            book.gallery(prices, out / "best", n=gallery, order="best")
    return summ
