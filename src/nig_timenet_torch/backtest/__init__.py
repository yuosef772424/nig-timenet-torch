"""PURPOSE: A separate historical backtest for any model and any strategy: one input (the signal contract), raw candles
    on demand, entry / exit rules as data, a trade explorer with charts and the effect of every rule on the book.
TAGS: backtest, strategy test, signal contract, entry exit rules, take profit atr, trade explorer, rule impact,
    portfolio, separate system
PITFALLS: imports only core, stats, data (candle download) and eval (tearsheet, candle drawing): it never imports
    the model, the trainer or the run steps (rule G7), so any model reaches it only through the signal table
    (``contract.describe()``). Exploratory (rule C5): a strategy tuned on a stage is a hypothesis for the next one.

Use:

    from nig_timenet_torch import backtest as bt
    sig = bt.SignalSet(frame=signals)                    # or SignalSet(path="signals.parquet", use=("sigma",))
    px = bt.ArchivePrices("4h", cache_dir="candles")     # or FilePrices(dir) / FramePrices({coin: frame})
    s = bt.strategy_from_dict({"entry": {"top_share": 0.1}, "exit": {"max_bars": 3, "take_profit": 1.5}})
    run = bt.run_backtest(sig, px, s)
    bt.TradeBook(run.trades).where(outcome="loss", coins=["BTCUSDT"]).plot(0, px, "charts")
    bt.write_report(run, "out", prices=px)
    table, _ = bt.ladder(sig, px, s, [("tp 1.5 atr", {"exit.take_profit": 1.5}), ("stop 1 atr", {"exit.stop_loss": 1})])

Command line: ``python -m nig_timenet_torch.backtest {contract,run,compare,ladder,from-battery} ...``.
"""

from nig_timenet_torch.backtest.adapters import from_battery_table
from nig_timenet_torch.backtest.contract import OPTIONAL, REQUIRED, SignalSet, check_signals, describe
from nig_timenet_torch.backtest.engine import Backtest, run_backtest, simulate_coin
from nig_timenet_torch.backtest.explorer import TradeBook
from nig_timenet_torch.backtest.impact import apply_changes, compare, ladder, leave_one_out
from nig_timenet_torch.backtest.portfolio import by_group, cap_open, daily, equity, summary
from nig_timenet_torch.backtest.prices import ArchivePrices, FilePrices, FramePrices, PriceSource, atr
from nig_timenet_torch.backtest.report import write_report
from nig_timenet_torch.backtest.rules import (
    ENTRY_FILTERS,
    Book,
    Costs,
    Entry,
    Exit,
    Strategy,
    strategy_from_dict,
    strategy_to_dict,
)

__all__ = ["from_battery_table", "OPTIONAL", "REQUIRED", "SignalSet", "check_signals", "describe", "Backtest",
           "run_backtest", "simulate_coin", "TradeBook", "apply_changes", "compare", "ladder", "leave_one_out",
           "by_group", "cap_open", "daily", "equity", "summary", "ArchivePrices", "FilePrices", "FramePrices",
           "PriceSource", "atr", "write_report", "ENTRY_FILTERS", "Book", "Costs", "Entry", "Exit", "Strategy",
           "strategy_from_dict", "strategy_to_dict"]
