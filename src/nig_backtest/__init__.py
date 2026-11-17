"""PURPOSE: A separate historical backtest for any model and any strategy: one input (the signal contract), raw candles
    on demand, entry / exit rules as data, a trade explorer with charts and the effect of every rule on the book.
TAGS: backtest, strategy test, signal contract, entry exit rules, take profit atr, trade explorer, rule impact,
    portfolio, separate system
PITFALLS: a SEPARATE package (numpy / pandas, matplotlib for charts): no module imports a host project, except the
    bridges in ``nig_backtest/bridges/`` (lazily, inside their functions), so the folder moves to any project as it is
    and a model reaches it only through the signal table (``contract.describe()``). Exploratory: a strategy tuned on
    one stage is a hypothesis for the next one.

Use:

    import nig_backtest as bt
    sig = bt.SignalSet(frame=signals)                    # or SignalSet(path="signals.parquet", use=("sigma",))
    px = bt.ArchivePrices("4h", cache_dir="candles")     # or FilePrices(dir) / FramePrices({coin: frame})
    s = bt.strategy_from_dict({"entry": {"top_share": 0.1}, "exit": {"max_bars": 3, "take_profit": 1.5}})
    run = bt.run_backtest(sig, px, s)
    bt.TradeBook(run.trades).where(outcome="loss", coins=["BTCUSDT"]).plot(0, px, "charts")
    bt.write_report(run, "out", prices=px)
    table, _ = bt.ladder(sig, px, s, [("tp 1.5 atr", {"exit.take_profit": 1.5}), ("stop 1 atr", {"exit.stop_loss": 1})])

Command line: ``nig-backtest {contract,run,compare,ladder,from-battery} ...``. Host projects connect through
``nig_backtest.bridges.<project>`` (here ``bridges.nig_timenet_torch``: run outputs, the model on chosen coins,
the artifact's candles).
"""

from nig_backtest.align import check_alignment
from nig_backtest.contract import OPTIONAL, REQUIRED, SignalSet, check_signals, describe
from nig_backtest.engine import Backtest, admit, run_backtest, simulate_coin
from nig_backtest.explorer import TradeBook
from nig_backtest.impact import Protocol, apply_changes, compare, ladder, leave_one_out
from nig_backtest.kinds import OUTPUT_KINDS, decode, is_absolute
from nig_backtest.periods import Periods
from nig_backtest.portfolio import by_group, by_period, daily, equity, summary
from nig_backtest.prices import ArchivePrices, FilePrices, FramePrices, PriceSource, atr
from nig_backtest.report import write_report
from nig_backtest.rules import (
    ADMISSION,
    COST_MODES,
    ENTRY_FILTERS,
    SIDE_RULES,
    Book,
    Costs,
    Entry,
    Exit,
    Strategy,
    strategy_from_dict,
    strategy_to_dict,
)
from nig_backtest.store import StorePrices, build_store, store_info

__all__ = ["check_alignment", "OPTIONAL", "REQUIRED", "SignalSet", "check_signals", "describe", "Backtest", "admit",
           "run_backtest", "simulate_coin", "TradeBook", "Protocol", "apply_changes", "compare", "ladder",
           "leave_one_out", "OUTPUT_KINDS", "decode", "is_absolute", "Periods", "by_group", "by_period", "daily",
           "equity", "summary", "ArchivePrices", "FilePrices", "FramePrices", "PriceSource", "atr", "write_report",
           "ADMISSION", "COST_MODES", "ENTRY_FILTERS", "SIDE_RULES", "Book", "Costs", "Entry", "Exit", "Strategy",
           "strategy_from_dict", "strategy_to_dict", "StorePrices", "build_store", "store_info"]
