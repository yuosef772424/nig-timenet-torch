"""PURPOSE: Command line of the backtest: print the input contract, run one strategy, compare strategies, ladder the
    rules of a strategy, and turn a run's battery tables into a signal table.
TAGS: backtest cli, run strategy, compare strategies, rule ladder, contract, battery to signals, command line
PITFALLS: ``--prices`` is ``files:DIR`` (TF raw layout ``{coin}.csv.gz``), ``files:DIR:PATTERN`` or
    ``archive:INTERVAL[:CACHE_DIR]`` (needs the network on the first read of a coin); strategies are JSON files of the
    strict Strategy form (unknown keys raise); a ladder file is a JSON list of [label, {dotted key: value}].
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from nig_backtest.adapters import from_battery_table
from nig_backtest.contract import SignalSet, describe
from nig_backtest.engine import run_backtest
from nig_backtest.impact import compare, ladder
from nig_backtest.portfolio import cap_open, equity
from nig_backtest.prices import ArchivePrices, FilePrices
from nig_backtest.report import md_table, plot_curves, write_report
from nig_backtest.rules import strategy_from_dict

__all__ = ["main", "price_source"]


def _files(rest: str):
    root, _, pattern = rest.partition(":")
    return FilePrices(root, pattern or "{coin}.csv.gz")


def _archive(rest: str):
    interval, _, cache = rest.partition(":")
    return ArchivePrices(interval or "4h", cache_dir=cache or None)


PRICE_SPECS = {"files": _files, "archive": _archive}


def price_source(spec: str):
    kind, _, rest = spec.partition(":")
    if kind not in PRICE_SPECS:
        raise ValueError(f"--prices must be files:DIR[:PATTERN] or archive:INTERVAL[:CACHE], not {spec!r}")
    return PRICE_SPECS[kind](rest)


def _signals(a) -> SignalSet:
    def split(v):
        return tuple(x for x in v.split(",") if x) if v else None

    return SignalSet(path=a.signals, use=split(a.use) if a.use is not None else None, stages=split(a.stages))


def _strategy(path):
    return strategy_from_dict(json.loads(Path(path).read_text()))


def _common(p):
    p.add_argument("--signals", required=True, help="parquet file in the contract form")
    p.add_argument("--prices", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--use", help="optional features to keep, comma separated ('' = none); default all present")
    p.add_argument("--stages", help="stages to read, comma separated; default all")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m nig_backtest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("contract")
    r = sub.add_parser("run")
    _common(r)
    r.add_argument("--strategy", required=True)
    r.add_argument("--gallery", type=int, default=12)
    c = sub.add_parser("compare")
    _common(c)
    c.add_argument("--strategies", nargs="+", required=True)
    lad = sub.add_parser("ladder")
    _common(lad)
    lad.add_argument("--strategy", required=True)
    lad.add_argument("--steps", required=True)
    fb = sub.add_parser("from-battery")
    fb.add_argument("folder")
    fb.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "contract":
        print(describe())
        return
    if a.cmd == "from-battery":
        parts = [from_battery_table(pd.read_parquet(Path(a.folder) / f"battery_table_{st}.parquet"), stage=st)
                 for st in ("val", "test") if (Path(a.folder) / f"battery_table_{st}.parquet").is_file()]
        pd.concat(parts, ignore_index=True).to_parquet(a.out, index=False)
        print(f"wrote {sum(map(len, parts))} signals to {a.out}")
        return
    sig, px, out = _signals(a), price_source(a.prices), Path(a.out)
    if a.cmd == "run":
        summ = write_report(run_backtest(sig, px, _strategy(a.strategy)), out, prices=px, gallery=a.gallery)
        print(f"{summ['trades']} trades, final equity {summ['final_equity']:.4f}; report in {out}")
        return
    if a.cmd == "compare":
        strategies = {Path(p).stem: _strategy(p) for p in a.strategies}
        table, runs = compare(sig, px, strategies)
    else:
        table, runs = ladder(sig, px, _strategy(a.strategy), [tuple(x) for x in json.loads(Path(a.steps).read_text())])
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text(md_table(table) + f"\n\nvariants tried: {len(runs)}\n", encoding="utf-8")
    plot_curves({n: equity(cap_open(bt.trades, bt.strategy.book.max_open), bt.strategy.book.risk_per_trade)
                 for n, bt in runs.items()},
                out / "comparison.png", "equity per variant")
    print(md_table(table))


if __name__ == "__main__":
    main()
