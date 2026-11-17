"""PURPOSE: Command line of the backtest: print the contract, run one strategy, compare / ladder strategies under the
    VAL -> TEST protocol, build and inspect a compressed price store, check candles against the training data.
TAGS: backtest cli, run strategy, compare strategies, rule ladder, protocol, price store, alignment, command line
PITFALLS: ``--prices`` is ``files:DIR[:PATTERN]`` (TF raw layout ``{coin}.csv.gz``), ``archive:INTERVAL[:CACHE]``
    (network on the first read of a coin) or ``store:FILE`` (one compressed file, read once); strategies are JSON of
    the strict Strategy form; a ladder file is a JSON list of [label, {dotted key: value}]; ``--no-protocol`` shows the
    TEST numbers of every variant (exploration only).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from nig_backtest.align import check_alignment
from nig_backtest.contract import SignalSet, describe
from nig_backtest.engine import run_backtest
from nig_backtest.impact import Protocol, compare, ladder
from nig_backtest.periods import Periods
from nig_backtest.portfolio import equity
from nig_backtest.prices import ArchivePrices, FilePrices
from nig_backtest.report import md_table, plot_curves, write_report
from nig_backtest.rules import strategy_from_dict
from nig_backtest.store import StorePrices, build_store, store_info

__all__ = ["main", "price_source", "PRICE_SPECS"]


def _files(rest: str):
    root, _, pattern = rest.partition(":")
    return FilePrices(root, pattern or "{coin}.csv.gz")


def _archive(rest: str):
    interval, _, cache = rest.partition(":")
    return ArchivePrices(interval or "4h", cache_dir=cache or None)


def _store(rest: str):
    return StorePrices(rest)


PRICE_SPECS = {"files": _files, "archive": _archive, "store": _store}


def price_source(spec: str):
    kind, _, rest = spec.partition(":")
    if kind not in PRICE_SPECS:
        raise ValueError(f"--prices must be one of {sorted(PRICE_SPECS)}:..., not {spec!r}")
    return PRICE_SPECS[kind](rest)


def _split(v):
    return tuple(x for x in v.split(",") if x) if v else None


def _signals(a) -> SignalSet:
    periods = Periods(a.train_end, a.val_end) if a.train_end or a.val_end else None
    return SignalSet(path=a.signals, use=_split(a.use) if a.use is not None else None, stages=_split(a.stages),
                     kind=a.kind, horizon=a.horizon, periods=periods)


def _strategy(path):
    return strategy_from_dict(json.loads(Path(path).read_text()))


def _common(p):
    p.add_argument("--signals", required=True, help="parquet file in the contract form")
    p.add_argument("--prices", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--use", help="optional features to keep, comma separated ('' = none); default all present")
    p.add_argument("--stages", help="periods to read (train, val, test), comma separated; default all")
    p.add_argument("--kind", default="score", help="what score means (nig-backtest contract lists the kinds)")
    p.add_argument("--horizon", help="the model's horizon, e.g. 8h (for exit.hold 'horizon')")
    p.add_argument("--train-end", help="split date train | val (when the signals have no stage column)")
    p.add_argument("--val-end", help="split date val | test")
    p.add_argument("--merge-val-test", action="store_true", help="also report val + test as one period")


def _parser():
    ap = argparse.ArgumentParser(prog="nig-backtest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("contract")
    r = sub.add_parser("run")
    _common(r)
    r.add_argument("--strategy", required=True)
    r.add_argument("--gallery", type=int, default=12)
    r.add_argument("--batch-coins", type=int, default=50, help="coins per chart batch")
    for name in ("compare", "ladder"):
        c = sub.add_parser(name)
        _common(c)
        c.add_argument("--no-protocol", action="store_true", help="show TEST for every variant (exploration only)")
        c.add_argument("--metric", default="final_equity", help="the VAL number that chooses the variant")
        if name == "compare":
            c.add_argument("--strategies", nargs="+", required=True)
        else:
            c.add_argument("--strategy", required=True)
            c.add_argument("--steps", required=True)
    st = sub.add_parser("store", help="build or inspect a compressed price store")
    st.add_argument("action", choices=("build", "info"))
    st.add_argument("file")
    st.add_argument("--prices", help="the source to build from")
    st.add_argument("--coins", help="comma separated, or @file with one coin per line")
    st.add_argument("--start")
    st.add_argument("--end")
    al = sub.add_parser("align", help="candles against the last candle and target kept in the training data")
    al.add_argument("--reference", required=True, help="parquet: coin, time, last_high / low / close, future_close / "
                    "high_max / low_min (bridge: alignment_reference)")
    al.add_argument("--prices", required=True)
    al.add_argument("--horizon", required=True)
    al.add_argument("--interval", required=True, help="the reference bar length, e.g. 4h")
    fb = sub.add_parser("from-battery")
    fb.add_argument("folder")
    fb.add_argument("out")
    return ap


def _comparison(a, sig, px, out: Path) -> None:
    protocol = Protocol(enabled=not a.no_protocol, metric=a.metric, merge_val_test=a.merge_val_test)
    if a.cmd == "compare":
        table, runs = compare(sig, px, {Path(p).stem: _strategy(p) for p in a.strategies}, protocol)
    else:
        steps = [tuple(x) for x in json.loads(Path(a.steps).read_text())]
        table, runs = ladder(sig, px, _strategy(a.strategy), steps, protocol)
    out.mkdir(parents=True, exist_ok=True)
    note = f"\n\nprotocol {table.attrs.get('protocol')}; variants tried: {len(runs)}\n"
    (out / "comparison.md").write_text(md_table(table) + note, encoding="utf-8")
    shown = table.attrs.get("chosen")
    for period in pd.unique(table["period"]):
        if a.no_protocol or period in ("train", "val") or shown is not None:
            curves = {n: equity(bt.trades[bt.trades["period"] == period], bt.strategy.book.risk_per_trade)
                      for n, bt in runs.items() if a.no_protocol or period in ("train", "val") or n == shown}
            plot_curves(curves, out / f"comparison_{period.replace('+', '_')}.png", f"equity per variant, {period}")
    print(md_table(table) + note)


def _coins(spec: str | None):
    if spec and spec.startswith("@"):
        return [x.strip() for x in Path(spec[1:]).read_text().splitlines() if x.strip()]
    return list(_split(spec) or ())


def main(argv=None) -> None:
    a = _parser().parse_args(argv)
    if a.cmd == "contract":
        print(describe())
    elif a.cmd == "from-battery":
        from nig_backtest.bridges.nig_timenet_torch import signals_from_folder
        signals = signals_from_folder(a.folder)
        signals.to_parquet(a.out, index=False)
        print(f"wrote {len(signals)} signals to {a.out} (kind 'return'; pass --kind return)")
    elif a.cmd == "store":
        if a.action == "info":
            print(json.dumps(store_info(a.file), indent=1))
        else:
            print(json.dumps(build_store(price_source(a.prices), _coins(a.coins), a.file, start=a.start, end=a.end)))
    elif a.cmd == "align":
        res = check_alignment(price_source(a.prices), pd.read_parquet(a.reference), a.horizon, a.interval)
        print(json.dumps({k: v for k, v in res.items() if k != "worst"}, indent=1, default=str))
        print(md_table(res["worst"]))
    elif a.cmd == "run":
        sig, px, out = _signals(a), price_source(a.prices), Path(a.out)
        periods = write_report(run_backtest(sig, px, _strategy(a.strategy)), out, prices=px, gallery=a.gallery,
                               merge_val_test=a.merge_val_test, batch_coins=a.batch_coins)
        print("; ".join(f"{p}: {s['trades']} trades, final equity {s['final_equity']:.4f}" for p, s in periods.items()))
    else:
        _comparison(a, _signals(a), price_source(a.prices), Path(a.out))


if __name__ == "__main__":
    main()
