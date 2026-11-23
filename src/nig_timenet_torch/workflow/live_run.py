"""PURPOSE: Live prediction end to end: fetch the latest closed candles from Binance, build the model windows with the
    training data code, load the model from a folder, predict the last N bars (the older ones judged, the newest live).
TAGS: live run, live trading, binance fetch, prepare live data, predict latest bars, historical test and live, cli
PITFALLS: needs ``live_model.json`` beside ``best.pt`` (``workflow.live_model.export_live_model``) and the data settings
    the model trained with; reads the TRAINING universe when the model has cross-asset columns (breadth, rank, orth),
    so a few coins alone would be wrong inputs; the open bar is never used (``data.fetch.recent_klines``); a coin
    with a stale newest bar or a gap is listed in ``skipped`` and gets NO row; rows without a finished horizon have no
    truth (status ``live``); the history fetched (``bars``) must cover the window plus the indicators' warm-up
    (1000 bars is ample for the 1h / 4h feature sets measured; check ``eval/`` parity before trusting a new feature
    set); network code, not exercised by CI (tests inject a fake Http); never write the API response or a key to a
    file; the signals are exploratory (rule C5) and not an order: nothing here places a trade.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX, TS_COL
from nig_timenet_torch.data.build import build_live
from nig_timenet_torch.data.fetch import FUTURES_KLINES, Http, recent_klines
from nig_timenet_torch.data.settings import DataSettings
from nig_timenet_torch.data.sources import frames_loader, normalize_tf, resample_ohlcv
from nig_timenet_torch.data_io.asset_index import asset_ids
from nig_timenet_torch.eval.battery_lib import DEFAULT_COST_PER_SIDE
from nig_timenet_torch.trainer.groups import row_groups
from nig_timenet_torch.workflow import flat_run
from nig_timenet_torch.workflow.battery_run import StageInputs, stage_atr
from nig_timenet_torch.workflow.latest_report import latest_report
from nig_timenet_torch.workflow.live_model import LiveModel, load_live_model
from nig_timenet_torch.workflow.prediction_table import predict_table, table_meta

__all__ = ["live_table", "drop_stale", "fetch_universe", "live_run", "main"]

MAX_PRINT_COINS = 12          # more coins than this: print only the selection (the whole table goes to the CSV)


def live_table(lm: LiveModel, dataset: dict, device: str = "cpu") -> pd.DataFrame:
    """The battery table of a live dataset (``data.build.build_live``) under ``lm``: one row per coin and bar, time
    major (a cross-section model gets whole timestamps), ``y_ret`` NaN where the horizon is not over."""
    spec = lm.spec
    lc = np.asarray(dataset["last_candles"])
    n = len(lc)
    order = np.lexsort((asset_ids(dataset, np.arange(n)), lc[:, TS_COL]))            # by time, then coin
    x, y = flat_run.flat_inputs(dataset, order, spec["cross_section"], spec["coin_channel"],
                                tuple(spec["input_timeframes"]))
    meta = table_meta(dataset, order, stage="live", fold=0, train_end=spec["train_end"])
    with torch.no_grad():
        table = predict_table(lm.model, x.to(device), meta, lm.identity,
                              groups=row_groups(x) if spec["cross_section"] else None)
    anchors = flat_run.anchors_by_name(lc, order, lm.targets, lm.codec_state)
    inputs = StageInputs(table, {k: v.numpy() for k, v in y.items()}, anchors,
                         lc[order, LAST_COLUMN_INDEX["last_close"]], lm.codec_state,
                         atr=stage_atr(dataset, order, tuple(spec["input_timeframes"])))
    return inputs.battery_table()


def drop_stale(table: pd.DataFrame, clock: pd.Timestamp, bar: str, max_bars: float = 2.0):
    """(table without the coins whose newest row is older than ``max_bars`` base bars before ``clock``, {coin: why}): a
    halted, delisted or lagging coin must not look like a live prediction."""
    now = clock.tz_convert("UTC").tz_localize(None) if clock.tzinfo else clock
    newest = table.groupby("coin")["end_time"].max()
    limit = pd.Timedelta(normalize_tf(bar)) * max_bars
    late = {c: f"stale: its newest bar closed {now - t} before the clock (limit {limit})"
            for c, t in newest.items() if now - t > limit}
    return table[~table["coin"].isin(late)].reset_index(drop=True), late


def fetch_universe(coins, interval: str, bars: int, http: Http | None = None, now=None, workers: int = 8,
                   url: str = FUTURES_KLINES):
    """({coin: closed-candle frame}, {coin: error text}) for ``coins`` from the live API; one failed coin does not stop
    the others (it is listed with its error)."""
    http = http or Http()

    def one(coin):
        try:
            return coin, recent_klines(coin, interval, bars, http, now, url=url), ""
        except Exception as e:                       # a delisted or renamed coin, a rate limit: report, carry on
            return coin, None, f"{type(e).__name__}: {str(e)[:200]}"
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(coins)))) as pool:
        got = list(pool.map(one, coins))
    return {c: f for c, f, err in got if f is not None}, {c: err for c, f, err in got if f is None}


def _settings(lm: LiveModel, data_settings) -> DataSettings:
    if data_settings is None:
        raw = lm.spec.get("data_settings")
        if not raw:
            raise ValueError(f"{lm.folder}: the model folder holds no data settings (a TF-exported artifact): pass "
                             "data_settings= (the DataSettings the artifact was built with)")
        return DataSettings.from_dict(raw)
    return data_settings if isinstance(data_settings, DataSettings) else DataSettings.from_dict(dict(data_settings))


def _universe(lm: LiveModel, s: DataSettings, coins) -> list[str]:
    """The coins to read: the training universe when a cross-asset column needs it, else the asked coins and the
    market reference coin."""
    f = s.features
    cross = f.breadth.enabled or f.momentum_rank.enabled or f.orth_natr.enabled
    names = list(lm.spec["coins"]) if cross else list(coins or lm.spec["coins"])
    names += [c for c in (coins or []) if c not in names]
    ref = f.market.reference_symbol
    return names + ([ref] if f.market.enabled and ref not in names else [])


def live_run(model_dir, coins=None, *, data_settings=None, last: int = 6, bars: int = 1000, http: Http | None = None,
             now=None, device: str = "cpu", out_dir=None, k: float = 1.0, rank_by: str = "combined",
             top: int | None = None, cost: float = DEFAULT_COST_PER_SIDE, charts: bool = True, echo=print,
             klines_url: str = FUTURES_KLINES) -> dict:
    """Fetch -> prepare -> load -> predict -> report. ``coins``: the coins to show (default: the training universe);
    ``last``: how many of the newest bars per coin to predict (the ones whose horizon is over are judged, the rest are
    live); ``bars``: closed raw candles read per coin; ``now``: the clock (tests). Returns {"table": the battery table
    of every coin read, "report": ``latest_report``'s dict, "skipped": {coin: why}, "fetched": n coins, "now": time}."""
    lm = load_live_model(model_dir, device)
    s = _settings(lm, data_settings)
    clock = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    names = _universe(lm, s, coins)
    frames, errors = fetch_universe(names, s.source.interval, bars, http, clock, url=klines_url)
    want = [c for c in (coins or lm.spec["coins"])]
    if not [c for c in want if c in frames]:
        raise RuntimeError(f"no candles fetched for any asked coin; errors: {errors}")
    ds = build_live(s, frames_loader(frames, s.source), [c for c in names if c in frames], last_n=last)
    tf = s.windows.tf_order[0]
    table, stale = drop_stale(live_table(lm, ds, device), clock, tf)
    skipped = {**errors, **ds["skipped_assets"], **stale, **{c: "sealed coin" for c in ds["sealed_dropped"]}}
    shown = [c for c in want if c in set(table["coin"])]
    if not shown:
        raise RuntimeError(f"no coin has a fresh live row; skipped: {skipped}")
    candles = {c: resample_ohlcv(frames[c], tf, s.source.interval) for c in shown}
    rep = latest_report(table, out_dir, n=last, coins=shown, k=k, rank_by=rank_by, top=top, cost=cost, now=clock,
                        candles=candles, charts=charts, title="LIVE", echo=echo,
                        show_latest=len(shown) <= MAX_PRINT_COINS)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "live_run.json").write_text(json.dumps({
            "model_dir": str(model_dir), "identity": lm.identity.fingerprint(), "clock": str(clock), "bars": bars,
            "last": last, "coins_read": len(frames), "coins_shown": shown, "skipped": skipped,
            "bar": normalize_tf(tf)}, indent=1), encoding="utf-8")
    if skipped:
        echo(f"skipped {len(skipped)} coin(s): " + "; ".join(f"{c}: {w}" for c, w in list(skipped.items())[:5]))
    return {"table": table, "report": rep, "skipped": skipped, "fetched": len(frames), "now": clock}


def main(argv=None) -> None:
    """``python -m nig_timenet_torch.workflow.live_run --model DIR [--coins A,B] [--last 6] [--bars 1000] [--k 1.0]
    [--rank-by combined] [--top N] [--data-settings JSON_FILE] [--out DIR] [--no-charts] [--device cpu]
    [--klines-url URL]``"""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--coins", default="")
    ap.add_argument("--last", type=int, default=6)
    ap.add_argument("--bars", type=int, default=1000)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--rank-by", default="combined")
    ap.add_argument("--top", type=int)
    ap.add_argument("--cost", type=float, default=DEFAULT_COST_PER_SIDE)
    ap.add_argument("--data-settings", help="a JSON file with the DataSettings (only when the model folder has none)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--klines-url", default=FUTURES_KLINES,
                    help="klines endpoint; the futures API answers HTTP 451 from some countries")
    a = ap.parse_args(argv)
    ds = json.loads(Path(a.data_settings).read_text()) if a.data_settings else None
    live_run(a.model, [c for c in a.coins.split(",") if c] or None, data_settings=ds, last=a.last, bars=a.bars,
             k=a.k, rank_by=a.rank_by, top=a.top, cost=a.cost, device=a.device, out_dir=a.out, charts=not a.no_charts,
             klines_url=a.klines_url)


if __name__ == "__main__":
    main()
