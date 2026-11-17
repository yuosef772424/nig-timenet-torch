"""PURPOSE: The bridge to nig_timenet_torch: a run's outputs (battery tables on Drive, or the model run now on chosen
    coins) become a signal table with each output's time, kind and horizon; the artifact's own candles become prices.
TAGS: bridge, nig timenet torch, battery table, run model on one coin, prediction time, artifact candles, drive,
    output kind, horizon
PITFALLS: imports nig_timenet_torch only inside functions. The close head's mu is a decoded simple RETURN over the
    artifact's horizon (kind 'return', absolute); its confidence head is the optional ``confidence``. ``time`` is the
    prediction time (the close of the last input bar). Running on some coins narrows the stage rows BEFORE the model
    runs; a cross-section (panel) model needs whole timestamps, so it refuses ``coins``. Artifact candles are the
    model's own frame (high / low / close of the last bar of each window; open = the previous close), one candle per
    window, so an artifact with ``stride`` > 1 gives candles ``stride`` bars apart; full or finer candles come from the
    archive (``nig_backtest.prices.ArchivePrices``).
"""

import numpy as np
import pandas as pd

from nig_backtest.contract import SignalSet, check_signals
from nig_backtest.periods import Periods
from nig_backtest.prices import FramePrices

__all__ = ["KIND", "from_battery_table", "signals_from_folder", "signals_from_run", "run_on_coins", "horizon_of",
           "artifact_prices", "signal_set", "periods_of", "alignment_reference"]

KIND = "return"


def horizon_of(dataset: dict) -> str:
    """The prediction horizon of an artifact as a duration string (forecast_horizon bars of base_tf)."""
    return str(int(dataset["forecast_horizon"]) * pd.Timedelta(dataset["base_tf"]))


def from_battery_table(table: pd.DataFrame, stage: str | None = None, confidence=None) -> pd.DataFrame:
    """A contract signal table from a battery table: score = mu (a return), sigma, p_up = p_dir, y_true = y_ret,
    time = end_time (the prediction time), pred_high / pred_low (returns, clipped to their side of 0) when present,
    ``stage`` for every row and an optional ``confidence`` array aligned to the rows."""
    out = pd.DataFrame({"coin": table["coin"].astype(str).to_numpy(), "time": table["end_time"].to_numpy(),
                        "score": table["mu"].to_numpy()})
    for src, dst in (("sigma", "sigma"), ("p_dir", "p_up"), ("y_ret", "y_true")):
        if src in table:
            out[dst] = table[src].to_numpy()
    if {"pred_high", "pred_low"} <= set(table.columns):
        out["pred_high"] = np.maximum(table["pred_high"].to_numpy(), 0.0)
        out["pred_low"] = np.minimum(table["pred_low"].to_numpy(), 0.0)
    if stage is not None:
        out["stage"] = stage
    if confidence is not None:
        out["confidence"] = np.asarray(confidence, dtype=float)
    return check_signals(out)


def signals_from_folder(folder, stages=("val", "test")) -> pd.DataFrame:
    """The signals of a finished run from the battery tables it wrote (``battery_table_<stage>.parquet``, e.g. a run
    folder on Drive)."""
    from pathlib import Path
    parts = [from_battery_table(pd.read_parquet(Path(folder) / f"battery_table_{st}.parquet"), stage=st)
             for st in stages if (Path(folder) / f"battery_table_{st}.parquet").is_file()]
    if not parts:
        raise FileNotFoundError(f"no battery_table_<stage>.parquet in {folder}")
    return pd.concat(parts, ignore_index=True)


def signals_from_run(run, stage: str = "test", coins=None, confidence: bool = True) -> tuple[pd.DataFrame, dict]:
    """(signals, meta) of a run whose model holds its weights (after ``workflow.run.train`` or ``restore``): the model
    runs on the rows of ``stage`` (only those of ``coins`` when given); meta = {kind, horizon}."""
    from nig_timenet_torch.model.heads import head_spec
    from nig_timenet_torch.workflow import battery_run
    from nig_timenet_torch.workflow.flat_run import row_coins

    saved = None
    if coins is not None:
        if run.settings.split.cross_section:
            raise ValueError("a cross-section (panel) run needs whole timestamps: run it on all coins")
        idx = run.idx[stage]
        keep = np.isin(row_coins(run.dataset, idx).astype(str), [str(c) for c in coins])
        if not keep.any():
            raise ValueError(f"no {stage} rows for the coins {list(coins)}")
        saved = (idx, run.x.pop(stage, None), run.y.pop(stage, None), run.extra.get("tables", {}).pop(stage, None))
        run.idx[stage] = idx[keep]
    try:
        inputs = battery_run.stage_inputs(run, stage)
        conf_key = head_spec("nig_regression", "close").output_keys.get("confidence")
        conf = inputs.table[conf_key].to_numpy() if confidence and conf_key in inputs.table else None
        signals = from_battery_table(inputs.battery_table(), stage=stage, confidence=conf)
    finally:
        if saved is not None:
            run.idx[stage] = saved[0]
            for store, value in ((run.x, saved[1]), (run.y, saved[2]), (run.extra.setdefault("tables", {}), saved[3])):
                store.pop(stage, None)
                if value is not None:
                    store[stage] = value
    return signals, {"kind": KIND, "horizon": horizon_of(run.dataset)}


def run_on_coins(settings, coins, stage: str = "test", weights: str = "best", out=None):
    """Prepare and build a run from ``settings``, restore its trained weights from ``settings.run_dir`` and run the
    model on ``coins`` only: (signals, meta, run)."""
    from nig_timenet_torch.workflow import run as steps
    run = steps.restore(steps.build(steps.prepare(settings, out)), weights=weights)
    signals, meta = signals_from_run(run, stage, coins)
    return signals, meta, run


def artifact_prices(dataset: dict, coins=None) -> FramePrices:
    """The model's own candles from an artifact (``last_candles``: high / low / close of each window's last bar at its
    open time; open = the previous close of the same coin), one frame per coin."""
    from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX, TS_COL
    lc = np.asarray(dataset["last_candles"])
    frames = {}
    for b in dataset["asset_bounds"]:
        if coins is not None and b["name"] not in coins:
            continue
        part = lc[b["start"]:b["end"]]
        t = pd.to_datetime(part[:, TS_COL].astype("int64"))         # ns since epoch, UTC (the project's convention)
        col = {k: part[:, LAST_COLUMN_INDEX[f"last_{k}"]] for k in ("high", "low", "close")}
        f = pd.DataFrame(col, index=t)
        f = f[~f.index.duplicated()].sort_index()
        f.insert(0, "open", f["close"].shift(1).fillna(f["close"]))
        frames[b["name"]] = f
    return FramePrices(frames)


def signal_set(signals: pd.DataFrame, meta: dict, **kw) -> SignalSet:
    """A SignalSet of bridge output with its kind and horizon."""
    return SignalSet(frame=signals, kind=meta["kind"], horizon=meta["horizon"], **kw)


def periods_of(run) -> Periods:
    """The run's split dates (train_end, val_end) as backtest periods (UTC, naive)."""
    def naive(t):
        t = pd.Timestamp(t)
        return str(t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t)
    return Periods(naive(run.bounds[0]), naive(run.bounds[1]))


def alignment_reference(dataset: dict, coins=None) -> pd.DataFrame:
    """Per sample the last candle and the target as kept BEFORE encoding (``last_candles``), for
    ``nig_backtest.align.check_alignment``: coin, time (the last bar's open), last high / low / close, future close /
    max high / min low."""
    from nig_timenet_torch.core.schema import LAST_COLUMN_INDEX, TS_COL
    lc = np.asarray(dataset["last_candles"])
    parts = []
    for b in dataset["asset_bounds"]:
        if coins is not None and b["name"] not in coins:
            continue
        part = lc[b["start"]:b["end"]]
        f = pd.DataFrame({k: part[:, LAST_COLUMN_INDEX[k]] for k in ("last_high", "last_low", "last_close",
                                                                     "future_close", "future_high_max",
                                                                     "future_low_min")})
        f.insert(0, "time", pd.to_datetime(part[:, TS_COL].astype("int64")))
        f.insert(0, "coin", b["name"])
        parts.append(f)
    return pd.concat(parts, ignore_index=True)
