"""PURPOSE: The ONE input of the backtest: a signal table (coin, time, score + optional features) checked against a
    declared contract; a SignalSet reads it from a frame or a parquet file, coin batches on demand, features on / off.
TAGS: backtest contract, signal table, input tunnel, optional features, feature toggle, lazy parquet, coin batches,
    stages, validation
PITFALLS: ``time`` is the moment the signal is KNOWN (the close of the last input bar, UTC); the engine enters after
    it, never at it. A column outside the contract raises unless it starts with ``x_`` (free columns, carried to the
    trades). An optional feature that is absent or switched off is dropped before any rule sees it, and a rule that
    needs it raises (no silent default). A parquet source is read per coin batch with a filter, never whole.

The backtest is a separate package: it never imports a host project (only its bridges do). Any model (or a
hand-written rule, or the training labels themselves) reaches it only through this table.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nig_backtest.kinds import OUTPUT_KINDS, decode, is_absolute
from nig_backtest.periods import Periods, assign

__all__ = ["REQUIRED", "OPTIONAL", "FREE_PREFIX", "SignalSet", "check_signals", "describe", "to_utc_naive"]

#: column -> meaning; every signal row needs these
REQUIRED = {
    "coin": "instrument name as the price source knows it (e.g. BTCUSDT)",
    "time": "UTC time the signal is known (the close of the last input bar); entry happens after it",
    "score": "the directional output: > 0 = up, < 0 = down; its size ranks the rows (e.g. the predicted return)",
}
#: optional column -> meaning; each one is a FEATURE that can be absent, or present and switched off
OPTIONAL = {
    "confidence": "higher = surer (any scale); rules: min_confidence, confidence_top_share",
    "sigma": "predicted sd of the move, return units; rules: score_over_sigma, exits in 'sigma' units",
    "p_up": "probability of an up move in [0, 1]; rule: side='p_up'",
    "pred_high": "predicted largest rise over the horizon, return units (>= 0); exits in 'pred' units",
    "pred_low": "predicted largest fall over the horizon, return units (<= 0); exits in 'pred' units",
    "y_true": "the realised label the model was trained on (any units); trade column label_right",
    "stage": "train / val / test (or any split name); rule: stages",
    "ref_price": "the coin's price at time (the anchor of an output of kind 'price')",
}
FREE_PREFIX = "x_"


def to_utc_naive(values) -> pd.Series:
    """Times as naive UTC datetime64[ns] (aware values are converted to UTC first)."""
    t = pd.to_datetime(pd.Series(values), utc=True)
    return t.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")


def check_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame with the contract enforced: required columns present and non-null, no unknown column, coin as str,
    time as naive UTC, score / numeric features as float64. Raises ValueError naming what is wrong."""
    missing = [c for c in REQUIRED if c not in frame.columns]
    unknown = [c for c in frame.columns if c not in REQUIRED and c not in OPTIONAL and not str(c).startswith(
        FREE_PREFIX)]
    if missing or unknown:
        raise ValueError(f"signal table: missing {missing}, unknown {unknown} (contract: {list(REQUIRED)} + optional "
                         f"{list(OPTIONAL)} + free columns '{FREE_PREFIX}*')")
    out = frame.copy()
    out["coin"] = out["coin"].astype(str)
    out["time"] = to_utc_naive(out["time"]).to_numpy()
    for c in ("score", *[k for k in OPTIONAL if k != "stage"]):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="raise").astype("float64")
    if "stage" in out:
        out["stage"] = out["stage"].astype(str)
    bad = out[list(REQUIRED)].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"signal table: {int(bad.sum())} rows with an empty coin / time / score")
    if "p_up" in out and not out["p_up"].dropna().between(0, 1).all():
        raise ValueError("signal table: p_up outside [0, 1]")
    return out


@dataclass
class SignalSet:
    """The signals of one backtest: from a frame (``frame``) or a parquet file (``path``, read per coin batch).
    ``use`` = the optional features to keep (None = every one present); ``stages`` = the periods to read (None = all);
    ``kind`` = what ``score`` means (``kinds.OUTPUT_KINDS``: absolute kinds decode to a predicted move, the others
    give a direction and a ranking only); ``horizon`` = the model's prediction horizon (e.g. "8h"; an exit
    ``hold: "horizon"`` uses it); ``periods`` = the split dates (``periods.Periods``) when the table has no stage."""
    frame: pd.DataFrame | None = None
    path: str | None = None
    use: tuple | None = None
    stages: tuple | None = None
    kind: str = "score"
    horizon: str | None = None
    periods: Periods | None = None

    def __post_init__(self):
        if (self.frame is None) == (self.path is None):
            raise ValueError("SignalSet needs exactly one of frame / path")
        if self.frame is not None:
            self.frame = check_signals(self.frame)
        cols = list(self.frame.columns) if self.frame is not None else _parquet_columns(self.path)
        present = tuple(c for c in OPTIONAL if c in cols)
        if self.use is not None:
            unknown = sorted(set(self.use) - set(OPTIONAL))
            if unknown:
                raise ValueError(f"SignalSet.use: unknown features {unknown} (optional: {list(OPTIONAL)})")
            missing = sorted(set(self.use) - set(present))
            if missing:
                raise ValueError(f"SignalSet.use asks for {missing}, absent from the signal table (present: {present})")
        self.present = present
        self.enabled = present if self.use is None else tuple(c for c in present if c in self.use)
        if self.stages is not None and "stage" not in self.enabled and self.periods is None:
            raise ValueError("SignalSet.stages needs the 'stage' column (present and enabled) or periods (dates)")
        need = [c for c in OUTPUT_KINDS.entry(self.kind).meta["needs"] if c not in self.enabled]
        if need:
            raise ValueError(f"output kind {self.kind!r} needs the features {need} (present and enabled)")
        if self.horizon is not None:
            pd.Timedelta(self.horizon)                                  # raises on a bad duration

    @property
    def absolute(self) -> bool:
        """True when the output decodes to a price without loss (kinds.is_absolute)."""
        return is_absolute(self.kind)

    def coins(self) -> list[str]:
        if self.frame is not None:
            return sorted(self.frame["coin"].unique())
        return sorted(pd.read_parquet(self.path, columns=["coin"])["coin"].astype(str).unique())

    def read(self, coins=None) -> pd.DataFrame:
        """The rows of ``coins`` (None = all) and of the chosen stages, with the disabled features dropped and the
        decoded columns of the output kind (dir, strength, pred_move)."""
        if self.frame is not None:
            out = self.frame if coins is None else self.frame[self.frame["coin"].isin(list(coins))]
        else:
            filters = None if coins is None else [("coin", "in", list(coins))]
            out = check_signals(pd.read_parquet(self.path, filters=filters))
        drop = [c for c in self.present if c not in self.enabled]
        out = out.drop(columns=drop).sort_values(["coin", "time"], kind="stable").reset_index(drop=True)
        out = decode(out, self.kind)
        out["period"] = assign(out, self.periods)
        if self.stages is not None:
            out = out[out["period"].isin(list(self.stages))].reset_index(drop=True)
        return out

    def batches(self, size: int = 50, coins=None):
        """Yield (coins, rows) per batch of ``size`` coins: the memory bound of a large backtest."""
        names = self.coins() if coins is None else sorted(set(map(str, coins)))
        for i in range(0, len(names), max(1, size)):
            part = names[i:i + size]
            yield part, self.read(part)


def _parquet_columns(path) -> list[str]:
    import pyarrow.parquet as pq
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return list(pq.read_schema(path).names)


def describe() -> str:
    """The contract as text (what ``python -m nig_backtest contract`` prints)."""
    lines = ["required columns:"] + [f"  {k}: {v}" for k, v in REQUIRED.items()]
    lines += ["optional features (absent = off; present = on unless SignalSet.use leaves them out):"]
    lines += [f"  {k}: {v}" for k, v in OPTIONAL.items()]
    lines += [f"free columns: any name starting with '{FREE_PREFIX}' (carried to the trades, never read by a rule)"]
    lines += ["output kinds (SignalSet(kind=...), what score means; absolute = decodes to a price without loss):"]
    lines += [f"  {k}: {'absolute' if is_absolute(k) else 'direction + ranking only'}"
              + (f", needs {list(OUTPUT_KINDS.entry(k).meta['needs'])}" if OUTPUT_KINDS.entry(k).meta["needs"] else "")
              for k in OUTPUT_KINDS.names()]
    return "\n".join(lines)
