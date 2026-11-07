"""PURPOSE: The one batched predict that turns a model and its inputs into the canonical PredictionTable, with one
    writer and one validating reader.
TAGS: prediction table, batched predict, table meta, table writer, table reader
PITFALLS: meta arrays must align with x row by row; the timestamp column is taken by name (core.schema.TS_COL), never
    by an integer subscript.

ONE batched predict -> canonical PredictionTable, one writer, one validating reader (C5; BUILD_PLAN.md P27).

predict_table runs the model over ALL rows in fixed-size batches (call count ceil(N / batch_size), independent of
the number of assets); per-asset evaluation is a groupby on the table. ``meta`` holds per-row arrays aligned to x:
timestamp, prediction_made_at, label_available_at (int64 ns UTC, or datetime64), asset (str), plus scalars or
per-row arrays horizon_ns, stage, fold, train_end. ``table_meta`` builds it from a dataset + SplitPlan times.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nig_timenet_torch.core.prediction_schema import SCHEMA_VERSION, validate_table
from nig_timenet_torch.core.schema import TS_COL
from nig_timenet_torch.data_io.split_plan import sample_times
from nig_timenet_torch.trainer.identity import RunIdentity

__all__ = ["predict_table", "write_table", "read_table", "table_meta"]


def _utc(a) -> pd.DatetimeIndex:
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.datetime64):
        a = a.astype("datetime64[ns]").astype(np.int64)
    return pd.to_datetime(a.astype(np.int64), utc=True).as_unit("ns")


def table_meta(dataset: dict, idx: np.ndarray, *, stage: str, fold: int, train_end) -> dict:
    """meta for rows ``idx`` of a dataset: asset from asset_bounds, times from split_plan.sample_times."""
    idx = np.asarray(idx)
    _, feature_end, label_end = sample_times(dataset)
    ts = np.asarray(dataset["last_candles"])[:, TS_COL].astype(np.int64)
    asset = np.empty(ts.shape[0], dtype=object)
    for b in dataset["asset_bounds"]:
        asset[b["start"]:b["end"]] = b["name"]
    bar = int(pd.Timedelta(dataset["base_tf"]).value)
    return {"timestamp": ts[idx], "asset": asset[idx], "prediction_made_at": feature_end[idx],
            "label_available_at": label_end[idx], "horizon_ns": int(dataset["forecast_horizon"]) * bar,
            "stage": stage, "fold": int(fold), "train_end": train_end}


def predict_table(model: torch.nn.Module, x, meta: dict, identity: RunIdentity,
                  batch_size: int = 512) -> pd.DataFrame:
    """``x``: (N, T, F) tensor, or a dict {timeframe: (N, T_tf, F_tf)} for a multi-timeframe model (P45); the
    model runs on whatever device ``x`` is on, outputs come back to CPU."""
    first = next(iter(x.values())) if isinstance(x, dict) else x
    n = first.shape[0]
    if isinstance(x, dict) and any(v.shape[0] != n for v in x.values()):
        raise ValueError(f"timeframe inputs have different row counts: { {k: v.shape[0] for k, v in x.items()} }")
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, start + batch_size)
            out = model({k: v[sl] for k, v in x.items()} if isinstance(x, dict) else x[sl])
            for k, v in out.items():
                chunks.setdefault(k, []).append(v.detach().cpu().numpy().reshape(v.shape[0], -1))
    full = lambda v: np.full(n, v) if np.ndim(v) == 0 else np.asarray(v)  # noqa: E731
    train_end = pd.Timestamp(meta["train_end"])
    train_end = train_end.tz_localize("UTC") if train_end.tzinfo is None else train_end.tz_convert("UTC")
    df = pd.DataFrame({
        "timestamp": _utc(meta["timestamp"]),
        "asset": pd.array(full(meta["asset"]).astype(str), dtype="str"),
        "horizon_ns": full(meta["horizon_ns"]).astype(np.int64),
        "prediction_made_at": _utc(meta["prediction_made_at"]),
        "label_available_at": _utc(meta["label_available_at"]),
        "identity_hash": pd.array(np.full(n, identity.fingerprint()), dtype="str"),
        "stage": pd.array(full(meta["stage"]).astype(str), dtype="str"),
        "fold": full(meta["fold"]).astype(np.int64),
        "train_end": pd.DatetimeIndex(np.full(n, train_end.value)).tz_localize("UTC").as_unit("ns"),
        "schema_version": np.full(n, SCHEMA_VERSION, dtype=np.int64),
    })
    for k in sorted(chunks):
        arr = np.concatenate(chunks[k], axis=0)
        if arr.shape[1] != 1:
            raise ValueError(f"output {k!r} has {arr.shape[1]} columns per row; only scalar outputs are tabulated")
        df[k] = arr[:, 0].astype(np.float64)
    validate_table(df)
    return df


def write_table(path, df: pd.DataFrame, identity: RunIdentity) -> None:
    """``path`` (parquet) + ``path + ".meta.json"`` (identity, its fingerprint, rows, schema version)."""
    validate_table(df)
    fp = identity.fingerprint()
    if len(df) and set(df["identity_hash"].unique()) != {fp}:
        raise ValueError("table identity_hash does not match the identity being written")
    path = Path(path)
    df.to_parquet(path, index=False)
    meta = {"identity": identity.to_json(), "identity_hash": fp, "n_rows": len(df), "schema_version": SCHEMA_VERSION,
            "columns": list(df.columns)}
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def read_table(path) -> pd.DataFrame:
    """Reads, validates, and checks the table against its .meta.json (rows, identity hash, columns)."""
    path = Path(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    df = pd.read_parquet(path)
    validate_table(df)
    if len(df) != meta["n_rows"] or list(df.columns) != meta["columns"]:
        raise ValueError(f"{path}: rows/columns differ from its .meta.json")
    if len(df) and set(df["identity_hash"].unique()) != {meta["identity_hash"]}:
        raise ValueError(f"{path}: identity_hash differs from its .meta.json")
    if RunIdentity.from_json(meta["identity"]).fingerprint() != meta["identity_hash"]:
        raise ValueError(f"{path}: .meta.json identity does not hash to its identity_hash")
    return df
