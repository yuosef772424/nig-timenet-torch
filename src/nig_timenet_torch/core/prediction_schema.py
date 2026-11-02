"""PredictionTable schema (C5; BUILD_PLAN.md P27): pure constants + one validator. Besides the mandatory columns a
table carries one float column per model output, named exactly like the output key (``y_*``); nothing else."""

__all__ = ["PREDICTION_TABLE_COLUMNS", "COLUMN_KINDS", "OUTPUT_PREFIX", "SCHEMA_VERSION", "PredictionTableError",
           "validate_table"]

SCHEMA_VERSION = 1
PREDICTION_TABLE_COLUMNS = ("timestamp", "asset", "horizon_ns", "prediction_made_at", "label_available_at",
                            "identity_hash", "stage", "fold", "train_end", "schema_version")
COLUMN_KINDS = {"timestamp": "datetime", "asset": "string", "horizon_ns": "int", "prediction_made_at": "datetime",
                "label_available_at": "datetime", "identity_hash": "string", "stage": "string", "fold": "int",
                "train_end": "datetime", "schema_version": "int"}
OUTPUT_PREFIX = "y_"


class PredictionTableError(ValueError):
    pass


def _kind_ok(series, kind: str) -> bool:
    import pandas as pd

    t = series.dtype
    if kind == "datetime":
        return isinstance(t, pd.DatetimeTZDtype) and str(t.tz) == "UTC" and t.unit == "ns"
    if kind == "int":
        return pd.api.types.is_integer_dtype(t) and t.itemsize == 8
    if kind == "float":
        return pd.api.types.is_float_dtype(t)
    if kind == "string":
        return pd.api.types.is_string_dtype(t) or pd.api.types.is_object_dtype(t)
    raise AssertionError(kind)


def validate_table(df) -> None:
    """Raises PredictionTableError listing EVERY missing / extra / mistyped column (datetimes: tz-aware UTC ns;
    ints: int64; outputs ``y_*``: float), plus the leakage rule label_available_at >= prediction_made_at >=
    timestamp."""
    problems = []
    missing = [c for c in PREDICTION_TABLE_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in COLUMN_KINDS and not str(c).startswith(OUTPUT_PREFIX)]
    if missing:
        problems.append(f"missing columns: {missing}")
    if extra:
        problems.append(f"extra columns: {extra}")
    for c in df.columns:
        kind = COLUMN_KINDS.get(c, "float" if str(c).startswith(OUTPUT_PREFIX) else None)
        if kind and not _kind_ok(df[c], kind):
            problems.append(f"column {c!r}: dtype {df[c].dtype} is not {kind}")
    if not problems and len(df):
        made, avail = df["prediction_made_at"], df["label_available_at"]
        if (avail < made).any() or (made < df["timestamp"]).any():
            problems.append("rows with label_available_at < prediction_made_at or prediction_made_at < timestamp")
    if problems:
        raise PredictionTableError("invalid prediction table: " + "; ".join(problems))
