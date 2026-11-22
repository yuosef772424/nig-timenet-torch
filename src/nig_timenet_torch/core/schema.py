"""Data schema names shared with the TF side (BUILD_PLAN.md P03).

PURPOSE: The last_candles column schema, price target columns, target modes and raw OHLCV column names, defined once
    so no package copies them.
TAGS: schema, constants, last candles, price targets, target modes, ohlcv columns
PITFALLS: TF values are pinned by tests/test_schema_pinned.py; OHLCV_COLUMNS, OHLCV_AGG, the kline column names,
    LAST_OF / FUTURE_OF and the label-key helpers are this project's own (one copy each).

SOURCE: values copied verbatim from yuosef772424/crypto-signal-prediction core/schema.py at
aeafc564f25c78f252936affbd374f68aa49b8d9. Pinned by tests/test_schema_pinned.py: change them only together with the
source, as a deliberate diff. Not copied (yet): ENTRY_CLOSE_REGS and the PRICE_NORM_* constants (added by the step that
needs them).
"""

from types import MappingProxyType

#: Columns of the ``last_candles`` (N, 7) array, in their physical order in every saved dataset. Only ever APPEND.
LAST_COLUMNS = ('last_high', 'last_low', 'last_close', 'timestamp',
                'future_close', 'future_low_min', 'future_high_max')

#: Column name -> index inside ``last_candles`` (read only).
LAST_COLUMN_INDEX = MappingProxyType({name: i for i, name in enumerate(LAST_COLUMNS)})

#: Index of the timestamp column inside ``last_candles``.
TS_COL = LAST_COLUMNS.index('timestamp')

#: ``last_candles`` is stored as float64: the ns timestamp (~1.7e18) loses ~12 s of precision in float32.
LAST_DTYPE = 'float64'

#: Price columns needed to compute the targets.
TARGET_COLUMNS = ('high', 'low', 'close')

#: Base target modes of retarget_splits (+ optionally "+relative"; "relative" alone = "return+relative").
#: NOT from the TF schema: the raw candle columns every data source produces (open, high, low, close, volume), in
#: the TF CSV order, so the native data pipeline names them once.
OHLCV_COLUMNS = ('open', 'high', 'low', 'close', 'volume')

TARGET_MODES = ("return", "return_close", "scaled", "magnitude", "volnorm", "entry_range")

#: Modes that do not accept the "+relative" suffix.
NO_RELATIVE_BASES = ("entry_range",)

#: How base bars combine into a higher-timeframe bar (the old project's OHLCV_AGG, verbatim); one copy for
#: data.align and data.sources.
OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

#: Column names of the Binance kline files that more than one module reads: the quote-currency volume (the TF CSV
#: header) and the open time of the monthly archive files.
QUOTE_VOLUME_COL = "quote_volume"
OPEN_TIME_COL = "open_time"

#: Price target -> its OWN last price column and its future price column in ``last_candles`` (the label's anchor and
#: source); ``data_io.target_codec`` and ``data.targets`` both read these maps.
LAST_OF = MappingProxyType({"high": "last_high", "low": "last_low", "close": "last_close"})
FUTURE_OF = MappingProxyType({"high": "future_high_max", "low": "future_low_min", "close": "future_close"})

#: Label keys of a price target: the regression label (``nig_regression`` head) and the direction label
#: (``binary_classification`` head). ``model.heads`` registers its head kinds with these same strings, so data
#: tools get a key without importing torch.
REG_LABEL_KEY = "y_{target}_reg"
CLASS_LABEL_KEY = "y_{target}_class"


def head_names(targets) -> list[str]:
    """Target heads in dataset order: high -> high_class, high_reg, low_class, ... (the ``targets`` meta)."""
    return [f"{t}_{kind}" for t in targets for kind in ("class", "reg")]


def reg_label_key(target: str) -> str:
    """The dataset key of ``target``'s regression label, e.g. ``y_close_reg``."""
    return REG_LABEL_KEY.format(target=target)


def class_label_key(target: str) -> str:
    """The dataset key of ``target``'s direction label, e.g. ``y_close_class``."""
    return CLASS_LABEL_KEY.format(target=target)
