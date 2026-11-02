"""Data schema names shared with the TF side (BUILD_PLAN.md P03).

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
TARGET_MODES = ("return", "return_close", "scaled", "magnitude", "volnorm", "entry_range")

#: Modes that do not accept the "+relative" suffix.
NO_RELATIVE_BASES = ("entry_range",)
