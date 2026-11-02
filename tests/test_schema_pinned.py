"""P03: constants pinned to their TF-side source, so a typo cannot become a silent incompatibility."""

import ast
from pathlib import Path

from nig_timenet_torch.core import constants, schema

CORE = Path(__file__).resolve().parent.parent / "src" / "nig_timenet_torch" / "core"


def test_last_columns_matches_source():
    """Source: crypto-signal-prediction core/schema.py @ aeafc564f25c78f252936affbd374f68aa49b8d9."""
    assert schema.LAST_COLUMNS == ('last_high', 'last_low', 'last_close', 'timestamp',
                                   'future_close', 'future_low_min', 'future_high_max')
    assert dict(schema.LAST_COLUMN_INDEX) == {n: i for i, n in enumerate(schema.LAST_COLUMNS)}
    assert schema.TS_COL == 3
    assert schema.LAST_DTYPE == 'float64'
    assert schema.TARGET_COLUMNS == ('high', 'low', 'close')
    assert schema.TARGET_MODES == ("return", "return_close", "scaled", "magnitude", "volnorm", "entry_range")
    assert schema.NO_RELATIVE_BASES == ("entry_range",)


def test_nig_alpha_den_min_matches_source():
    """Source: crypto-signal-prediction core/constants.py @ aeafc564f25c78f252936affbd374f68aa49b8d9."""
    assert constants.NIG_ALPHA_DEN_MIN == 1e-2


def test_modules_have_no_dependencies():
    """P03 DoD: nothing beyond types.MappingProxyType (stdlib)."""
    for name, allowed in (("schema.py", {"types"}), ("constants.py", set())):
        mods = set()
        for node in ast.parse((CORE / name).read_text()).body:
            if isinstance(node, ast.Import):
                mods |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add(node.module)
        assert mods == allowed, name
