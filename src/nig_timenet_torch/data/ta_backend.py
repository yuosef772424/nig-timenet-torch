"""The technical-analysis backend (pandas_ta_classic), imported on first use so the rest of the package works
without it. Importing the module registers the ``DataFrame.ta`` accessor the indicators are called through."""

from types import ModuleType

__all__ = ["ta_module"]

_INSTALL_HINT = ("the native data pipeline computes indicators with pandas_ta_classic: "
                 "pip install 'nig-timenet-torch[data]'  (or: pip install pandas-ta-classic==0.8.32)")


def ta_module() -> ModuleType:
    try:
        import pandas_ta_classic as ta
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return ta
