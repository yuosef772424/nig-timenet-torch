"""Data roles (C6; BUILD_PLAN.md P25): every split is stamped with what it may be used for. A learned-state
component's fit() requires a role (a calibrator requires CALIBRATION, never FIT)."""

import enum

__all__ = ["Role"]


class Role(enum.Enum):
    FIT = "fit"
    CALIBRATION = "calibration"
    SELECTION = "selection"
    REPORT = "report"
    SEALED = "sealed"
