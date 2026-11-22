"""PURPOSE: The ``Role`` enum: what a data split may be used for (fit, calibration, selection, report, sealed).
TAGS: role, data role, split, calibration, selection, sealed, learned state
PITFALLS: a learned-state component's ``fit`` requires a role (a calibrator takes CALIBRATION, never FIT); today
    ``data_io.split_plan`` gives SEALED no rows (empty array).

Data roles (C6; BUILD_PLAN.md P25): every split is stamped with what it may be used for. A learned-state
component's fit() requires a role (a calibrator requires CALIBRATION, never FIT)."""

import enum

__all__ = ["Role"]


class Role(enum.Enum):
    FIT = "fit"
    CALIBRATION = "calibration"
    SELECTION = "selection"
    REPORT = "report"
    SEALED = "sealed"
