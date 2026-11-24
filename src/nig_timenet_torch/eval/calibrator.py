"""PURPOSE: The role-checked confidence calibrators of the TF ``calibration_method``: "isotonic" (increasing map in [0,
    1], inputs clipped to the fitted range) and "platt" (logistic fit on the logit); fit on the CALIBRATION role
    only.
TAGS: calibrator, isotonic, platt, role check, learned state, confidence calibration, registry, tf parity
PITFALLS: fit() raises for any role other than Role.CALIBRATION (never fit on train or test); pure numpy (pool
    adjacent violators; Newton steps for Platt), no scikit-learn at run time (it is the dev-only reference of the
    tests); Platt clips the score to [1e-4, 1 - 1e-4] before the logit, as the TF code did, and its tiny L2 term
    (C = 1e6) keeps a separable val sample finite; the state is plain JSON data (lists and numbers).

P29 (C6): the calibrator is ONE registered component per method (rule B4): ``fit`` -> state, ``apply`` on any
scores. The returned state records fit_role and fit_rows.
"""

import numpy as np

from nig_timenet_torch.core.registry import Contract, Registry
from nig_timenet_torch.core.roles import Role

__all__ = ["CALIBRATORS", "Calibrator", "PlattCalibrator", "CalibratorRoleError", "PLATT_CLIP", "PLATT_C"]

#: Platt scaling as the TF code fitted it: the score is clipped to [PLATT_CLIP, 1 - PLATT_CLIP] before the logit and
#: the regularisation constant C of its logistic regression is PLATT_C (a tiny L2 term)
PLATT_CLIP, PLATT_C = 1e-4, 1e6

CALIBRATORS = Registry("calibrator", Contract("calibrator", required_meta=("method",)))


class CalibratorRoleError(ValueError):
    pass


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators for a non-decreasing fit; returns one fitted value per input point."""
    vals, wts, sizes = [], [], []
    for yi, wi in zip(y, w, strict=True):
        vals.append(float(yi)), wts.append(float(wi)), sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v, ww, s = vals.pop(), wts.pop(), sizes.pop()
            tot = wts[-1] + ww
            vals[-1] = (vals[-1] * wts[-1] + v * ww) / tot
            wts[-1], sizes[-1] = tot, sizes[-1] + s
    return np.repeat(vals, sizes)


class Calibrator:
    method = "isotonic"

    def fit(self, scores: np.ndarray, labels: np.ndarray, *, role: Role) -> dict:
        if role is not Role.CALIBRATION:
            raise CalibratorRoleError(f"Calibrator.fit requires role=Role.CALIBRATION, got {role!r}")
        x = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        if x.shape != y.shape or x.size == 0:
            raise ValueError(f"scores/labels must be same non-empty length, got {x.shape} vs {y.shape}")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("scores/labels must be finite")
        ux, inv = np.unique(x, return_inverse=True)  # ties in x pooled to their mean label (weight = count)
        w = np.bincount(inv).astype(np.float64)
        ym = np.bincount(inv, weights=y) / w
        fitted = np.clip(_pav(ym, w), 0.0, 1.0)
        return {"method": self.method, "fit_role": role.value, "fit_rows": int(x.size),
                "x": ux.tolist(), "y": fitted.tolist()}

    def apply(self, scores: np.ndarray, state: dict) -> np.ndarray:
        if state.get("method") != self.method or state.get("fit_role") != Role.CALIBRATION.value:
            got = (state.get("method"), state.get("fit_role"))
            raise ValueError(f"not an isotonic calibration-role state: {got!r}")
        s = np.asarray(scores, dtype=np.float64)
        return np.interp(s, np.asarray(state["x"]), np.asarray(state["y"]))  # np.interp clips out of range


def _logit(scores: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(scores, dtype=np.float64), PLATT_CLIP, 1.0 - PLATT_CLIP)
    return np.log(v / (1.0 - v))


def _logistic_fit(z: np.ndarray, y: np.ndarray, c: float, steps: int = 100) -> tuple[float, float]:
    """(slope, intercept) minimising sum logloss + slope^2 / (2 c) by Newton steps (the intercept is not penalised)."""
    w, b = 0.0, float(np.log(max(y.mean(), 1e-9) / max(1.0 - y.mean(), 1e-9)))
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(w * z + b)))
        g = np.array([np.sum((p - y) * z) + w / c, np.sum(p - y)])
        s = p * (1.0 - p)
        h = np.array([[np.sum(s * z * z) + 1.0 / c, np.sum(s * z)], [np.sum(s * z), np.sum(s)]])
        step = np.linalg.solve(h + 1e-12 * np.eye(2), g)
        w, b = w - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-12:
            break
    return float(w), float(b)


class PlattCalibrator:
    method = "platt"

    def fit(self, scores: np.ndarray, labels: np.ndarray, *, role: Role) -> dict:
        if role is not Role.CALIBRATION:
            raise CalibratorRoleError(f"PlattCalibrator.fit requires role=Role.CALIBRATION, got {role!r}")
        x = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        if x.shape != y.shape or x.size == 0:
            raise ValueError(f"scores/labels must be same non-empty length, got {x.shape} vs {y.shape}")
        if not (np.isfinite(x).all() and np.isfinite(y).all()) or not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("scores must be finite and labels 0 / 1")
        w, b = _logistic_fit(_logit(x), y, PLATT_C)
        return {"method": self.method, "fit_role": role.value, "fit_rows": int(x.size), "slope": w, "intercept": b}

    def apply(self, scores: np.ndarray, state: dict) -> np.ndarray:
        if state.get("method") != self.method or state.get("fit_role") != Role.CALIBRATION.value:
            got = (state.get("method"), state.get("fit_role"))
            raise ValueError(f"not a platt calibration-role state: {got!r}")
        return 1.0 / (1.0 + np.exp(-(state["slope"] * _logit(scores) + state["intercept"])))


CALIBRATORS.register("isotonic", Calibrator, version=1, method="isotonic")
CALIBRATORS.register("platt", PlattCalibrator, version=1, method="platt")
