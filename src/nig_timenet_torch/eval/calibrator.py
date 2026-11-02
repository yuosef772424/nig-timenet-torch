"""P29: role-checked learned-state calibrator (C6). Isotonic regression only, mirroring the TF side's
calibration_method="isotonic" (increasing, outputs clipped to [0, 1], out-of-range inputs clipped to the fitted
range). Pure numpy pool-adjacent-violators; no scikit-learn at runtime (it is a dev-only reference in tests).

fit() refuses any role other than Role.CALIBRATION: a calibrator is never fit on FIT (train) data, enforced
here by a raise rather than by convention. The returned state records fit_role and fit_rows.
"""

import numpy as np

from nig_timenet_torch.core.registry import Contract, Registry
from nig_timenet_torch.core.roles import Role

__all__ = ["CALIBRATORS", "Calibrator", "CalibratorRoleError"]

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


CALIBRATORS.register("isotonic", Calibrator, version=1, method="isotonic")
