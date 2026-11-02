import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from nig_timenet_torch.core.roles import Role
from nig_timenet_torch.eval.calibrator import CALIBRATORS, Calibrator, CalibratorRoleError


def _data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    s = np.round(rng.normal(size=n), 2)  # rounding creates ties
    y = (rng.random(n) < 1 / (1 + np.exp(-2 * s))).astype(float)
    return s, y


@pytest.mark.parametrize("role", [r for r in Role if r is not Role.CALIBRATION])
def test_fit_on_wrong_role_raises(role):
    s, y = _data()
    with pytest.raises(CalibratorRoleError):
        Calibrator().fit(s, y, role=role)


def test_fit_on_calibration_role_succeeds_and_applies():
    s, y = _data()
    state = Calibrator().fit(s, y, role=Role.CALIBRATION)
    assert state["fit_role"] == "calibration" and state["fit_rows"] == len(s) and state["method"] == "isotonic"
    q = np.linspace(-5, 5, 1001)  # includes out-of-range points
    out = Calibrator().apply(q, state)
    assert np.all(np.diff(out) >= 0) and out.min() >= 0 and out.max() <= 1
    ref = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(s, y)
    np.testing.assert_allclose(out, ref.predict(q), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(Calibrator().apply(s, state), ref.predict(s), rtol=1e-5, atol=1e-6)
    assert CALIBRATORS.entry("isotonic").obj is Calibrator
