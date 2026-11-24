import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from nig_timenet_torch.core.roles import Role
from nig_timenet_torch.eval.calibrator import CALIBRATORS, Calibrator, CalibratorRoleError, PlattCalibrator


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


def _probs(n=800, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.001, 0.999, n)
    return x, (rng.random(n) < 0.2 + 0.6 * x).astype(float)


@pytest.mark.parametrize("role", [r for r in Role if r is not Role.CALIBRATION])
def test_platt_fit_on_wrong_role_raises(role):
    x, y = _probs()
    with pytest.raises(CalibratorRoleError):
        PlattCalibrator().fit(x, y, role=role)


def test_platt_matches_the_tf_logistic_regression_on_the_logit():
    """The TF code fitted ``LogisticRegression(C=1e6)`` on the logit of the clipped score (scikit-learn's lbfgs stops at
    its own tolerance, so the maps agree to ~1e-4)."""
    from sklearn.linear_model import LogisticRegression

    x, y = _probs()
    state = PlattCalibrator().fit(x, y, role=Role.CALIBRATION)
    assert state["method"] == "platt" and state["fit_role"] == "calibration" and state["fit_rows"] == len(x)
    z = lambda v: np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4))).reshape(-1, 1)  # noqa: E731
    ref = LogisticRegression(C=1e6, max_iter=1000).fit(z(x), y.astype(int))
    q = np.linspace(0.0, 1.0, 101)                      # includes the clipped ends
    np.testing.assert_allclose(PlattCalibrator().apply(q, state), ref.predict_proba(z(q))[:, 1], atol=1e-4)
    assert CALIBRATORS.entry("platt").obj is PlattCalibrator


def test_platt_state_is_plain_data_and_a_state_of_another_method_is_refused():
    import json

    x, y = _probs()
    state = PlattCalibrator().fit(x, y, role=Role.CALIBRATION)
    assert json.loads(json.dumps(state)) == state
    with pytest.raises(ValueError, match="platt"):
        PlattCalibrator().apply(x, Calibrator().fit(x, y, role=Role.CALIBRATION))
    with pytest.raises(ValueError, match="0 / 1"):
        PlattCalibrator().fit(x, y * 0.5, role=Role.CALIBRATION)
