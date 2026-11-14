"""model/scalers.py: the train-fitted state lives in the model (buffers), an unfitted module raises, and the defaults
add nothing to the state_dict (old checkpoints still load)."""

import math

import numpy as np
import pytest
import torch

from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.model.architectures import ANTI_MEMORIZATION_CONFIG, ModelSettings, build_model
from nig_timenet_torch.model.nig_layers import NIGHead, OrderedMeans
from nig_timenet_torch.model.scalers import (
    FITTED_SCALE,
    INPUT_SCALERS,
    FittedScale,
    InputScaler,
    fit_scalers,
    label_scale,
)
from nig_timenet_torch.workflow.settings import MODEL_FIRST_FIXES


def _fixed_model(seq=32, feat=3):
    seed_everything(0)
    return build_model(ModelSettings("nig_timenet_v2", {**ANTI_MEMORIZATION_CONFIG, **MODEL_FIRST_FIXES}), seq, feat)


def test_unfitted_input_scaler_raises_on_forward():
    with pytest.raises(RuntimeError, match="not fitted"):
        InputScaler(3)(torch.zeros(2, 4, 3))


def test_unfitted_fitted_scale_raises_on_forward():
    with pytest.raises(RuntimeError, match="not fitted"):
        FittedScale()()


def test_fit_matches_numpy_and_clips_at_five():
    rng = np.random.default_rng(0)
    x = (rng.normal(size=(50, 7, 3)) * np.array([0.01, 1.0, 30.0]) + np.array([0.5, -2.0, 4.0])).astype(np.float32)
    sc = InputScaler(3)
    state = sc.fit(torch.from_numpy(x), chunk=13)  # several chunks
    flat = x.reshape(-1, 3).astype(np.float64)
    np.testing.assert_allclose(sc.mean.numpy(), flat.mean(0), rtol=1e-5)
    np.testing.assert_allclose(sc.std.numpy(), flat.std(0), rtol=1e-5)
    assert state["rows"] == 50 and len(state["mean"]) == 3
    z = sc(torch.from_numpy(x))
    np.testing.assert_allclose(z.numpy(), np.clip((x - flat.mean(0)) / flat.std(0), -5, 5), atol=1e-4)
    out = sc(torch.tensor([[[1e6, -1e6, 0.0]]]))
    assert out[0, 0, 0] == 5.0 and out[0, 0, 1] == -5.0


def test_fit_rejects_wrong_shape():
    with pytest.raises(ValueError):
        InputScaler(3).fit(torch.zeros(4, 3))
    with pytest.raises(ValueError):
        InputScaler(3).fit(torch.zeros(4, 5, 2))


def test_state_dict_round_trip_gives_identical_outputs():
    m = _fixed_model().eval()
    x = torch.randn(40, 32, 3) * 0.03
    y = {f"y_{t}_reg": torch.randn(40) * 0.03 for t in ("high", "low", "close")}
    fit_scalers(m, x, y)
    new = _fixed_model().eval()
    with pytest.raises(RuntimeError):
        new(x)  # unfitted until the state is loaded
    new.load_state_dict(m.state_dict())
    with torch.no_grad():
        a, b = m(x), new(x)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_label_scale_is_sqrt_of_mean_variance():
    rng = np.random.default_rng(1)
    y = {"y_a_reg": rng.normal(size=500) * 0.02, "y_b_reg": rng.normal(size=500) * 0.06}
    expect = math.sqrt((y["y_a_reg"].var() + y["y_b_reg"].var()) / 2)
    assert label_scale(y, ["y_a_reg", "y_b_reg"]) == pytest.approx(expect, rel=1e-9)
    with pytest.raises(ValueError):
        label_scale(y, [])


def test_fitted_scale_set_validates():
    fs = FittedScale()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            fs.set(bad)
    fs.set(0.5)
    assert float(fs()) == 0.5


def test_fit_scalers_fits_every_scaler():
    m = _fixed_model()
    x = torch.randn(60, 32, 3) * 0.03 + 0.001
    y = {f"y_{t}_reg": torch.randn(60) * s for t, s in (("high", 0.02), ("low", 0.04), ("close", 0.03))}
    state = fit_scalers(m, x, y)
    scalers = [mod for mod in m.modules() if isinstance(mod, InputScaler)]
    scales = [mod for mod in m.modules() if isinstance(mod, FittedScale)]
    assert len(scalers) == 1 and len(scales) == 4  # 3 NIG heads + ordered means
    assert all(bool(torch.isfinite(s.std).all() and torch.isfinite(s.mean).all()) for s in scalers)
    expect = label_scale(y, ["y_high_reg", "y_low_reg", "y_close_reg"])
    assert state["nig_label_scale"] == pytest.approx(expect)
    assert all(float(s.value) == pytest.approx(expect, rel=1e-6) for s in scales)
    assert "input_scaler" in state
    m.eval()
    with torch.no_grad():
        out = m(x[:5])
    assert all(bool(torch.isfinite(v).all()) for v in out.values())


def test_fit_scalers_is_a_noop_for_a_plain_model():
    m = build_model(ModelSettings("nig_timenet_v2", dict(ANTI_MEMORIZATION_CONFIG)), 32, 3)
    assert fit_scalers(m, torch.zeros(4, 32, 3), {}) == {}


def test_nighead_train_std_equals_float_scale_with_same_weights():
    seed_everything(0)
    fitted = NIGHead(8, 16, 0.0, scale=FITTED_SCALE)
    fixed = NIGHead(8, 16, 0.0, scale=0.037)
    fixed.load_state_dict(fitted.state_dict(), strict=False)
    with pytest.raises(RuntimeError):
        fitted(torch.randn(4, 8))
    fitted.fitted_scale.set(0.037)
    h = torch.randn(5, 8)
    for a, b in zip(fitted(h), fixed(h), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_ordered_means_train_std_equals_float_scale():
    fitted, fixed = OrderedMeans(FITTED_SCALE), OrderedMeans(0.05)
    fitted.fitted_scale.set(0.05)
    mu, rh, rl = torch.randn(6, 1) * 0.05, torch.randn(6, 1) * 0.05, torch.randn(6, 1) * 0.05
    for a, b in zip(fitted(mu, rh, rl), fixed(mu, rh, rl), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    with pytest.raises(RuntimeError):
        OrderedMeans(FITTED_SCALE)(mu, rh, rl)


def test_bad_head_scales_raise():
    for bad in (0.0, -1.0, "oops"):
        with pytest.raises(ValueError):
            NIGHead(4, 4, scale=bad)


def test_defaults_add_no_state_dict_keys():
    m = build_model(ModelSettings("nig_timenet_v2", dict(ANTI_MEMORIZATION_CONFIG)), 32, 3)
    assert not [k for k in m.state_dict() if "input_scaler" in k or "fitted_scale" in k]
    h = NIGHead(4, 4)
    assert not [k for k in h.state_dict() if "fitted_scale" in k]
    fixed = build_model(ModelSettings("nig_timenet_v2", {**ANTI_MEMORIZATION_CONFIG, **MODEL_FIRST_FIXES}), 32, 3)
    assert {k for k in fixed.state_dict() if "input_scaler" in k or "fitted_scale" in k}


def test_unknown_input_scaler_raises():
    with pytest.raises(ValueError):
        build_model(ModelSettings("nig_timenet_v2", {**ANTI_MEMORIZATION_CONFIG, "input_scaler": "nope"}), 32, 3)
    assert set(INPUT_SCALERS.names()) >= {"none", "train_std"}


def test_default_model_settings_carry_the_model_first_fixes():
    from nig_timenet_torch.workflow.settings import default_model_settings

    ms = default_model_settings()
    assert ms.architecture == "nig_timenet_v2"
    assert ms.overrides == {**ANTI_MEMORIZATION_CONFIG, **MODEL_FIRST_FIXES}
    assert MODEL_FIRST_FIXES["input_scaler"] == "train_std" and MODEL_FIRST_FIXES["nig_scale"] == FITTED_SCALE
