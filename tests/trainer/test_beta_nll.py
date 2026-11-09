"""FX1: beta-NLL in the evidential task (TF ``beta_nll``, torch ``beta_nll_normalize``) and its TrainSettings wiring."""

import pytest
import torch

from nig_timenet_torch.trainer.tasks import TASK_REGISTRY, nig_w_st, with_defaults
from nig_timenet_torch.workflow.flat_run import trainer_config
from nig_timenet_torch.workflow.settings import TrainSettings

KEYS = {"mu": "mu", "nu": "nu", "alpha": "al", "beta": "be"}


def _batch():
    g = torch.Generator().manual_seed(0)
    y = 0.02 * torch.randn(64, generator=g)
    out = {"mu": 0.01 * torch.randn(64, 1, generator=g), "nu": 0.1 + torch.rand(64, 1, generator=g),
           "al": 2.0 + torch.rand(64, 1, generator=g), "be": 1e-4 + 1e-3 * torch.rand(64, 1, generator=g)}
    return y, out


def _loss(y, out, **opts):
    cfg = with_defaults("evidential", {"output_keys": KEYS, "use_calibration_loss": False, **opts})
    return TASK_REGISTRY["evidential"]["fn"](y, out, cfg, {"lambda_reg": 0.0, "lambda_calib": 0.0})[0]


def test_beta_nll_zero_is_the_plain_loss():
    y, out = _batch()
    assert torch.equal(_loss(y, out), _loss(y, out, beta_nll=0.0, beta_nll_normalize=True))


def test_beta_nll_weights_each_row_by_its_detached_variance():
    y, out = _batch()
    base = _loss(y, out)  # mean of the per-row NLL
    w = nig_w_st(*(out[k].ravel() for k in ("nu", "al", "be"))).square()
    got = _loss(y, out, beta_nll=1.0)
    assert not torch.isclose(got, base)
    mu = out["mu"].clone().requires_grad_(True)
    g_plain = torch.autograd.grad(_loss(y, {**out, "mu": mu}), mu)[0].ravel()
    g_beta = torch.autograd.grad(_loss(y, {**out, "mu": mu}, beta_nll=1.0), mu)[0].ravel()
    torch.testing.assert_close(g_beta, g_plain * w)  # the weight is a constant for the gradient


def test_beta_nll_normalize_keeps_the_mean_weight_one():
    y, out = _batch()
    w = nig_w_st(*(out[k].ravel() for k in ("nu", "al", "be"))).square()
    mu = out["mu"].clone().requires_grad_(True)
    g_plain = torch.autograd.grad(_loss(y, {**out, "mu": mu}), mu)[0].ravel()
    g_norm = torch.autograd.grad(_loss(y, {**out, "mu": mu}, beta_nll=1.0, beta_nll_normalize=True), mu)[0].ravel()
    torch.testing.assert_close(g_norm, g_plain * w / w.mean())


def test_train_settings_wire_beta_nll_only_when_set():
    targets = ("high", "low", "close")
    plain = trainer_config(targets, 0.1, TrainSettings())
    assert all("beta_nll" not in c for c in plain["targets"].values())
    on = trainer_config(targets, 0.1, TrainSettings(beta_nll=0.5, beta_nll_normalize=True))
    reg = [c for c in on["targets"].values() if "beta_nll" in c]
    assert len(reg) == 3 and all(c["beta_nll"] == 0.5 and c["beta_nll_normalize"] is True for c in reg)
    with pytest.raises(ValueError, match="beta_nll"):
        TrainSettings(beta_nll=-1.0)
