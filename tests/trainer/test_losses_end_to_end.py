"""P18 (M3 exit gate): compute_losses on the oracle model + batch vs losses.json (both scenarios) and grads.npz
(PRIMARY scenario). Model in eval() (the oracle ran training=False)."""

import pytest
import torch

from tests.oracle import oracle_loader as ol
from tests.trainer.loss_helpers import (
    LOSSES,
    PRIMARY,
    START,
    grads_vs_oracle,
    make_trainer,
    named_trainable,
    run,
    value,
)


def check_losses(scenario):
    _, sc, (total, raw, per_target, stats, constraints) = run(scenario)
    ol.assert_close(total, value(sc["total_loss"]), "loss", f"{scenario} total_loss")
    ol.assert_close(raw, value(sc["raw_loss"]), "loss", f"{scenario} raw_loss")
    for t, v in sc["task_losses_weighted"].items():
        ol.assert_close(per_target[t], value(v), "loss", f"{scenario} {t}")
    for k, v in sc["stats"].items():
        ol.assert_close(stats[k], value(v), "loss", f"{scenario} {k}")
    assert constraints == {} and not sc["constraints"]


def test_total_loss_matches_oracle_primary_scenario():
    assert LOSSES["scenarios"][PRIMARY]["scheduled_vars"] == pytest.approx({"lambda_reg": 0.05, "lambda_calib": 0.1})
    check_losses(PRIMARY)


def test_total_loss_matches_oracle_start_scenario():
    assert LOSSES["scenarios"][START]["scheduled_vars"] == {"lambda_reg": 0.0, "lambda_calib": 0.0}
    check_losses(START)


def test_regularization_term_matches_oracle():
    trainer = make_trainer()
    for sc in LOSSES["scenarios"].values():
        ol.assert_close(trainer.model.regularization_loss(), value(sc["model_regularization_losses"]), "loss", "reg")






def test_grads_match_oracle_primary_scenario():
    trainer, _, (total, *_) = run(PRIMARY)
    named = named_trainable(trainer)
    grads = torch.autograd.grad(total, list(named.values()))
    grads_vs_oracle(trainer, dict(zip(named, grads, strict=True)))
