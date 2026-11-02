"""P17: task losses recomputed on the oracle's outputs.npz + labels.npz vs losses.json stats (both scenarios)."""

import pytest
import torch

from nig_timenet_torch.core.registry import ContractError, UnknownNameError
from nig_timenet_torch.trainer.tasks import TASK_REGISTRY, register_task_type
from tests.oracle import oracle_loader as ol

SCENARIOS = list(ol.load_losses()["scenarios"])


def targets_cfg():
    return ol.load_config()["trainer"]["targets"]


def run_task(t, scenario):
    cfg = targets_cfg()[t]
    sc = ol.load_losses()["scenarios"][scenario]
    entry = TASK_REGISTRY[cfg["task_type"]]
    loss, stats = entry["fn"](ol.load_labels()[cfg["true_key"]], ol.load_outputs(), cfg, sc["scheduled_vars"])
    assert sorted(stats) == sorted(entry["stat_keys"])
    return loss, stats, sc


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("t", ["high_reg", "low_reg", "close_reg"])
def test_evidential_loss_matches_oracle_nig_base(t, scenario):
    loss, stats, sc = run_task(t, scenario)
    for k, v in stats.items():
        ol.assert_close(v, torch.tensor(sc["stats"][f"{t}_{k}"]["value"]), "loss", f"{scenario} {t}_{k}")
    ol.assert_close(loss, torch.tensor(sc["task_losses_weighted"][t]["value"]), "loss", f"{scenario} {t} loss")


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("t", ["high_class", "low_class", "close_class"])
def test_classification_loss_matches_oracle_accuracy(t, scenario):
    loss, stats, sc = run_task(t, scenario)
    ol.assert_close(stats["accuracy"], torch.tensor(sc["stats"][f"{t}_accuracy"]["value"]), "loss", t)
    # task_losses_weighted = raw loss * loss_weights (all 1.0); Kendall is applied later, in compute_losses (P18)
    ol.assert_close(loss, torch.tensor(sc["task_losses_weighted"][t]["value"]), "loss", f"{t} loss")


def test_regression_task_loss_kinds():
    fn = TASK_REGISTRY["regression"]["fn"]
    y, out = torch.tensor([0.0, 2.0]), {"p": torch.tensor([1.0, 0.0])}
    for kind, expected in [("mse", 2.5), ("mae", 1.5), ("huber", 1.0)]:
        loss, stats = fn(y, out, {"output_keys": {"pred": "p"}, "loss_fn": kind}, {})
        torch.testing.assert_close(loss, torch.tensor(expected))
        torch.testing.assert_close(stats["mae"], torch.tensor(1.5))


def test_unported_options_raise():
    cfg = dict(targets_cfg()["close_reg"], beta_nll=0.5)
    with pytest.raises(NotImplementedError):
        TASK_REGISTRY["evidential"]["fn"](ol.load_labels()["y_close_reg"], ol.load_outputs(), cfg, {})


def test_unknown_task_type_raises():
    with pytest.raises(UnknownNameError, match="evidential"):
        TASK_REGISTRY["evidentail"]


def test_sample_weight_contract_checked_at_registration():
    with pytest.raises(ContractError):
        register_task_type("bad_task", lambda y, o, c, s: (0, {}), [], supports_sample_weight=True)
    assert "bad_task" not in TASK_REGISTRY
