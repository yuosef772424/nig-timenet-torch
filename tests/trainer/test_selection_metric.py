"""P22: SELECTION_METRICS — named, fixed, independent of loss weights."""

import pytest
import torch

from nig_timenet_torch.core.registry import UnknownNameError
from nig_timenet_torch.stats import metrics
from nig_timenet_torch.trainer.selection_metric import SELECTION_METRICS
from nig_timenet_torch.trainer.trainer import Trainer
from tests.trainer.fixture_setup import build_model, load_fixture, trainer_config


def test_default_is_val_raw_loss_and_registered_names():
    model = build_model(16, 18)
    assert Trainer(model, trainer_config()).selection_metric == "val_raw_loss"
    assert SELECTION_METRICS.entry("val_raw_loss").meta["higher_is_better"] is False
    for t in ("high", "low", "close"):
        assert SELECTION_METRICS.entry(f"val_auc_{t}_class").meta["higher_is_better"] is True
        assert SELECTION_METRICS.entry(f"val_pinball_{t}_reg").meta["higher_is_better"] is False


def test_unknown_selection_metric_raises():
    with pytest.raises(UnknownNameError, match="val_raw_loss"):
        Trainer(build_model(16, 18), trainer_config(), selection_metric="val_auc_close")


@pytest.mark.parametrize("name", ["val_auc_close_class", "val_pinball_close_reg"])
def test_changing_loss_weight_does_not_change_selection_metric(name):
    torch.manual_seed(0)
    x, y = load_fixture(128)
    model = build_model(16, 18)
    values = []
    for w in (1.0, 5.0):
        trainer = Trainer(model, trainer_config(), selection_metric=name)
        trainer.loss_weights["close_class"] = w
        trainer.loss_weights["close_reg"] = w
        logs = trainer.eval_step(x, y)
        values.append((logs["selection_metric"], logs["raw_loss"]))
    assert values[0][0] == values[1][0]
    assert values[0][1] != values[1][1]  # the loss itself did change: the metric is truly independent of it
    with torch.no_grad():
        out = model.eval()(x)
    if name.startswith("val_auc"):
        assert values[0][0] == metrics.auc(y["y_close_class"].numpy(), out["y_close_class_logits"].numpy())
