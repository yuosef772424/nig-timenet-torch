"""trainer/targets.py: strict trainer config (HF-20), targets_config from HeadSpec (HF-21), defaults from the registry.

PURPOSE:  Unknown trainer-config keys raise with a did-you-mean, defaults come from TASK_REGISTRY / LOSS_DEFAULTS and
  never enter ``Trainer.config``, schedules are parsed once, ``targets_config`` equals the pinned baseline targets.
TAGS:     trainer config validation targets_config HeadSpec default_options schedules selection metric
PITFALLS: the golden dict is tests/pins/trainer_config.json (taken before the refactor, by the pin test).
"""

import copy
import json
from pathlib import Path

import pytest

from nig_timenet_torch.core.registry import UnknownNameError
from nig_timenet_torch.trainer.selection_metric import NOT_SELECTABLE, SELECTION_METRICS
from nig_timenet_torch.trainer.targets import resolve_loss, resolve_targets, targets_config, validate_trainer_config
from nig_timenet_torch.trainer.trainer import Trainer
from nig_timenet_torch.workflow import flat_run
from tests.trainer.fixture_setup import build_model

GOLDEN = json.loads((Path(__file__).resolve().parents[1] / "pins" / "trainer_config.json").read_text("utf-8"))
ALL = ("high", "low", "close")


def _cfg():
    return flat_run.trainer_config(ALL, 0.1)


def test_targets_config_equals_the_pre_refactor_dict():
    assert flat_run.trainer_config(ALL, 0.1) == GOLDEN["high_low_close"]["config"]
    assert flat_run.trainer_config(("close",), 0.1) == GOLDEN["close_only"]["config"]
    head_types = {t: ["nig_regression", "binary_classification"] for t in ALL}
    assert targets_config(head_types, options_by_kind={"nig_regression": {"use_calibration_loss": True},
                                                       "binary_classification": {"binary": True}}).keys() == \
        GOLDEN["high_low_close"]["config"]["targets"].keys()


@pytest.mark.parametrize("typo,hint", [(("targets", "close_class", "label_smoothin", 0.5), "label_smoothing"),
                                       (("loss", None, "use_uncertainty_weightin", False), "use_uncertainty_weighting"),
                                       (("top", None, "constraint", []), "constraints")])
def test_mistyped_keys_raise_with_a_did_you_mean(typo, hint):
    where, target, key, value = typo
    cfg = _cfg()
    (cfg["targets"][target] if where == "targets" else cfg["loss"] if where == "loss" else cfg)[key] = value
    with pytest.raises(ValueError, match=key) as e:
        validate_trainer_config(cfg)
    assert hint in str(e.value)
    with pytest.raises(ValueError, match=key):  # the Trainer validates first
        Trainer(build_model(16, 18), cfg)


def test_missing_keys_unknown_task_and_unscheduled_lambda_raise():
    cfg = _cfg()
    del cfg["targets"]["close_reg"]["output_keys"]
    with pytest.raises(ValueError, match="lacks"):
        validate_trainer_config(cfg)
    cfg = _cfg()
    cfg["targets"]["close_reg"]["task_type"] = "evidentail"
    with pytest.raises(UnknownNameError):
        validate_trainer_config(cfg)
    cfg = _cfg()
    cfg["targets"]["close_reg"]["lambda_reg_var"] = "lambda_rg"
    with pytest.raises(ValueError, match="not a scheduled variable"):
        validate_trainer_config(cfg)
    cfg = _cfg()
    del cfg["loss"]["schedules"]["lambda_calib"]
    with pytest.raises(ValueError, match="lambda_calib"):
        validate_trainer_config(cfg)


def test_targets_config_refuses_an_option_the_task_does_not_register():
    with pytest.raises(ValueError, match="label_smoothin"):
        targets_config({"close": ["binary_classification"]}, options_by_kind={"binary_classification":
                                                                              {"label_smoothin": 0.1}})
    with pytest.raises(ValueError, match="does not use"):
        targets_config({"close": ["binary_classification"]}, options_by_kind={"nig_regression": {}})


def test_defaults_come_from_the_registry_and_never_enter_the_config():
    cfg = {"targets": {"c": {"true_key": "y_close_class", "task_type": "classification",
                             "output_keys": {"logits": "y_close_class_logits"}}}, "loss": {}}
    before = copy.deepcopy(cfg)
    validate_trainer_config(cfg)
    assert resolve_targets(cfg)["c"] == {"true_key": "y_close_class", "task_type": "classification",
                                         "output_keys": {"logits": "y_close_class_logits"}, "loss_weight": 1.0,
                                         "flood_level": None, "binary": True, "label_smoothing": 0.0}
    loss = resolve_loss(cfg)
    assert loss["use_uncertainty_weighting"] is False and loss["schedules"] == {} and loss["all_constraints"] == []
    assert cfg == before, "resolution must not write defaults into the config (it is hashed into the RunIdentity)"


def test_schedules_are_parsed_once_in_init():
    t = Trainer(build_model(16, 18), _cfg())
    assert set(t.schedules) == {"lambda_reg", "lambda_calib"}
    assert t.scheduled_vars == {"lambda_reg": 0.0, "lambda_calib": 0.0}
    assert t.config == _cfg(), "Trainer.config stays the dict as given"
    bad = _cfg()
    bad["loss"]["schedules"]["lambda_reg"]["schedule"] = "cosin"
    with pytest.raises(ValueError, match="cosin"):
        Trainer(build_model(16, 18), bad)  # at construction, not at fit


@pytest.mark.parametrize("name", NOT_SELECTABLE)
def test_direction_metrics_cannot_select_but_stay_registered(name):
    assert name in SELECTION_METRICS
    with pytest.raises(ValueError, match="reporting only"):
        Trainer(build_model(16, 18), _cfg(), selection_metric=name)
    assert Trainer(build_model(16, 18), _cfg(), selection_metric="val_raw_loss").selection_metric == "val_raw_loss"
    Trainer(build_model(16, 18), _cfg(), selection_metric="val_auc_close_class")

