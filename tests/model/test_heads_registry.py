"""P17: HEAD_REGISTRY / HeadSpec — one vocabulary for output keys, label key and loss task type."""

import pytest

from nig_timenet_torch.core.registry import DuplicateNameError
from nig_timenet_torch.model.heads import HEAD_REGISTRY, head_spec
from nig_timenet_torch.trainer.tasks import TASK_REGISTRY
from tests.oracle import oracle_loader as ol


def test_duplicate_head_registration_raises():
    with pytest.raises(DuplicateNameError):
        HEAD_REGISTRY.register("binary_classification", None, version=1, output_keys={}, label_key="", task_type="")


def test_head_specs_match_the_oracle_trainer_targets():
    """Every oracle target is exactly what head_spec(kind, target) says (keys, label, task type)."""
    targets = ol.load_config()["trainer"]["targets"]
    for t in ("high", "low", "close"):
        for kind in ("nig_regression", "binary_classification"):
            spec = head_spec(kind, t)
            cfg = targets[spec.name]
            assert cfg["true_key"] == spec.label_key
            assert cfg["task_type"] == spec.task_type and spec.task_type in TASK_REGISTRY
            assert cfg["output_keys"] == spec.output_keys
