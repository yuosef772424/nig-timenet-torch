"""Oracle loss/gradient helpers shared by the trainer tests (HF-32)."""

import torch

from nig_timenet_torch.model.builder import build_nig_timenet_v2, keras_to_torch_value, torch_param_name
from nig_timenet_torch.trainer.trainer import Trainer
from tests.oracle import oracle_loader as ol

LOSSES = ol.load_losses()
PRIMARY, START = LOSSES["primary"], next(k for k in LOSSES["scenarios"] if k.startswith("schedules_start"))
KENDALL_PREFIX = "trainer/uncertainty_weighted_loss/"


def make_trainer() -> Trainer:
    kw = dict(ol.load_config()["constructor_kwargs"])
    model = build_nig_timenet_v2(kw.pop("seq_len"), kw.pop("n_features"), **kw)
    model.load_oracle_weights(ol.load_weights())
    trainer = Trainer(model.eval(), ol.load_config()["trainer"])
    with torch.no_grad():
        for path, v in ol.load_weights().items():
            if path.startswith(KENDALL_PREFIX):
                trainer.uncertainty_layer.log_var(path.removeprefix(KENDALL_PREFIX + "log_var_")).copy_(v)
    return trainer


def run(scenario: str):
    trainer = make_trainer()
    sc = LOSSES["scenarios"][scenario]
    trainer.scheduled_vars = dict(sc["scheduled_vars"])
    assert trainer.kendall_task_names == sc["kendall_tasks"] and trainer.fixed_task_names == sc["fixed_tasks"]
    assert trainer.loss_weights == sc["loss_weights"]
    return trainer, sc, trainer.compute_losses(ol.load_inputs(), ol.load_labels())


def value(entry) -> torch.Tensor:
    return torch.tensor(entry["value"], dtype=torch.float32)


def grads_vs_oracle(trainer: Trainer, grads: dict[str, torch.Tensor]):
    """grads: torch parameter name (model) or 'trainer/...' attr_path -> grad; compared in torch layout."""
    oracle = ol.load_grads()
    assert len(oracle) == 75
    for path, g_tf in oracle.items():
        if path.startswith(KENDALL_PREFIX):
            got = grads[path]
        else:
            got = grads[torch_param_name(path)]
        ol.assert_close(got, keras_to_torch_value(path, g_tf), "grad", path)


def named_trainable(trainer: Trainer):
    named = dict(trainer.model.named_parameters())
    for t in trainer.kendall_task_names:
        named[f"{KENDALL_PREFIX}log_var_{t}"] = trainer.uncertainty_layer.log_var(t)
    return named
