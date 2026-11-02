"""P22: Trainer.train_step / eval_step on the fixture (sanity, not parity: no TF oracle for a multi-step loop)."""

import torch

from nig_timenet_torch.core.seeds import set_determinism
from nig_timenet_torch.trainer.trainer import Trainer
from tests.trainer.fixture_setup import build_model, load_fixture, trainer_config


def test_train_step_reduces_loss_on_fixture():
    set_determinism(0)
    x, y = load_fixture(64)
    model = build_model(16, 18)
    trainer = Trainer(model, trainer_config(), optimizer=lambda ps: torch.optim.AdamW(ps, lr=3e-3))
    first = trainer.eval_step(x, y)["total_loss"]
    for _ in range(30):
        logs = trainer.train_step(x, y)
    assert set(logs) >= {"total_loss", "raw_loss", "close_reg_loss", "close_class_accuracy"}
    assert trainer.eval_step(x, y)["total_loss"] < first


def test_optimizer_instance_gets_kendall_params():
    model = build_model(16, 18)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, trainer_config(), optimizer=opt)
    ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) in ids for p in trainer.uncertainty_layer.parameters())
    x, y = load_fixture(16)
    before = trainer.uncertainty_layer.log_var("close_class").detach().clone()
    trainer.train_step(x, y)
    assert not torch.equal(before, trainer.uncertainty_layer.log_var("close_class").detach())
