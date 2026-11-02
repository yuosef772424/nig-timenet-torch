"""P23: seed_everything makes the first batch and the first step reproducible."""

import torch

from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.trainer.trainer import Trainer
from tests.trainer.fixture_setup import build_model, load_fixture, trainer_config


def first_batch_and_step(seed: int):
    seed_everything(seed)
    x, y = load_fixture(128)
    trainer = Trainer(build_model(16, 18, dropout=0.1), trainer_config(),
                      optimizer=lambda ps: torch.optim.AdamW(ps, lr=1e-3))
    idx = trainer.batch_indices(x.shape[0], 32)[0]
    logs = trainer.train_step(x[idx], {k: v[idx] for k, v in y.items()})
    return idx, logs["total_loss"]


def test_seed_everything_gives_identical_first_batch():
    i1, l1 = first_batch_and_step(0)
    i2, l2 = first_batch_and_step(0)
    assert torch.equal(i1, i2) and l1 == l2
    i3, l3 = first_batch_and_step(1)
    assert not torch.equal(i1, i3) and l1 != l3
