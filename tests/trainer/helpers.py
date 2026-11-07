"""Shared trainer-test helpers (HF-32). The ``data`` fixture is registered in tests/trainer/conftest.py."""

import dataclasses

import pytest  # noqa: F401
import torch

from nig_timenet_torch.core.seeds import seed_everything
from nig_timenet_torch.trainer.identity import RunIdentity, settings_hash
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer
from tests.trainer.fixture_setup import build_model, load_fixture, trainer_config

N_TRAIN, N_VAL = 96, 48


@pytest.fixture(scope="module")
def data():
    x, y = load_fixture(N_TRAIN + N_VAL)
    cut = N_TRAIN
    return x[:cut], {k: v[:cut] for k, v in y.items()}, x[cut:], {k: v[cut:] for k, v in y.items()}


def make_trainer(selection_metric="val_raw_loss", init_seed=0, config=None, lr=1e-3) -> Trainer:
    seed_everything(init_seed)  # model init draws from the global RNG
    model = build_model(16, 18, dropout=0.1)
    return Trainer(model, trainer_config() if config is None else config,
                   optimizer=lambda ps: torch.optim.AdamW(ps, lr=lr), selection_metric=selection_metric)


def identity_for(settings: FitSettings, seed=0) -> RunIdentity:
    return RunIdentity(schema_version=1, settings_hash=settings_hash(dataclasses.asdict(settings)), seed=seed,
                       data_fingerprint="fixture", code_sha="0" * 40)


def fit(trainer, data, settings, **kw):
    return trainer.fit(*data, settings, **kw)


def state_equal(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
