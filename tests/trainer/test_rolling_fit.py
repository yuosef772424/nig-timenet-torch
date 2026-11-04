"""fit_fold: per-fold run folder, identity and seed; fresh and warm starts."""

import pytest
import torch

from nig_timenet_torch.trainer.identity import RunIdentity
from nig_timenet_torch.trainer.rolling_fit import fit_fold, fold_dir
from nig_timenet_torch.trainer.tracker import EarlyStopSettings
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 1)

    def forward(self, x):
        return {"y_reg": self.lin(x)[:, 0]}


def _make(seed):
    return Trainer(_Net(), {"targets": {"r": {"true_key": "y", "task_type": "regression",
                                              "output_keys": {"pred": "y_reg"}}}, "loss": {}},
                   optimizer=lambda ps: torch.optim.SGD(ps, lr=0.05))


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(64, 3, generator=g)
    y = {"y": x @ torch.tensor([1.0, -2.0, 0.5])}
    return x[:48], y_cut(y, 0, 48), x[48:], y_cut(y, 48, 64)


def y_cut(y, a, b):
    return {k: v[a:b] for k, v in y.items()}


def _identity(fold, seed=0):
    return RunIdentity(1, "h", seed, "fp", "sha", fold=fold, data_cut="2025-01-01")


SETTINGS = FitSettings(epochs=3, batch_size=16, early_stop=EarlyStopSettings(patience=10))


def test_each_fold_gets_its_own_run_folder_and_identity_seed(tmp_path):
    t0, r0 = fit_fold(_make, _data(), SETTINGS, _identity(0, 5), run_dir=tmp_path)
    assert r0.epochs_done == 3 and any(fold_dir(tmp_path, 0, 5).iterdir())
    t1, _ = fit_fold(_make, _data(1), SETTINGS, _identity(1, 5), run_dir=tmp_path)
    assert fold_dir(tmp_path, 1, 5).exists() and t0 is not t1
    # the same fold again resumes from its checkpoint instead of retraining
    again, r = fit_fold(_make, _data(), SETTINGS, _identity(0, 5), run_dir=tmp_path)
    assert r.epochs_done == 3


def test_fresh_folds_are_independent_and_reproducible():
    a, _ = fit_fold(_make, _data(), SETTINGS, _identity(0, 1))
    b, _ = fit_fold(_make, _data(), SETTINGS, _identity(0, 1))
    c, _ = fit_fold(_make, _data(), SETTINGS, _identity(0, 2))
    assert torch.equal(a.model.lin.weight, b.model.lin.weight)
    assert not torch.equal(a.model.lin.weight, c.model.lin.weight)


def test_warm_start_continues_the_previous_trainer_for_warm_epochs():
    first, _ = fit_fold(_make, _data(), SETTINGS, _identity(0), init="warm", warm_epochs=1)
    w = first.model.lin.weight.detach().clone()
    second, res = fit_fold(_make, _data(1), SETTINGS, _identity(1), previous=first, init="warm", warm_epochs=1)
    assert second is first and res.epochs_done == 1 and not torch.equal(w, second.model.lin.weight)
    fresh, res = fit_fold(_make, _data(1), SETTINGS, _identity(1), previous=first, init="fresh")
    assert fresh is not first and res.epochs_done == 3
    full, res = fit_fold(_make, _data(1), SETTINGS, _identity(2), previous=first, init="warm")
    assert res.epochs_done == 3                                      # warm_epochs 0 = the full epochs


def test_bad_init_is_refused():
    with pytest.raises(ValueError, match="init must be one of"):
        fit_fold(_make, _data(), SETTINGS, _identity(0), init="cold")
