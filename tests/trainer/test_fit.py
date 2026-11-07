"""T1: Trainer.fit on the fixture - epoch loop, schedules applied, early stopping, best weights by the registered
selection metric, checkpoint + resume (equal to an uninterrupted run), identity check, seeds."""

import copy
import dataclasses
import json

import pytest
import torch

from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.trainer import checkpoint
from nig_timenet_torch.trainer.identity import (
    ResumeMismatchError,
    RunIdentity,
    SeedMismatchError,
    build_run_identity,
)
from nig_timenet_torch.trainer.schedules import LrSettings, lr_at
from nig_timenet_torch.trainer.tracker import EarlyStopSettings
from nig_timenet_torch.trainer.trainer import FitSettings, Trainer
from tests.trainer.fixture_setup import build_model, trainer_config
from tests.trainer.helpers import fit, identity_for, make_trainer, state_equal

N_TRAIN, N_VAL = 96, 48
LR = LrSettings(lr_initial=3e-3, lr_min=1e-5, warmup_epochs=1, kind="cosine", total_epochs=3)


def test_three_epochs_run_and_apply_the_schedules(data):
    settings = FitSettings(epochs=3, batch_size=32, lr=LR, early_stop=EarlyStopSettings(restore_best_weights=False))
    t = make_trainer()
    result = fit(t, data, settings)
    assert result.epochs_done == 3 and [h["epoch"] for h in result.history] == [1, 2, 3]
    assert [h["lr"] for h in result.history] == [lr_at(LR, e) for e in range(3)]
    # the trainer config schedules lambda_reg 0 -> 0.05 and lambda_calib 0 -> 0.1 over 5 epochs: epoch index 2 of 5
    assert t.scheduled_vars["lambda_reg"] == pytest.approx(0.05 * 2 / 5)
    assert 0 < t.scheduled_vars["lambda_calib"] < 0.1
    row = result.history[-1]
    assert {"train_total_loss", "val_total_loss", "val_raw_loss", "val_selection_metric"} <= set(row)
    assert all(torch.isfinite(torch.tensor(v)) for k, v in row.items() if k != "epoch")
    assert result.history[-1]["val_total_loss"] < result.history[0]["val_total_loss"]


def test_resume_after_epoch_one_equals_an_uninterrupted_run(data, tmp_path):
    settings = FitSettings(epochs=3, batch_size=32, lr=LR)
    ident = identity_for(settings)
    full = make_trainer()
    full_result = fit(full, data, settings, run_dir=tmp_path / "full", identity=ident)

    first = make_trainer()
    fit(first, data, dataclasses.replace(settings, epochs=1), run_dir=tmp_path / "split", identity=ident)
    assert checkpoint.load_checkpoint(tmp_path / "split")["epochs_done"] == 1
    # a NEW process: other init seed, new optimizer; everything must come from the checkpoint
    second = make_trainer(init_seed=123)
    resumed = fit(second, data, settings, run_dir=tmp_path / "split", identity=ident)

    assert resumed.epochs_done == 3 and resumed.history == full_result.history
    assert state_equal(second.model, full.model)
    for name in full.uncertainty_layer.task_names:
        assert torch.equal(full.uncertainty_layer.log_var(name), second.uncertainty_layer.log_var(name))
    a, b = (checkpoint.load_checkpoint(tmp_path / d) for d in ("full", "split"))
    assert a["tracker"] == b["tracker"] and a["history"] == b["history"]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


def test_resume_refuses_a_changed_identity_and_an_overwrite(data, tmp_path):
    settings = FitSettings(epochs=1, batch_size=32)
    fit(make_trainer(), data, settings, run_dir=tmp_path, identity=identity_for(settings))
    other = dataclasses.replace(settings, batch_size=16)
    with pytest.raises(ResumeMismatchError, match="settings_hash"):
        fit(make_trainer(), data, other, run_dir=tmp_path, identity=identity_for(other))
    changed_data = dataclasses.replace(identity_for(settings), data_fingerprint="other")
    with pytest.raises(ResumeMismatchError, match="data_fingerprint"):
        fit(make_trainer(), data, settings, run_dir=tmp_path, identity=changed_data)
    with pytest.raises(FileExistsError):
        fit(make_trainer(), data, settings, run_dir=tmp_path, identity=identity_for(settings), resume=False)
    with pytest.raises(ValueError, match="identity"):
        fit(make_trainer(), data, settings, run_dir=tmp_path / "x")


def test_more_epochs_do_not_change_the_identity(data, tmp_path):
    """`epochs` is not part of the run identity, so `fit(epochs=1)` then `fit(epochs=2)` is a legal resume."""
    s1 = FitSettings(epochs=1, batch_size=32)
    s2 = dataclasses.replace(s1, epochs=2)
    assert identity_for(s1) == identity_for(s2)
    fit(make_trainer(), data, s1, run_dir=tmp_path, identity=identity_for(s1))
    result = fit(make_trainer(), data, s2, run_dir=tmp_path, identity=identity_for(s1))
    assert result.epochs_done == 2


def test_early_stopping_stops_and_restores_the_best_weights(data, tmp_path):
    # min_delta is huge, so epoch 1 is the best and every later epoch is a wasted one
    settings = FitSettings(epochs=6, batch_size=32, early_stop=EarlyStopSettings(patience=2, min_delta=1e9,
                                                                                   smoothing="none"))
    t = make_trainer()
    result = fit(t, data, settings, run_dir=tmp_path, identity=identity_for(settings))
    assert result.stopped_early and result.epochs_done == 3 and result.best_epoch == 1 and result.restored_best
    best = checkpoint.load_best(tmp_path)
    assert all(torch.equal(best[k], v) for k, v in t.model.state_dict().items())
    assert json.loads((tmp_path / checkpoint.BEST_META).read_text())["metric"] == "val_raw_loss"
    # the checkpoint keeps the LAST epoch's weights, not the best ones (so a resume equals an uninterrupted run)
    last = checkpoint.load_checkpoint(tmp_path)["model"]
    assert not all(torch.equal(best[k], last[k]) for k in best)
    # a finished (early-stopped) run resumed with more epochs stays finished
    again = fit(make_trainer(), data, dataclasses.replace(settings, epochs=10), run_dir=tmp_path,
                identity=identity_for(settings))
    assert again.epochs_done == 3 and again.stopped_early


def test_best_epoch_is_chosen_by_the_registered_selection_metric(data):
    """A metric with higher_is_better=True (AUC) is maximised, not minimised."""
    settings = FitSettings(epochs=3, batch_size=32, early_stop=EarlyStopSettings(smoothing="none", min_delta=0.0))
    t = make_trainer(selection_metric="val_auc_close_class")
    result = fit(t, data, settings)
    aucs = [h["val_selection_metric"] for h in result.history]
    assert result.best_epoch == 1 + aucs.index(max(aucs)) and result.best_value == max(aucs)
    with pytest.raises(KeyError):
        make_trainer(selection_metric="val_hit_rate")


def test_same_seed_same_run_different_seed_different_run(data):
    settings = FitSettings(epochs=2, batch_size=32)
    a, b, c = make_trainer(), make_trainer(), make_trainer()
    ha, hb = fit(a, data, settings).history, fit(b, data, settings).history
    hc = fit(c, data, dataclasses.replace(settings, seed=1)).history
    assert ha == hb and state_equal(a.model, b.model)
    assert hc != ha


def test_fit_needs_an_optimizer_and_valid_settings(data):
    t = Trainer(build_model(16, 18), trainer_config())
    with pytest.raises(RuntimeError, match="optimizer"):
        fit(t, data, FitSettings(epochs=1))
    with pytest.raises(ValueError):
        FitSettings(epochs=0)


def test_fit_refuses_an_identity_with_another_seed(data, tmp_path):
    """T2: identity.seed must equal settings.seed; checked before anything runs, with or without a run folder."""
    settings = FitSettings(epochs=1, batch_size=32, seed=3)
    wrong = identity_for(settings, seed=9)
    with pytest.raises(SeedMismatchError, match=r"identity\.seed \(9\) != settings\.seed \(3\)"):
        fit(make_trainer(), data, settings, identity=wrong)
    with pytest.raises(SeedMismatchError):
        fit(make_trainer(), data, settings, run_dir=tmp_path / "r", identity=wrong)
    assert not (tmp_path / "r").exists()
    assert fit(make_trainer(), data, settings, identity=identity_for(settings, seed=3)).epochs_done == 1
    assert fit(make_trainer(), data, settings).epochs_done == 1  # no identity: nothing to compare


def real_identity(trainer: Trainer, settings: FitSettings) -> RunIdentity:
    """The identity a real run gets: built from the trainer's whole config and its optimizer (T2 review)."""
    return build_run_identity(settings, ModelSettings(), trainer.config, trainer.optimizer, "fixture",
                              code_sha="0" * 40)


def test_resume_refuses_a_changed_loss_weight_or_optimizer_lr(data, tmp_path):
    """T2 review: the identity covers the trainer config and the optimizer, so a run whose objective or learning rate
    changed cannot resume as the same run; an unchanged one still resumes (and may get more epochs)."""
    settings = FitSettings(epochs=1, batch_size=32)
    first = make_trainer()
    fit(first, data, settings, run_dir=tmp_path, identity=real_identity(first, settings))

    more = dataclasses.replace(settings, epochs=2)
    same = make_trainer(init_seed=5)
    assert fit(same, data, more, run_dir=tmp_path, identity=real_identity(same, more)).epochs_done == 2

    heavier = copy.deepcopy(trainer_config())
    target = next(iter(heavier["targets"]))
    heavier["targets"][target]["loss_weight"] = heavier["targets"][target].get("loss_weight", 1.0) + 1.0
    for kwargs in ({"config": heavier}, {"lr": 2e-3}):
        t = make_trainer(**kwargs)
        ident = real_identity(t, dataclasses.replace(settings, epochs=3))
        with pytest.raises(ResumeMismatchError, match="settings_hash"):
            fit(t, data, dataclasses.replace(settings, epochs=3), run_dir=tmp_path, identity=ident)
    assert checkpoint.load_checkpoint(tmp_path)["epochs_done"] == 2  # the refused resumes touched nothing
