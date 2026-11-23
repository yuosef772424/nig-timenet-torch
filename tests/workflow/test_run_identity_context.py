"""Guard of D-042 (HB2, audit 2026-11b C-03 / C-04 / C-05; rule B3): every RunSettings field that changes what a run
learns is in its run identity, and every field at its default leaves the identity of the run before D-042 unchanged.

PURPOSE:  A class guard, not an instance check: EVERY ``TrainSettings`` / ``SplitSettings`` field is set to a
  non-default value and the settings hash of a built trainer must move, unless the field is listed in
  ``NOT_IN_IDENTITY`` with its reason (and then it must NOT move); a new field fails until it is classified here.
TAGS:     run identity, settings hash, resume safety, split settings, selection metric, staged plan, guard, d-042
PITFALLS: the trainer is built through ``flat_run._trainer`` on a small GRU model with the auxiliary heads, so the
  fields read only by those heads (aux_*) reach the trainer config; ``cross_section_weight`` and ``lr_min_ratio``
  are read only when their switch is on, so their case sets the switch in both arms.
"""

import dataclasses

import pytest
import torch

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.trainer.day_weights import DayWeightSettings
from nig_timenet_torch.trainer.identity import _payload, configuration_hash, settings_hash
from nig_timenet_torch.trainer.staged import StageSettings
from nig_timenet_torch.trainer.tracker import EarlyStopSettings
from nig_timenet_torch.workflow import flat_run, run_common
from nig_timenet_torch.workflow.settings import RunSettings, SplitSettings, TrainSettings, default_model_settings

#: fields that do not change what a run learns, so a resume may change them
NOT_IN_IDENTITY = {"epochs": "a resumed run may get more epochs", "device": "where it runs, not what it learns",
                   "trace_representative": "log only (D-019)",
                   "steps_per_execution": "speed only: the same updates in the same order"}

#: field -> (non-default value, settings both arms share so the field is read)
TRAIN_CASES = {
    "epochs": (7, {}), "batch_size": (128, {}), "seed": (1, {}), "lr": (2e-3, {}), "weight_decay": (2e-4, {}),
    "label_smoothing": (0.05, {}), "selection_metric": ("val_rank_ic_reg", {}),
    "early_stop": (EarlyStopSettings(patience=3), {}), "device": ("cpu", {}), "trace_representative": (True, {}),
    "steps_per_execution": (2, {}), "grad_clip_norm": (1.0, {}), "ema_decay": (0.99, {}), "init": ("keras", {}),
    "weight_decay_groups": ("exclude_norm_bias", {}), "uncertainty_weighting": (False, {}),
    "class_loss_weight": (0.5, {}), "lambda_reg_end": (0.1, {}), "lambda_calib_end": (0.2, {}),
    "beta_nll": (0.5, {}), "beta_nll_normalize": (True, {}),
    "loss_balance": ("equal_grad_norm", {"uncertainty_weighting": False}),
    "kendall_init_logvar": (0.5, {}), "staged": (StageSettings(), {}),
    "cross_section_loss": ("ic", {"split.cross_section": True}),
    "cross_section_weight": (2.0, {"split.cross_section": True, "cross_section_loss": "ic"}),
    "direction_balance": (0.1, {}), "batch_by": ("timestamp", {}), "batch_order": ("time_blocks", {}),
    "time_block_bars": (2, {}), "day_weights": (DayWeightSettings(), {}), "aux_loss_weight": (0.5, {}),
    "aux_quantile": (0.8, {}), "aux_variance_weight": (1.0, {}), "lr_schedule": ("cosine", {}),
    "lr_min_ratio": (0.2, {"lr_schedule": "cosine"}),
}

SPLIT_CASES = {"train_end": {"train_end": "2024-01-01", "val_end": "2024-06-01"},
               "val_end": {"train_end": "2024-01-01", "val_end": "2024-06-01"},
               "train_share": {"train_share": 0.7}, "val_share": {"val_share": 0.15},
               "embargo_hours": {"embargo_hours": 4},
               "max_rows": {"max_rows": 1000}, "exclude_coins": {"exclude_coins": ("BTC",)},
               "cross_section": {"cross_section": True}, "top_coins": {"top_coins": 10},
               "coin_channel": {"cross_section": True, "coin_channel": True},
               "target_mode": {"target_mode": "scaled"}, "entry_close_reg": {"entry_close_reg": "range_pos"},
               "target_clip": {"target_clip": 5.0}}


@pytest.fixture(scope="module")
def model():
    base = default_model_settings()
    ms = ModelSettings(base.architecture, {**base.overrides, "encoder": "gru",
                                           "aux_heads": ["side_payoff", "upper_quantile"]})
    torch.manual_seed(0)
    return ms, build_model(ms, 24, 3)


def _settings(train: dict, split: dict | None = None, data: LoaderSettings | None = None) -> RunSettings:
    split = dict(split or {})
    split.update({k.removeprefix("split."): train.pop(k) for k in list(train) if k.startswith("split.")})
    return RunSettings(run_dir="unused", train=TrainSettings(**train), split=SplitSettings(**split),
                       data=data or LoaderSettings())


def _hashes(model, s: RunSettings) -> tuple[str, str]:
    ms, m = model
    tr = flat_run._trainer(s, m, tuple(TARGET_COLUMNS))
    fs = flat_run.fit_settings_of(s)
    ident = settings_hash(_payload(fs, ms, tr.config, tr.optimizer, run_common.identity_context(s)))
    gate = configuration_hash(fs, ms, tr.config, tr.optimizer, run_common.identity_context(s, selection=False))
    return ident, gate


def test_every_train_and_split_field_is_classified():
    assert set(TRAIN_CASES) == {f.name for f in dataclasses.fields(TrainSettings)}, (
        "a TrainSettings field was added or removed: classify it in TRAIN_CASES (and NOT_IN_IDENTITY if a resume may "
        "change it)")
    assert set(SPLIT_CASES) == {f.name for f in dataclasses.fields(SplitSettings)}
    assert set(run_common.IDENTITY_SELECTION["split"]) == set(SPLIT_CASES)


@pytest.mark.parametrize("name", sorted(TRAIN_CASES))
def test_a_train_field_moves_the_identity_unless_it_is_listed(model, name):
    value, shared = TRAIN_CASES[name]
    base, changed = _hashes(model, _settings(dict(shared))), _hashes(model, _settings({**shared, name: value}))
    if name in NOT_IN_IDENTITY:
        assert changed == base, f"{name} is listed as not in the identity ({NOT_IN_IDENTITY[name]}) but moves it"
    else:
        assert changed[0] != base[0], f"train.{name} changes what is learned but not the run identity"
        if name != "seed":   # the gate key covers every seed of one configuration
            assert changed[1] != base[1], f"train.{name} is configuration but not in the gate key"


@pytest.mark.parametrize("name", sorted(SPLIT_CASES))
def test_a_split_field_moves_the_identity_not_the_gate_key(model, name):
    base, changed = _hashes(model, _settings({})), _hashes(model, _settings({}, SPLIT_CASES[name]))
    assert changed[0] != base[0], f"split.{name} chooses the rows but does not move the run identity"
    assert changed[1] == base[1], f"split.{name} moved the gate key (the gates run on their own data)"


def test_input_frames_move_the_identity_and_a_default_run_has_no_context(model):
    assert run_common.identity_context(_settings({})) == {}      # the identity of a run before D-042 (B-091)
    base = _hashes(model, _settings({}))
    frames = _hashes(model, _settings({}, data=LoaderSettings(input_timeframes=("4h", "1h"))))
    swapped = _hashes(model, _settings({}, data=LoaderSettings(input_timeframes=("1h", "4h"))))
    assert len({base[0], frames[0], swapped[0]}) == 3 and base[1] == frames[1] == swapped[1]
    # loader fields that only say where / how the artifact is read stay out (the data fingerprint names the data)
    assert _hashes(model, _settings({}, data=LoaderSettings(directory="/elsewhere", mmap=False))) == base
