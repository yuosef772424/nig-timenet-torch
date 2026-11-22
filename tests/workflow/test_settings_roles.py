"""The role table classifies every TrainSettings field and reproduces the hand-kept lists it replaced (C-15)."""

from dataclasses import fields

from nig_timenet_torch.jobs import gates as gates_job
from nig_timenet_torch.workflow import gate_run, run_common, study_settings
from nig_timenet_torch.workflow import settings_roles as roles
from nig_timenet_torch.workflow.settings import TrainSettings

_KNOWN = {"seed", "resume", "gate-run", "baseline-run", "job-key", "job-set", "configuration"}


def test_every_train_field_is_classified_and_nothing_else():
    assert set(roles.TRAIN_FIELD_ROLES) == {f.name for f in fields(TrainSettings)}, \
        "a new TrainSettings field needs a row in workflow/settings_roles.py (``configuration`` if nothing applies)"


def test_tags_are_known_and_dormant_switches_exist():
    names = {f.name for f in fields(TrainSettings)}
    for field, tags in roles.TRAIN_FIELD_ROLES.items():
        assert tags, field
        for t in tags:
            if t.startswith("dormant:"):
                switch = t.split(":", 1)[1]
                assert switch in names or switch.startswith("model."), (field, t)
            else:
                assert t in _KNOWN, (field, t)
        assert ("configuration" in tags) == (len(tags) == 1 and tags[0] == "configuration")


def test_derived_lists_equal_the_hand_kept_lists():
    # the lists as they were written by hand before the table (membership lists: order is not read)
    assert roles.RESUMABLE_TRAIN_FIELDS == ("epochs", "device", "trace_representative", "steps_per_execution")
    assert run_common.RESUMABLE_TRAIN_FIELDS == roles.RESUMABLE_TRAIN_FIELDS
    assert set(gate_run._PER_RUN_KEYS) == {"seed", "epochs", "device", "trace_representative"}
    assert set(roles.BASELINE_PER_RUN_FIELDS) == {"seed", "epochs", "batch_size", "device", "trace_representative"}
    assert set(study_settings._OWN_KEYS) == {"device", "batch_size", "lr", "seed"}
    assert set(gates_job._NOT_CONFIGURATION) == {"seed", "epochs", "device"}
    assert set(roles.STUDY_HASH_SKIP_FIELDS) == {"epochs", "device", "trace_representative", "steps_per_execution",
                                                 "seed"}
