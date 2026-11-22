"""PURPOSE: The one table of what each TrainSettings field is FOR (per-run, job-owned, configuration, dormant) and
    the field lists every consumer (resume, study hash, gate configuration, baseline arm, job keys) derives from it.
TAGS: train settings roles, per-run fields, resumable fields, configuration fields, dormant switches, field lists
PITFALLS: a new TrainSettings field fails tests/workflow/test_settings_roles.py until it has a row here
    (``configuration`` when nothing else applies); the tags are NOT one partition: the consumers disagree on purpose
    (``trace_representative`` is per-run for the gate run but not a job key; ``batch_size`` is exempt from the baseline
    arm but is configuration for the gate report; ``steps_per_execution`` is resumable but part of a gate
    configuration), and every derived list equals the hand-kept list it replaced (the test pins them); the identity
    payload's own pops live in trainer/identity.py (FitSettings fields, not TrainSettings) and are not derived here.

Tags: ``seed`` the run's seed; ``resume`` a resume or another process may change it (speed or length only: neither the
run identity nor a rolling study hash holds it); ``gate-run`` set per environment / seed / job, so not part of a gate
configuration; ``baseline-run`` may differ from today's default configuration without making it another arm;
``job-key`` the study job's config has its own key for it (``config.train`` may not set it); ``job-set`` the gate job
sets it per environment / seed (the configuration may not); ``configuration`` none of these; ``dormant:<switch>`` read
only while ``<switch>`` (another TrainSettings field, or ``model.<override>``) is on.
"""

from dataclasses import fields

from nig_timenet_torch.workflow.settings import TrainSettings

__all__ = ["TRAIN_FIELD_ROLES", "train_fields_tagged", "RESUMABLE_TRAIN_FIELDS", "GATE_PER_RUN_FIELDS",
           "BASELINE_PER_RUN_FIELDS", "JOB_KEY_FIELDS", "JOB_SET_FIELDS", "STUDY_HASH_SKIP_FIELDS"]

_C = ("configuration",)

TRAIN_FIELD_ROLES: dict[str, tuple[str, ...]] = {
    "epochs": ("resume", "gate-run", "baseline-run", "job-set"),
    "batch_size": ("baseline-run", "job-key"),
    "seed": ("seed", "gate-run", "baseline-run", "job-key", "job-set"),
    "lr": ("job-key",),
    "weight_decay": _C,
    "label_smoothing": _C,
    "selection_metric": _C,
    "early_stop": _C,
    "device": ("resume", "gate-run", "baseline-run", "job-key", "job-set"),
    "trace_representative": ("resume", "gate-run", "baseline-run"),
    "steps_per_execution": ("resume",),
    "grad_clip_norm": _C,
    "ema_decay": _C,
    "init": _C,
    "weight_decay_groups": _C,
    "uncertainty_weighting": _C,
    "class_loss_weight": _C,
    "lambda_reg_end": _C,
    "lambda_calib_end": _C,
    "beta_nll": _C,
    "beta_nll_normalize": ("dormant:beta_nll",),
    "loss_balance": _C,
    "kendall_init_logvar": _C,
    "staged": _C,
    "cross_section_loss": _C,
    "cross_section_weight": ("dormant:cross_section_loss",),
    "base_loss_weight": ("dormant:cross_section_loss",),
    "direction_balance": _C,
    "batch_by": _C,
    "batch_order": _C,
    "time_block_bars": ("dormant:batch_order",),
    "day_weights": _C,
    "aux_loss_weight": ("dormant:model.aux_heads",),
    "aux_quantile": ("dormant:model.aux_heads",),
    "aux_variance_weight": ("dormant:model.aux_heads",),
    "lr_schedule": _C,
    "lr_min_ratio": ("dormant:lr_schedule",),
}


def train_fields_tagged(*tags: str) -> tuple[str, ...]:
    """The TrainSettings fields (in declaration order) that carry any of ``tags``."""
    order = [f.name for f in fields(TrainSettings)]
    return tuple(n for n in order if n in TRAIN_FIELD_ROLES and set(tags) & set(TRAIN_FIELD_ROLES[n]))


#: a resume may change these (run_common.identity_context, tests/workflow/test_run_identity_context.py)
RESUMABLE_TRAIN_FIELDS = train_fields_tagged("resume")
#: left out of a gate run's configuration (workflow/gate_run._configuration)
GATE_PER_RUN_FIELDS = train_fields_tagged("gate-run")
#: left at the default when asking "is this today's baseline arm" (workflow/gate_run.baseline_arm)
BASELINE_PER_RUN_FIELDS = train_fields_tagged("baseline-run")
#: a study job's ``config.train`` may not set these (workflow/study_settings.job_run_settings)
JOB_KEY_FIELDS = train_fields_tagged("job-key")
#: a gate job's configuration may not set these (jobs/gates.GatesConfig)
JOB_SET_FIELDS = train_fields_tagged("job-set")
#: left out of a rolling study hash (workflow/rolling_run.study_hash)
STUDY_HASH_SKIP_FIELDS = train_fields_tagged("resume", "seed")
