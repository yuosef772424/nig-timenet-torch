"""PURPOSE: The job-config-to-RunSettings mapping the study entries share (jobs/staged.py, jobs/rolling.py): the
    default model plus overrides, the strict train / split sections, the artifact and the smoke stride.
TAGS: job settings, study entries, run settings from config, model overrides, strict sections, smoke mode
PITFALLS: every dict section is built with the strict ``from_dict`` (an unknown key raises, rule B5); ``train`` may not
    set device, batch_size or lr (the config has its own keys, one source); a model override replaces the key in the
    default model (preset + switches), it does not start from nothing; another ``architecture`` (e.g.
    ``snapshot_panel``) starts from ITS registered default config plus the overrides (the default model's preset
    belongs to nig_timenet_v2).
"""

from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.model.architectures import ModelSettings
from nig_timenet_torch.workflow.settings import (
    EvalSettings,
    RunSettings,
    SplitSettings,
    TrainSettings,
    default_model_settings,
)

__all__ = ["job_run_settings"]

_OWN_KEYS = ("device", "batch_size", "lr", "seed")


def job_run_settings(*, device: str, run_dir: str, artifact: str, base_tf: str | None, max_rows: int | None,
                     batch_size: int, lr: float, model_overrides: dict, train: dict, split: dict,
                     code_sha: str = "", architecture: str | None = None, input_timeframes: tuple = ()) -> RunSettings:
    """The RunSettings of a study job: everything the config does not name is the RunSettings default.
    ``architecture`` None = the default model's."""
    bad = sorted(set(train) & set(_OWN_KEYS))
    if bad:
        raise ValueError(f"config.train may not set {bad}: the config has its own keys for them")
    base = default_model_settings()
    arch = base.architecture if architecture is None else architecture
    model = (ModelSettings(arch, {**base.overrides, **model_overrides}) if arch == base.architecture
             else ModelSettings(arch, dict(model_overrides)))
    return RunSettings(
        run_dir=run_dir, code_sha=code_sha,
        data=LoaderSettings(name=artifact, mmap=True, expect_base_tf=base_tf, input_timeframes=tuple(input_timeframes)),
        split=from_dict(SplitSettings, {**split, "max_rows": max_rows}),
        model=model,
        train=from_dict(TrainSettings, {**train, "device": device, "batch_size": batch_size, "lr": lr}),
        evaluation=EvalSettings())
