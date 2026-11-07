"""PURPOSE: The trainer config schema: ``validate_trainer_config`` (unknown keys raise), ``resolve_targets`` /
    ``resolve_loss`` (defaults filled once) and ``targets_config``, the ONE builder of the targets (from HeadSpec).
TAGS: trainer config, strict config, validate, targets_config, HeadSpec, task config_keys, loss defaults, did you mean
PITFALLS: validation and resolution NEVER change the dict they read: ``Trainer.config`` is hashed into the RunIdentity,
    so a default filled into it would orphan every existing checkpoint (tests/trainer/test_trainer_config_pins.py).
    ``LOSS_DEFAULTS`` / ``TARGET_DEFAULTS`` are pinned defaults (decision file); a task option's default lives in
    ``TASK_REGISTRY`` (``default_options``), not here.

Trainer config shape: ``{"targets": {name: target}, "loss": {...}, optional "constraints": [...]}``. A target needs
``true_key``, ``task_type`` and ``output_keys``; it may carry ``loss_weight``, ``flood_level`` and the options
registered as the task's ``config_keys``. ``loss`` may carry ``use_uncertainty_weighting`` (Kendall),
``uncertainty_weighting_exclude_task_types`` (Kendall weights only the OTHER task types; the excluded ones are added
unweighted), ``schedules`` and ``constraints``. Every lambda variable a target names (``lambda_reg_var``,
``lambda_calib_var``) must be a scheduled variable.
"""

import difflib

from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.trainer.tasks import TASK_REGISTRY, with_defaults

__all__ = ["validate_trainer_config", "resolve_targets", "resolve_loss", "targets_config", "LOSS_DEFAULTS",
           "TARGET_DEFAULTS"]

TOP_KEYS = ("targets", "loss", "constraints")
LOSS_KEYS = ("use_uncertainty_weighting", "uncertainty_weighting_exclude_task_types", "schedules", "constraints")
TARGET_REQUIRED = ("true_key", "task_type", "output_keys")
TARGET_OPTIONAL = ("loss_weight", "flood_level")
#: the value of each ``loss`` key a config leaves out (the TF trainer's ``.get`` defaults)
LOSS_DEFAULTS = {"use_uncertainty_weighting": False, "uncertainty_weighting_exclude_task_types": ["evidential"],
                 "schedules": {}, "constraints": []}
#: the value of each target-level key a config leaves out
TARGET_DEFAULTS = {"loss_weight": 1.0, "flood_level": None}


def _unknown(where: str, keys, known) -> ValueError:
    parts = []
    for k in keys:
        close = difflib.get_close_matches(str(k), known, n=1)
        parts.append(f"{k!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    return ValueError(f"{where}: unknown key(s) {', '.join(parts)}. Known keys: {sorted(known)}")


def _check_keys(where: str, d, known) -> None:
    if not isinstance(d, dict):
        raise TypeError(f"{where} must be a dict, got {type(d).__name__}")
    extra = [k for k in d if k not in known]
    if extra:
        raise _unknown(where, extra, known)


def validate_trainer_config(cfg: dict) -> None:
    """Raise ValueError (unknown key, missing key, a lambda variable that is not scheduled) or UnknownNameError (task
    type) for a malformed trainer config; return None and change nothing otherwise."""
    _check_keys("trainer config", cfg, TOP_KEYS)
    for k in ("targets", "loss"):
        if k not in cfg:
            raise ValueError(f"trainer config lacks {k!r}")
    _check_keys("trainer config['loss']", cfg["loss"], LOSS_KEYS)
    schedules = cfg["loss"]["schedules"] if "schedules" in cfg["loss"] else LOSS_DEFAULTS["schedules"]
    _check_keys("trainer config['loss']['schedules']", schedules, set(schedules))  # must be a dict
    for name, target in cfg["targets"].items():
        where = f"trainer config target {name!r}"
        if not isinstance(target, dict):
            raise TypeError(f"{where} must be a dict, got {type(target).__name__}")
        missing = [k for k in TARGET_REQUIRED if k not in target]
        if missing:
            raise ValueError(f"{where} lacks {missing}")
        entry = TASK_REGISTRY.entry(target["task_type"])  # unknown task type raises UnknownNameError
        _check_keys(where + f" (task_type {target['task_type']!r})", target,
                    {*TARGET_REQUIRED, *TARGET_OPTIONAL, *entry.meta["config_keys"]})
        resolved = with_defaults(target["task_type"], target)
        used = [v for v in ("lambda_reg_var", "lambda_calib_var") if v in resolved
                and (v == "lambda_reg_var" or resolved["use_calibration_loss"])]
        for var in used:
            if resolved[var] not in schedules:
                raise ValueError(f"{where}: {var}={resolved[var]!r} is not a scheduled variable; "
                                 f"loss.schedules has {sorted(schedules)}")


def resolve_targets(cfg: dict) -> dict:
    """{target name: target config with the base and task defaults filled}; a new dict, ``cfg`` is not changed."""
    return {name: {**TARGET_DEFAULTS, **with_defaults(t["task_type"], t)} for name, t in cfg["targets"].items()}


def resolve_loss(cfg: dict) -> dict:
    """``cfg['loss']`` with ``LOSS_DEFAULTS`` filled, plus all constraints (loss and top level): ``all_constraints``."""
    loss = {**LOSS_DEFAULTS, **cfg["loss"]}
    loss["all_constraints"] = [*loss["constraints"], *(cfg["constraints"] if "constraints" in cfg else [])]
    return loss


def targets_config(head_types: dict, *, options_by_kind: dict | None = None) -> dict:
    """{head name: target config} for ``head_types = {price target: [head kind, ...]}``, every entry derived from the
    kind's HeadSpec (``true_key`` = label key, ``task_type``, ``output_keys``) plus ``options_by_kind[kind]``
    (per-target options such as ``label_smoothing``); an option the head's task does not register raises."""
    options = options_by_kind or {}
    used = {kind for kinds in head_types.values() for kind in kinds}
    if set(options) - used:
        raise ValueError(f"options_by_kind names head kind(s) {sorted(set(options) - used)} "
                         f"that head_types does not use")
    out = {}
    for target, kinds in head_types.items():
        for kind in kinds:
            spec = head_spec(kind, target)
            if spec.name in out:
                raise ValueError(f"two heads share the name {spec.name!r}")
            cfg = {"true_key": spec.label_key, "task_type": spec.task_type, "output_keys": spec.output_keys,
                   **(options[kind] if kind in options else {})}
            _check_keys(f"options of head {spec.name!r} (task_type {spec.task_type!r})", cfg,
                        {*TARGET_REQUIRED, *TARGET_OPTIONAL, *TASK_REGISTRY.entry(spec.task_type).meta["config_keys"]})
            out[spec.name] = cfg
    return out
