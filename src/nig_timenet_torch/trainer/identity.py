"""PURPOSE: RunIdentity: the persisted, immutable fingerprint of a run (settings hash incl. the WHOLE trainer config,
    seed, data fingerprint, code sha); ``check_resume`` refuses a resume whose identity changed.
TAGS: run identity, fingerprint, resume, settings hash, checkpoint, ResumeMismatchError
PITFALLS: the trainer config is hashed AS GIVEN: a default added to it (or a key renamed) changes every identity and
    orphans existing runs; tests/trainer/test_trainer_config_pins.py pins the baseline fingerprint.

RunIdentity: a persisted, immutable run fingerprint so resume-with-a-changed-objective fails loudly (C3;
BUILD_PLAN.md P23). Field names follow crypto-signal-prediction docs/research/hardening/audit_2026-10-31.md C3;
fresh implementation (no TF code). settings_hash covers every objective/arch/optimizer/data setting except
epochs and verbosity, so RunIdentity has no epochs field."""

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import torch

__all__ = ["RunIdentity", "ResumeMismatchError", "SeedMismatchError", "build_run_identity", "check_resume",
           "optimizer_settings", "settings_hash"]

IDENTITY_SCHEMA_VERSION = 1


class ResumeMismatchError(ValueError):
    """Saved and current RunIdentity differ outside ``allow_changes``; the message has a field-level diff."""


class SeedMismatchError(ValueError):
    """``RunIdentity.seed`` differs from the seed the run is configured with (``FitSettings.seed``)."""


def settings_hash(settings: dict, exclude: tuple[str, ...] = ("epochs", "verbose", "verbosity")) -> str:
    """sha256 of the canonical JSON of ``settings`` without the top-level keys in ``exclude``."""
    kept = {k: v for k, v in settings.items() if k not in exclude}
    return hashlib.sha256(json.dumps(kept, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class RunIdentity:
    schema_version: int
    settings_hash: str
    seed: int
    data_fingerprint: str
    code_sha: str
    parent_hashes: tuple[str, ...] = ()
    # P37 rolling seam: set only for a rolling-study fold; None is omitted from to_json, so a single-split
    # identity (and its fingerprint) is unchanged
    fold: int | None = None
    data_cut: str | None = None

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["parent_hashes"] = list(self.parent_hashes)
        for k in ("fold", "data_cut"):
            if d[k] is None:
                del d[k]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "RunIdentity":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown RunIdentity fields: {sorted(unknown)}")
        d = dict(d)
        if "parent_hashes" in d:  # absent -> the dataclass default ()
            d["parent_hashes"] = tuple(d["parent_hashes"])
        return cls(**d)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()


def check_resume(saved: RunIdentity, current: RunIdentity, allow_changes: tuple[str, ...] = ("epochs",)) -> None:
    """Raises ResumeMismatchError with a field-level diff if any field outside ``allow_changes`` differs.
    RunIdentity has no ``epochs`` field (settings_hash excludes it), so the default allow-list is a no-op kept for
    parity with the TF-side design and for future fields."""
    a, b = saved.to_json(), current.to_json()
    diff = {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k) and k not in allow_changes}
    if diff:
        lines = "\n".join(f"  {k}: saved={s!r} current={c!r}" for k, (s, c) in diff.items())
        raise ResumeMismatchError(f"cannot resume: run identity changed\n{lines}")


def _as_dict(obj, what: str) -> dict:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    raise TypeError(f"{what} must be a dataclass instance or a mapping, got {type(obj).__name__}")


def optimizer_settings(optimizer) -> dict:
    """What identifies an optimizer for a RunIdentity: its class, its ``defaults`` and, for EVERY param group, all
    hyper-parameters (lr, weight_decay, betas, ...) plus the number of tensors in it (a group added or dropped changes
    it). Tensors themselves are not part of it. Build the identity from the optimizer BEFORE ``fit`` (``fit`` rewrites
    the group lr each epoch when ``FitSettings.lr`` is set)."""
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(f"optimizer must be a torch.optim.Optimizer, got {type(optimizer).__name__}")
    cls = type(optimizer)
    groups = [{**{k: v for k, v in g.items() if k != "params"}, "n_params": len(g["params"])}
              for g in optimizer.param_groups]
    for where, hp in [("defaults", optimizer.defaults), *[(f"param_groups[{i}]", g) for i, g in enumerate(groups)]]:
        absent = [k for k in ("lr", "weight_decay") if k not in hp]
        if absent:
            raise ValueError(f"{cls.__name__}.{where} has no {absent}: the run identity needs lr and weight_decay")
    return {"type": f"{cls.__module__}.{cls.__qualname__}", "defaults": dict(optimizer.defaults),
            "param_groups": groups}


def build_run_identity(fit_settings, model_settings, trainer_config, optimizer, data_fingerprint: str, *,
                       code_sha: str) -> RunIdentity:
    """The one place a RunIdentity is built for a real run (ROADMAP T2).

    ``fit_settings``: a ``FitSettings``; its ``epochs`` and ``trace`` are left out of the hash (a resumed run may get
    more epochs; the trace is log-only) and its ``seed`` becomes ``identity.seed`` (``Trainer.fit`` raises
    ``SeedMismatchError`` if they ever differ).
    ``model_settings``: the model's settings (``ModelSettings`` or a plain dict of the model config).
    ``trainer_config``: the WHOLE ``tasks_cfg`` the Trainer is built with (targets with their loss weights and task
    types, loss with its schedules and Kendall switch, constraints); it must be plain JSON data, so a value that would
    be hashed through ``repr`` is refused. ``optimizer``: the ``torch.optim.Optimizer`` (see ``optimizer_settings``).
    ``data_fingerprint``: ``ArtifactManifest.data_fingerprint`` of the artifact the run trains on.
    ``code_sha``: the git commit of this repository.
    All of these are required: a run whose objective or optimizer changed must not resume as the same run.
    """
    fit = _as_dict(fit_settings, "fit_settings")
    fit.pop("epochs", None)
    fit.pop("trace", None)  # log-only (T4): switching the trace on must not change a run's identity
    model = _as_dict(model_settings, "model_settings")
    trainer = _as_dict(trainer_config, "trainer_config")
    try:
        json.dumps(trainer, sort_keys=True)
    except (TypeError, ValueError) as e:
        raise TypeError(f"trainer_config must be plain JSON data (it is hashed into the run identity): {e}") from e
    if not isinstance(data_fingerprint, str) or not data_fingerprint:
        raise ValueError("data_fingerprint must be the non-empty fingerprint string from the artifact manifest")
    if not isinstance(code_sha, str) or not code_sha:
        raise ValueError("code_sha must be the non-empty git commit of this repository")
    payload = {"fit": fit, "model": model, "trainer": trainer, "optimizer": optimizer_settings(optimizer)}
    return RunIdentity(schema_version=IDENTITY_SCHEMA_VERSION, settings_hash=settings_hash(payload),
                       seed=int(fit["seed"]), data_fingerprint=data_fingerprint, code_sha=code_sha)
