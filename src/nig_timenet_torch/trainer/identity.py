"""RunIdentity: a persisted, immutable run fingerprint so resume-with-a-changed-objective fails loudly (C3;
BUILD_PLAN.md P23). Field names follow crypto-signal-prediction docs/research/hardening/audit_2026-10-31.md C3;
fresh implementation (no TF code). settings_hash covers every objective/arch/optimizer/data setting except
epochs and verbosity, so RunIdentity has no epochs field."""

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["RunIdentity", "ResumeMismatchError", "SeedMismatchError", "build_run_identity", "check_resume",
           "settings_hash"]

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
        d["parent_hashes"] = tuple(d.get("parent_hashes", ()))
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


def build_run_identity(fit_settings, model_settings, data_fingerprint: str, *, code_sha: str) -> RunIdentity:
    """The one place a RunIdentity is built for a real run (ROADMAP T2).

    ``fit_settings``: a ``FitSettings``; its ``epochs`` is left out of the hash (a resumed run may get more epochs) and
    its ``seed`` becomes ``identity.seed`` (``Trainer.fit`` raises ``SeedMismatchError`` if they ever differ).
    ``model_settings``: the model's settings (``ModelSettings`` or a plain dict of the model config).
    ``data_fingerprint``: ``ArtifactManifest.data_fingerprint`` of the artifact the run trains on.
    ``code_sha``: the git commit of this repository.
    """
    fit = _as_dict(fit_settings, "fit_settings")
    fit.pop("epochs", None)
    model = _as_dict(model_settings, "model_settings")
    if not isinstance(data_fingerprint, str) or not data_fingerprint:
        raise ValueError("data_fingerprint must be the non-empty fingerprint string from the artifact manifest")
    if not isinstance(code_sha, str) or not code_sha:
        raise ValueError("code_sha must be the non-empty git commit of this repository")
    return RunIdentity(schema_version=IDENTITY_SCHEMA_VERSION,
                       settings_hash=settings_hash({"fit": fit, "model": model}), seed=int(fit["seed"]),
                       data_fingerprint=data_fingerprint, code_sha=code_sha)
