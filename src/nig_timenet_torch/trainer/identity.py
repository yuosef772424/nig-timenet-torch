"""RunIdentity: a persisted, immutable run fingerprint so resume-with-a-changed-objective fails loudly (C3;
BUILD_PLAN.md P23). Field names follow crypto-signal-prediction docs/research/hardening/audit_2026-10-31.md C3;
fresh implementation (no TF code). settings_hash covers every objective/arch/optimizer/data setting except
epochs and verbosity, so RunIdentity has no epochs field."""

import dataclasses
import hashlib
import json
from dataclasses import dataclass

__all__ = ["RunIdentity", "ResumeMismatchError", "check_resume", "settings_hash"]


class ResumeMismatchError(ValueError):
    """Saved and current RunIdentity differ outside ``allow_changes``; the message has a field-level diff."""


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

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["parent_hashes"] = list(self.parent_hashes)
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
