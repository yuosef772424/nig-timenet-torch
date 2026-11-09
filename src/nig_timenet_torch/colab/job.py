"""PURPOSE: The job a Colab run executes: ``job.json`` on the ``colab-queue`` branch, parsed and validated, including
    the entry's own strict config.
TAGS: colab job, job validation, queue branch, job config, expected outputs
PITFALLS: validating the config imports the entry module (jobs.<entry>, torch included) and builds its ``CONFIG``
    dataclass strictly; an entry without ``CONFIG`` is refused; ``check-job`` runs all of this.

The job a Colab run executes: ``job.json`` on the ``colab-queue`` branch (ROADMAP T3).

Same idea as the TF runner (queue branch -> run -> results branch), with the two T3 changes built into the schema:
``device`` is required (rule E1: the run asserts and records it) and the entry is a module of
``nig_timenet_torch.jobs`` run by name, so nothing is looked up under a temporary directory (rule E2).

job.json (every key required except ``rerun`` and ``on_existing``):
    {"job_id": "smoke-001", "repo_ref": "<40-hex commit>", "entry": "device_smoke", "args": [], "config": {},
     "max_minutes": 5, "device": "cuda", "needs_drive": false, "expected_outputs": ["smoke.json"], "note": "free text",
     "rerun": false, "on_existing": "refuse"}
``on_existing`` says what happens when ``runs/<job_id>/`` already exists on the results branch: ``"refuse"`` (default,
the behaviour before D-021) or ``"new_version"``: the job runs as ``<job_id>-v2`` (then -v3, ...), the next free id,
with its own bundle folder and its own ``config.run_dir`` (the ``/<job_id>/`` segment of the path is renamed), so a
new version never resumes or overwrites an old one. ``versioned`` builds that job; it records ``base_job_id``.
The entry runs as ``python -m nig_timenet_torch.jobs.<entry> --config <json> --out <dir> --max-minutes <n> <args...>``
with ``PYTHONPATH=<checkout>/src`` and must write the files named in ``expected_outputs`` plus ``device.json`` and
``manifest.json`` (``colab/contract.py``).
"""

import importlib
import json
import re

from nig_timenet_torch.colab.device import EXPECTED
from nig_timenet_torch.core.config import from_dict

__all__ = ["JobError", "parse_job", "validate_job", "check_entry_config", "versioned", "version_id", "REQUIRED",
           "OPTIONAL", "ON_EXISTING", "MAX_MINUTES_CAP"]

REQUIRED = ("job_id", "repo_ref", "entry", "args", "config", "max_minutes", "device", "needs_drive",
            "expected_outputs", "note")
OPTIONAL = ("rerun", "on_existing", "base_job_id")
ON_EXISTING = ("refuse", "new_version")
RESERVED_FLAGS = ("--config", "--out", "--max-minutes")
BUNDLE_NAMES = ("run_log.txt", "env.json", "bundle_summary.md", "publish.json", "job.json")
RESERVED_NAMES = (*BUNDLE_NAMES, "manifest.json", "device.json")
MAX_MINUTES_CAP = 180
JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,63}")
ENTRY = re.compile(r"[a-z][a-z0-9_]{0,62}")
SHA40 = re.compile(r"[0-9a-f]{40}")
OUTPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}")


class JobError(ValueError):
    """job.json is not a valid job."""


def _no_duplicates(pairs):
    keys = [k for k, _ in pairs]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise JobError(f"duplicate keys in job.json: {dup}")
    return dict(pairs)


def parse_job(text: str) -> dict:
    try:
        obj = json.loads(text, object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError as e:
        raise JobError(f"job.json is not valid JSON: {e.msg} (line {e.lineno})") from None
    return validate_job(obj)


def validate_job(obj) -> dict:
    """Returns the job with ``rerun`` / ``on_existing`` defaulted; raises JobError naming the first problem."""
    if not isinstance(obj, dict):
        raise JobError("job.json must be a JSON object")
    missing = [k for k in REQUIRED if k not in obj]
    extra = [k for k in obj if k not in REQUIRED + OPTIONAL]
    if missing or extra:
        raise JobError(f"job.json keys: missing {missing}, unknown {extra}")
    job = dict(obj)
    job.setdefault("rerun", False)
    job.setdefault("on_existing", "refuse")
    checks = [
        (isinstance(job["job_id"], str) and JOB_ID.fullmatch(job["job_id"]) and ".." not in job["job_id"],
         "job_id must be 3-64 chars of A-Za-z0-9._- and start with a letter or digit"),
        (isinstance(job["repo_ref"], str) and SHA40.fullmatch(job["repo_ref"]),
         "repo_ref must be a full 40-hex lowercase commit"),
        (isinstance(job["entry"], str) and ENTRY.fullmatch(job["entry"]),
         "entry must be a module name of nig_timenet_torch.jobs"),
        (isinstance(job["args"], list) and all(isinstance(a, str) for a in job["args"]),
         "args must be a list of strings"),
        (isinstance(job["config"], dict), "config must be an object"),
        (isinstance(job["max_minutes"], int) and not isinstance(job["max_minutes"], bool)
         and 1 <= job["max_minutes"] <= MAX_MINUTES_CAP, f"max_minutes must be an integer in 1..{MAX_MINUTES_CAP}"),
        (job["device"] in EXPECTED, f"device must be one of {EXPECTED} (rule E1)"),
        (isinstance(job["needs_drive"], bool), "needs_drive must be true or false"),
        (isinstance(job["rerun"], bool), "rerun must be true or false"),
        (job["on_existing"] in ON_EXISTING, f"on_existing must be one of {ON_EXISTING}"),
        (not (job["rerun"] and job["on_existing"] == "new_version"),
         "rerun (repeat the same job_id) and on_existing new_version (a new id) exclude each other"),
        (job["on_existing"] == "refuse" or _run_dir_segment_ok(job),
         "on_existing new_version needs config.run_dir (when set) to contain the folder /<job_id>/, so every version "
         "gets its own checkpoint folder"),
        (isinstance(job["note"], str), "note must be a string"),
        (isinstance(job.get("base_job_id", ""), str), "base_job_id must be a string"),
        (isinstance(job["expected_outputs"], list) and job["expected_outputs"]
         and all(isinstance(n, str) and OUTPUT_NAME.fullmatch(n) and n not in RESERVED_NAMES
                 for n in job["expected_outputs"])
         and len(set(job["expected_outputs"])) == len(job["expected_outputs"]),
         f"expected_outputs must be a non-empty list of distinct plain file names, none of {RESERVED_NAMES}"),
    ]
    for ok, message in checks:
        if not ok:
            raise JobError(message)
    bad = [a for a in job["args"] if a.split("=")[0] in RESERVED_FLAGS]
    if bad:
        raise JobError(f"args may not set {bad}: the runner passes {RESERVED_FLAGS}")
    check_entry_config(job["entry"], job["config"], job["device"])
    return job


def _run_dir_segment_ok(job: dict) -> bool:
    run_dir = job["config"].get("run_dir") if isinstance(job["config"], dict) else None  # job file, not a setting
    return run_dir is None or f"/{job['job_id']}/" in f"{run_dir}/"


def version_id(job_id: str, n: int) -> str:
    """The id of version ``n`` (>= 2) of ``job_id``: ``<job_id>-v<n>``."""
    if n < 2:
        raise ValueError("a version number is >= 2 (version 1 is the job_id itself)")
    return f"{job_id}-v{n}"


def versioned(job: dict, new_id: str) -> dict:
    """The job as version ``new_id``: job_id renamed, the ``/<job_id>/`` segment of ``config.run_dir`` renamed (a fresh
    checkpoint folder), ``base_job_id`` recorded. The result is validated again."""
    old = job["job_id"]
    out = {**job, "job_id": new_id, "base_job_id": job.get("base_job_id", old), "config": dict(job["config"])}
    run_dir = out["config"].get("run_dir")
    if run_dir is not None:
        out["config"]["run_dir"] = f"{run_dir}/".replace(f"/{old}/", f"/{new_id}/", 1)[:-1]
    return validate_job(out)


def check_entry_config(entry: str, config: dict, device: str | None = None) -> None:
    """Build the entry module's ``CONFIG`` (a frozen settings dataclass) strictly from ``config``: an unknown or missing
    key, or a bad value, is a JobError now instead of a failure on the GPU (rule B5). An existing entry module without
    ``CONFIG`` is refused (HF-17: every entry exposes one); a module that does not exist is reported by the runner. If
    both the job and the config name a device they must agree (``device`` is the one the run asserts, rule E1)."""
    name = f"nig_timenet_torch.jobs.{entry}"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name == name:
            return  # no such entry module: the runner reports it when it runs the job
        raise
    cls = getattr(module, "CONFIG", None)
    if cls is None:
        raise JobError(f"entry {entry!r} declares no CONFIG (every jobs/ entry exposes its strict config class)")
    try:
        cfg = from_dict(cls, config)
    except (ValueError, TypeError) as e:
        raise JobError(f"config of entry {entry!r}: {e}") from None
    if device is not None and getattr(cfg, "device", device) != device:
        raise JobError(f"job.device {device!r} differs from config.device {cfg.device!r} (same device)")
