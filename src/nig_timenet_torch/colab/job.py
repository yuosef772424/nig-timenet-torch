"""The job a Colab run executes: ``job.json`` on the ``colab-queue`` branch (ROADMAP T3).

Same idea as the TF runner (queue branch -> run -> results branch), with the two T3 changes built into the schema:
``device`` is required (rule E1: the run asserts and records it) and the entry is a module of
``nig_timenet_torch.jobs`` run by name, so nothing is looked up under a temporary directory (rule E2).

job.json (every key required except ``rerun``):
    {"job_id": "smoke-001", "repo_ref": "<40-hex commit>", "entry": "device_smoke", "args": [], "config": {},
     "max_minutes": 5, "device": "cuda", "needs_drive": false, "expected_outputs": ["smoke.json"], "note": "free text",
     "rerun": false}
The entry runs as ``python -m nig_timenet_torch.jobs.<entry> --config <json> --out <dir> --max-minutes <n> <args...>``
with ``PYTHONPATH=<checkout>/src`` and must write the files named in ``expected_outputs`` plus ``device.json`` and
``manifest.json`` (``colab/contract.py``).
"""

import json
import re

from nig_timenet_torch.colab.device import EXPECTED

REQUIRED = ("job_id", "repo_ref", "entry", "args", "config", "max_minutes", "device", "needs_drive",
            "expected_outputs", "note")
OPTIONAL = ("rerun",)
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
    """Returns the job with ``rerun`` defaulted; raises JobError naming the first problem."""
    if not isinstance(obj, dict):
        raise JobError("job.json must be a JSON object")
    missing = [k for k in REQUIRED if k not in obj]
    extra = [k for k in obj if k not in REQUIRED + OPTIONAL]
    if missing or extra:
        raise JobError(f"job.json keys: missing {missing}, unknown {extra}")
    job = dict(obj)
    job.setdefault("rerun", False)
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
        (isinstance(job["note"], str), "note must be a string"),
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
    return job
