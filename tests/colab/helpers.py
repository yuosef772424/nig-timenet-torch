"""Local git remotes for the Colab runner tests: a bare repo holding a copy of this package at one commit, plus a
``colab-queue`` branch with a job.json. ``file://`` URLs, so no network and no token."""

import json
import shutil
import subprocess
from pathlib import Path

from tests.scan import PKG_DIR

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]


def git(*args, cwd) -> str:
    r = subprocess.run([*GIT, *args], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, f"git {args}: {r.stderr}"
    return r.stdout.strip()


def make_origin(base: Path) -> tuple[str, str]:
    """-> (file:// url of a bare repo, sha of its main commit); the repo holds ``src/nig_timenet_torch`` as it is."""
    work = base / "origin_work"
    shutil.copytree(PKG_DIR, work / "src" / "nig_timenet_torch", ignore=shutil.ignore_patterns("__pycache__"))
    git("init", "-q", "-b", "main", cwd=work)
    git("add", "-A", cwd=work)
    git("commit", "-q", "-m", "package copy", cwd=work)
    sha = git("rev-parse", "HEAD", cwd=work)
    bare = base / "origin.git"
    git("clone", "-q", "--bare", str(work), str(bare), cwd=base)
    return f"file://{bare}", sha


def queue_job(base: Path, url: str, job: dict) -> str:
    """Create/replace branch colab-queue with job.json; returns the queue commit sha."""
    work = base / "queue_work"
    shutil.rmtree(work, ignore_errors=True)
    git("clone", "-q", url, str(work), cwd=base)
    git("checkout", "-q", "--orphan", "colab-queue", cwd=work)
    git("rm", "-rq", "--cached", ".", cwd=work)
    for p in work.iterdir():
        if p.name != ".git":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    (work / "job.json").write_text(json.dumps(job))
    git("add", "job.json", cwd=work)
    git("commit", "-q", "-m", "queue", cwd=work)
    git("push", "-q", "-f", "origin", "colab-queue", cwd=work)
    return git("rev-parse", "HEAD", cwd=work)


def make_job(sha: str, **over) -> dict:
    job = {"job_id": "smoke-001", "repo_ref": sha, "entry": "device_smoke", "args": [], "config": {"device": "cpu"},
           "max_minutes": 2, "device": "cpu", "needs_drive": False, "expected_outputs": ["smoke.json"],
           "note": "test"}
    job.update(over)
    return job


def results_files(base: Path, url: str, job_id: str) -> set[str]:
    """File names under runs/<job_id>/ on colab-results (fresh clone)."""
    dest = base / "results_check"
    shutil.rmtree(dest, ignore_errors=True)
    git("clone", "-q", "--branch", "colab-results", url, str(dest), cwd=base)
    return {p.name for p in (dest / "runs" / job_id).iterdir()} if (dest / "runs" / job_id).is_dir() else set()
