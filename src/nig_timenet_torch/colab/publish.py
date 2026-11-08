"""PURPOSE: Results of a Colab run: the bundle folder on Drive (its own files listed in publish.json), the push of
    those files to colab-results and republish from the bundle alone (rules E1/E2, ROADMAP T3).
TAGS: colab publish, bundle, results branch, drive copy, republish, git size limit, run folder, checkpoints
PITFALLS: the job's run_dir (checkpoints) lives inside the bundle folder (<drive_runs>/<job_id>/train), so the bundle
    never deletes or pushes a file it did not write (B-050); outputs above GIT_MAX_BYTES stay on Drive only; a bundle
    written before the files record is pushed whole (every file in the folder).

Results of a Colab run: bundle, Drive copy, push to ``colab-results``, and ``republish`` (rules E1/E2, ROADMAP T3).

Order matters and is the fix for TF profile-005 (results stuck on Drive when ``/tmp/askpass.sh`` vanished during a long
job): the bundle is written FIRST, to Drive when it is mounted (else under the work dir), with a ``publish.json`` that
says whether it reached GitHub; the push comes second and may fail without losing anything. ``republish`` needs only
that bundle folder and a token: it rebuilds the git credentials from scratch, so it works after the VM's temp files,
the work dir's clones or the runner object are gone.

Layout on the results branch: ``runs/<job_id>/`` (job.json, env.json, run_log.txt, bundle_summary.md, manifest.json,
device.json and the outputs). An output above ``contract.GIT_MAX_BYTES`` stays on Drive only and is listed in the
summary.

The bundle folder on Drive is ``<drive_runs>/<job_id>``, and a job's ``run_dir`` is ``<drive_runs>/<job_id>/train``
(``docs/jobs``): the run folder lives INSIDE the bundle folder. A bundle therefore owns only the files it wrote, listed
in ``publish.json`` (``files``): a rebuild removes exactly those, and the push sends only those. Until 2026-11-08 the
rebuild removed the whole folder, which sent every run's checkpoints to the Drive trash at publish time (B-050).
"""

import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from nig_timenet_torch.colab.bootstrap import BootstrapError, Secrets, git_env, run_git
from nig_timenet_torch.colab.contract import GIT_MAX_BYTES
from nig_timenet_torch.colab.job import BUNDLE_NAMES

QUEUE_BRANCH = "colab-queue"
RESULTS_BRANCH = "colab-results"
MAX_LOG_BYTES = 200 * 1024
NOT_PUSHED = ("publish.json",)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def tail_text(path, max_bytes: int = MAX_LOG_BYTES) -> str:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return "(no log)"
    cut = len(data) > max_bytes
    text = data[-max_bytes:].decode("utf-8", errors="replace")
    return ("[... earlier output cut ...]\n" if cut else "") + text


def bundle_files(bundle_dir) -> list[str] | None:
    """The files the last ``build_bundle`` wrote into ``bundle_dir`` (relative posix paths, from ``publish.json``);
    None for a folder without that record (no bundle yet, or a bundle written before the record existed)."""
    try:
        files = json.loads((Path(bundle_dir) / "publish.json").read_text(encoding="utf-8")).get("files")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(files, list):
        return None
    return [f for f in files if isinstance(f, str) and f and not Path(f).is_absolute() and ".." not in Path(f).parts]


def _clear_bundle(dest: Path) -> None:
    """Remove the files of the previous bundle in ``dest`` (and the folders they leave empty), nothing else: the job's
    run folder (checkpoints) may live inside ``dest``."""
    for name in bundle_files(dest) or ():
        p = dest / name
        if p.is_file():
            p.unlink()
            for parent in p.parents:
                if parent == dest or not parent.is_relative_to(dest) or any(parent.iterdir()):
                    break
                parent.rmdir()
    (dest / "publish.json").unlink(missing_ok=True)


def build_bundle(dest, job: dict, out_dir, log_path, env_info: dict, status: str, problems: list[str],
                 secrets: Secrets | None = None) -> list[str]:
    """Write the bundle folder ``dest`` (replacing the previous bundle's files, keeping every other file there, e.g. the
    job's run folder); every text is masked. Records the file names in ``publish.json`` and returns them."""
    secrets = secrets or Secrets()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _clear_bundle(dest)
    written: list[Path] = []
    if out_dir and Path(out_dir).is_dir():
        for p in Path(out_dir).rglob("*"):
            if p.is_file() and p.name not in BUNDLE_NAMES:
                target = dest / p.relative_to(out_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
                written.append(target)
    (dest / "job.json").write_text(json.dumps(job, indent=1, sort_keys=True), encoding="utf-8")
    (dest / "env.json").write_text(secrets.mask(json.dumps(env_info, indent=1, sort_keys=True)), encoding="utf-8")
    (dest / "run_log.txt").write_text(secrets.mask(tail_text(log_path)), encoding="utf-8")
    written += [dest / "job.json", dest / "env.json", dest / "run_log.txt"]
    large = sorted(p.relative_to(dest).as_posix() for p in written if p.stat().st_size > GIT_MAX_BYTES)
    dev = env_info.get("device") or {}
    lines = [f"# {job['job_id']}: {status}", "", f"- repo_ref: `{job['repo_ref']}`",
             f"- device: {job['device']} (used {dev.get('used')}, {dev.get('gpu_name')})",
             f"- note: {job['note']}", *(f"- problem: {secrets.mask(p)}" for p in problems)]
    if large:
        lines += ["", "Kept on Drive only (above the git size limit): " + ", ".join(large)]
    (dest / "bundle_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    names = sorted({p.relative_to(dest).as_posix() for p in [*written, dest / "bundle_summary.md"]})
    write_publish_state(dest, files=names)
    return names


def write_publish_state(bundle_dir, **fields) -> dict:
    path = Path(bundle_dir) / "publish.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    state.update(fields, updated_utc=_now())
    path.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    return state


def clone_branch(url: str, branch: str, dest: str, env=None, secrets: Secrets | None = None) -> bool:
    """Shallow clone of one branch; False (nothing cloned) when the branch does not exist on the remote."""
    r = subprocess.run(["git", "ls-remote", "--exit-code", "--heads", url, branch], env=env or dict(os.environ),
                       capture_output=True, text=True, errors="replace", timeout=300)
    if r.returncode == 2:
        return False
    if r.returncode != 0:
        tail = r.stderr.strip()[-400:]
        raise BootstrapError(f"git ls-remote failed (exit {r.returncode}): {secrets.mask(tail) if secrets else tail}")
    shutil.rmtree(dest, ignore_errors=True)
    run_git(["clone", "--quiet", "--depth", "1", "--branch", branch, url, dest], env=env, secrets=secrets)
    return True


def fetch_queue(url: str, branch: str, workdir: str, env=None, secrets: Secrets | None = None) -> tuple[str, str]:
    """-> (job.json text, sha of the queue commit) from the queue branch; raises when the branch or file is absent."""
    dest = os.path.join(workdir, "queue_probe")
    try:
        if not clone_branch(url, branch, dest, env=env, secrets=secrets):
            raise BootstrapError(f"queue branch {branch} does not exist on the remote")
        job = Path(dest) / "job.json"
        if not job.is_file():
            raise BootstrapError(f"{branch} has no job.json at its root")
        return job.read_text(encoding="utf-8"), run_git(["rev-parse", "HEAD"], cwd=dest, env=env, secrets=secrets)
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def results_has_job(url: str, branch: str, job_id: str, workdir: str, env=None, secrets: Secrets | None = None) -> bool:
    dest = os.path.join(workdir, "results_probe")
    try:
        return clone_branch(url, branch, dest, env=env, secrets=secrets) and \
            os.path.isfile(os.path.join(dest, "runs", job_id, "env.json"))
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def _copy_pushable(bundle_dir: Path, target: Path) -> None:
    """The bundle's own files (its ``publish.json`` record; every file for a bundle written before the record) up to
    the git size limit; ``target`` is the job's folder in a clone of the results branch."""
    shutil.rmtree(target, ignore_errors=True)
    names = bundle_files(bundle_dir)
    paths = (bundle_dir.rglob("*") if names is None else (bundle_dir / n for n in names))
    for p in paths:
        if p.is_file() and p.name not in NOT_PUSHED and p.stat().st_size <= GIT_MAX_BYTES:
            (target / p.relative_to(bundle_dir)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target / p.relative_to(bundle_dir))


def push_results(url: str, branch: str, bundle_dir, job: dict, status: str, workdir: str, env_fn,
                 secrets: Secrets | None = None, attempts: int = 4) -> str:
    """Commit the bundle as ``runs/<job_id>/`` on ``branch`` (created as an orphan when missing) and push without
    force; returns the commit sha. A rejected push is retried from a fresh clone. ``env_fn()`` is called before EVERY
    git command group, so the credentials are rebuilt each attempt (nothing is assumed to survive between them)."""
    dest = os.path.join(workdir, "results_repo")
    for attempt in range(attempts):
        env = env_fn()
        shutil.rmtree(dest, ignore_errors=True)
        if not clone_branch(url, branch, dest, env=env, secrets=secrets):
            os.makedirs(dest)
            run_git(["init", "--quiet"], cwd=dest, env=env, secrets=secrets)
            run_git(["symbolic-ref", "HEAD", f"refs/heads/{branch}"], cwd=dest, env=env, secrets=secrets)
            run_git(["remote", "add", "origin", url], cwd=dest, env=env, secrets=secrets)
        _copy_pushable(Path(bundle_dir), Path(dest) / "runs" / job["job_id"])
        run_git(["add", "-A"], cwd=dest, env=env, secrets=secrets)
        message = f"colab-results: {job['job_id']} {status}\n\njob_id: {job['job_id']}\nrepo_ref: {job['repo_ref']}\n"
        run_git(["-c", "user.name=colab-runner", "-c", "user.email=colab-runner@users.noreply.github.com",
                 "-c", "commit.gpgsign=false", "commit", "--quiet", "--allow-empty", "-m", message],
                cwd=dest, env=env, secrets=secrets)
        try:
            run_git(["push", "--quiet", "origin", f"HEAD:refs/heads/{branch}"], cwd=dest, env=env, secrets=secrets)
        except BootstrapError as e:
            if attempt + 1 == attempts:
                raise BootstrapError(f"push to {branch} failed after {attempts} attempts: {e}") from None
            time.sleep(attempt + 1)
            continue
        return run_git(["rev-parse", "HEAD"], cwd=dest, env=env, secrets=secrets)
    raise AssertionError("unreachable")  # pragma: no cover


def republish(bundle_dir, url: str, token: str, workdir: str, branch: str = RESULTS_BRANCH, force: bool = False,
              attempts: int = 4, echo=print) -> dict:
    """Push an existing bundle folder (the Drive copy of a run whose push failed) to the results branch.

    Needs only the bundle and a token. Refuses a job that is already published unless ``force``. Returns the updated
    ``publish.json`` state."""
    bundle = Path(bundle_dir)
    secrets = Secrets(token)
    try:
        job = json.loads((bundle / "job.json").read_text(encoding="utf-8"))
        state = json.loads((bundle / "publish.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise BootstrapError(f"{bundle} is not a bundle folder (job.json / publish.json missing or "
                             f"unreadable)") from None
    os.makedirs(workdir, exist_ok=True)
    askpass = os.path.join(workdir, "git_askpass")

    def env_fn():
        return git_env(token, askpass)

    if not force and results_has_job(url, branch, job["job_id"], workdir, env=env_fn(), secrets=secrets):
        echo(f"job {job['job_id']} is already on {branch}; nothing to do (use force to replace it)")
        return write_publish_state(bundle, pushed=True)
    sha = push_results(url, branch, bundle, job, state.get("status", "FAIL"), workdir, env_fn, secrets, attempts)
    echo(f"republished {job['job_id']} to {branch} at {sha[:12]}")
    return write_publish_state(bundle, pushed=True, results_sha=sha, branch=branch)

