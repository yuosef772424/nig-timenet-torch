"""One Colab job from queue to results (ROADMAP T3): ``fetch`` job.json from ``colab-queue`` -> ``check_rerun`` ->
``checkout`` the pinned commit -> ``execute`` it under a wall-clock guard -> ``publish`` to ``colab-results`` and Drive.

Design copied from the TF runner (``tools/colab_runner.py`` there), with the two T3 fixes:
  * Rule E1: ``execute`` asserts the job's device in this process BEFORE starting the entry (a GPU job on a CPU VM
    fails without running anything), records the device report in ``env.json``, and ``publish`` validates the entry's
    own ``device.json`` against the job (``contract.validate_outputs``).
  * Rule E2: nothing lives in ``/tmp``. The work dir (default ``/content/nig_runner``) holds the clones, the job's
    scratch (``TMPDIR`` of the entry points there), the log and the credentials helper, which ``git_env`` rewrites on
    every use. The bundle is written before the push, to Drive when mounted, and ``colab/publish.py::republish`` can
    push it later from nothing but that folder.

Steps never raise: a failure is recorded and later steps are skipped, so ``publish`` can always push the log.
Runner code and job code are different commits on purpose: the runner comes from the notebook's own clone, the job
runs at ``repo_ref``. The job cannot weaken the validation that judges it.
"""

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from nig_timenet_torch.colab import device as device_mod
from nig_timenet_torch.colab import publish
from nig_timenet_torch.colab.bootstrap import BootstrapError, Secrets, clean_clone, git_env
from nig_timenet_torch.colab.contract import validate_outputs
from nig_timenet_torch.colab.job import JobError, parse_job

DEFAULT_WORKDIR = "/content/nig_runner"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive"
DEFAULT_DRIVE_RUNS = DEFAULT_DRIVE_ROOT + "/nig_timenet_runs"
ENV_PACKAGES = ("torch", "numpy", "pandas", "pyarrow", "scikit-learn", "scipy")
SECRET_ENV_KEYS = ("GIT_ASKPASS", "COLAB_RUNNER_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")


class RunnerError(BootstrapError):
    """A step failed; the message is already masked."""


def run_logged(cmd, cwd, env, log_path, wall_seconds, secrets: Secrets, echo=print, echo_lines=400):
    """Run ``cmd``, append masked output to ``log_path`` (echoing the first ``echo_lines`` lines); kill the process
    group after ``wall_seconds``. Returns (exit code, timed_out)."""
    state = {"timed_out": False}
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)

        def kill():
            state["timed_out"] = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()

        timer = threading.Timer(wall_seconds, kill)
        timer.daemon = True
        timer.start()
        shown = 0
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = secrets.mask(raw.decode("utf-8", "replace"))
                log.write(line.encode("utf-8"))
                log.flush()
                if shown < echo_lines:
                    echo(line.rstrip("\n"))
                    shown += 1
        finally:
            timer.cancel()
            proc.stdout.close()
            code = proc.wait()
    return code, state["timed_out"]


def package_versions(names=ENV_PACKAGES) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


class Runner:
    """One job, one Runner. Order: fetch/load, check_rerun, checkout, execute, publish."""

    def __init__(self, repo_url: str, token: str, workdir: str = DEFAULT_WORKDIR, drive_runs: str = DEFAULT_DRIVE_RUNS,
                 drive_root: str = DEFAULT_DRIVE_ROOT, queue_branch: str = publish.QUEUE_BRANCH,
                 results_branch: str = publish.RESULTS_BRANCH, echo=print, python: str | None = None,
                 grace_seconds: int = 120, device_check=device_mod.assert_device, runner_sha: str = "",
                 push_attempts: int = 4):
        if not isinstance(token, str) or len(token) < 8:
            raise RunnerError("a GitHub token is required")
        self.repo_url, self.workdir = repo_url, os.path.abspath(workdir)
        self.drive_runs, self.drive_root = drive_runs, drive_root
        self.queue_branch, self.results_branch = queue_branch, results_branch
        self.secrets, self.echo, self.python = Secrets(token), echo, python or sys.executable
        self._token, self.grace_seconds, self.device_check = token, grace_seconds, device_check
        self.runner_sha, self.push_attempts = runner_sha, push_attempts
        self.job = None
        self.problems: list[str] = []
        self.refused = False
        self.queue_sha = ""
        self.repo_dir = self.out_dir = self.log_path = None
        self.device_report: dict = {}
        self.exit_code, self.timed_out, self.run_seconds = None, False, None
        self.t_start = datetime.now(UTC)
        os.makedirs(self.workdir, exist_ok=True)

    # -- helpers
    def git(self) -> dict:
        """Fresh git environment (the askpass helper is rewritten, so it cannot have vanished)."""
        return git_env(self._token, os.path.join(self.workdir, "git_askpass"))

    def _say(self, text) -> None:
        self.echo(self.secrets.mask(text))

    def _log(self, text) -> None:
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(self.secrets.mask(text) + "\n")

    def _fail(self, phase: str, message: str) -> None:
        self.problems.append(f"{phase}: {message}")
        self._log(f"== {phase} FAILED: {message}")
        self._say(f"[{phase}] FAILED: {message}")

    def _step(self, phase: str, fn) -> None:
        if self.problems or self.refused or self.job is None:
            return
        try:
            fn()
        except (BootstrapError, device_mod.DeviceError, OSError) as e:
            self._fail(phase, self.secrets.mask(f"{type(e).__name__}: {e}" if isinstance(e, OSError) else e))

    def bundle_dir(self, job_id: str) -> str:
        """Drive when mounted, else the work dir: both survive a cleaned temp dir."""
        base = self.drive_runs if os.path.isdir(self.drive_root) else os.path.join(self.workdir, "bundle")
        return os.path.join(base, job_id)

    # -- steps
    def fetch(self) -> dict | None:
        text, sha = publish.fetch_queue(self.repo_url, self.queue_branch, self.workdir, self.git(), self.secrets)
        return self.load(text, sha)

    def load(self, job_text: str, queue_sha: str = "") -> dict | None:
        self.queue_sha = queue_sha
        try:
            self.job = parse_job(job_text)
        except JobError as e:
            self.problems.append(f"load: {e}")
            self._say(f"[load] job rejected: {e}")
            return None
        os.makedirs(os.path.join(self.workdir, "logs"), exist_ok=True)
        self.log_path = os.path.join(self.workdir, "logs", f"{self.job['job_id']}.log")
        Path(self.log_path).write_text(f"# run log of {self.job['job_id']} started {self.t_start.isoformat()}\n",
                                       encoding="utf-8")
        self._say(f"job {self.job['job_id']} @ {self.job['repo_ref'][:12]}: {self.job['entry']} "
                  f"on {self.job['device']}")
        return self.job

    def check_rerun(self) -> None:
        def go():
            if not self.job["rerun"] and publish.results_has_job(self.repo_url, self.results_branch,
                                                                 self.job["job_id"], self.workdir, self.git(),
                                                                 self.secrets):
                self.refused = True
                self.problems.append(f"job {self.job['job_id']} is already finished (runs/{self.job['job_id']}/ exists "
                                     f"on {self.results_branch}); queue a new job_id or set \"rerun\": true")
                self._say(f"[refused] {self.problems[-1]}")
        self._step("check_rerun", go)

    def checkout(self) -> None:
        def go():
            self.repo_dir = os.path.join(self.workdir, "repo")
            clean_clone(self.repo_url, self.repo_dir, self.job["repo_ref"], self.git(), self.secrets, exact_sha=True)
            entry = Path(self.repo_dir, "src", "nig_timenet_torch", "jobs", f"{self.job['entry']}.py")
            if not entry.is_file():
                raise RunnerError(f"entry {self.job['entry']} not found in nig_timenet_torch/jobs at the "
                                  f"checked-out commit")
            self._log(f"== checkout ok: {self.job['repo_ref']}")
            self._say("checked out " + self.job["repo_ref"])
        self._step("checkout", go)

    def _job_env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV_KEYS}
        scratch = os.path.join(self.workdir, "tmp")
        os.makedirs(scratch, exist_ok=True)
        src = os.path.join(self.repo_dir, "src")
        inherited = [env["PYTHONPATH"]] if env.get("PYTHONPATH") else []
        env.update({"PYTHONUNBUFFERED": "1", "NIG_REPO_DIR": self.repo_dir, "NIG_JOB_ID": self.job["job_id"],
                    "TMPDIR": scratch, "PYTHONPATH": os.pathsep.join([src, *inherited])})
        if self.job["needs_drive"]:
            env["NIG_DRIVE_ROOT"] = self.drive_root
        return env

    def execute(self) -> None:
        def go():
            job = self.job
            try:
                self.device_report = self.device_check(job["device"])
            except device_mod.DeviceError:
                self.device_report = device_mod.device_report(job["device"])  # the failed state goes on record too
                raise
            if job["needs_drive"] and not os.path.isdir(self.drive_root):
                raise RunnerError(f"needs_drive is true but Drive is not mounted at {self.drive_root}")
            cfg = os.path.join(self.workdir, "tmp", f"{job['job_id']}.config.json")
            os.makedirs(os.path.dirname(cfg), exist_ok=True)
            Path(cfg).write_text(json.dumps(job["config"]), encoding="utf-8")
            self.out_dir = os.path.join(self.workdir, "out", job["job_id"])
            shutil.rmtree(self.out_dir, ignore_errors=True)
            os.makedirs(self.out_dir)
            cmd = [self.python, "-m", f"nig_timenet_torch.jobs.{job['entry']}", "--config", cfg, "--out", self.out_dir,
                   "--max-minutes", str(job["max_minutes"]), *job["args"]]
            self._log("== run: " + " ".join(cmd))
            self._say("running: " + " ".join(cmd[1:]))
            t0 = time.time()
            self.exit_code, self.timed_out = run_logged(
                cmd, self.repo_dir, self._job_env(), self.log_path, job["max_minutes"] * 60 + self.grace_seconds,
                self.secrets, self._say)
            self.run_seconds = round(time.time() - t0, 1)
            if self.timed_out:
                raise RunnerError(f"killed by the wall-clock guard after {self.run_seconds} s (max_minutes "
                                  f"{job['max_minutes']} + {self.grace_seconds} s grace)")
            if self.exit_code != 0:
                raise RunnerError(f"exit code {self.exit_code}")
        self._step("execute", go)

    def env_info(self, end: datetime) -> dict:
        return {"device": self.device_report or None, "python": sys.version.split()[0], "platform": platform.platform(),
                "packages": package_versions(), "start_utc": self.t_start.isoformat(), "end_utc": end.isoformat(),
                "duration_s": round((end - self.t_start).total_seconds(), 1), "run_seconds": self.run_seconds,
                "exit_code": self.exit_code, "timed_out": self.timed_out, "queue_sha": self.queue_sha,
                "runner_sha": self.runner_sha, "workdir": self.workdir}

    def _block(self, status: str, results_sha: str, lines=(), bundle: str | None = None) -> dict:
        job_id = self.job["job_id"] if self.job else "-"
        for ln in ("=" * 44, "COLAB RUNNER STATUS", f"job_id   : {job_id}", f"status   : {status}",
                   f"results  : {results_sha} ({self.results_branch})", *lines, "=" * 44):
            self._say(ln)
        return {"status": status, "job_id": job_id, "results_sha": results_sha, "pushed": results_sha != "-",
                "bundle": bundle, "problems": list(self.problems)}

    def publish(self) -> dict:
        """Validate, write the bundle (Drive first), push to the results branch, print the status block."""
        if self.job is None or self.refused:
            return self._block("REFUSED", "-", [f"problem  : {p}" for p in self.problems] + ["nothing was pushed"])
        if not self.problems:
            self.problems += [f"validate: {p}" for p in validate_outputs(self.out_dir, self.job, self.job["repo_ref"])]
        status = "OK" if not self.problems else "FAIL"
        bundle = self.bundle_dir(self.job["job_id"])
        publish.build_bundle(bundle, self.job, self.out_dir, self.log_path, self.env_info(datetime.now(UTC)), status,
                             self.problems, self.secrets)
        publish.write_publish_state(bundle, status=status, job_id=self.job["job_id"], pushed=False)
        lines, sha = [f"bundle   : {bundle}"], "-"
        try:
            sha = publish.push_results(self.repo_url, self.results_branch, bundle, self.job, status, self.workdir,
                                       self.git, self.secrets, self.push_attempts)
            publish.write_publish_state(bundle, pushed=True, results_sha=sha, branch=self.results_branch)
        except BootstrapError as e:
            self.problems.append(f"publish: {self.secrets.mask(e)}")
            status = "FAIL"
            lines.append(f"push     : FAILED; the bundle is saved at {bundle}; run republish to retry")
        lines += [f"problem  : {p}" for p in self.problems]
        return self._block(status, sha, lines, bundle)

    def run_all(self) -> dict:
        """The whole sequence for the notebook's one cell."""
        self.fetch()
        self.check_rerun()
        self.checkout()
        self.execute()
        return self.publish()
