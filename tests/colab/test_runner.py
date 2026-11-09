"""colab/runner.py + colab/publish.py end to end on local git remotes: queue -> run -> results, the device rule (E1),
no dependence on /tmp and the republish path (E2). The job is the real ``device_smoke`` entry, run in a subprocess
against a checkout of this package's own files."""

import json
import os
import re
import stat
import sys
import time

import pytest

from nig_timenet_torch.colab import __main__ as cli
from nig_timenet_torch.colab import bootstrap, publish
from nig_timenet_torch.colab.device import DeviceError
from nig_timenet_torch.colab.runner import Runner, run_logged
from tests.colab.helpers import make_job, make_origin, queue_job, results_files
from tests.scan import PKG_DIR

TOKEN = "ghp_abcdefghijklmnop1234"


@pytest.fixture()
def world(tmp_path):
    url, sha = make_origin(tmp_path)
    return {"base": tmp_path, "url": url, "sha": sha, "work": str(tmp_path / "work"), "echo": []}


def runner(w, **kw):
    kw.setdefault("drive_root", str(w["base"] / "no_drive"))
    kw.setdefault("push_attempts", 1)
    return Runner(w["url"], TOKEN, w["work"], echo=w["echo"].append, **kw)


def queue(w, **over):
    queue_job(w["base"], w["url"], make_job(w["sha"], **over))


def test_ok_run_goes_queue_to_run_to_results_and_records_the_device(world):
    queue(world)
    r = runner(world).run_all()
    assert r["status"] == "OK" and r["problems"] == [], world["echo"][-15:]
    assert r["pushed"] is True and r["bundle"].endswith(os.path.join("bundle", "smoke-001"))
    files = results_files(world["base"], world["url"], "smoke-001")
    assert {"job.json", "env.json", "run_log.txt", "bundle_summary.md", "manifest.json", "device.json",
            "smoke.json"} <= files
    assert "publish.json" not in files  # local state, not a result
    bundle = world["base"] / "work" / "bundle" / "smoke-001"  # no Drive here: the work dir, never /tmp
    env = json.loads((bundle / "env.json").read_text())
    assert env["device"]["used"] == "cpu" and env["device"]["torch_version"] and env["exit_code"] == 0
    smoke = json.loads((bundle / "smoke.json").read_text())
    pinned = os.path.join(world["work"], "repo", "src")
    assert smoke["package_file"].startswith(pinned), "the job must run the pinned checkout"
    assert smoke["tempdir"].startswith(os.path.join(world["work"], "tmp")), "rule E2: no /tmp for the job"
    state = json.loads((bundle / "publish.json").read_text())
    assert state["pushed"] is True and len(state["results_sha"]) == 40


def test_a_finished_job_is_refused_unless_it_sets_rerun(world):
    queue(world)
    assert runner(world).run_all()["status"] == "OK"
    again = runner(world).run_all()
    assert again["status"] == "REFUSED" and "already finished" in again["problems"][0]
    assert again["pushed"] is False and again["bundle"] is None  # nothing to republish for a refused job
    queue(world, rerun=True)
    assert runner(world).run_all()["status"] == "OK"


def test_on_existing_new_version_runs_a_finished_job_as_its_next_free_version(world):
    queue(world, on_existing="new_version")
    assert runner(world).run_all()["status"] == "OK"                     # first run keeps its id
    second = runner(world).run_all()
    assert second["status"] == "OK" and second["job_id"] == "smoke-001-v2", world["echo"][-15:]
    assert second["bundle"].endswith(os.path.join("bundle", "smoke-001-v2"))
    third = runner(world).run_all()
    assert third["job_id"] == "smoke-001-v3"
    for jid in ("smoke-001", "smoke-001-v2", "smoke-001-v3"):            # every version kept side by side
        assert "env.json" in results_files(world["base"], world["url"], jid)
    job = json.loads((world["base"] / "work" / "bundle" / "smoke-001-v3" / "job.json").read_text())
    assert job["job_id"] == "smoke-001-v3" and job["base_job_id"] == "smoke-001"


def test_a_gpu_job_on_a_cpu_vm_fails_before_anything_runs_and_the_failure_is_published(world):
    queue(world, device="cuda", config={"device": "cuda"})
    r = runner(world).run_all()  # the default device check is the real one: this sandbox has no GPU
    assert r["status"] == "FAIL" and any("execute: " in p and "job expects cuda" in p for p in r["problems"])
    assert not (world["base"] / "work" / "out" / "smoke-001").exists(), "the entry must not have started"
    assert {"env.json", "run_log.txt", "bundle_summary.md"} <= results_files(world["base"], world["url"], "smoke-001")


def test_an_entry_that_cannot_get_its_device_fails_the_job_even_when_the_runner_check_passes(world):
    """Second layer of rule E1: the entry asserts the device itself (here the runner's own check is bypassed)."""
    queue(world, device="cuda", config={"device": "cuda"})
    r = runner(world, device_check=lambda expected: {"used": expected}).run_all()
    assert r["status"] == "FAIL" and any("exit code" in p for p in r["problems"])
    log = (world["base"] / "work" / "bundle" / "smoke-001" / "run_log.txt").read_text()
    assert "job expects cuda" in log


def test_a_bad_commit_and_a_bad_job_are_failures_not_crashes(world):
    queue(world, repo_ref="0" * 40)
    r = runner(world).run_all()
    assert r["status"] == "FAIL" and any("checkout" in p for p in r["problems"])
    queue_job(world["base"], world["url"], {"job_id": "x"})
    assert runner(world).run_all()["status"] == "REFUSED"


def test_the_token_never_appears_in_output_or_results(world):
    queue(world)
    runner(world).run_all()
    assert TOKEN not in "\n".join(world["echo"])
    for f in (world["base"] / "work" / "bundle" / "smoke-001").iterdir():
        assert TOKEN not in f.read_text(errors="replace")


# ------------------------------------------------------------------------------------------------ rule E2
def reject_pushes(w, on: bool):
    flag = w["base"] / "reject_pushes"
    hook = w["base"] / "origin.git" / "hooks" / "pre-receive"
    hook.write_text(f'#!/bin/sh\n[ -f "{flag}" ] && {{ echo "rejected by test" >&2; exit 1; }}\nexit 0\n')
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR)
    flag.write_text("x") if on else flag.unlink(missing_ok=True)


def test_a_failed_push_keeps_the_bundle_and_republish_needs_nothing_else(world):
    queue(world)
    reject_pushes(world, True)
    r = runner(world).run_all()
    assert r["status"] == "FAIL" and any(p.startswith("publish:") for p in r["problems"])
    assert r["pushed"] is False and r["bundle"], "the notebook's republish cell keys on these two"
    assert any("run republish" in line for line in world["echo"])
    bundle = world["base"] / "work" / "bundle" / "smoke-001"
    assert json.loads((bundle / "publish.json").read_text())["pushed"] is False
    # everything the run used is gone: clones, scratch, credentials helper, the Runner object itself
    for gone in ("repo", "out", "tmp", "git_askpass", "results_repo", "logs"):
        __import__("shutil").rmtree(world["base"] / "work" / gone, ignore_errors=True)
    reject_pushes(world, False)
    state = publish.republish(bundle, world["url"], TOKEN, str(world["base"] / "fresh_workdir"), echo=lambda *_: None)
    assert state["pushed"] is True
    assert {"job.json", "manifest.json", "smoke.json"} <= results_files(world["base"], world["url"], "smoke-001")
    fresh = str(world["base"] / "fresh_workdir")
    again = publish.republish(bundle, world["url"], TOKEN, fresh, echo=world["echo"].append)
    assert "already on" in world["echo"][-1] and again["pushed"] is True


def test_republish_command_line(world, tmp_path, capsys):
    queue(world)
    reject_pushes(world, True)
    runner(world).run_all()
    reject_pushes(world, False)
    token_file = tmp_path / "tok.txt"
    bootstrap.save_token(str(token_file), TOKEN)
    bundle = str(world["base"] / "work" / "bundle" / "smoke-001")
    assert cli.main(["republish", bundle, "--url", world["url"], "--token-path", str(token_file),
                     "--workdir", str(tmp_path / "w2")]) == 0
    assert "republished smoke-001" in capsys.readouterr().out
    assert cli.main(["republish", str(tmp_path / "not_a_bundle"), "--url", world["url"], "--token-path",
                     str(token_file), "--workdir", str(tmp_path / "w2")]) == 1
    assert cli.main(["republish", bundle, "--url", world["url"], "--token-path", str(tmp_path / "none"),
                     "--workdir", str(tmp_path / "w2")]) == 1


def test_check_job_command_line(tmp_path, capsys):
    good = tmp_path / "job.json"
    good.write_text(json.dumps(make_job("a" * 40)))
    assert cli.main(["check-job", str(good)]) == 0 and "OK: job smoke-001" in capsys.readouterr().out
    good.write_text(json.dumps(make_job("a" * 40, device="gpu")))
    assert cli.main(["check-job", str(good)]) == 1


def test_large_outputs_stay_on_drive_and_are_listed(world, tmp_path):
    from nig_timenet_torch.colab import contract

    out = tmp_path / "out"
    out.mkdir()
    (out / "big.bin").write_bytes(b"0" * (contract.GIT_MAX_BYTES + 1))
    (out / "small.txt").write_text("s")
    log = tmp_path / "l.log"
    log.write_text("log")
    bundle = tmp_path / "bundle"
    job = make_job(world["sha"])
    publish.build_bundle(bundle, job, out, log, {}, "OK", [])
    publish.write_publish_state(bundle, status="OK", pushed=False)
    assert "big.bin" in (bundle / "bundle_summary.md").read_text()
    publish.push_results(world["url"], "colab-results", bundle, job, "OK", str(tmp_path / "w"),
                         lambda: bootstrap.git_env(TOKEN, str(tmp_path / "w" / "ask")))
    assert results_files(world["base"], world["url"], "smoke-001") >= {"small.txt", "job.json"}
    assert "big.bin" not in results_files(world["base"], world["url"], "smoke-001")


def test_the_publish_keeps_the_run_folder_that_lives_inside_the_bundle_folder(world):
    """B-050: docs/jobs put run_dir at <drive_runs>/<job_id>/train, inside the bundle folder; the publish used to
    replace the whole bundle folder, so every run's checkpoints went to the Drive trash at the end of the job."""
    from nig_timenet_torch.colab import contract

    drive_root = world["base"] / "drive"
    train = drive_root / "nig_timenet_runs" / "smoke-001" / "train"  # what the job's trainer writes during the run
    train.mkdir(parents=True)
    (train / "best.pt").write_bytes(b"best")
    (train / "checkpoint.pt").write_bytes(b"c" * (contract.GIT_MAX_BYTES + 1))
    queue(world)
    r = runner(world, drive_root=str(drive_root), drive_runs=str(drive_root / "nig_timenet_runs")).run_all()
    assert r["status"] == "OK" and r["bundle"] == str(train.parent), world["echo"][-15:]
    assert (train / "best.pt").read_bytes() == b"best" and (train / "checkpoint.pt").stat().st_size > 1
    assert "smoke.json" in results_files(world["base"], world["url"], "smoke-001")
    assert "train" not in results_files(world["base"], world["url"], "smoke-001")  # not the bundle's file
    assert "checkpoint.pt" not in (train.parent / "bundle_summary.md").read_text()


def test_a_rebuilt_bundle_replaces_only_its_own_files(world, tmp_path):
    bundle, log = tmp_path / "bundle", tmp_path / "l.log"
    log.write_text("log")
    (bundle / "train").mkdir(parents=True)
    (bundle / "train" / "best.pt").write_bytes(b"w")
    job = make_job(world["sha"])
    for i, name in enumerate(("first.txt", "sub/second.txt")):
        out = tmp_path / f"out{i}"
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(name)
        names = publish.build_bundle(bundle, job, out, log, {}, "OK", [])
    assert not (bundle / "first.txt").exists() and (bundle / "sub" / "second.txt").is_file()
    assert (bundle / "train" / "best.pt").read_bytes() == b"w"
    assert publish.bundle_files(bundle) == names and "train/best.pt" not in names and "publish.json" not in names
    publish.write_publish_state(bundle, status="OK", pushed=False)
    assert publish.bundle_files(bundle) == names  # the runner's state update keeps the record
    publish.push_results(world["url"], "colab-results", bundle, job, "OK", str(tmp_path / "w"),
                         lambda: bootstrap.git_env(TOKEN, str(tmp_path / "w" / "ask")))
    pushed = results_files(world["base"], world["url"], "smoke-001")
    assert {"sub", "job.json", "bundle_summary.md"} <= pushed and not pushed & {"train", "first.txt", "publish.json"}


def test_a_bundle_without_the_files_record_is_pushed_whole(world, tmp_path):
    """A bundle folder written before the record (an old Drive bundle given to republish) keeps the old behaviour."""
    bundle = tmp_path / "old_bundle"
    bundle.mkdir()
    for name in ("job.json", "env.json", "run_log.txt", "bundle_summary.md", "result.json"):
        (bundle / name).write_text("{}")
    (bundle / "publish.json").write_text(json.dumps({"status": "OK", "pushed": False}))
    assert publish.bundle_files(bundle) is None
    publish.push_results(world["url"], "colab-results", bundle, make_job(world["sha"]), "OK", str(tmp_path / "w"),
                         lambda: bootstrap.git_env(TOKEN, str(tmp_path / "w" / "ask")))
    assert "result.json" in results_files(world["base"], world["url"], "smoke-001")


def test_the_wall_clock_guard_kills_a_runaway_job(tmp_path):
    t0 = time.time()
    log = tmp_path / "l.log"
    cmd = [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(60)"]
    code, timed_out = run_logged(cmd, str(tmp_path), dict(os.environ), str(log), 1, bootstrap.Secrets(TOKEN),
                                 echo=lambda *_: None)
    assert timed_out and code != 0 and time.time() - t0 < 20 and "start" in log.read_text()


def test_the_runner_refuses_a_missing_token(tmp_path):
    with pytest.raises(bootstrap.BootstrapError, match="token is required"):
        Runner("file:///x", "short", str(tmp_path))


# ------------------------------------------------------------------------------------------------ guards (B3)
def test_no_module_of_the_runner_depends_on_tmp():
    """Rule E2 as a scan of the class: no '/tmp' path and no tempfile/mkdtemp use anywhere in colab/ or jobs/
    (the one exception is jobs/device_smoke.py READING tempfile.gettempdir() to show where TMPDIR points)."""
    offenders = []
    for folder in ("colab", "jobs"):
        for p in (PKG_DIR / folder).glob("*.py"):
            text = p.read_text(encoding="utf-8")
            code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
            if re.search(r"""["']/tmp\b""", code) or "mkdtemp" in code or "TemporaryDirectory" in code \
                    or ("tempfile" in code and p.name != "device_smoke.py"):
                offenders.append(p.name)
    assert not offenders, f"rule E2: {offenders} use a temp location"


def test_every_failing_step_reaches_the_notebook_as_a_message(world):
    """DeviceError is caught per step (not leaked as a traceback that skips publish)."""
    queue(world)
    r = runner(world, device_check=lambda expected: (_ for _ in ()).throw(DeviceError("boom"))).run_all()
    assert r["status"] == "FAIL" and any("boom" in p for p in r["problems"])
