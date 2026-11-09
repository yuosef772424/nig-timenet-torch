"""colab/job.py and colab/contract.py: the job schema and the output contract (rule E1 in the validator)."""

import json
import shutil

import pytest

from nig_timenet_torch.colab import contract
from nig_timenet_torch.colab.job import JobError, parse_job, validate_job
from tests.colab.helpers import git, make_job

SHA = "a" * 40


def job(**over):
    return make_job(SHA, **over)


def test_a_valid_job_parses_and_defaults_rerun():
    j = parse_job(json.dumps(job()))
    assert j["rerun"] is False and j["device"] == "cpu"


@pytest.mark.parametrize("change,match", [
    ({"device": "tpu"}, "device must be one of"),
    ({"device": None}, "device must be one of"),
    ({"repo_ref": "main"}, "40-hex"),
    ({"repo_ref": "A" * 40}, "40-hex"),
    ({"entry": "../x"}, "entry must be a module name"),
    ({"entry": "tools/x.py"}, "entry must be a module name"),
    ({"job_id": "a"}, "job_id"),
    ({"job_id": "a..b"}, "job_id"),
    ({"max_minutes": 0}, "max_minutes"),
    ({"max_minutes": True}, "max_minutes"),
    ({"max_minutes": 999}, "max_minutes"),
    ({"args": ["--out=/x"]}, "may not set"),
    ({"args": [1]}, "args must be"),
    ({"expected_outputs": []}, "expected_outputs"),
    ({"expected_outputs": ["manifest.json"]}, "expected_outputs"),
    ({"expected_outputs": ["a.json", "a.json"]}, "expected_outputs"),
    ({"expected_outputs": ["../x"]}, "expected_outputs"),
    ({"needs_drive": "yes"}, "needs_drive"),
    ({"rerun": 1}, "rerun"),
])
def test_invalid_jobs_are_refused(change, match):
    with pytest.raises(JobError, match=match):
        validate_job(job(**change))


def test_missing_unknown_and_duplicate_keys_are_refused():
    j = job()
    del j["device"]  # rule E1: a job that does not name its device is refused, not defaulted
    with pytest.raises(JobError, match="missing \\['device'\\]"):
        validate_job(j)
    with pytest.raises(JobError, match="unknown \\['colour'\\]"):
        validate_job(job(colour="red"))
    with pytest.raises(JobError, match="duplicate keys"):
        parse_job('{"job_id": "abc", "job_id": "abd"}')
    with pytest.raises(JobError, match="not valid JSON"):
        parse_job("{nope")
    with pytest.raises(JobError, match="JSON object"):
        parse_job("[]")


# ------------------------------------------------------------------------------------------------ contract
@pytest.fixture()
def repo(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    r.mkdir()
    git("init", "-q", cwd=r)
    (r / "f").write_text("x")
    git("add", "f", cwd=r)
    git("commit", "-q", "-m", "c", cwd=r)
    monkeypatch.setenv("NIG_REPO_DIR", str(r))
    return git("rev-parse", "HEAD", cwd=r)


def finished_run(tmp_path, sha, device="cpu"):
    out = tmp_path / "out"
    out.mkdir()
    contract.write_device_json(out, device)
    (out / "smoke.json").write_text("{}")
    contract.write_manifest(out, {"device": device}, "fp-1")
    return out


def test_a_finished_run_validates(tmp_path, repo):
    out = finished_run(tmp_path, repo)
    assert contract.validate_outputs(out, job(), repo) == []


def test_a_changed_output_a_missing_output_and_a_wrong_commit_are_problems(tmp_path, repo):
    out = finished_run(tmp_path, repo)
    (out / "smoke.json").write_text('{"tampered": 1}')
    (out / "extra.csv").write_text("a")
    p = contract.validate_outputs(out, job(expected_outputs=["smoke.json", "gone.json"]), "b" * 40)
    text = " | ".join(p)
    assert "does not match its sha256" in text and "extra.csv is in the folder but not in manifest" in text
    assert "expected output gone.json was not written" in text and "is not the job's repo_ref" in text


def test_a_gpu_job_whose_device_json_says_cpu_fails_validation(tmp_path, repo):
    """Rule E1, validator side: the entry cannot report success on the wrong device."""
    out = finished_run(tmp_path, repo, device="cpu")
    p = contract.validate_outputs(out, job(device="cuda"), repo)
    assert any("the job asked for 'cuda'" in x for x in p) and any("cuda_available" in x for x in p)
    assert any("smoke computation did not run on 'cuda'" in x for x in p)


def test_device_json_edited_after_the_manifest_is_caught(tmp_path, repo):
    out = finished_run(tmp_path, repo)
    d = json.loads((out / "device.json").read_text())
    d["gpu_name"] = "Tesla T4"
    (out / "device.json").write_text(json.dumps(d))
    p = contract.validate_outputs(out, job(), repo)
    assert any("device.json does not match" in x or "manifest.device.gpu_name" in x for x in p)


def test_missing_manifest_or_device_json_is_a_problem(tmp_path, repo):
    out = finished_run(tmp_path, repo)
    shutil.copy(out / "manifest.json", tmp_path / "m.json")
    (out / "manifest.json").unlink()
    assert any("manifest.json is missing" in x for x in contract.validate_outputs(out, job(), repo))
    shutil.copy(tmp_path / "m.json", out / "manifest.json")
    (out / "device.json").unlink()
    assert any("device.json is missing" in x for x in contract.validate_outputs(out, job(), repo))


def test_entry_args_reads_the_runner_flags(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text('{"k": 1}')
    config, out, minutes, rest = contract.entry_args(["--config", str(cfg), "--out", str(tmp_path / "o"),
                                                      "--max-minutes", "7", "--extra", "1"])
    assert (config, minutes, rest) == ({"k": 1}, 7, ["--extra", "1"]) and out.is_dir()


def test_job_device_and_config_device_must_agree():
    import pytest

    from nig_timenet_torch.colab import job as job_mod
    with pytest.raises(job_mod.JobError, match="differs from config.device"):
        job_mod.check_entry_config("device_smoke", {"device": "cpu"}, "cuda")
    job_mod.check_entry_config("device_smoke", {"device": "cuda"}, "cuda")


def test_every_jobs_entry_exposes_a_strict_frozen_config():
    """HF-17: validation at queue time needs ``CONFIG`` in every entry module of jobs/."""
    import dataclasses
    import importlib
    import pkgutil

    from nig_timenet_torch import jobs
    entries = [m.name for m in pkgutil.iter_modules(jobs.__path__) if not m.name.startswith("_")]
    assert entries, "jobs/ has no entry"
    for name in entries:
        cfg = getattr(importlib.import_module(f"nig_timenet_torch.jobs.{name}"), "CONFIG", None)
        assert cfg is not None and dataclasses.is_dataclass(cfg), f"jobs/{name}.py must expose CONFIG (a dataclass)"
        assert cfg.__dataclass_params__.frozen, f"jobs/{name}.py CONFIG must be frozen"


def _vjob(**over):
    return job(entry="no_entry_module", **over)


def test_versioned_renames_the_job_and_its_run_dir_segment_only():
    from nig_timenet_torch.colab.job import version_id, versioned
    j = validate_job(_vjob(job_id="base-smoke", on_existing="new_version",
                           config={"run_dir": "/d/runs/base-smoke/train"}))
    v2 = versioned(j, version_id("base-smoke", 2))
    assert v2["job_id"] == "base-smoke-v2" and v2["base_job_id"] == "base-smoke"
    assert v2["config"]["run_dir"] == "/d/runs/base-smoke-v2/train" and j["config"]["run_dir"].endswith("smoke/train")
    assert versioned(v2, version_id("base-smoke", 3))["base_job_id"] == "base-smoke"
    with pytest.raises(JobError, match="run_dir"):           # a run_dir without the id: versions would share it
        validate_job(_vjob(on_existing="new_version", config={"run_dir": "/d/runs/other/train"}))
    with pytest.raises(JobError, match="exclude"):
        validate_job(_vjob(on_existing="new_version", rerun=True))
    with pytest.raises(JobError, match="on_existing"):
        validate_job(_vjob(on_existing="overwrite"))
    assert validate_job(_vjob())["on_existing"] == "refuse"  # default: the behaviour before D-021
