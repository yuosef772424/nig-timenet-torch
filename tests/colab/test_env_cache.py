"""colab/env_cache.py: the cache key, building into a tar, restoring from it, a key change, a corrupt tar. No network:
the subprocess runner is a fake that makes the folders pip and venv would make."""

import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from nig_timenet_torch.colab import env_cache as ec

PYPROJECT = """[project]
name = "x"
dependencies = ["numpy==2.4.6", "pandas==3.0.6", "torch>=2.15,<2.16"]
[project.optional-dependencies]
data = ["requests>=2.31"]
dev = ["pytest"]
"""


class FakeRun:
    """Plays `python -m venv ...` and `pip install ...`; records every command."""

    def __init__(self, fail_pip=False):
        self.cmds, self.fail_pip = [], fail_pip

    def __call__(self, cmd, capture_output=True, text=True):
        self.cmds.append(cmd)
        if "venv" in cmd:
            venv = Path(cmd[-1])
            (venv / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
            (venv / "bin").mkdir()
            (venv / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
            (venv / "bin" / "python").symlink_to(sys.executable)
        elif self.fail_pip:
            return subprocess.CompletedProcess(cmd, 1, "", "no matching distribution")
        else:
            (Path(cmd[0]).parents[1] / "lib/python3.13/site-packages/requests.py").write_text("# fake wheel\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture()
def project(tmp_path):
    p = tmp_path / "pyproject.toml"
    p.write_text(PYPROJECT)
    return str(p)


def test_requirements_drop_torch_and_take_the_extra():
    assert ec.requirements_of(PYPROJECT) == ["numpy==2.4.6", "pandas==3.0.6", "requests>=2.31"]
    assert ec.requirements_of(PYPROJECT, extras=()) == ["numpy==2.4.6", "pandas==3.0.6"]
    with pytest.raises(ec.EnvCacheError, match="no extra"):
        ec.requirements_of(PYPROJECT, extras=("nope",))


def test_the_key_changes_with_python_torch_or_a_requirement_and_not_with_order():
    base = ec.cache_key("3.12.1", "2.15.0", ["a==1", "b==2"])
    assert base == ec.cache_key("3.12.1", "2.15.0", ["b==2", "a==1"])
    assert len(base) == 16
    assert base != ec.cache_key("3.12.2", "2.15.0", ["a==1", "b==2"])
    assert base != ec.cache_key("3.12.1", "2.15.1+cu126", ["a==1", "b==2"])
    assert base != ec.cache_key("3.12.1", "2.15.0", ["a==1", "b==3"])
    assert base != ec.cache_key("3.12.1", "2.15.0", ["a==1"])


def test_the_key_of_a_pyproject_ignores_the_torch_pin_but_not_the_installed_torch(project):
    k1, _ = ec.environment_key(project, python_version="3.12.0", torch_version="2.15.0")
    k2, _ = ec.environment_key(project, python_version="3.12.0", torch_version="2.15.0")
    k3, _ = ec.environment_key(project, python_version="3.12.0", torch_version="2.15.1")
    assert k1 == k2 != k3
    Path(project).write_text(PYPROJECT.replace("torch>=2.15,<2.16", "torch>=2.16"))
    assert ec.environment_key(project, python_version="3.12.0", torch_version="2.15.0")[0] == k1  # torch line is no key


def test_first_call_builds_and_packs_the_second_restores_without_pip(tmp_path, project):
    cache, venv, out = tmp_path / "drive_cache", str(tmp_path / "venv"), []
    kw = dict(python_version="3.12.0", torch_version="2.15.0+cu126", echo=out.append)
    run = FakeRun()
    first = ec.ensure_env(project, venv, str(cache), run=run, **kw)
    assert first["source"] == "built" and Path(first["tar"]).is_file() and len(run.cmds) == 2
    pip = run.cmds[1]
    assert pip[:3] == [str(Path(venv, "bin", "python")), "-m", "pip"] and "-c" in pip
    constraints = Path(pip[pip.index("-c") + 1]).read_text()
    assert constraints == "torch==2.15.0+cu126\n"  # torch is pinned, never installed
    assert "torch" not in Path(pip[pip.index("-r") + 1]).read_text()
    assert not list(cache.glob("*.tmp"))
    import shutil

    shutil.rmtree(venv)  # a new VM: nothing local
    run2 = FakeRun()
    second = ec.ensure_env(project, venv, str(cache), run=run2, **kw)
    assert second["source"] == "cache" and run2.cmds == [] and second["key"] == first["key"]
    assert (Path(venv) / "lib/python3.13/site-packages/requests.py").is_file()
    assert Path(venv, "bin", "python").is_symlink()  # symlinks survive the tar


def test_a_changed_requirement_or_torch_rebuilds_and_removes_the_stale_tar(tmp_path, project):
    cache, venv = tmp_path / "c", str(tmp_path / "venv")
    kw = dict(python_version="3.12.0", echo=lambda m: None)
    a = ec.ensure_env(project, venv, str(cache), run=FakeRun(), torch_version="2.15.0", **kw)
    b = ec.ensure_env(project, venv, str(cache), run=FakeRun(), torch_version="2.15.1", **kw)  # Colab updated torch
    assert b["source"] == "built" and b["key"] != a["key"] and b["removed"] == [Path(a["tar"]).name]
    assert [p.name for p in cache.glob("env-*.tar")] == [Path(b["tar"]).name]
    Path(project).write_text(PYPROJECT.replace("2.4.6", "2.4.7"))
    c = ec.ensure_env(project, venv, str(cache), run=FakeRun(), torch_version="2.15.1", **kw)
    assert c["source"] == "built" and len(list(cache.glob("env-*.tar"))) == 1


def test_a_corrupt_tar_is_rebuilt_and_a_failed_pip_leaves_no_tar(tmp_path, project):
    cache, venv = tmp_path / "c", str(tmp_path / "venv")
    kw = dict(python_version="3.12.0", torch_version="2.15.0", echo=lambda m: None)
    key, _ = ec.environment_key(project, python_version="3.12.0", torch_version="2.15.0")
    cache.mkdir()
    ec.tar_path(str(cache), key).write_bytes(b"not a tar at all")
    assert ec.ensure_env(project, venv, str(cache), run=FakeRun(), **kw)["source"] == "built"
    other = tmp_path / "c2"
    with pytest.raises(ec.EnvCacheError, match="pip install failed"):
        ec.ensure_env(project, venv, str(other), run=FakeRun(fail_pip=True), **kw)
    assert not other.exists() or not list(other.glob("env-*"))


def test_unpack_refuses_paths_outside_the_venv(tmp_path):
    bad = tmp_path / "bad.tar"
    with tarfile.open(bad, "w") as tf:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 0
        tf.addfile(info)
    with pytest.raises(ec.EnvCacheError, match="unsafe path"):
        ec.unpack(bad, str(tmp_path / "venv"))
    assert not (tmp_path / "escape.txt").exists()


def test_activate_puts_site_packages_first_and_reports_modules_already_imported(tmp_path, monkeypatch):
    sp = tmp_path / "venv/lib/python3.13/site-packages"
    sp.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", list(sys.path))
    got = ec.activate_env(str(tmp_path / "venv"), names=("sys", "no_such_module_xyz"))
    assert sys.path[0] == str(sp) and got["already_imported"] == ["sys"]
    with pytest.raises(ec.EnvCacheError, match="no site-packages"):
        ec.activate_env(str(tmp_path / "missing"))
