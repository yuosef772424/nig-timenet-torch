"""PURPOSE: A cached Python environment on Drive for the notebook: a venv (--system-site-packages, so Colab's CUDA
    torch stays) with the project's requirements, as ONE tar keyed by Python + torch + requirements; new key = rebuild.
TAGS: env cache, venv tar, cache key, requirements hash, drive cache, restore environment, activate venv
PITFALLS: the key covers Python, torch and the requirement lines only, not the platform; numpy/pandas that the kernel
    already imported are not swapped by activate_env (it lists them: restart the runtime); the venv is restored to
    the same path it was built at (its scripts hold absolute paths); torch is never installed (a constraint pins it).

Why: a fresh Colab VM spends minutes on ``pip install`` of the same wheels every session. ``ensure_env`` builds
``python -m venv --system-site-packages`` once, installs the requirements of ``pyproject.toml`` (dependencies plus the
``data`` extra, minus torch: Colab's own build is the base and is pinned by a constraint file), packs the venv into
``<cache_dir>/env-<key>.tar`` through a temp name, and later sessions just unpack it. Any change of Python, torch or a
requirement line gives a new key, builds again and removes the stale tar. The subprocess runner is injected, so the
tests need no network.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from importlib import metadata
from pathlib import Path

__all__ = ["EnvCacheError", "requirements_of", "cache_key", "environment_key", "tar_path", "pack", "unpack",
           "ensure_env", "activate_env", "KEY_FILE"]

KEY_FILE = ".nig_env_key"
_TORCH = re.compile(r"^\s*torch\b", re.IGNORECASE)


class EnvCacheError(Exception):
    """The environment cannot be built or restored; the message says what to do."""


def requirements_of(pyproject_text: str, extras: tuple = ("data",)) -> list[str]:
    """The requirement lines to install: ``[project].dependencies`` plus the named extras, minus torch, sorted."""
    project = tomllib.loads(pyproject_text).get("project", {})
    lines = list(project.get("dependencies", []))
    for extra in extras:
        if extra not in project.get("optional-dependencies", {}):
            raise EnvCacheError(f"pyproject.toml has no extra {extra!r}")
        lines += project["optional-dependencies"][extra]
    return sorted({ln.strip() for ln in lines if ln.strip() and not _TORCH.match(ln)})


def cache_key(python_version: str, torch_version: str, requirements: list[str]) -> str:
    """16 hex characters: the same Python, torch and requirement lines (any order) give the same key."""
    blob = json.dumps({"python": python_version, "torch": torch_version, "requirements": sorted(requirements)},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _torch_version() -> str:
    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        raise EnvCacheError("torch is not installed here; the cached env sits on top of Colab's torch") from None


def environment_key(pyproject: str, extras: tuple = ("data",), python_version: str | None = None,
                    torch_version: str | None = None) -> tuple[str, list[str]]:
    reqs = requirements_of(Path(pyproject).read_text(encoding="utf-8"), extras)
    py = python_version or "%d.%d.%d" % sys.version_info[:3]
    return cache_key(py, torch_version or _torch_version(), reqs), reqs


def tar_path(cache_dir: str, key: str) -> Path:
    return Path(cache_dir) / f"env-{key}.tar"


def pack(venv_dir: str, tar_file: Path) -> None:
    """Write the venv into ``tar_file`` through a temp name, so a cut write never leaves a half tar under the name."""
    tar_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = tar_file.with_name(tar_file.name + ".tmp")
    with tarfile.open(tmp, "w") as tf:
        tf.add(venv_dir, arcname=".")
    os.replace(tmp, tar_file)


def unpack(tar_file: Path, venv_dir: str) -> None:
    """Replace ``venv_dir`` by the tar's content. Refuses members that would land outside it."""
    root = Path(venv_dir).resolve()
    with tarfile.open(tar_file) as tf:
        for m in tf.getmembers():
            if os.path.isabs(m.name) or not (root / m.name).resolve().is_relative_to(root):
                raise EnvCacheError(f"{tar_file} holds an unsafe path {m.name!r}; delete it to rebuild the env")
        shutil.rmtree(venv_dir, ignore_errors=True)
        os.makedirs(venv_dir)
        opts = {"filter": "tar"} if hasattr(tarfile, "tar_filter") else {}   # keep the venv's symlinks
        tf.extractall(venv_dir, members=tf.getmembers(), **opts)  # noqa: S202 - members checked above, the tar is ours


def _site_packages(venv_dir: str) -> Path:
    found = sorted(Path(venv_dir).glob("lib/python*/site-packages"))
    if not found:
        raise EnvCacheError(f"no site-packages in {venv_dir}")
    return found[0]


def _pip(run, venv_dir: str, reqs: list[str], torch_version: str, workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "requirements.txt").write_text("\n".join(reqs) + "\n", encoding="utf-8")
    (workdir / "constraints.txt").write_text(f"torch=={torch_version}\n", encoding="utf-8")
    cmd = [os.path.join(venv_dir, "bin", "python"), "-m", "pip", "install", "-q", "-r",
           str(workdir / "requirements.txt"), "-c", str(workdir / "constraints.txt")]
    r = run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise EnvCacheError(f"pip install failed (exit {r.returncode}): {(r.stderr or r.stdout)[-500:]}")


def ensure_env(pyproject: str, venv_dir: str, cache_dir: str, *, extras: tuple = ("data",),
               python_version: str | None = None, torch_version: str | None = None, python: str | None = None,
               run=subprocess.run, echo=print) -> dict:
    """The environment of ``pyproject`` at ``venv_dir``: unpacked from ``cache_dir`` when a tar with this key exists,
    else built, packed and the stale tars removed. -> {"key", "source": "cache"|"built", "tar", "removed"}."""
    key, reqs = environment_key(pyproject, extras, python_version, torch_version)
    torch_v = torch_version or _torch_version()
    tar = tar_path(cache_dir, key)
    if tar.is_file():
        try:
            unpack(tar, venv_dir)
            if Path(venv_dir, KEY_FILE).read_text(encoding="utf-8").strip() == key:
                echo(f"env restored from the cache ({tar.name}, {tar.stat().st_size / 1e6:.0f} MB)")
                return {"key": key, "source": "cache", "tar": str(tar), "removed": []}
        except (tarfile.TarError, OSError) as e:
            echo(f"cached env unusable ({type(e).__name__}): rebuilding")
        tar.unlink(missing_ok=True)
    shutil.rmtree(venv_dir, ignore_errors=True)
    cmd = [python or sys.executable, "-m", "venv", "--system-site-packages", venv_dir]
    r = run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise EnvCacheError(f"venv creation failed (exit {r.returncode}): {(r.stderr or r.stdout)[-500:]}")
    _pip(run, venv_dir, reqs, torch_v, Path(venv_dir).with_name(Path(venv_dir).name + "_build"))
    Path(venv_dir, KEY_FILE).write_text(key + "\n", encoding="utf-8")
    pack(venv_dir, tar)
    removed = [p.name for p in Path(cache_dir).glob("env-*.tar") if p != tar]
    for name in removed:
        Path(cache_dir, name).unlink()
    echo(f"env built and cached ({tar.name}); stale caches removed: {removed or 'none'}")
    return {"key": key, "source": "built", "tar": str(tar), "removed": removed}


def activate_env(venv_dir: str, names: tuple = ("numpy", "pandas", "pyarrow")) -> dict:
    """Put the venv's site-packages first on ``sys.path``. -> {"site_packages", "already_imported"}: the ``names``
    the kernel imported before, which keep their old version until the runtime restarts."""
    sp = str(_site_packages(venv_dir))
    if sp not in sys.path:
        sys.path.insert(0, sp)
    return {"site_packages": sp, "already_imported": [n for n in names if n in sys.modules]}
