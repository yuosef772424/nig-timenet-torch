"""The run contract between the runner and a job entry (ROADMAP T3), both sides in one place.

Entry side: ``entry_args`` reads the flags the runner passes, ``write_device_json`` records the device (rule E1) and
proves a tensor lands on it, ``write_manifest`` hashes every output. Runner side: ``validate_outputs`` recomputes
every hash and checks the manifest, the expected outputs and the recorded device against what the JOB asked for, so
an entry cannot report success on the wrong device (a GPU job whose ``device.json`` says cpu FAILS).
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from nig_timenet_torch.colab import device
from nig_timenet_torch.colab.job import BUNDLE_NAMES

MANIFEST_REQUIRED = ("git_sha", "config", "input_fingerprint", "outputs", "device")
GIT_MAX_BYTES = 1024 * 1024  # an output above this stays on Drive; the git results branch gets the small ones


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _outputs(out_dir) -> list[Path]:
    root = Path(out_dir)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json")


# ------------------------------------------------------------------------------------------------ entry side
def entry_args(argv: list[str] | None = None) -> tuple[dict, Path, int, list[str]]:
    """-> (config dict, out dir, max minutes, remaining args) from the flags the runner passes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=int, required=True)
    ns, rest = ap.parse_known_args(argv)
    config = json.loads(Path(ns.config).read_text(encoding="utf-8"))
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    return config, out, ns.max_minutes, rest


def write_device_json(out_dir, expected: str, torch_mod=None) -> dict:
    """Assert ``expected`` on this VM, compute on it, and write ``device.json``. Raises DeviceError (the entry then
    exits non-zero and writes nothing else, so the job fails loudly)."""
    rep = device.assert_device(expected, torch_mod)
    rep["smoke"] = device.smoke_compute(expected, torch_mod)
    (Path(out_dir) / "device.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def write_manifest(out_dir, config: dict, input_fingerprint: str, started: float | None = None) -> dict:
    """Hash every output file and write ``manifest.json`` (last: it is the completion marker). The commit is read
    from the checkout the runner points at (``NIG_REPO_DIR``), not from an argument."""
    repo_dir = os.environ["NIG_REPO_DIR"]
    sha = subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "HEAD"], text=True).strip()
    root = Path(out_dir)
    dev = json.loads((root / "device.json").read_text(encoding="utf-8"))
    man = {"git_sha": sha, "config": config, "input_fingerprint": input_fingerprint,
           "outputs": {p.relative_to(root).as_posix(): sha256_file(p) for p in _outputs(root)},
           "device": {k: dev[k] for k in ("expected", "used", "gpu_name", "torch_version", "torch_cuda_build")},
           "runtime_seconds": round(time.time() - started, 1) if started else None}
    (root / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
    return man


# ------------------------------------------------------------------------------------------------ runner side
def _read_json(path: Path, problems: list[str], what: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        problems.append(f"{what} is missing or not valid JSON")
        return None


def _check_device(job: dict, dev: dict, man: dict, problems: list[str]) -> None:
    want = job["device"]
    if dev.get("expected") != want or dev.get("used") != want:
        problems.append(f"device.json says expected={dev.get('expected')!r} used={dev.get('used')!r}, "
                        f"the job asked for {want!r} (rule E1)")
    if want == "cuda" and not (dev.get("cuda_available") and dev.get("gpu_name") and dev.get("torch_cuda_build")):
        problems.append("device.json: a cuda job must record cuda_available, gpu_name and torch_cuda_build")
    if (dev.get("smoke") or {}).get("device") != want:
        problems.append(f"device.json: the smoke computation did not run on {want!r}")
    for k, v in (man.get("device") or {}).items():
        if dev.get(k) != v:
            problems.append(f"manifest.device.{k} differs from device.json")


def validate_outputs(out_dir, job: dict, repo_sha: str) -> list[str]:
    """Problems found in a finished run's output folder (empty list = valid)."""
    root = Path(out_dir)
    problems: list[str] = []
    man = _read_json(root / "manifest.json", problems, "manifest.json")
    dev = _read_json(root / "device.json", problems, "device.json")
    if man is None or dev is None:
        return problems
    missing = [k for k in MANIFEST_REQUIRED if k not in man]
    if missing:
        return problems + [f"manifest.json lacks {missing}"]
    if man["git_sha"] != repo_sha:
        problems.append(f"manifest git_sha {man['git_sha']} is not the job's repo_ref {repo_sha}")
    if not isinstance(man["input_fingerprint"], str) or not man["input_fingerprint"]:
        problems.append("manifest input_fingerprint must be a non-empty string")
    actual = {p.relative_to(root).as_posix() for p in _outputs(root)}
    for name in job["expected_outputs"]:
        if name not in actual:
            problems.append(f"expected output {name} was not written")
    for name in sorted(actual - set(man["outputs"])):
        problems.append(f"{name} is in the folder but not in manifest.outputs")
    for name, want in man["outputs"].items():
        p = root / name
        if not p.is_file():
            problems.append(f"manifest lists {name} but it is missing")
        elif sha256_file(p) != want:
            problems.append(f"{name} does not match its sha256 in the manifest")
        if name in BUNDLE_NAMES:
            problems.append(f"{name} collides with a file the bundle writes itself")
    _check_device(job, dev, man, problems)
    return problems
