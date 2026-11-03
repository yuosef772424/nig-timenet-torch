"""Section 1 of the notebook, second half (ROADMAP T3): check the device the run will use and record the whole setup.

The clones and the install happen in ``bootstrap.py`` (nothing of this package exists on the VM before them). This
module then asserts the expected device (rule E1), checks that the torch Colab ships can see CUDA without being
reinstalled, and writes ``setup.json``: commits of both repos, device report, package versions. The record goes to
the work dir and, when given, to a Drive folder, so the next session can read it without the owner copying anything.
"""

import json
import os
from pathlib import Path

from nig_timenet_torch.colab import device
from nig_timenet_torch.colab.runner import package_versions


def check_environment(expect_device: str, commits: dict, install: dict, workdir: str, record_dir: str | None = None,
                      echo=print, device_check=device.assert_device) -> dict:
    """Assert ``expect_device``, write ``setup.json`` and return the record. Raises DeviceError on a mismatch AFTER
    writing the record, so the failure itself is on file."""
    record = {"commits": dict(commits), "install": dict(install), "packages": package_versions(),
              "device": device.device_report(expect_device)}
    error = None
    try:
        record["device"] = device_check(expect_device)
    except device.DeviceError as e:
        error = e
        record["device_error"] = str(e)
    for folder in filter(None, (workdir, record_dir)):
        try:
            os.makedirs(folder, exist_ok=True)
            Path(folder, "setup.json").write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
        except OSError as e:
            echo(f"could not write setup.json to {folder}: {type(e).__name__}")
    d = record["device"]
    same = install.get("torch_before") == install.get("torch_after")
    echo(f"torch {d['torch_version']} (cuda build {d['torch_cuda_build']}, unchanged by the install: {same}) | "
         f"device used {d['used']} | gpu {d['gpu_name']}")
    for name, sha in commits.items():
        echo(f"commit {name}: {sha}")
    if error is not None:
        raise error
    return record
