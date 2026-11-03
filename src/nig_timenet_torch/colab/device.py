"""The device a Colab job runs on: report it, assert it, record it (rule E1).

TF's "GPU" runs were CPU runs for months (a ``tensorflow-cpu`` pin). Here a job names the device it expects
(``"cuda"`` or ``"cpu"``); ``assert_device`` fails BEFORE the job starts when the VM cannot give it, ``device_report``
is the record written next to the results (``env.json`` of the bundle, ``device.json`` of the job, ``setup.json`` of
the notebook), and a smoke computation proves a tensor really lands on the device. Colab ships a CUDA build of
torch, so nothing here ever installs or replaces torch: when the installed build has no CUDA, the error says so and
stops.
"""

import platform
import sys
from datetime import UTC, datetime

EXPECTED = ("cuda", "cpu")


class DeviceError(RuntimeError):
    """The VM cannot provide the device a job expects (the message says what to change)."""


def _torch():
    import torch

    return torch


def device_report(expected: str | None = None, torch_mod=None) -> dict:
    """Plain-dict record of the device situation; JSON-serialisable."""
    torch = torch_mod or _torch()
    available = bool(torch.cuda.is_available())
    rep = {"expected": expected, "used": "cuda" if available else "cpu", "cuda_available": available,
           "torch_version": str(torch.__version__), "torch_cuda_build": getattr(torch.version, "cuda", None),
           "device_count": int(torch.cuda.device_count()) if available else 0, "gpu_name": None,
           "gpu_capability": None, "gpu_memory_gb": None, "python": sys.version.split()[0],
           "platform": platform.platform(), "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    if available:
        props = torch.cuda.get_device_properties(0)
        rep.update(gpu_name=torch.cuda.get_device_name(0), gpu_capability=list(torch.cuda.get_device_capability(0)),
                   gpu_memory_gb=round(props.total_memory / 2**30, 2))
    return rep


def assert_device(expected: str, torch_mod=None) -> dict:
    """The device report if the VM gives ``expected``; else DeviceError. A GPU job never falls back to CPU."""
    if expected not in EXPECTED:
        raise DeviceError(f"expected device must be one of {EXPECTED}, got {expected!r}")
    rep = device_report(expected, torch_mod)
    if expected == "cuda" and not rep["cuda_available"]:
        if rep["torch_cuda_build"] is None:
            raise DeviceError(f"the job expects cuda but torch {rep['torch_version']} is a CPU-only build; Colab's own "
                              f"torch is CUDA-enabled, so this VM's torch was replaced (do not reinstall torch: use a "
                              f"fresh runtime)")
        raise DeviceError("the job expects cuda but no GPU is visible: Runtime > Change runtime type > T4 GPU, "
                          "then restart the session")
    if expected == "cpu":
        rep["used"] = "cpu"
    return rep


def smoke_compute(expected: str, torch_mod=None) -> dict:
    """A small matmul on the expected device; returns where the result tensor really lives. Raises DeviceError when
    that is not the expected device (e.g. cuda reported but the allocation lands on cpu)."""
    torch = torch_mod or _torch()
    dev = torch.device(expected)
    x = torch.ones((64, 64), device=dev)
    y = (x @ x).sum()
    if expected == "cuda":
        torch.cuda.synchronize()
    landed = y.device.type
    if landed != expected:
        raise DeviceError(f"a tensor allocated for {expected} landed on {landed}")
    return {"device": landed, "checksum": float(y)}
