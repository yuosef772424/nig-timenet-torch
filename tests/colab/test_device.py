"""colab/device.py (rule E1): the device is asserted, never silently replaced, and recorded."""

import json
import re
from types import SimpleNamespace

import pytest

from nig_timenet_torch.colab import contract, device


def fake_torch(cuda: bool, build="12.6"):
    props = SimpleNamespace(total_memory=15 * 2**30)

    class T:  # a "tensor" that always lands on the device it was asked for
        def __init__(self, dev):
            self.device = SimpleNamespace(type=dev if not isinstance(dev, SimpleNamespace) else dev.type)

        def __matmul__(self, other):
            return self

        def sum(self):
            return self

        def __float__(self):
            return 4096.0

    return SimpleNamespace(
        __version__="2.11.0+cu126" if build else "2.11.0+cpu", version=SimpleNamespace(cuda=build),
        cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: 1, get_device_properties=lambda i: props,
                             get_device_name=lambda i: "Tesla T4", get_device_capability=lambda i: (7, 5),
                             synchronize=lambda: None),
        device=lambda name: SimpleNamespace(type=name), ones=lambda shape, device: T(device))


def test_report_on_a_gpu_vm_names_the_gpu_and_the_cuda_build():
    rep = device.device_report("cuda", fake_torch(True))
    got = (rep["used"], rep["gpu_name"], rep["torch_cuda_build"], rep["gpu_capability"])
    assert got == ("cuda", "Tesla T4", "12.6", [7, 5])
    json.dumps(rep)  # serialisable


def test_a_cuda_job_on_a_cpu_vm_fails_and_says_what_to_change():
    with pytest.raises(device.DeviceError, match="no GPU is visible.*T4 GPU"):
        device.assert_device("cuda", fake_torch(False))


def test_a_cuda_job_with_a_cpu_only_torch_build_says_torch_was_replaced():
    with pytest.raises(device.DeviceError, match="CPU-only build.*do not reinstall torch"):
        device.assert_device("cuda", fake_torch(False, build=None))


def test_unknown_device_is_refused():
    with pytest.raises(device.DeviceError, match="must be one of"):
        device.assert_device("tpu", fake_torch(True))


def test_cuda_passes_on_a_gpu_vm_and_cpu_passes_everywhere():
    assert device.assert_device("cuda", fake_torch(True))["used"] == "cuda"
    assert device.assert_device("cpu", fake_torch(True))["used"] == "cpu"
    assert device.assert_device("cpu", fake_torch(False))["used"] == "cpu"


def test_smoke_compute_proves_the_tensor_lands_on_the_device():
    assert device.smoke_compute("cuda", fake_torch(True))["device"] == "cuda"
    bad = fake_torch(True)
    bad.ones = lambda shape, device: _CpuTensor()
    with pytest.raises(device.DeviceError, match="landed on cpu"):
        device.smoke_compute("cuda", bad)


class _CpuTensor:
    device = SimpleNamespace(type="cpu")

    def __matmul__(self, other):
        return self

    def sum(self):
        return self

    def __float__(self):
        return 1.0


def test_real_torch_cpu_smoke_and_device_json(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    rep = contract.write_device_json(out, "cpu")
    on_disk = json.loads((out / "device.json").read_text())
    assert on_disk["used"] == "cpu" and on_disk["smoke"]["device"] == "cpu" and rep["torch_version"]
    with pytest.raises(device.DeviceError):
        contract.write_device_json(out, "cuda")  # this sandbox has no GPU: a cuda job must fail, not fall back


def test_the_runner_package_never_installs_or_replaces_torch():
    """The "no reinstall" claim: nothing under colab/ may pip-install torch."""
    from tests.scan import PKG_DIR

    for p in (PKG_DIR / "colab").glob("*.py"):
        assert not re.search(r"pip.{0,40}install.{0,60}['\"]torch", p.read_text(encoding="utf-8")), p.name
