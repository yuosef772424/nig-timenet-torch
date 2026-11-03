"""colab/session.py: notebook section 1, second half - assert the device, write setup.json (rule E1)."""

import json

import pytest

from nig_timenet_torch.colab import device, session

COMMITS = {"torch": "a" * 40, "tf": "b" * 40}
INSTALL = {"torch_before": "2.11.0+cu126", "torch_after": "2.11.0+cu126"}


def test_cpu_check_writes_setup_json_to_workdir_and_drive(tmp_path):
    out = []
    rec = session.check_environment("cpu", COMMITS, INSTALL, str(tmp_path / "w"), str(tmp_path / "drive" / "runs"),
                                    echo=out.append)
    for folder in ("w", "drive/runs"):
        on_disk = json.loads((tmp_path / folder / "setup.json").read_text())
        assert on_disk["commits"] == COMMITS and on_disk["device"]["used"] == "cpu" and on_disk["install"] == INSTALL
        assert on_disk["packages"]["torch"]
    assert rec["device"]["expected"] == "cpu"
    text = "\n".join(out)
    assert f"commit torch: {'a' * 40}" in text and f"commit tf: {'b' * 40}" in text
    assert "unchanged by the install: True" in text


def test_a_cuda_expectation_on_a_cpu_vm_raises_after_the_failure_is_on_file(tmp_path):
    with pytest.raises(device.DeviceError, match="expects cuda"):
        session.check_environment("cuda", COMMITS, INSTALL, str(tmp_path), echo=lambda *_: None)
    on_disk = json.loads((tmp_path / "setup.json").read_text())
    assert "expects cuda" in on_disk["device_error"] and on_disk["device"]["used"] == "cpu"


def test_a_torch_changed_by_the_install_is_visible_in_the_summary(tmp_path):
    out = []
    session.check_environment("cpu", COMMITS, {"torch_before": "2.11.0", "torch_after": "2.15.0"}, str(tmp_path),
                              echo=out.append)
    assert "unchanged by the install: False" in "\n".join(out)
