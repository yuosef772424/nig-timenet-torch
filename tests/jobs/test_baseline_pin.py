"""Rule B2: the baseline job's summary and test report on the fixture artifact are pinned (the run-step refactor must
reproduce them). Floats compare at rtol 1e-4 (CPU builds differ in the last digits); timings, paths and the code
fingerprint (it follows the git HEAD) are left out. Regenerate on purpose with ``PIN_UPDATE=1 pytest <this file>``."""

import json
import os
import pathlib

import numpy as np
import pytest

from nig_timenet_torch.jobs import baseline
from tests.data_io.test_artifact import art  # noqa: F401  (fixture)

ROOT = pathlib.Path(__file__).resolve().parents[2]
PIN = ROOT / "tests" / "pins" / "baseline_fixture.json"
SKIP = {"step_seconds_per_batch", "epoch_seconds_median", "run_dir", "identity_fingerprint", "gpu_name",
        "trace_epoch_seconds", "trace_epoch_start", "trace_forward_seconds", "trace_backward_seconds"}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k not in SKIP and not k.startswith("trace_")}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def _same(a, b, path=""):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            _same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or (isinstance(a, float) and np.isnan(a)):
            assert (a is None or np.isnan(a)) and (b is None or np.isnan(b)), path
        else:
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-7, err_msg=path)
    else:
        assert a == b, path


def test_baseline_outputs_equal_the_pin(tmp_path, art, monkeypatch):  # noqa: F811
    monkeypatch.setenv("NIG_REPO_DIR", str(ROOT))
    cfg = {"device": "cpu", "seed": 0, "epochs": 2, "batch_size": 64, "run_dir": str(tmp_path / "train"),
           "artifact": str(art), "health_max_rows": 50}
    cp, out = tmp_path / "cfg.json", tmp_path / "out"
    cp.write_text(json.dumps(cfg))
    assert baseline.main(["--config", str(cp), "--out", str(out), "--max-minutes", "5"]) == 0
    got = {name: _clean(json.loads((out / name).read_text())) for name in ("baseline_summary.json", "test_report.json")}
    if os.environ.get("PIN_UPDATE"):
        PIN.write_text(json.dumps(got, indent=1, sort_keys=True, default=float), encoding="utf-8")
        pytest.skip("pin rewritten")
    _same(json.loads(json.dumps(got, default=float)), json.loads(PIN.read_text()))
