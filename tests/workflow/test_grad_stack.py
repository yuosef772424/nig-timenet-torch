"""grad_stack (workflow/grad_stack.py, jobs/grad_stack.py): time halves, per-row gradients, a tiny end-to-end run."""

import json

import numpy as np
import pytest
import torch

from nig_timenet_torch.core.config import from_dict
from nig_timenet_torch.jobs.grad_stack import CONFIG
from nig_timenet_torch.workflow.grad_stack import GradStackConfig, grad_features, grad_stack_battery, run_grad_stack

TINY = {"e1": {"n_coins": 8, "n_bars": 400}, "epochs": 1, "batch_size": 128}


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("gs")
    return run_grad_stack(GradStackConfig(**TINY), out), out


def test_end_to_end_writes_every_variant(result):
    meta, out = result
    assert set(meta["gate_fails"]) == {"A", "data_h2", "grad", "grad_data", "full"}
    s = meta["split"]
    assert s["H1"]["to"] < s["H2"]["from"] < s["val"]["from"], "halves are time ordered, before val"
    z = np.load(out / "modelA_grads_preds.npz")
    assert z["grad_train"].shape[1] == len(meta["grad_features"]) and np.isfinite(z["grad_test"]).all()
    assert json.loads((out / "summary.json").read_text())["config"]["grad_target"] == "output"


def test_per_row_gradients_do_not_depend_on_the_batch(result):
    """Eval mode: a row's gradient features are the same alone or inside a batch (no cross-row term)."""
    _, out = result
    from nig_timenet_torch.data_io.loading import LoaderSettings
    from nig_timenet_torch.model.architectures import ModelSettings
    from nig_timenet_torch.workflow import run
    from nig_timenet_torch.workflow.settings import RunSettings, TrainSettings
    s = RunSettings(run_dir=str(out / "chk"), code_sha="x", model=ModelSettings("tiny_mlp"),
                    data=LoaderSettings(name=str(out / "E1"), local_dir="", mmap=False),
                    train=TrainSettings(epochs=1, device="cpu"))
    r = run.build(run.prepare(s))
    x, y = r.x["val"][:16], {k: v[:16] for k, v in r.y["val"].items()}
    many, _, _ = grad_features(r.trainer, x, y, target="output", batch=16, device=torch.device("cpu"))
    one, _, _ = grad_features(r.trainer, x[:1], y, target="output", batch=1, device=torch.device("cpu"))
    np.testing.assert_allclose(many[0], one[0], rtol=1e-4, atol=1e-6)


def test_config_is_strict():
    assert CONFIG is GradStackConfig
    with pytest.raises(ValueError):
        from_dict(GradStackConfig, {"no_such": 1})
    with pytest.raises(ValueError):
        GradStackConfig(grad_target="labels")
    with pytest.raises(ValueError):
        GradStackConfig(variants=("grad",))


def test_battery_runs_on_every_variant(result):
    _, out = result
    res = grad_stack_battery(out, show=False)
    assert set(res) == {"A", "data_h2", "grad", "grad_data", "full"}
    assert all("verdict" in sec for sec in res.values())
    assert (out / "runs" / "grad" / "battery_report.txt").is_file()
