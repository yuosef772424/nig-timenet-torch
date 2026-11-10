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
    assert set(meta["gate_fails"]) == {"A", "data_h2", "grad", "grad_data", "grad_full", "full"}
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
    assert set(res) == {"A", "data_h2", "grad", "grad_data", "grad_full", "full"}
    assert all("verdict" in sec for sec in res.values())
    assert (out / "runs" / "grad" / "battery_report.txt").is_file()


def test_grad_net_sees_every_layer_and_only_gradients(result):
    """GradNet: one token per A learning layer (+ input); A frozen and in eval mode; B's prediction depends on the
    window only through A's gradients."""
    from nig_timenet_torch.data_io.loading import LoaderSettings
    from nig_timenet_torch.model.architectures import ModelSettings
    from nig_timenet_torch.workflow import run
    from nig_timenet_torch.workflow.grad_net import GradNet
    from nig_timenet_torch.workflow.settings import RunSettings, TrainSettings
    _, out = result
    s = RunSettings(run_dir=str(out / "chk_gn"), code_sha="x", model=ModelSettings("tiny_mlp"),
                    data=LoaderSettings(name=str(out / "E1"), local_dir="", mmap=False),
                    train=TrainSettings(epochs=1, device="cpu"))
    r = run.build(run.prepare(s))
    a = r.trainer.model
    g = GradNet(a, r.x["train"], ModelSettings("tiny_mlp"))
    n_layers = sum(1 for m in a.modules() if not list(m.children()) and list(m.parameters(recurse=False)))
    assert len(g.names) == n_layers + 1 and g.names[0] == "input"
    assert not any(p.requires_grad for p in g.a.parameters())
    g.train()
    assert not g.a.training
    x = r.x["val"][:8]
    grads = g.gradients(x)
    assert len(grads) == len(g.names) and all(torch.isfinite(t).all() for t in grads)
    with torch.no_grad():                       # prediction path (no_grad) still computes A's gradients
        y = g(x)
    assert all(torch.isfinite(v).all() for v in y.values())
    fixed = g.gradients(x)                       # B's only input: with the gradients held fixed, the window
    g.gradients = lambda _x: fixed               # itself (and so A's outputs) cannot change B's prediction
    with torch.no_grad():
        y2 = g(torch.randn_like(x))
    for k in y:
        torch.testing.assert_close(y[k], y2[k])


def test_single_model_mode_trains_one_model(tmp_path):
    """variants=("full",): one model on all train rows, no model A, no gradients; the test and the battery run."""
    meta = run_grad_stack(GradStackConfig(**{**TINY, "architecture": "conv1d_net", "variants": ("full",)}), tmp_path)
    assert set(meta["gate_fails"]) == {"full"} and meta["grad_features"] == []
    assert not (tmp_path / "modelA_grads_preds.npz").exists()
    assert set(grad_stack_battery(tmp_path, show=False)) == {"full"}
    with pytest.raises(ValueError):
        GradStackConfig(variants=("grad",))


def test_encoder_name_as_architecture_and_lr_decay(tmp_path):
    """architecture="itransformer" means the default model with that encoder; lr / lr_min decay by cosine from lr
    at the first epoch to lr_min at the last."""
    cfg = GradStackConfig(**{**TINY, "architecture": "itransformer", "variants": ("full",), "epochs": 3,
                             "lr": 1e-5, "lr_min": 1e-7})
    assert cfg.architecture == "" and cfg.model_overrides["encoder"] == "itransformer"
    run_grad_stack(cfg, tmp_path)
    import torch as _t
    hist = _t.load(tmp_path / "runs" / "full" / "checkpoint.pt", weights_only=False)["history"]
    lrs = [h["lr"] for h in hist]
    assert lrs[0] == pytest.approx(1e-5) and lrs[-1] == pytest.approx(1e-7) and lrs[0] > lrs[1] > lrs[2]
    with pytest.raises(ValueError):
        GradStackConfig(lr=1e-5, lr_min=1e-3)
