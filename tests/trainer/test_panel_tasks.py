"""Panel task losses (masked, per column) and the trainer's row access (trainer/batching.py)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nig_timenet_torch.trainer.batching import n_rows, take, to_device
from nig_timenet_torch.trainer.panel_tasks import panel_classification_loss, panel_rank_loss, panel_regression_loss
from nig_timenet_torch.trainer.tasks import TASK_REGISTRY


def _grid(b=3, n=5, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(b, n, generator=g) > 0.3
    mask[:, :3] = True
    return mask, torch.randn(b, n, k, generator=g), torch.randn(b, n, k, generator=g)


def test_the_three_tasks_are_registered_with_their_stats():
    for name, stat in (("panel_classification", "accuracy"), ("panel_regression", "mae"), ("panel_rank_ic", "ic")):
        assert TASK_REGISTRY.entry(name).meta["stat_keys"] == [stat]


def test_classification_is_a_masked_binary_cross_entropy_on_the_chosen_column():
    mask, logit, _ = _grid()
    y = (torch.rand(3, 5, 3) > 0.5).float()
    cfg = {"output_keys": {"logits": "logit", "mask": "mask"}, "column": 1}
    loss, stats = panel_classification_loss(y, {"logit": logit, "mask": mask}, cfg, {})
    want = F.binary_cross_entropy_with_logits(logit[..., 1][mask], y[..., 1][mask])
    torch.testing.assert_close(loss, want)
    assert 0.0 <= float(stats["accuracy"]) <= 1.0
    junk = logit.clone()
    junk[~mask] = 1e4                                               # padding must not matter
    torch.testing.assert_close(panel_classification_loss(y, {"logit": junk, "mask": mask}, cfg, {})[0], want)
    smooth, _ = panel_classification_loss(y, {"logit": logit, "mask": mask}, {**cfg, "label_smoothing": 0.2}, {})
    assert not torch.isclose(smooth, want)


@pytest.mark.parametrize("kind", ["mse", "mae", "huber"])
def test_regression_losses_count_real_slots_only(kind):
    mask, mu, y = _grid()
    cfg = {"output_keys": {"pred": "mu", "mask": "mask"}, "column": 2, "loss_fn": kind, "huber_delta": 0.5}
    loss, stats = panel_regression_loss(y, {"mu": mu, "mask": mask}, cfg, {})
    e = (y[..., 2] - mu[..., 2])[mask]
    want = {"mse": e.square(), "mae": e.abs(),
            "huber": torch.where(e.abs() <= 0.5, 0.5 * e.square(), 0.5 * (e.abs() - 0.25))}[kind].mean()
    torch.testing.assert_close(loss, want)
    torch.testing.assert_close(stats["mae"], e.abs().mean())


def test_rank_ic_loss_is_minus_the_mean_daily_correlation():
    mask, pred, _ = _grid(b=4, n=6)
    rank = pred.clone() * 2 + 1                                       # a perfect linear relation inside each day
    cfg = {"output_keys": {"pred": "logit", "mask": "mask"}, "column": 0}
    loss, stats = panel_rank_loss(rank, {"logit": pred, "mask": mask}, cfg, {})
    assert float(loss) == pytest.approx(-1.0, abs=1e-4) and float(stats["ic"]) == pytest.approx(1.0, abs=1e-4)
    anti, _ = panel_rank_loss(-rank, {"logit": pred, "mask": mask}, cfg, {})
    assert float(anti) == pytest.approx(1.0, abs=1e-4)
    few = mask.clone()
    few[0, 2:] = False                                                # day 0 has 2 real slots: under min_slots
    both, _ = panel_rank_loss(rank, {"logit": pred, "mask": few}, {**cfg, "min_slots": 3}, {})
    assert float(both) == pytest.approx(-1.0, abs=1e-4)
    p = pred.clone().requires_grad_()
    panel_rank_loss(torch.randn(4, 6, 3), {"logit": p, "mask": mask}, cfg, {})[0].backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


class _Lazy:
    def __len__(self):
        return 6

    def __getitem__(self, ix):
        ix = np.arange(6)[ix] if isinstance(ix, slice) else np.asarray(ix)
        return {"x": np.ones((len(ix), 2), np.float32) * ix[:, None], "mask": np.ones((len(ix), 2), bool)}


def test_batching_handles_tensors_dicts_nested_dicts_and_lazy_inputs():
    t = torch.arange(10.0).reshape(5, 2)
    assert n_rows(t) == 5 and take(t, slice(1, 3)).shape == (2, 2) and to_device(t, torch.device("cpu")) is t
    d = {"x": t, "m": torch.ones(5, 3)}
    assert n_rows(d) == 5 and take(d, torch.tensor([0, 4]))["m"].shape == (2, 3)
    nested = {"x": {"1h": t, "4h": t}, "mask": torch.ones(5)}
    assert take(nested, slice(0, 2))["x"]["4h"].shape == (2, 2)
    moved = to_device(_Lazy()[np.array([1, 2])], torch.device("cpu"))
    assert isinstance(moved["x"], torch.Tensor) and moved["x"][1, 0] == 2.0
    assert n_rows(_Lazy()) == 6
