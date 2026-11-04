"""predict_grid / panel_report: grids in, grids scored; daily rank IC, AUC, accuracy."""

import numpy as np
import torch

from nig_timenet_torch.eval.panel_eval import panel_report, predict_grid
from nig_timenet_torch.model.grid_panel import GridPanelModel
from nig_timenet_torch.model.panel import build_panel_model, panel_encoder


class _Inputs:
    def __init__(self, x, mask):
        self.x, self.mask = x, mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, ix):
        return {"x": self.x[ix].numpy(), "mask": self.mask[ix].numpy()}


def test_predict_grid_equals_one_forward_whatever_the_batching():
    torch.manual_seed(0)
    model = GridPanelModel(build_panel_model(panel_encoder(6, 4), d_model=16, num_heads=2)).eval()
    x = torch.randn(7, 5, 6, 4)
    mask = torch.rand(7, 5) > 0.3
    mask[:, 0] = True
    whole = model({"x": x, "mask": mask})
    for bd in (1, 3, 100):
        pred = predict_grid(model, _Inputs(x, mask), batch_days=bd)
        assert pred["logit"].shape == (7, 5, 3) and pred["mask"].dtype == bool
        np.testing.assert_allclose(pred["logit"][mask.numpy()], whole["logit"][mask].detach().numpy(), atol=1e-5)


def _grids(d=30, n=8, k=3, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.ones((d, n), bool)
    mask[::5, 5:] = False
    rank = np.zeros((d, n, k), np.float32)
    for i in range(d):
        for j in range(k):
            r = rng.permutation(mask[i].sum())
            rank[i, mask[i], j] = (r + 0.5) / len(r) - 0.5
    cls = ((rank > 0) & mask[..., None]).astype(np.float32)
    return mask, rank, cls


def test_perfect_predictions_score_one_and_noise_scores_about_zero():
    mask, rank, cls = _grids()
    targets = {"y_rank": rank, "y_cls": cls}
    perfect = panel_report({"logit": rank * 10 - 1e-3, "mu": rank, "mask": mask}, targets, ("high", "low", "close"))
    assert perfect["n_rows"] == int(mask.sum()) and perfect["n_days"] == 30
    for t in ("high", "low", "close"):
        assert perfect[t]["ic_logit"] == 1.0 and perfect[t]["ic_mu"] == 1.0
        assert perfect[t]["auc"] == 1.0 and perfect[t]["accuracy"] == 1.0
    noise = np.random.default_rng(1).normal(size=rank.shape)
    rep = panel_report({"logit": noise, "mu": noise, "mask": mask}, targets, ("high", "low", "close"))
    assert all(abs(rep[t]["ic_logit"]) < 0.25 and abs(rep[t]["auc"] - 0.5) < 0.12 for t in ("high", "low", "close"))


def test_a_market_wide_shift_does_not_change_the_daily_ic():
    mask, rank, cls = _grids(seed=2)
    shift = np.random.default_rng(3).normal(size=(30, 1, 1)) * 5
    base = {"logit": rank + 0.0, "mu": rank, "mask": mask}
    shifted = {**base, "logit": rank + shift}
    a = panel_report(base, {"y_rank": rank, "y_cls": cls}, ("high", "low", "close"))
    b = panel_report(shifted, {"y_rank": rank, "y_cls": cls}, ("high", "low", "close"))
    assert a["close"]["ic_logit"] == b["close"]["ic_logit"] == 1.0 and a["close"]["auc"] > b["close"]["auc"]
