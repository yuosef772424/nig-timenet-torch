"""GridPanelModel: the same network as PanelModel, with days x slots in and out."""

import pytest
import torch

from nig_timenet_torch.model.grid_panel import GridPanelModel
from nig_timenet_torch.model.panel import PanelModel, build_panel_model, panel_encoder

T, F, B, N = 8, 5, 3, 4


def _core(**kw) -> PanelModel:
    torch.manual_seed(0)
    enc = panel_encoder(T, F, **kw.pop("enc", {}))
    return build_panel_model(enc, d_model=16, num_heads=2, n_cross_layers=kw.pop("layers", 1), dropout=0.0,
                             attn_dropout=0.0, **kw).eval()


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    return {"x": torch.randn(B, N, T, F, generator=g), "mask": mask}


def test_matches_the_flat_panel_model_on_the_real_slots():
    core, batch = _core(), _batch()
    out = GridPanelModel(core)(batch)
    day, pos = batch["mask"].nonzero(as_tuple=True)
    flat = core({"x": batch["x"][batch["mask"]], "day": day, "pos": pos})
    for k in ("logit", "mu"):
        assert out[k].shape == (B, N, 3)
        torch.testing.assert_close(out[k][batch["mask"]], flat[k], atol=1e-5, rtol=1e-5)
    assert out["mask"].dtype == torch.bool and out["mask"].equal(batch["mask"])


def test_padding_values_and_extra_padding_change_nothing():
    core, batch = _core(), _batch()
    model = GridPanelModel(core)
    base = model(batch)
    noisy = {"x": torch.where(batch["mask"][..., None, None], batch["x"], torch.randn_like(batch["x"]) * 50),
             "mask": batch["mask"]}
    wider = {"x": torch.cat([batch["x"], torch.randn(B, 3, T, F)], 1),
             "mask": torch.cat([batch["mask"], torch.zeros(B, 3, dtype=torch.bool)], 1)}
    m = batch["mask"]
    torch.testing.assert_close(model(noisy)["logit"][m], base["logit"][m], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model(wider)["mu"][:, :N][m], base["mu"][m], atol=1e-5, rtol=1e-5)


def test_a_day_does_not_see_another_day_but_sees_its_own_coins():
    model, batch = GridPanelModel(_core()), _batch()
    base = model(batch)["logit"]
    other = {"x": batch["x"].clone(), "mask": batch["mask"]}
    other["x"][0] += 3.0                                           # change day 0 only
    changed = model(other)["logit"]
    assert torch.allclose(changed[1:], base[1:], atol=1e-5)
    assert not torch.allclose(changed[0], base[0], atol=1e-5)
    mate = {"x": batch["x"].clone(), "mask": batch["mask"]}
    mate["x"][0, 3] += 3.0                                         # change coin 3 of day 0: coin 0 of day 0 reacts
    assert not torch.allclose(model(mate)["logit"][0, 0], base[0, 0], atol=1e-6)


def test_without_cross_layers_coins_are_independent_and_gradients_flow():
    model = GridPanelModel(_core(layers=0))
    batch = _batch()
    base = model(batch)["logit"]
    mate = {"x": batch["x"].clone(), "mask": batch["mask"]}
    mate["x"][0, 3] += 3.0
    assert torch.allclose(model(mate)["logit"][0, 0], base[0, 0], atol=1e-6)
    model.train()
    out = model(batch)
    (out["logit"][batch["mask"]].sum() + out["mu"][batch["mask"]].sum()).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_multi_timeframe_input_and_empty_batch():
    batch = _batch()
    assert GridPanelModel(_core()).targets == ("high", "low", "close")
    with pytest.raises(ValueError, match="at least one real coin"):
        GridPanelModel(_core())({"x": batch["x"], "mask": torch.zeros(B, N, dtype=torch.bool)})
