"""LS1 (owner 2026-11-19): the auxiliary close heads side_payoff / upper_quantile, their losses and their wiring."""

import math

import numpy as np
import pytest
import torch

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.model.architectures import ModelSettings, build_model
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.model.scalers import fit_scalers
from nig_timenet_torch.trainer.tasks import quantile_task_loss, side_payoff_task_loss, side_payoffs, with_defaults
from nig_timenet_torch.workflow import flat_run
from nig_timenet_torch.workflow.settings import RunSettings, TrainSettings, default_model_settings

SIDE = head_spec("side_payoff", "close")
UPQ = head_spec("upper_quantile", "close")


def _cfg(kind: str, **opts) -> dict:
    spec = head_spec(kind, "close")
    return with_defaults(spec.task_type, {"output_keys": spec.output_keys, **opts})


def test_side_payoffs_are_what_each_side_is_paid_after_cost():
    y = torch.tensor([math.log(1.5), 0.0, math.log(0.5)])
    long, short = side_payoffs(y, 0.001)
    assert torch.allclose(long, torch.tensor([0.5 - 0.002, -0.002, -0.5 - 0.002]))
    assert torch.allclose(short, torch.tensor([-0.5 - 0.002, -0.002, 0.5 - 0.002]))
    # the asymmetry the short pays: an equal log move up costs more than the same move down gains
    up, down = side_payoffs(torch.tensor([0.4, -0.4]), 0.0)[1]
    assert -up > down


def test_side_payoff_loss_is_zero_at_the_payoffs_and_its_minimiser_is_the_mean_payoff():
    y = torch.randn(4096, generator=torch.Generator().manual_seed(0)) * 0.05
    long, short = side_payoffs(y, 0.0008)
    cfg = _cfg("side_payoff")
    loss, stats = side_payoff_task_loss(y, {SIDE.output_keys["long"]: long, SIDE.output_keys["short"]: short}, cfg, {})
    assert float(loss) == pytest.approx(0.0) and float(stats["mae_short"]) == pytest.approx(0.0)
    # a constant prediction: the squared loss is lowest at the MEAN payoff (the conditional mean is what it learns)
    const = lambda v: torch.full_like(y, v)  # noqa: E731
    at = lambda lv, sv: float(side_payoff_task_loss(  # noqa: E731
        y, {SIDE.output_keys["long"]: const(lv), SIDE.output_keys["short"]: const(sv)}, cfg, {})[0])
    m_l, m_s = float(long.mean()), float(short.mean())
    assert at(m_l, m_s) < min(at(m_l + 0.002, m_s), at(m_l - 0.002, m_s), at(m_l, m_s + 0.002), at(m_l, m_s - 0.002))
    # the short's mean payoff is below the long's by twice the exp() part of a symmetric label (E e^y - 1 > 0)
    assert m_s < -m_l - 2 * 0.0016 + 1e-9


def test_pinball_loss_is_lowest_at_the_quantile_and_reports_coverage():
    y = torch.randn(20000, generator=torch.Generator().manual_seed(1))
    cfg = _cfg("upper_quantile", quantile=0.9)
    at = lambda v: quantile_task_loss(y, {UPQ.output_keys["q"]: torch.full_like(y, v)}, cfg, {})  # noqa: E731
    q90 = float(torch.quantile(y, 0.9))
    assert float(at(q90)[0]) < min(float(at(q90 - 0.1)[0]), float(at(q90 + 0.1)[0]))
    assert float(at(q90)[1]["coverage"]) == pytest.approx(0.9, abs=0.01)


def _model(aux=(), grad=1.0):
    base = default_model_settings()
    over = {**base.overrides, "encoder": "gru"}
    if aux:
        over.update(aux_heads=list(aux), aux_trunk_grad=grad)
    torch.manual_seed(0)
    m = build_model(ModelSettings(base.architecture, over), 24, 3)
    x = torch.randn(64, 24, 3, generator=torch.Generator().manual_seed(2))
    y = {head_spec(k, t).label_key: torch.randn(64) * 0.03 for t in TARGET_COLUMNS for k in ("nig_regression",)}
    y.update({head_spec("binary_classification", t).label_key: (torch.rand(64) > 0.5).float() for t in TARGET_COLUMNS})
    fit_scalers(m, x, y)
    return m, x, y


def test_default_model_and_trainer_config_have_no_aux_heads():
    m, x, _ = _model()
    assert m.aux_kinds == {} and not any("payoff" in k or "upper_q" in k for k in m(x))
    cfg = flat_run.trainer_config(TARGET_COLUMNS, 0.1, TrainSettings())
    assert cfg == flat_run.trainer_config(TARGET_COLUMNS, 0.1, TrainSettings(), aux_kinds={})   # config pinned
    assert cfg["loss"]["uncertainty_weighting_exclude_task_types"] == ["evidential"]


def test_aux_heads_join_the_trainer_with_a_fixed_weight_and_zero_trunk_gradient_at_grad_0():
    m, x, y = _model(("side_payoff", "upper_quantile"), grad=0.0)
    out = m(x)
    assert {*SIDE.output_keys.values(), *UPQ.output_keys.values()} <= set(out)
    assert float(out[SIDE.output_keys["short"]].detach().abs().max()) < 0.05     # small init: starts near a zero payoff
    tr = flat_run._trainer(RunSettings(run_dir="unused", train=TrainSettings(aux_loss_weight=0.5, aux_quantile=0.8)),
                           m, tuple(TARGET_COLUMNS))
    assert tr.targets_cfg[SIDE.name]["loss_weight"] == 0.5 and tr.targets_cfg[UPQ.name]["quantile"] == 0.8
    assert tr.targets_cfg[SIDE.name]["cost_per_side"] == RunSettings(run_dir="x").evaluation.battery.cost_per_side
    assert SIDE.name not in tr.kendall_task_names and UPQ.name not in tr.kendall_task_names
    _, _, per_target, stats, _ = tr.compute_losses(x, y, training=True)
    m.zero_grad()
    (per_target[SIDE.name] + per_target[UPQ.name]).backward()
    named = dict(m.named_parameters())
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for n, p in named.items() if not n.startswith("aux_"))
    assert all(float(p.grad.abs().sum()) > 0 for n, p in named.items() if n.startswith("aux_fc_"))
    assert {f"{SIDE.name}_mae_short", f"{UPQ.name}_coverage"} <= set(stats)


def test_aux_heads_shape_the_trunk_when_their_gradient_passes():
    m, x, y = _model(("side_payoff",), grad=0.3)
    tr = flat_run._trainer(RunSettings(run_dir="unused"), m, tuple(TARGET_COLUMNS))
    m.zero_grad()
    tr.compute_losses(x, y, training=True)[2][SIDE.name].backward()
    trunk = [p for n, p in m.named_parameters() if not n.startswith(("aux_", "nig_", "conf_", "class_", "y_"))]
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in trunk)


def test_aux_heads_refuse_a_non_auxiliary_kind_and_bad_settings():
    with pytest.raises(ValueError, match="auxiliary head kinds"):
        _model(("binary_classification",))
    with pytest.raises(ValueError, match="aux_quantile"):
        TrainSettings(aux_quantile=1.0)
    with pytest.raises(ValueError, match="aux_loss_weight"):
        TrainSettings(aux_loss_weight=-1.0)


def test_battery_table_carries_the_aux_columns():
    from nig_timenet_torch.eval.battery_adapter import to_battery_table
    from tests.eval.synth import ANCHOR, codec_state, make

    table, labels, anchors = make()
    adapt = lambda t: to_battery_table(t, labels, anchors=anchors, last_close=np.full(len(t), ANCHOR),  # noqa: E731
                                       codec_state=codec_state())
    assert not {"pay_long", "pay_short", "q_up"} & set(adapt(table).columns)      # no aux heads: no columns
    n = len(table)
    table[SIDE.output_keys["long"]], table[SIDE.output_keys["short"]] = np.full(n, 0.01), np.full(n, -0.02)
    table[UPQ.output_keys["q"]] = np.full(n, 0.1)
    out = adapt(table)
    assert (out["pay_long"] == 0.01).all() and (out["pay_short"] == -0.02).all()
    np.testing.assert_allclose(out["q_up"], math.expm1(0.1))


def test_variance_weight_reweights_rows_by_the_nig_variance_and_reaches_the_trainer_config():
    n = 1000
    y = torch.zeros(n)
    keys = {"nu": "nu", "alpha": "alpha", "beta": "beta"}
    calm, wild = torch.full((n // 2,), 1e-4), torch.full((n // 2,), 4e-4)
    nig = {"nu": torch.ones(n), "alpha": torch.full((n,), 2.0), "beta": torch.cat([calm, wild]) / 2.0}
    err = torch.cat([torch.zeros(n // 2), torch.full((n // 2,), 0.02)])   # only the volatile rows are wrong
    long, short = side_payoffs(y, 0.0)
    outs = {SIDE.output_keys["long"]: long + err, SIDE.output_keys["short"]: short + err, **nig}
    plain = float(side_payoff_task_loss(y, outs, _cfg("side_payoff", cost_per_side=0.0), {})[0])
    wls = float(side_payoff_task_loss(y, outs, _cfg("side_payoff", cost_per_side=0.0, variance_weight=1.0,
                                                    nig_keys=keys), {})[0])
    assert wls == pytest.approx(plain * 2 * (1 / 4) / (1 + 1 / 4))   # weights 1 / var scaled to mean 1: wild rows 0.4
    on = flat_run.trainer_config(TARGET_COLUMNS, 0.1, TrainSettings(aux_variance_weight=1.0),
                                 aux_kinds={"close": ["side_payoff"]})["targets"][SIDE.name]
    assert on["variance_weight"] == 1.0 and on["nig_keys"]["beta"] == head_spec("nig_regression", "close").output_keys[
        "beta"]
    off = flat_run.trainer_config(TARGET_COLUMNS, 0.1, TrainSettings(), aux_kinds={"close": ["side_payoff"]})
    assert "variance_weight" not in off["targets"][SIDE.name]


def test_lr_schedule_is_constant_by_default_and_cosine_decays_to_the_floor():
    from nig_timenet_torch.trainer.schedules import lr_at

    assert flat_run.fit_settings_of(RunSettings(run_dir="x")).lr is None          # as every run before LR1
    s = RunSettings(run_dir="x", train=TrainSettings(lr=3e-4, epochs=8, lr_schedule="cosine", lr_min_ratio=0.1))
    lr = flat_run.fit_settings_of(s).lr
    steps = [lr_at(lr, e) for e in range(8)]
    assert steps[0] == pytest.approx(3e-4) and all(a > b for a, b in zip(steps, steps[1:], strict=False))
    assert steps[-1] == pytest.approx(3e-5)
    with pytest.raises(ValueError, match="lr_schedule"):
        TrainSettings(lr_schedule="nope")
    with pytest.raises(ValueError, match="lr_min_ratio"):
        TrainSettings(lr_min_ratio=0.0)
