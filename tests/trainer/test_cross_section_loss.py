"""The cross-section loss term (trainer/cross_section_loss.py, D-032) and the long / short selection metric: the
definitions, the invariances that make the term safe next to the NIG loss, and that OFF leaves the trainer as it is."""

import numpy as np
import pytest
import torch

from nig_timenet_torch.core.schema import TARGET_COLUMNS
from nig_timenet_torch.model.heads import head_spec
from nig_timenet_torch.trainer.cross_section_loss import (
    CROSS_SECTION_LOSSES,
    cross_section_loss,
    group_correlation,
    group_ranks,
)
from nig_timenet_torch.trainer.selection_metric import GROUP_KEY, SELECTION_METRICS, compute_selection_metric
from nig_timenet_torch.trainer.targets import LOSS_DEFAULTS, resolve_targets
from nig_timenet_torch.workflow.flat_run import trainer_config
from nig_timenet_torch.workflow.settings import TrainSettings


def _panel(groups=40, coins=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    gid = torch.arange(groups).repeat_interleave(coins)[torch.randperm(groups * coins, generator=g)]
    return gid, torch.randn(groups * coins, generator=g)


def test_group_ranks_are_ranks_inside_the_group():
    gid = torch.tensor([7, 7, 7, 7, 3, 3, 3])
    y = torch.tensor([3.0, 1.0, 2.0, 5.0, -1.0, 4.0, 0.0])
    assert torch.allclose(group_ranks(y, gid), torch.tensor([0.125, -0.375, -0.125, 0.375, -1 / 3, 1 / 3, 0.0]))


def test_the_term_is_one_minus_the_correlation_inside_the_timestamp_and_ignores_shift_and_scale():
    gid, y = _panel()
    g = torch.Generator().manual_seed(1)
    market, noise = torch.randn(40, generator=g)[gid], torch.randn(len(y), generator=g)
    assert CROSS_SECTION_LOSSES["none"] is None
    for kind in ("ic", "rank_ic"):
        fn = CROSS_SECTION_LOSSES[kind]
        base = fn(y, y + market, gid)                        # the label's market move does not enter the term
        assert float(base) < 0.05
        assert float(fn(-y, y + market, gid)) > 1.9          # the reversed ranking
        assert abs(float(fn(noise, y, gid)) - 1.0) < 0.2       # an unrelated mean (sd of the term ~ 0.05 here)
        assert torch.allclose(fn(3.0 * y + 5.0 * market, y + market, gid), base, atol=1e-5)   # shift + scale
    assert float(group_correlation(y, y, torch.arange(len(y)))) == 0.0          # every row alone: no group qualifies
    assert float(group_correlation(y, torch.ones(len(y)), gid)) == 0.0          # a constant label: nothing to rank


def test_the_gradient_is_relative_only():
    gid, y = _panel()
    mu = torch.zeros(len(y), requires_grad=True)
    start = 0.01 * torch.randn(len(y), generator=torch.Generator().manual_seed(2))
    CROSS_SECTION_LOSSES["rank_ic"](mu + start, y, gid).backward()
    per_group = torch.zeros(40).index_add_(0, gid, mu.grad)
    assert torch.allclose(per_group, torch.zeros(40), atol=1e-5)                # it never pushes a timestamp's level


def test_it_reads_every_nig_head_and_refuses_a_config_without_one():
    cfg = resolve_targets(trainer_config(TARGET_COLUMNS, 0.1))
    gid, y = _panel()
    out, labels = {}, {}
    for t in TARGET_COLUMNS:
        spec = head_spec("nig_regression", t)
        out[spec.output_keys["mu"]], labels[spec.label_key] = y[:, None], y
    assert float(cross_section_loss("ic", out, labels, cfg, gid)) == pytest.approx(0.0, abs=1e-5)
    only_class = {k: v for k, v in cfg.items() if v["task_type"] != "evidential"}
    with pytest.raises(ValueError, match="evidential"):
        cross_section_loss("ic", out, labels, only_class, gid)


def test_off_by_default_and_only_written_into_the_config_when_set():
    assert LOSS_DEFAULTS["cross_section"] == "none" and TrainSettings().cross_section_loss == "none"
    assert "cross_section" not in trainer_config(TARGET_COLUMNS, 0.1, TrainSettings())["loss"]     # config pinned
    on = trainer_config(TARGET_COLUMNS, 0.1, TrainSettings(cross_section_loss="rank_ic", cross_section_weight=0.5))
    assert on["loss"]["cross_section"] == "rank_ic" and on["loss"]["cross_section_weight"] == 0.5
    with pytest.raises(ValueError, match="cross_section_weight"):
        TrainSettings(cross_section_weight=-1.0)


def test_long_short_selection_metric_needs_timestamps_and_ranks_a_good_mean_above_noise():
    entry = SELECTION_METRICS.entry("val_ls_sharpe_close")
    assert entry.meta["higher_is_better"] and entry.meta["selectable"]
    spec = head_spec("nig_regression", "close")
    gid, y = _panel(groups=200)
    labels = {spec.label_key: y, GROUP_KEY: gid}
    name, mu = "val_ls_sharpe_close", spec.output_keys["mu"]
    g = torch.Generator().manual_seed(3)
    good = compute_selection_metric(name, {mu: y + torch.randn(len(y), generator=g)}, labels, {})
    noise = compute_selection_metric(name, {mu: torch.randn(len(y), generator=g)}, labels, {})
    assert good > 1.0 > abs(noise)
    assert np.isnan(compute_selection_metric("val_ls_sharpe_close", {spec.output_keys["mu"]: torch.zeros(len(y))},
                                             labels, {}))
    with pytest.raises(ValueError, match="cross_section"):
        compute_selection_metric("val_ls_sharpe_close", {spec.output_keys["mu"]: y}, {spec.label_key: y}, {})


def test_direction_balance_term_is_zero_for_a_balanced_batch_and_off_by_default():
    from nig_timenet_torch.trainer.direction_balance import direction_balance_loss

    assert LOSS_DEFAULTS["direction_balance"] == 0.0 and TrainSettings().direction_balance == 0.0
    assert "direction_balance" not in trainer_config(TARGET_COLUMNS, 0.1, TrainSettings())["loss"]   # config pinned
    on = trainer_config(TARGET_COLUMNS, 0.1, TrainSettings(direction_balance=2.0))["loss"]
    reg, cls = head_spec("nig_regression", "close"), head_spec("binary_classification", "close")
    assert on["direction_balance"] == 2.0 and on["direction_balance_targets"] == [reg.name, cls.name]   # close only
    cfg = resolve_targets(trainer_config(TARGET_COLUMNS, 0.1, TrainSettings(direction_balance=2.0)))
    mu_key, p_key = reg.output_keys["mu"], cls.output_keys["logits"]
    mu = torch.randn(2000, generator=torch.Generator().manual_seed(4))
    mu = mu - mu.mean()
    names = on["direction_balance_targets"]
    balanced = direction_balance_loss({mu_key: mu, p_key: torch.full((2000,), 0.5)}, cfg, names)
    leaning = direction_balance_loss({mu_key: mu.abs(), p_key: torch.full((2000,), 0.7)}, cfg, names)
    assert float(balanced) < 1e-3 < 0.1 < float(leaning)
    scaled = direction_balance_loss({mu_key: 100.0 * mu.abs(), p_key: torch.full((2000,), 0.7)}, cfg, names)
    assert torch.allclose(scaled, leaning)                         # shrinking every mean does not meet the term
    lean = (mu.abs() * 0.01).requires_grad_()
    direction_balance_loss({mu_key: lean, p_key: torch.full((2000,), 0.5)}, cfg, names).backward()
    assert float(lean.grad.sum()) > 0                              # the gradient pushes the long lean down
    with pytest.raises(ValueError, match="direction_balance"):
        TrainSettings(direction_balance=-1.0)


def test_base_loss_weight_zero_leaves_the_cross_section_term_alone_and_one_changes_nothing():
    """DY1 (owner 2026-11-22, "each loss on its own"): base_loss_weight 0 trains the rank term alone, 1.0 (the
    default) gives the same total as before, and the setting is refused without a cross-section term."""
    import copy

    from tests.trainer.fixture_setup import load_fixture
    from tests.trainer.fixture_setup import trainer_config as fixture_config
    from tests.trainer.helpers import make_trainer

    assert TrainSettings().base_loss_weight == 1.0 and LOSS_DEFAULTS["base_loss_weight"] == 1.0
    assert "base_loss_weight" not in trainer_config(TARGET_COLUMNS, 0.1, TrainSettings(cross_section_loss="rank_ic"))[
        "loss"]                                                                                  # config pinned
    on = trainer_config(TARGET_COLUMNS, 0.1, TrainSettings(cross_section_loss="rank_ic", base_loss_weight=0.0))
    assert on["loss"]["base_loss_weight"] == 0.0
    with pytest.raises(ValueError, match="base_loss_weight"):
        TrainSettings(base_loss_weight=0.0)
    with pytest.raises(ValueError, match="base_loss_weight"):
        TrainSettings(cross_section_loss="rank_ic", base_loss_weight=1.5)
    x, y = load_fixture(64)
    gid = torch.arange(8).repeat_interleave(8)

    def losses(**loss):
        cfg = copy.deepcopy(fixture_config())
        cfg["loss"].update(loss)
        tr = make_trainer(config=cfg)
        tr.model.eval()
        with torch.no_grad():
            total, raw, _, stats = tr.compute_losses(x, y, groups=gid)
        return total, raw, stats

    base, base_raw, _ = losses()
    one, one_raw, stats = losses(cross_section="rank_ic")
    same, same_raw, _ = losses(cross_section="rank_ic", base_loss_weight=1.0)
    alone, alone_raw, alone_stats = losses(cross_section="rank_ic", base_loss_weight=0.0)
    term = stats["cross_section_loss"]
    assert torch.allclose(one, base + term) and torch.allclose(same, one) and torch.allclose(same_raw, one_raw)
    assert torch.allclose(alone, alone_stats["cross_section_loss"]) and torch.allclose(alone_raw, alone)
    assert torch.allclose(alone, term)
