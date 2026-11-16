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
    market = torch.randn(40)[gid]
    assert CROSS_SECTION_LOSSES["none"] is None
    for kind in ("ic", "rank_ic"):
        fn = CROSS_SECTION_LOSSES[kind]
        base = fn(y, y + market, gid)                        # the label's market move does not enter the term
        assert float(base) < 0.05
        assert float(fn(-y, y + market, gid)) > 1.9          # the reversed ranking
        assert abs(float(fn(torch.randn(len(y)), y, gid)) - 1.0) < 0.1
        assert torch.allclose(fn(3.0 * y + 5.0 * market, y + market, gid), base, atol=1e-5)   # shift + scale
    assert float(group_correlation(y, y, torch.arange(len(y)))) == 0.0          # every row alone: no group qualifies
    assert float(group_correlation(y, torch.ones(len(y)), gid)) == 0.0          # a constant label: nothing to rank


def test_the_gradient_is_relative_only():
    gid, y = _panel()
    mu = torch.zeros(len(y), requires_grad=True)
    CROSS_SECTION_LOSSES["rank_ic"](mu + 0.01 * torch.randn(len(y)), y, gid).backward()
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
    good = compute_selection_metric(name, {mu: y + torch.randn(len(y))}, labels, {})
    noise = compute_selection_metric(name, {mu: torch.randn(len(y))}, labels, {})
    assert good > 1.0 > abs(noise)
    assert np.isnan(compute_selection_metric("val_ls_sharpe_close", {spec.output_keys["mu"]: torch.zeros(len(y))},
                                             labels, {}))
    with pytest.raises(ValueError, match="cross_section"):
        compute_selection_metric("val_ls_sharpe_close", {spec.output_keys["mu"]: y}, {spec.label_key: y}, {})
