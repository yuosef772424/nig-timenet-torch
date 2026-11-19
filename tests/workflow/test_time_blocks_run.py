"""TrainSettings.batch_order "time_blocks" through the run steps on the fixture artifact (flat model, no channel)."""

import numpy as np
import pytest

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.flat_run import time_ids
from nig_timenet_torch.workflow.settings import RunSettings


def settings(art_dir, tmp_path, **over):  # noqa: F811
    base = RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(name=str(art_dir)))
    return updated(base, {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu", **over})


OVER = {"split.max_rows": 60, "evaluation.battery.enabled": False}


def test_time_blocks_run_trains_one_epoch_without_an_extra_channel(art, tmp_path):  # noqa: F811
    r = steps.build(steps.prepare(settings(art, tmp_path, **OVER, **{"train.batch_order": "time_blocks"}),
                                  tmp_path / "out"))
    assert np.array_equal(r.trainer.row_times.numpy(), time_ids(r.dataset, r.idx["train"]))
    assert np.array_equal(r.trainer.val_times.numpy(), time_ids(r.dataset, r.idx["val"]))
    plain = steps.prepare(settings(art, tmp_path / "b", **OVER), None)
    assert r.x["train"].shape[2] == plain.x["train"].shape[2]                 # no timestamp channel
    steps.train(r)
    assert len(r.result.history) == 1 and np.isfinite(r.result.history[0]["train_total_loss"])


def test_time_blocks_with_a_cross_section_loss_without_the_channel(art, tmp_path):  # noqa: F811
    over = {"train.batch_order": "time_blocks", "train.cross_section_loss": "rank_ic",
            "train.selection_metric": "val_rank_ic_reg"}
    r = steps.train(steps.build(steps.prepare(settings(art, tmp_path, **OVER, **over), tmp_path / "out")))
    row = r.result.history[0]
    assert np.isfinite(row["train_cross_section_loss"]) and np.isfinite(row["val_cross_section_loss"])
    assert -1.0 <= row["val_selection_metric"] <= 1.0


def test_cross_section_loss_without_channel_or_time_blocks_still_raises(art, tmp_path):  # noqa: F811
    r = steps.build(steps.prepare(settings(art, tmp_path, **OVER, **{"train.cross_section_loss": "rank_ic"}),
                                  tmp_path / "out"))
    with pytest.raises(ValueError, match="whole-timestamp"):
        steps.train(r)


def test_time_blocks_settings_are_strict(tmp_path):
    base = RunSettings(run_dir=str(tmp_path / "r"), data=LoaderSettings())
    with pytest.raises(ValueError, match="panel path"):
        updated(base, {"train.batch_order": "time_blocks", "split.cross_section": True})
    with pytest.raises(NotImplementedError, match="staged"):
        updated(base, {"train.batch_order": "time_blocks", "train.staged": {}})
    with pytest.raises(ValueError, match="time_block_bars"):
        updated(base, {"train.time_block_bars": 0})
    with pytest.raises(KeyError):
        updated(base, {"train.batch_order": "nope"})
