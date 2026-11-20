"""TrainSettings.day_weights (D-039) through the run steps on the fixture artifact: the fit gets the UTC day of every
training row, the measurement runs from epoch 1, and day_weights.csv lands next to the checkpoint."""

import numpy as np
import pandas as pd

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.workflow import flat_run
from nig_timenet_torch.workflow import run as steps
from nig_timenet_torch.workflow.settings import RunSettings

OVER = {"split.max_rows": 60, "evaluation.battery.enabled": False}


def settings(art_dir, tmp_path, **over):
    base = RunSettings(run_dir=str(tmp_path / "run"), data=LoaderSettings(name=str(art_dir)))
    return updated(base, {"train.epochs": 1, "train.batch_size": 64, "train.device": "cpu", **over})


def test_day_weights_run_writes_the_day_table(art, tmp_path):  # noqa: F811
    over = {"train.epochs": 2, "train.day_weights": {"strength": 1.0, "shrink_rows": 0}}
    r = steps.train(steps.build(steps.prepare(settings(art, tmp_path, **OVER, **over), tmp_path / "out")))
    hist = r.result.history
    assert len(hist) == 2 and all(np.isfinite(h["train_total_loss"]) for h in hist)
    assert hist[0]["train_day_w_ess_share"] > 0 and "train_day_w_max" in hist[1]
    table = pd.read_csv(tmp_path / "run" / steps.DAY_WEIGHTS_CSV)
    days = flat_run.batch_groups(r.dataset, r.idx["train"], "day").numpy()
    assert table["rows"].sum() == len(r.idx["train"]) and len(table) == len(np.unique(days))
    assert table["loss_ema"].notna().all() and table["date"].str.match(r"\d{4}-\d{2}-\d{2}").all()


def test_without_day_weights_no_table(art, tmp_path):  # noqa: F811
    steps.train(steps.build(steps.prepare(settings(art, tmp_path, **OVER), tmp_path / "out")))
    assert not (tmp_path / "run" / steps.DAY_WEIGHTS_CSV).exists()
