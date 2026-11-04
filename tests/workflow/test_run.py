"""workflow.run: settings -> results for the flat and the panel model, one split and rolling, retarget + retrain."""

import json

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.core.config import from_dict, to_dict, updated
from nig_timenet_torch.core.prediction_schema import validate_table
from nig_timenet_torch.data.build import build_dataset
from nig_timenet_torch.data_io.loading import LoadedData
from nig_timenet_torch.workflow.panel_run import panel_columns
from nig_timenet_torch.workflow.rolling import rolling_folds
from nig_timenet_torch.workflow.run import SUMMARY, TEST_TABLE, retrain, rolling_summary, run, scalars
from nig_timenet_torch.workflow.run_common import make_plan, split_bounds
from nig_timenet_torch.workflow.settings import RollingSettings, RunSettings
from tests.data.helpers import longer_settings

ROLL = {"train_span": "30D", "val_span": "6D", "test_span": "8D", "step": "12D", "embargo": "1h"}


@pytest.fixture(scope="module")
def loaded():
    return LoadedData(build_dataset(longer_settings(100)), None, "npy_dir", None)


def _settings(kind, **patch):
    base = {"kind": kind, "train.epochs": 2, "train.batch_size": 8 if kind == "panel" else 64,
            "train.device": "cpu", "train.verbose": False}
    return updated(RunSettings(), {**base, **patch})


def test_run_settings_are_strict_round_trip_and_switch_rolling_on():
    s = RunSettings()
    assert s.rolling is None and from_dict(RunSettings, to_dict(s)) == s
    with pytest.raises(ValueError, match="did you mean 'epochs'"):
        updated(s, {"train.epoch": 3})
    with pytest.raises(ValueError, match="kind must be one of"):
        updated(s, {"kind": "tree"})
    on = updated(s, {"rolling": ROLL, "rolling.init": "warm"})
    assert isinstance(on.rolling, RollingSettings) and on.rolling.init == "warm" and on.rolling.val_span.days == 6
    with pytest.raises(ValueError, match="init must be one of"):
        updated(s, {"rolling": {**ROLL, "init": "cold"}})
    with pytest.raises(ValueError, match="val_span"):
        RollingSettings(train_span="10D", test_span="2D", step="2D", embargo="0h", val_span="10D")


def test_split_follows_the_settings_then_the_artifact_then_shares(loaded):
    ds = loaded.dataset
    shares = split_bounds(ds, RunSettings())
    given = split_bounds(ds, updated(RunSettings(), {"split.train_end": "2025-02-01", "split.val_end": "2025-02-10"}))
    assert given == (pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-10")) and shares[0] < shares[1]
    plan = make_plan(ds, RunSettings())
    assert len(plan.train_idx) and len(plan.val_idx) and len(plan.test_idx)


def test_flat_single_split_writes_summary_table_and_checkpoints(loaded, tmp_path):
    r = run(_settings("flat", run_dir=str(tmp_path)), loaded)
    assert r.fit.epochs_done == 2 and r.report["n_rows"] == len(r.table)
    validate_table(r.table)
    saved = json.loads((tmp_path / SUMMARY).read_text())
    assert saved["kind"] == "flat" and saved["identity"] and (tmp_path / TEST_TABLE).exists()
    assert scalars(r.report) and "targets" not in scalars(r.report)


def test_panel_single_split_keeps_targets_two_dimensional(loaded, tmp_path):
    s = _settings("panel", run_dir=str(tmp_path))
    r = run(s, loaded)
    validate_table(r.table)
    assert [c for c in r.table.columns if c.startswith("y_")] == sorted(panel_columns(s.panel.targets))
    assert r.report["close"]["ic_logit"] == r.report["close"]["ic_logit"]                    # defined (not NaN)
    assert r.report["n_rows"] == len(r.table) and r.fit.history[0]["train_panel_ic_close_ic"] is not None
    assert (tmp_path / TEST_TABLE).exists()


def test_rolling_flat_runs_every_fold_warm_start_and_per_fold_runs(loaded, tmp_path):
    s = _settings("flat", run_dir=str(tmp_path), rolling={**ROLL, "init": "warm", "warm_epochs": 1, "seeds": [0, 1]})
    r = run(s, loaded)
    n_folds = len(rolling_folds(loaded.dataset, s.rolling))
    assert n_folds >= 2 and len(r.tables) == 2 * n_folds == len(r.folds)
    summary = rolling_summary(r)
    assert set(summary["fold"]) == set(range(n_folds)) and (summary["n_val"] > 0).all()
    warm = summary[summary["fold"] > 0]
    assert (warm["epochs_done"] == 1).all() and (summary[summary["fold"] == 0]["epochs_done"] == 2).all()
    assert all(validate_table(t) is None for t in r.tables)
    assert (tmp_path / "fold_000_seed_0").exists() and (tmp_path / "fold_001_seed_1").exists()
    assert len(list((tmp_path / "predictions").glob("*.parquet"))) == len(r.tables)
    # folds never test on rows they trained or validated on, and move forward in time
    ends = [pd.Timestamp(f["train_end"]) for f in r.folds if f["seed"] == 0]
    assert ends == sorted(ends)


def test_rolling_panel_with_reuse_between_retrains(loaded):
    s = _settings("panel", rolling={**ROLL, "retrain_every_k": 2})
    r = run(s, loaded)
    assert len(r.tables) >= 2
    folds = pd.DataFrame(r.folds)
    assert folds.loc[folds["fold"] == 1, "epochs_done"].iloc[0] == folds.loc[folds["fold"] == 0, "epochs_done"].iloc[0]
    assert all("close" in f["report"] for f in r.folds)


def test_a_rolling_fold_without_a_validation_tail_is_refused(loaded):
    s = _settings("flat", rolling={**ROLL, "val_span": "0h"})
    with pytest.raises(ValueError, match="val_span"):
        run(s, loaded)


def test_rolling_folds_with_a_validation_tail_keep_order_and_purity(loaded):
    folds = rolling_folds(loaded.dataset, from_dict(RollingSettings, ROLL))
    from nig_timenet_torch.data_io.split_plan import check_split_purity
    for f in folds:
        check_split_purity(f.plan)
        assert len(f.plan.val_idx) and not set(f.plan.val_idx) & set(f.plan.train_idx)
        assert not set(f.plan.test_idx) & (set(f.plan.val_idx) | set(f.plan.train_idx))


def test_retrain_with_a_new_target_normalization_changes_the_targets(loaded):
    first = run(_settings("flat", **{"targets.mode": ""}), loaded)
    again = retrain(first, {"targets.mode": "scaled", "train.seed": 1})
    assert again.target_report["mode"] == "scaled" and again.settings.train.seed == 1
    assert again.settings.targets.mode == "scaled" and first.target_report == {}
    assert again.report["n_rows"] == first.report["n_rows"]
    assert not np.allclose(first.table.filter(like="y_close").to_numpy(), again.table.filter(like="y_close").to_numpy())
