"""ROADMAP F2: the rolling walk-forward through the run steps, on the fixture artifact (CPU): folds never leak, the
scalers are fitted per fold on that fold's train rows, retrain_every_k reuses the model, seeds loop, per-fold and
across-fold summaries, resume."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from nig_timenet_torch.core.config import updated
from nig_timenet_torch.data_io.loading import LoaderSettings
from nig_timenet_torch.data_io.split_plan import check_split_purity
from nig_timenet_torch.trainer.staged import StageSettings
from nig_timenet_torch.workflow.prediction_table import read_table
from nig_timenet_torch.workflow.rolling import fold_with_val, rolling_folds
from nig_timenet_torch.workflow.rolling_run import fold_dir, run_rolling, summarize
from nig_timenet_torch.workflow.settings import RollingSettings, RunSettings


def rolling(**over):
    base = dict(train_span="20D", test_span="7D", step="10D", embargo="1D", max_folds=3)
    return RollingSettings(**{**base, **over})


def settings(art, tmp_path, **over):  # noqa: F811
    base = RunSettings(run_dir=str(tmp_path / "unused"), data=LoaderSettings(name=str(art)))
    return updated(base, {"train.epochs": 2, "train.batch_size": 64, "train.device": "cpu",
                          "evaluation.battery.enabled": False, **over})


def test_folds_never_leak_and_the_inner_validation_is_purged(art):  # noqa: F811
    from nig_timenet_torch.workflow import run_common
    ds = run_common.load_data(RunSettings(run_dir="x", data=LoaderSettings(name=str(art)))).dataset
    folds = rolling_folds(ds, rolling())
    assert len(folds) >= 2
    for f in folds:
        plan, cut_v = fold_with_val(f, "1D", 0.15)
        check_split_purity(plan)
        le, fs = plan.label_end, plan.feature_start
        assert le[plan.train_idx].max() <= cut_v.value < fs[plan.val_idx].min()
        assert le[plan.val_idx].max() <= f.train_end.value                    # train label_end <= cut
        assert fs[plan.test_idx].min() > (f.train_end + pd.Timedelta("1D")).value  # test starts after the embargo
        assert le[plan.test_idx].max() <= f.test_end.value
        assert not set(plan.train_idx) & set(plan.val_idx) and not set(plan.test_idx) & set(plan.val_idx)
    with pytest.raises(ValueError, match="val_share"):
        rolling(val_share=1.0)
    with pytest.raises(ValueError, match="validation rows"):
        fold_with_val(folds[0], "1D", 0.999999)


def test_run_rolling_trains_per_fold_with_scalers_of_that_fold(art, tmp_path):  # noqa: F811
    seen = []

    def grab(run, row):
        scaler = next(m for m in run.trainer.model.modules() if hasattr(m, "mean") and hasattr(m, "std"))
        x = run.x["train"].to(torch.float64)
        seen.append((row["fold"], run.extra["fitted_scalers"], scaler.mean.clone(), x.reshape(-1, x.shape[-1]).mean(0),
                     len(run.idx["train"]), run.ident.fold))

    out = tmp_path / "roll"
    summary = run_rolling(settings(art, tmp_path), rolling(), out, on_fold=grab)
    assert summary["n_folds"] == len(seen) >= 2
    sizes = [s[4] for s in seen]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)           # expanding: every fold has more rows
    for fold, fitted, mean, x_mean, n, ident_fold in seen:
        assert ident_fold == fold
        assert next(iter(fitted.values()))["rows"] == n                      # fitted on THIS fold's train rows only
        assert torch.allclose(mean.double(), x_mean, atol=1e-5)
    assert not torch.allclose(seen[0][2], seen[-1][2])                       # and the folds' scalers differ
    for i in range(summary["n_folds"]):
        folder = fold_dir(out, i, 0)
        assert (folder / "test_report.json").is_file() and (folder / "fold_row.json").is_file()
        table = read_table(folder / "test_table.parquet")
        assert set(table["fold"]) == {i} and set(table["stage"]) == {"test"}
        assert table["prediction_made_at"].min() > table["train_end"].iloc[0] + pd.Timedelta("1D")
    assert (out / "rolling_folds.csv").is_file() and "across folds" in (out / "rolling_summary.md").read_text()
    across = summary["across_folds"]
    key = next(k for k in across if k.endswith("rank_ic_by_timestamp"))
    assert across[key]["n"] == summary["n_folds"] and across[key]["sd"] is not None
    assert json.loads((out / "rolling_summary.json").read_text())["n_folds"] == summary["n_folds"]


def test_retrain_every_k_reuses_the_last_model_and_seeds_loop(art, tmp_path):  # noqa: F811
    rows = []
    out = tmp_path / "roll"
    summary = run_rolling(settings(art, tmp_path), rolling(retrain_every_k=2, seeds=(0, 1)), out,
                          on_fold=lambda run, row: rows.append((row, run.trainer.model)))
    n = summary["n_folds"]
    assert len(rows) == 2 * n and [r["seed"] for r, _ in rows[:2]] == [0, 1]
    for r, _ in rows:
        assert r["trained"] == (r["fold"] % 2 == 0) and r["trained_on_fold"] == r["fold"] - r["fold"] % 2
    by = {(r["fold"], r["seed"]): m for r, m in rows}
    assert by[(1, 0)] is by[(0, 0)] and by[(1, 1)] is by[(0, 1)] and by[(0, 0)] is not by[(0, 1)]
    if n > 2:
        assert by[(2, 0)] is not by[(0, 0)]
    fold0 = summary["per_fold"][0]["metrics"]
    key = next(k for k in fold0 if k.endswith("rank_ic_by_timestamp"))
    assert fold0[key]["n"] == 2 and fold0[key]["sd"] is not None            # the seed spread of a fold
    reused = json.loads((fold_dir(out, 1, 0) / "fold_row.json").read_text())
    assert reused["trained"] is False and reused["n_train"] > 0


def test_a_finished_study_resumes_without_retraining(art, tmp_path):  # noqa: F811
    out = tmp_path / "roll"
    first = run_rolling(settings(art, tmp_path), rolling(max_folds=2), out)
    again = run_rolling(settings(art, tmp_path), rolling(max_folds=2), out)
    assert again["across_folds"] == first["across_folds"]


def test_rolling_composes_with_staged_training_and_the_battery(art, tmp_path):  # noqa: F811
    st = StageSettings(start_share=0.5, eval_every=2, step_cap_epochs=2, warmup_steps=2, bn_refresh_steps=2, k=2,
                       probe_rows=16)
    s = settings(art, tmp_path, **{"train.staged": st, "evaluation.battery.enabled": True})
    summary = run_rolling(s, rolling(max_folds=2), tmp_path / "roll")
    assert summary["n_folds"] == 2 and (fold_dir(tmp_path / "roll", 0, 0) / "battery_report.txt").is_file()


def test_summarize_mean_and_spread():
    r = lambda f, s, v: {"fold": f, "seed": s, "trained": True, "n_test": 5, "train_end": "t", "test_start": "a",  # noqa: E731
                         "test_end": "b", "metrics": {"m": v}}
    out = summarize([r(0, 0, 1.0), r(0, 1, 3.0), r(1, 0, 5.0), r(1, 1, 5.0)], rolling())
    assert out["per_fold"][0]["metrics"]["m"] == {"mean": 2.0, "sd": pytest.approx(np.sqrt(2.0)), "min": 1.0,
                                                  "max": 3.0, "n": 2}
    assert out["across_folds"]["m"]["mean"] == 3.5 and out["across_folds"]["m"]["min"] == 2.0
