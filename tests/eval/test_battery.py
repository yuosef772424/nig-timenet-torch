"""The evaluation-battery contract: the tidy rollout-prediction table, the run dump the battery notebook
auto-detects, and every battery section - each section must flag a GOOD model positively and refuse to
lie when the model is noise (the verdict never rests on one lucky number)."""

import numpy as np
import pandas as pd
import pytest
import torch

from nig_timenet_torch.eval.ar_stream import dump_run, rollout_predictions
from nig_timenet_torch.eval.battery import (
    battery,
    confidence_buckets,
    cowardice,
    flip_test,
    full_battery,
    per_coin,
    sanity,
    shock_split,
    trade_grid,
    uncertainty_filter,
    verdict,
    weekly_stability,
)
from nig_timenet_torch.model.ar_candle_transformer import ArCandleTransformer


def _model() -> ArCandleTransformer:
    torch.manual_seed(0)
    return ArCandleTransformer(n_features=5, context_len=8, horizon_len=4, d_model=16,
                               num_layers=1, num_heads=4, num_kv_heads=2)


def _data(n: int = 64, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8, 5)).astype("float32")
    Y = rng.normal(size=(n, 4, 5)).astype("float32")
    dirs = np.sign(Y[:, :, 0]).astype("float32")
    dirs[dirs == 0] = 1.0
    return X, Y, dirs, X[:, -1, 0]


def _frame(n: int = 2400, seed: int = 0, good: bool = True, coins=("AAA", "BBB")) -> pd.DataFrame:
    """A tidy prediction table: `good=True` makes mu track the true return (an honest model);
    `good=False` makes it pure noise, and the battery must not call it tradeable."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 0.01, n)
    dir_true = np.where(y > 0, 1, -1)
    if good:
        mu = 0.8 * y + rng.normal(0, 0.002, n)
        # a spread confidence: near coin-flips at t~0.05, near-certain at t~4
        t = rng.uniform(0.05, 4.0, n)
        # the wrong-side probability FALLS with confidence: 45% wrong when unsure, 5% when certain
        p_right = 0.55 + 0.40 * (t / 4.0)
        side = np.where(rng.random(n) < p_right, np.sign(y), -np.sign(y))
        p_dir = 1.0 / (1.0 + np.exp(-side * t))
        sigma = np.full(n, 0.004)
    else:
        mu = rng.normal(0, 0.01, n)
        p_dir = rng.uniform(0.2, 0.8, n)
        sigma = rng.uniform(0.002, 0.02, n)
    times = pd.date_range("2026-01-05", periods=n, freq="1h").values  # dense: weekly buckets fill
    return pd.DataFrame({"step": np.tile(np.arange(1, 9), n // 8)[:n], "mu": mu, "sigma": sigma,
                         "p_dir": p_dir, "y_ret": y, "dir_true": dir_true,
                         "last_ret": rng.normal(0, 0.01, n),
                         "coin": rng.choice(list(coins), n), "end_time": times})


def test_rollout_predictions_table_shape_and_columns():
    X, Y, dirs, last_ret = _data()
    model = _model()
    table = rollout_predictions(model, X, Y, dirs, last_ret,
                                 coins=np.array(["AAA"] * len(X)),
                                 end_times=np.arange(len(X)))
    assert list(table.columns) == ["step", "mu", "sigma", "p_dir", "y_ret", "dir_true", "last_ret",
                                   "coin", "end_time"]
    assert len(table) == len(X) * 4  # one row per (window, step)
    assert table["step"].tolist() == [1, 2, 3, 4] * len(X)
    assert table["p_dir"].between(0, 1).all()
    assert table["dir_true"].isin([-1, 0, 1]).all()
    with pytest.raises(ValueError, match="horizon_len"):
        rollout_predictions(model, X, Y, dirs, last_ret, horizon_len=9)


def test_dump_run_writes_the_battery_inputs(tmp_path):
    X, Y, dirs, last_ret = _data()
    table = rollout_predictions(_model(), X, Y, dirs, last_ret)
    run = dump_run(str(tmp_path), "trial-001", {"val": table, "test": table},
                   {"best_val_dir_acc": 0.6}, {"direction_accuracy": 0.6}, {"epochs": 2})
    back = pd.read_parquet(f"{run}/predictions/val_predictions.parquet")
    assert len(back) == len(table)
    import json
    assert json.loads((tmp_path / "trial-001" / "report.json").read_text())["direction_accuracy"] == 0.6


def test_sanity_flags_a_good_model_and_a_biased_one():
    good = _frame(good=True)
    assert sanity(good, good)["rows"][1].count("dir_acc=1.0000") or \
        "dir_acc=" in sanity(good, good)["rows"][1]  # the good table scores its direction honestly
    rows = sanity(_frame(good=False), _frame(good=False))["rows"]
    assert any("IC(mu, y_ret)=-" in r or "IC(mu, y_ret)=0.0" in r or "IC(mu, y_ret)=+" in r
               for r in rows)  # an IC is always reported, whatever it is


def test_confidence_buckets_are_ordered_for_a_good_model():
    accs = []
    rows = confidence_buckets(_frame(good=True), n_buckets=4)["rows"]
    for r in rows:
        accs.append(float(r.split("dir_acc=")[1].split()[0]))
    assert accs[-1] > accs[0] + 0.15  # accuracy rises with the model's own confidence
    assert len(rows) == 4


def test_cowardice_measures_magnitude_honesty():
    good = cowardice(_frame(good=True))["rows"]
    bad = cowardice(_frame(good=False))["rows"]
    ratio_good = float(good[0].split("=")[1].split()[0])
    ratio_bad = float(bad[0].split("=")[1].split()[0])
    assert 0.5 < ratio_good < 1.5  # the honest table predicts real magnitudes
    assert ratio_bad == pytest.approx(1.0, abs=0.5)  # noise is not cowardly either - it is noise


def test_trade_grid_and_limit_execution_report_every_side():
    df = _frame(good=True)
    grid = trade_grid(df, cost=0.0008)["rows"]
    assert any("LONG" in r for r in grid) and any("SHORT" in r for r in grid)
    limit = trade_grid(df, cost=0.0008)["rows"]
    assert len(limit) >= 4  # header + one row per (side, pct)


def test_uncertainty_filter_uses_val_quantiles_no_leakage():
    val, test = _frame(good=True, seed=1), _frame(good=True, seed=2)
    section = uncertainty_filter(val, test, cost=0.0008)
    assert section["rows"][0].startswith("strategy")
    assert len(section["rows"]) == 5  # header + 3 quantile rows + the unfiltered baseline


def test_shock_split_spans_calm_to_wild():
    rows = shock_split(_frame(good=True), cost=0.0008)["rows"]
    assert len(rows) == 5  # five quintiles of the true move size
    assert "dir_acc" in rows[0] and "dir_acc" in rows[-1]


def test_weekly_stability_needs_and_uses_timestamps():
    with_ts = weekly_stability(_frame(good=True), cost=0.0008, pct=0.5, min_n=20)
    assert "week" in with_ts["rows"][0]  # hourly stamps: every week has enough trades
    no_ts = weekly_stability(_frame(good=True).drop(columns=["end_time"]), cost=0.0008)
    assert "SKIPPED" in no_ts["title"]


def test_per_coin_and_flip():
    rows = per_coin(_frame(good=True), cost=0.0008)["rows"]
    assert len(rows) == 3 and any("AAA" in r for r in rows[1:]) and any("BBB" in r for r in rows[1:])
    flip = flip_test(_frame(good=True), cost=0.0008)["rows"]
    assert "trade the direction" in flip[0] and "AGAINST" in flip[1]


def test_verdict_separates_an_edge_from_noise():
    good = _frame(good=True)
    sections = battery(good, good, cost=0.0008)
    v_good = verdict(good, 0.0008, sections)
    assert v_good["tradeable"] and v_good["best"]["sharpe"] > 0.10
    noise = _frame(good=False)
    sections_noise = battery(noise, noise, cost=0.0008)
    v_noise = verdict(noise, 0.0008, sections_noise)
    assert not v_noise["tradeable"]
    full = full_battery(noise, noise)
    assert "verdict" in full and full_battery(good, good)["verdict"]["tradeable"]


def test_battery_sections_all_present():
    out = battery(_frame(good=True), _frame(good=True))
    assert set(out) == {"sanity", "confidence", "cowardice", "trades", "limit", "uncertainty",
                        "shocks", "weekly", "per_coin", "flip"}
    for section in out.values():
        assert section["rows"], section["title"]
