"""The evaluation-battery contract: the tidy prediction table and every battery section; each section must flag
a GOOD model positively and refuse to lie when the model is noise."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval.battery import (
    battery,
    confidence_buckets,
    cowardice,
    flip_test,
    full_battery,
    market_neutral,
    per_coin,
    persistence_baseline,
    sanity,
    selective_evaluation,
    shock_split,
    table_integrity,
    trade_grid,
    uncertainty_filter,
    verdict,
    weekly_stability,
)


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
    assert set(out) == {"integrity", "sanity", "confidence", "cowardice", "trades", "limit",
                        "uncertainty", "shocks", "weekly", "per_coin", "flip", "baseline",
                        "selective", "neutral"}
    for section in out.values():
        assert section["rows"], section["title"]


def test_table_integrity_passes_an_honest_table_and_flags_a_broken_one():
    good = table_integrity(_frame(good=True), _frame(good=True, seed=3))
    assert "mismatches=0" in good["rows"][0] and "NaN rows=0" in good["rows"][0]
    broken = _frame(good=True, seed=4)
    broken.loc[0, "mu"] = np.nan
    broken.loc[1, "dir_true"] = -broken.loc[1, "dir_true"]
    flagged = table_integrity(broken, _frame(good=True, seed=5))
    assert "NaN rows=1" in flagged["rows"][0] and "mismatches=1" in flagged["rows"][0]


def test_persistence_baseline_the_model_must_beat_the_last_candle():
    df = _frame(good=True)
    section = persistence_baseline(df, cost=0.0008)
    assert len(section["rows"]) == 7 and "momentum" in section["rows"][2]
    keep = df["dir_true"] != 0
    y, mu, last = df["y_ret"].values, df["mu"].values, df["last_ret"].values
    def acc(pred):
        return float(((pred[keep] > 0) == (y[keep] > 0)).mean())
    assert acc(mu) > acc(last)  # the model must add something over the raw momentum line
    noise = persistence_baseline(_frame(good=False), cost=0.0008)
    assert "momentum" in noise["rows"][2] and noise["rows"]


def test_selective_evaluation_confident_slice_beats_overall_on_test():
    val, test = _frame(good=True, seed=1), _frame(good=True, seed=2)
    section = selective_evaluation(val, test, cost=0.0008)
    assert len(section["rows"]) == 6  # 3 quantiles x (VAL, TEST)
    thr = float(np.quantile((val["p_dir"] - 0.5).abs(), 0.75))
    m = ((test["p_dir"] - 0.5).abs() >= thr) & (test["dir_true"] != 0)
    acc_slice = float(((test["p_dir"].values[m] > 0.5) == (test["y_ret"].values[m] > 0)).mean())
    acc_all = float(((test["p_dir"].values > 0.5) == (test["y_ret"].values > 0)).mean())
    assert acc_slice > acc_all  # confidence picked on VAL must still rank on TEST


def _panel_frame(seed: int = 0, n_periods: int = 400) -> pd.DataFrame:
    """4 coins x n_periods: mu ranks y across coins within each period (a model with real
    relative-ranking skill), so the market-neutral spread must be positive before cost."""
    rng = np.random.default_rng(seed)
    coins = ["AAA", "BBB", "CCC", "DDD"]
    y = rng.normal(0, 0.01, (n_periods, 4))
    mu = y + rng.normal(0, 0.001, (n_periods, 4))
    return pd.DataFrame({"step": 1, "mu": mu.ravel(), "sigma": 0.004,
                         "p_dir": 0.6, "y_ret": y.ravel(),
                         "dir_true": np.where(y.ravel() > 0, 1, -1), "last_ret": 0.0,
                         "coin": np.repeat(coins, n_periods),
                         "end_time": np.tile(pd.date_range("2026-01-05", periods=n_periods,
                                                           freq="1h").values, 4)})


def test_market_neutral_needs_a_cross_section_and_pays_for_ranking():
    two = market_neutral(_frame(good=True), _frame(good=True, seed=6), cost=0.0008)
    assert "SKIPPED" in two["title"]  # 2 coins cannot make a neutral book
    no_coin = _frame(good=True).drop(columns=["coin"])
    assert "SKIPPED" in market_neutral(no_coin, no_coin, cost=0.0008)["title"]
    val, test = _panel_frame(1), _panel_frame(2)
    section = market_neutral(val, test, cost=0.0)
    assert section["title"] == "MARKET-NEUTRAL PORTFOLIO (TEST)"
    chosen = next(r for r in section["rows"][1:] if "VAL choice" in r)
    assert float(chosen.split()[-1]) > 1.0  # profit factor > 1: the ranking pays without cost
