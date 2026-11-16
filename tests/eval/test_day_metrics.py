"""Day metrics of the panel audit 2026-11-16 (eval/day_metrics.py, eval/uncertainty_use.py): definitions, trivial
scores (rule C1), cost accounting, paired comparison, baselines, the accept rule."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval import day_metrics as dm
from nig_timenet_torch.eval import uncertainty_use as uu


def frame(n_days=120, per_day=6, coins=20, ic=0.0, seed=0, market_sd=0.01) -> pd.DataFrame:
    """A panel: every coin at every 4h timestamp; y = market + specific; mu = ic-correlated with the specific part;
    sigma = the row's true specific sd (heteroscedastic across coins and across time)."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-01-01", periods=n_days * per_day, freq="4h")
    time = np.repeat(t.to_numpy(), coins)
    coin = np.tile([f"C{i:02d}" for i in range(coins)], len(t))
    scale = np.tile(np.linspace(0.5, 2.0, coins), len(t)) * np.repeat(np.exp(0.3 * rng.standard_normal(len(t))), coins)
    spec = 0.01 * scale * rng.standard_normal(len(time))
    signal = ic * spec + np.sqrt(1 - ic ** 2) * 0.01 * scale * rng.standard_normal(len(time))
    market = np.repeat(market_sd * rng.standard_normal(len(t)), coins)
    return pd.DataFrame({"time": time, "coin": coin, "mu": signal * max(ic, 0.05), "sigma": 0.01 * scale,
                         "y_ret": market + spec, "ret_1d": 0.02 * rng.standard_normal(len(time)),
                         "asym": 0.01 * rng.standard_normal(len(time))})


def test_a_score_without_signal_wins_half_the_days_and_loses_the_cost():
    f = frame(ic=0.0)
    for days in (dm.absolute_days(f, f["mu"]), dm.neutral_days(f, f["mu"])):
        s = dm.day_summary(days)
        assert s["n_days"] == 120 and s["win_lo"] < 0.5 < s["win_hi"] and s["mean_lo"] < 0 < s["mean_hi"]
        assert s["wilson_lo"] < s["win_share"] < s["wilson_hi"]
    gross, net = dm.absolute_days(f, f["mu"]), dm.absolute_days(f, f["mu"], cost=0.0008)
    assert np.allclose(gross - net, 0.0016)                                 # a row = a round trip = 2 sides
    assert np.allclose(dm.neutral_days(f, f["mu"]) - dm.neutral_days(f, f["mu"], cost=0.0008), 0.0016)
    assert abs(dm.relative_rank_ic(f, f["mu"])) < 0.02


def test_a_score_with_signal_wins_most_days_and_metric_2_ignores_the_market():
    f = frame(ic=0.3)
    s2 = dm.day_summary(dm.neutral_days(f, f["mu"]))
    assert s2["win_share"] > 0.9 and s2["mean_lo"] > 0 and dm.relative_rank_ic(f, f["mu"]) > 0.2
    shifted = f["mu"] + np.repeat(np.random.default_rng(1).normal(size=f["time"].nunique()), 20)
    assert np.allclose(dm.neutral_days(f, shifted), dm.neutral_days(f, f["mu"]))    # ranks inside the timestamp
    assert np.isclose(dm.relative_rank_ic(f, shifted), dm.relative_rank_ic(f, f["mu"]))
    assert not len(dm.neutral_days(f, np.ones(len(f))))                             # a constant score: no book
    assert dm.day_summary(dm.neutral_days(f, -f["mu"]))["win_share"] < 0.1          # the reversed ranking loses


def test_rows_without_a_score_do_not_trade_and_short_days_are_dropped():
    f = frame(n_days=4, coins=5)
    score = f["mu"].to_numpy().copy()
    score[: 6 * 5] = np.nan                                                        # the whole first day
    assert len(dm.absolute_days(f, score)) == 3
    score[6 * 5: 6 * 5 + 28] = 0.0                                                 # day 2 keeps 2 rows < MIN_ROWS
    assert len(dm.absolute_days(f, score)) == 2


def test_paired_days_and_the_accept_rule():
    f = frame(ic=0.3)
    model = dm.neutral_days(f, f["mu"])
    same = dm.paired_days(model, model)
    assert same["d_mean"] == 0 and same["d_win"] == 0 and not same["wins"]         # a tie is not a win
    noise = dm.neutral_days(f, f["ret_1d"])
    assert dm.paired_days(model, noise)["wins"] and not dm.paired_days(noise, model)["wins"]
    rep = dm.day_report(f, frame(ic=0.3, seed=5))
    assert rep["neutral"]["model"]["beats_every_baseline"] and set(rep["absolute"]) == {"model", *dm.ABSOLUTE_BASELINES}
    assert set(rep["neutral"]) == {"model", *dm.NEUTRAL_BASELINES}
    assert "model_minus_this" in rep["neutral"]["momentum_1d"]
    assert rep["absolute_relative_sign"]["gross"]["win_share"] > 0.9               # market-neutral by construction
    assert dm.accept([rep, rep])["neutral"]["accepted"] and not dm.accept([rep])["neutral"]["accepted"]   # rule C4
    text = dm.render(rep, "TEST")
    assert "Metric 1" in text and "Metric 2" in text and "always_long" in text and "WIN" in text
    with pytest.raises(KeyError, match="lacks"):
        dm.day_report(f.drop(columns=["asym"]))


def test_baselines_are_what_they_say_and_the_asymmetry_is_oriented_on_val():
    f = frame(n_days=30)
    f["asym"] = -(f["y_ret"] - f.groupby("time")["y_ret"].transform("mean"))       # a perfect REVERSED relative score
    b = dm.baseline_scores(f, f)
    assert (b["always_long"] == 1).all() and np.allclose(b["momentum_1d"], -b["reversal_1d"])
    market = f.groupby("time")["ret_1d"].transform("mean")
    assert np.array_equal(b["yesterday_market_sign"], np.sign(market))
    assert dm.orient(f, f["asym"]) == -1.0 and np.allclose(b["last_bar_asymmetry"], -f["asym"])
    assert np.allclose(dm.baseline_scores(f, None)["last_bar_asymmetry"], f["asym"])    # no VAL: as it is


def test_uncertainty_use_reads_a_useful_sd_and_fits_thresholds_on_val():
    f, val = frame(ic=0.3), frame(ic=0.3, seed=9)
    r = uu.uncertainty_report(f, val)
    sd = r["sd_ranks_error"]["close"]
    assert sd["across_coins"] > 0.15 and sd["across_time"] > 0.05 and sd["pooled"] > 0.15   # sigma = the true sd
    flat = uu.sd_ranks_error(f.assign(sigma=1.0))
    assert np.isnan(flat["across_coins"]) and np.isnan(flat["pooled"])              # a constant sd ranks nothing
    cal = r["sharp_and_calibrated"]["close"]
    assert 0.0 <= cal["t_sharp"] <= 1.0 and 0.0 < cal["coverage_80"] < 1.0
    bias = r["direction_bias"]
    assert 0.4 < bias["share_pred_up"] < 0.6 and bias["one_sign_timestamps"] < 0.05
    sel = r["confidence_selection"]
    assert sel["all_rows"]["rows_share"] == 1.0 and 0.1 < sel["top_confidence"]["rows_share"] < 0.3
    assert 0.1 < sel["low_sd"]["rows_share"] < 0.3 and "neutral_gross" in sel["top_abs_mu"]
    ranked = sel["ranked_by_mu_over_sd"]
    assert ranked["rows_share"] == 1.0 and ranked["absolute_gross"] == sel["all_rows"]["absolute_gross"]   # same signs
    assert ranked["neutral_gross"]["win_share"] > 0.8
    assert "T-sharp" in uu.render(r) and "top_confidence" in uu.render(r)
