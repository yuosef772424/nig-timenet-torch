"""The scorecard sections: days as the unit of judgment, VAL-only thresholds, and the long / short exp() payoff on a
known answer (symmetric log moves with no skill: a short loses about s^2 / 2 per trade, a long gains it)."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval.battery import full_battery
from nig_timenet_torch.eval.battery_scorecard import (
    day_profile,
    payoff_asymmetry,
    scorecard,
    scorecard_sections,
    scorecard_verdict,
    selections,
    uncertainty_buckets,
)
from nig_timenet_torch.eval.day_blocks import DAY_BLOCK_DAYS


def _frame(n: int = 6000, seed: int = 0, skill: float = 0.0, s: float = 0.05, per_day: int = 24) -> pd.DataFrame:
    """Log moves l ~ N(0, s) (symmetric); y_ret = e^l - 1 (what the adapter decodes). ``skill`` > 0 makes p_dir
    lean toward the true sign. mu is the predicted SIMPLE move, sigma its width as the adapter decodes it:
    (1 + mu) (e^s - 1)."""
    rng = np.random.default_rng(seed)
    lg = rng.normal(0.0, s, n)
    y = np.expm1(lg)
    z = skill * np.sign(lg) + rng.normal(0.0, 1.0, n)
    p_dir = 1.0 / (1.0 + np.exp(-z))
    mu = np.expm1(0.01 * z)
    times = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(n) // per_day, unit="D") \
        + pd.to_timedelta(np.arange(n) % per_day, unit="h")
    return pd.DataFrame({"step": 1, "mu": mu, "sigma": (1.0 + mu) * np.expm1(s), "p_dir": p_dir, "y_ret": y,
                         "dir_true": np.sign(y).astype(int), "last_ret": 0.0,
                         "coin": rng.choice(["A", "B", "C"], n), "end_time": times})


def test_day_profile_counts_days_tail_and_drawdown():
    days = np.repeat(np.arange(20), 5).astype("datetime64[D]")
    pnl = np.full(100, 0.01)
    pnl[:5] = -0.10                      # day 0 is the one bad day
    p = day_profile(pnl, days, level=0.95)
    assert p["days"] == 20 and p["win_days"] == pytest.approx(19 / 20)
    assert p["worst_day"] == pytest.approx(-0.10) and p["cvar"] == pytest.approx(-0.10)
    assert p["tail_loss_share"] == pytest.approx(1.0)       # the worst 5 % of days hold every loss
    assert p["max_dd"] == pytest.approx(0.10)
    assert p["mean_day"] == pytest.approx((19 * 0.01 - 0.10) / 20)
    assert p["mean_wo_best"] < p["mean_day"]


def test_day_profile_busy_days_expose_a_side_that_loses_when_crowded():
    """Thin days win, crowded days lose: the equal-weight mean day is positive, the busy-day mean is not."""
    days = np.concatenate([np.arange(10), np.repeat(np.arange(10, 20), 20)]).astype("datetime64[D]")
    pnl = np.concatenate([np.full(10, 0.02), np.full(200, -0.005)])
    p = day_profile(pnl, days, level=0.95)
    assert p["mean_day"] > 0 > p["mean_busy"]
    assert p["mean_busy"] == pytest.approx(-0.005)


def test_day_profile_trivial_baseline_constant_book_has_a_zero_width_ci():
    days = np.repeat(np.arange(4 * DAY_BLOCK_DAYS), 3).astype("datetime64[D]")
    p = day_profile(np.full(days.shape[0], 0.002), days, level=0.95)
    assert p["ci_lo"] == pytest.approx(0.002) and p["ci_hi"] == pytest.approx(0.002)
    assert p["tail_loss_share"] == 0.0 and p["max_dd"] == 0.0


def test_selections_take_their_thresholds_from_val_only():
    val, test = _frame(seed=1), _frame(seed=2)
    a = selections(val, test)
    shifted = test.assign(mu=test["mu"] * 10.0)                   # TEST moves grow: VAL edges stay where they were
    b = selections(val, shifted)
    assert b["top 5% |mu|"].sum() > 3 * a["top 5% |mu|"].sum()
    assert a["all rows"].sum() == int((test["dir_true"] != 0).sum())


def test_asymmetry_known_answer_short_pays_half_the_variance():
    """No skill, symmetric log moves: the log part is ~0 on both sides, the exp part is ~+s^2/2 for a long and
    ~-s^2/2 for a short, and the model's sigma predicts it."""
    s = 0.05
    val, test = _frame(n=40000, seed=3, s=s), _frame(n=40000, seed=4, s=s)
    st = payoff_asymmetry(val, test, cost=0.0)["stats"]
    for side, sign in (("LONG", 1.0), ("SHORT", -1.0)):
        r = st[f"{side} all rows"]
        assert abs(r["log_part"]) < 0.0015
        assert r["exp_part"] == pytest.approx(sign * s * s / 2, rel=0.15)
        assert r["pred_exp_part"] == pytest.approx(sign * s * s / 2, rel=0.05)
    assert st["SHORT all rows"]["loss_win"] > st["LONG all rows"]["loss_win"]   # same hit rate, bigger losses


def test_scorecard_flags_skill_and_refuses_noise():
    good = scorecard(_frame(seed=5, skill=2.0), _frame(seed=6, skill=2.0), cost=0.0008)
    noise = scorecard(_frame(seed=7), _frame(seed=8), cost=0.0008)
    assert good["stats"]["all rows | BOTH"]["ci_lo"] > 0
    assert noise["stats"]["all rows | BOTH"]["ci_lo"] < 0
    assert scorecard_verdict(good)["passed"] is True
    assert scorecard_verdict(noise)["passed"] is False


def test_uncertainty_buckets_show_the_short_exp_cost_growing_with_sigma():
    rng = np.random.default_rng(9)
    parts = []
    for s in (0.01, 0.03, 0.06, 0.09, 0.12):
        f = _frame(n=3000, seed=int(rng.integers(1e6)), s=s)
        parts.append(f)
    test = pd.concat(parts, ignore_index=True)
    st = uncertainty_buckets(test, test, cost=0.0008)["stats"]
    exp_costs = [st[f"sigma q{i}"]["short_exp_cost"] for i in range(1, 6)]
    assert exp_costs == sorted(exp_costs, reverse=True)   # more negative as sigma rises


@pytest.mark.slow  # rule E3: fast suite under 100 s on CI (owner 2026-11-19); still runs in CI
def test_sections_join_the_full_battery_and_skip_without_timestamps():
    val, test = _frame(seed=10), _frame(seed=11)
    out = full_battery(val, test, cost=0.0008)
    for key in ("scorecard", "buckets", "asymmetry", "scorecard_verdict"):
        assert key in out and out[key]["rows"]
    assert list(out)[-1] == "verdict"
    assert scorecard_sections(val, test.drop(columns="end_time"), cost=0.0008) == {}
