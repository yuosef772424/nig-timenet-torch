"""T-profit sections (rule C1): one ``test_trivial_baseline_*`` per statistic the battery adds, stating what a
constant / shuffled / always-up model scores. Tables are synthetic with a KNOWN answer; no model."""

import numpy as np
import pandas as pd
import pytest

from nig_timenet_torch.eval import battery_profit
from nig_timenet_torch.eval.battery_profit import limit_highlow, profit_close, profit_sections, sigma_sizing

COST = 0.0008


def _table(n: int = 8000, seed: int = 0, edge: float = 0.0, drift: float = 0.0, hetero: bool = False) -> pd.DataFrame:
    """y_ret ~ N(drift, s); the intrabar high / low bracket the close. p_dir is right w.p. 0.5 + edge."""
    rng = np.random.default_rng(seed)
    sigma = np.exp(rng.uniform(np.log(0.002), np.log(0.04), n)) if hetero else rng.uniform(0.004, 0.02, n)
    y = drift + (sigma if hetero else 0.01) * rng.standard_normal(n)
    high = np.maximum(y, 0) + np.abs(rng.normal(0, 0.005, n))
    low = np.minimum(y, 0) - np.abs(rng.normal(0, 0.005, n))
    right = rng.random(n) < 0.5 + edge
    side = np.where(right, np.sign(y), -np.sign(y))
    return pd.DataFrame({
        "step": 1, "mu": 0.5 * side * 0.005, "sigma": sigma if hetero else np.full(n, 0.01),
        "p_dir": 0.5 + 0.2 * side, "y_ret": y, "dir_true": np.sign(y).astype(int),
        "last_ret": rng.normal(0, 0.01, n), "coin": rng.choice(["A", "B", "C"], n),
        "end_time": pd.date_range("2026-01-01", periods=n, freq="1h").values,
        "pred_high": np.full(n, 0.01), "pred_low": np.full(n, -0.01), "fut_high": high, "fut_low": low,
        "sigma_high": 0.005, "sigma_low": 0.005})


def test_trivial_baseline_profit_close():
    noise = _table(edge=0.0)
    st = profit_close(noise, COST)["stats"]
    floor = -2 * COST
    for name in ("model (close p_dir)", "always up", "always down", "shuffled direction",
                 "last candle sign (momentum)"):
        s = st[name]
        assert abs(s["avg"] - floor) < 0.0006, name                    # no skill: every line pays only the cost
        assert s["ci_lo"] < floor < s["ci_hi"], name                   # and the CI contains that number
    assert st["always up"]["avg"] + st["always down"]["avg"] == pytest.approx(-4 * COST)
    good = profit_close(_table(edge=0.15), COST)["stats"]
    assert good["model (close p_dir)"]["ci_lo"] > 0.0
    for name in ("always up", "always down", "shuffled direction"):
        assert good[name]["avg"] < good["model (close p_dir)"]["avg"]
        assert good[name]["ci_lo"] < 0.0
    drift = profit_close(_table(edge=0.0, drift=0.002), COST)["stats"]
    assert drift["always up"]["ci_lo"] > 0 > drift["model (close p_dir)"]["avg"] - 0.0015   # a drift pays buy-and-hold


def test_profit_close_ci_widens_with_the_number_of_variants():
    t = _table(edge=0.05)
    one, many = profit_close(t, COST, n_variants=1)["stats"], profit_close(t, COST, n_variants=20)["stats"]
    m1, m20 = one["model (close p_dir)"], many["model (close p_dir)"]
    assert m20["level"] == pytest.approx(1 - 0.05 / 20) and m1["level"] == pytest.approx(0.95)
    assert m20["ci_hi"] - m20["ci_lo"] > m1["ci_hi"] - m1["ci_lo"]
    with pytest.raises(ValueError, match="n_variants"):
        profit_close(t, COST, n_variants=0)


def test_profit_close_without_timestamps_or_momentum_still_works():
    t = _table().drop(columns=["end_time", "last_ret"])
    st = profit_close(t, COST)["stats"]
    assert "last candle sign (momentum)" not in st and st["model (close p_dir)"]["n"] > 50


def test_trivial_baseline_limit_highlow():
    t = _table()
    st = limit_highlow(t, COST)["stats"]
    # a constant prediction IS the constant-offset baseline (same orders), and an uninformed random one scores alike
    assert st["LONG at pred_low - model"]["avg"] == pytest.approx(st["LONG at pred_low - constant offset"]["avg"])
    assert st["LONG at pred_low - model"]["fill"] == pytest.approx(st["LONG at pred_low - constant offset"]["fill"])
    rng = np.random.default_rng(1)
    noisy = t.assign(pred_low=-np.abs(rng.normal(0.01, 0.004, len(t))),
                     pred_high=np.abs(rng.normal(0.01, 0.004, len(t))))
    ns = limit_highlow(noisy, COST)["stats"]
    for side in ("LONG at pred_low", "SHORT at pred_high"):
        assert abs(ns[f"{side} - model"]["avg"] - ns[f"{side} - shuffled predictions"]["avg"]) < 0.0015
        assert abs(ns[f"{side} - model"]["avg"] - ns[f"{side} - constant offset"]["avg"]) < 0.0015
    # an oracle that knows this bar's low fills every order and beats the shuffled / constant lines
    oracle = limit_highlow(t.assign(pred_low=t["fut_low"], pred_high=t["fut_high"]), COST)["stats"]
    assert oracle["LONG at pred_low - model"]["fill"] == pytest.approx(1.0)
    for base in ("shuffled predictions", "constant offset"):
        assert oracle["LONG at pred_low - model"]["avg"] > oracle[f"LONG at pred_low - {base}"]["avg"]
        assert oracle["SHORT at pred_high - model"]["avg"] > oracle[f"SHORT at pred_high - {base}"]["avg"]


def test_limit_highlow_skips_without_high_low_and_never_trades_a_limit_through_the_market():
    t = _table()
    assert "SKIPPED" in limit_highlow(t.drop(columns=["pred_high"]), COST)["title"]
    inverted = t.assign(pred_low=0.01, pred_high=-0.01)      # buy limit above / short limit below the last close
    sec = limit_highlow(inverted, COST)
    assert all("(n<50)" in r or "fill=0.0%" in r for r in sec["rows"] if "model" in r)


def test_trivial_baseline_sigma_sizing():
    noise = sigma_sizing(_table(edge=0.1), COST)["stats"]          # sigma independent of the outcome
    sized, flat, shuf = (noise[k]["sharpe"] for k in
                         ("sized by 1/sigma", "equal weight", "sized by shuffled 1/sigma"))
    assert abs(sized - flat) < 0.04 and abs(sized - shuf) < 0.04   # an uninformative sigma changes nothing
    # the move size scales with sigma; cost 0 because 1/sigma sizing also scales the cost
    informed = sigma_sizing(_table(edge=0.1, hetero=True), 0.0)["stats"]
    assert informed["sized by 1/sigma"]["sharpe"] > informed["equal weight"]["sharpe"] + 0.02
    assert informed["sized by 1/sigma"]["sharpe"] > informed["sized by shuffled 1/sigma"]["sharpe"] + 0.02


def test_profit_sections_bundle_and_defaults():
    out = profit_sections(_table(), COST, n_variants=3)
    assert set(out) == {"profit_close", "profit_limit", "profit_sigma"}
    assert all(s["rows"] for s in out.values())
    assert battery_profit.DEFAULT_SIDES == 2
